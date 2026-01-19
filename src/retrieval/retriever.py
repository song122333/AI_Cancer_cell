# src/retrieval/retriever.py

"""
Retrieval 모듈 - Bio-Medical Specialized
기능:
1. Clinical Index 검색 (FAISS: 가이드라인, FDA 문서)
2. Literature Index 검색 (FAISS: PubMed 논문, 교과서)
3. Wikipedia 검색 (바이오마커, 약물 일반 정보)
4. Dense Reranking (Context 정확도 향상)
"""

from typing import List, Dict, Optional
import re
import wikipediaapi

from .embeddings import Embedder
from .vector_store import FaissIndex

# ---------------------------------------------------------
# Wikipedia API 설정
# ---------------------------------------------------------
wiki = wikipediaapi.Wikipedia(
    language="en",
    user_agent="bio-medical-rag/1.0 (internal-research)"
)

# ---------------------------------------------------------
# 1. FAISS 기반 검색 (내부 DB / 논문)
# ---------------------------------------------------------

def search_faiss_index(
    query: str,
    embedder: Embedder,
    index: FaissIndex,
    top_k: int = 5,
    source_tag: str = "InternalDB"
) -> List[Dict]:
    """
    FAISS 인덱스에서 유사한 청크를 검색합니다.
    """
    q_emb = embedder.encode(query)[0]
    results = index.search(q_emb, top_k=top_k)
    
    # 소스 태그 추가 (LLM이 출처를 알 수 있게)
    for r in results:
        r["source"] = source_tag
        
    return results

# ---------------------------------------------------------
# 2. Wikipedia 키워드 추출 (바이오 용어 특화)
# ---------------------------------------------------------

_STOPWORDS = {
    "what", "which", "that", "this", "these", "those",
    "who", "whom", "whose", "where", "when", "why", "how",
    "does", "do", "did", "is", "are", "was", "were", "be",
    "an", "a", "the", "of", "and", "or", "in", "on", "for",
    "to", "from", "with", "as", "by", "about", "describe", "explain"
}

def _tokenize_bio(text: str) -> List[str]:
    """
    [수정됨] 영문+숫자+하이픈 포함 (PD-1, CTLA-4, Phase-3 등 포착)
    """
    # 기존: r"[A-Za-z][A-Za-z\-']*" -> 숫자 누락됨
    # 수정: 숫자 포함 허용
    tokens = re.findall(r"[A-Za-z0-9][A-Za-z0-9\-']*", text)
    return [t for t in tokens if t.strip()]

def extract_candidate_titles(query: str, max_candidates: int = 5) -> List[str]:
    """
    질문에서 위키백과 검색용 키워드(타이틀 후보)를 추출합니다.
    """
    tokens = _tokenize_bio(query)
    if not tokens:
        return []

    tokens_lower = [t.lower() for t in tokens]
    
    candidates = []

    # 1. N-gram (Bigram/Trigram) 우선 - 약물명, 병명은 보통 길다
    # 예: "Lung Cancer", "Immune Checkpoint"
    for i in range(len(tokens) - 1):
        bigram = f"{tokens[i]} {tokens[i+1]}"
        if len(bigram) > 4: 
            candidates.append(bigram)
            
    # 2. 중요한 Unigram (Bio terms)
    # 짧아도 대문자가 섞여있거나 숫자가 있으면 중요 단어일 확률 높음 (p53, PD-1)
    for t in tokens:
        if t.lower() in _STOPWORDS:
            continue
        # 조건: 길이가 4 이상이거나, 숫자가 포함되어 있거나, 모두 대문자인 경우(약어)
        if len(t) >= 4 or any(c.isdigit() for c in t) or t.isupper():
            candidates.append(t)

    # 중복 제거 및 정렬
    seen = set()
    final_candidates = []
    for c in candidates:
        # Wikipedia 형식에 맞게 Title Case 변환
        c_fmt = c.title() if not c.isupper() else c # PD-1 등은 유지
        if c_fmt not in seen:
            seen.add(c_fmt)
            final_candidates.append(c_fmt)

    return final_candidates[:max_candidates]

# ---------------------------------------------------------
# 3. Wikipedia 검색 실행
# ---------------------------------------------------------

def search_wikipedia_chunks(query: str, max_pages: int = 3) -> List[Dict]:
    """
    추출된 키워드로 위키백과를 검색하고 본문을 가져옵니다.
    """
    chunks = []
    candidates = extract_candidate_titles(query, max_candidates=max_pages)

    if not candidates:
        return chunks

    for title in candidates:
        page = wiki.page(title)
        if not page.exists():
            continue

        # 문단 단위 분할 (너무 짧은 문단 제외)
        paragraphs = [p for p in page.text.split("\n") if len(p.strip()) > 50]
        
        # 상위 3개 문단만 사용 (Introduction이 가장 중요)
        for p in paragraphs[:3]:
            chunks.append({
                "text": p,
                "source": f"Wikipedia ({title})",
                "title": title
            })

    return chunks

# ---------------------------------------------------------
# 4. Dense Reranking (유사도 재정렬)
# ---------------------------------------------------------

def rerank_results(
    query: str,
    candidates: List[Dict],
    embedder: Embedder,
    top_k: int = 8
) -> List[Dict]:
    """
    다양한 소스(FAISS, Wiki)에서 가져온 청크들을 
    질문과의 유사도 순으로 다시 정렬합니다.
    """
    if not candidates:
        return []

    q_emb = embedder.encode(query)[0]
    
    scored_items = []
    for item in candidates:
        # 텍스트 임베딩 계산 (캐싱되어 있다면 더 좋음)
        c_text = item.get("text", "")
        c_emb = embedder.encode(c_text)[0]
        
        # Cosine Similarity (Normalized vector assumed)
        score = float((q_emb * c_emb).sum())
        item["score"] = score
        scored_items.append(item)

    # 점수 내림차순 정렬
    scored_items.sort(key=lambda x: x["score"], reverse=True)
    
    return scored_items[:top_k]

# ---------------------------------------------------------
# 5. Main Context Builder
# ---------------------------------------------------------

def get_relevant_context(
    query: str,
    embedder: Embedder,
    clinical_index: Optional[FaissIndex] = None,   # 임상 가이드라인
    literature_index: Optional[FaissIndex] = None, # 논문
    use_wiki: bool = True
) -> str:
    """
    최종 RAG Context 생성 함수
    """
    
    all_chunks = []

    # 1. 임상 가이드라인 검색 (가장 신뢰도 높음)
    if clinical_index:
        guideline_chunks = search_faiss_index(
            query, embedder, clinical_index, top_k=4, source_tag="Clinical_Guideline"
        )
        all_chunks.extend(guideline_chunks)

    # 2. 논문 검색 (최신 연구)
    if literature_index:
        paper_chunks = search_faiss_index(
            query, embedder, literature_index, top_k=4, source_tag="PubMed_Paper"
        )
        all_chunks.extend(paper_chunks)

    # 3. 위키백과 검색 (배경 지식 보완)
    if use_wiki:
        wiki_chunks = search_wikipedia_chunks(query, max_pages=2)
        all_chunks.extend(wiki_chunks)

    # 4. 통합 Reranking
    # 검색된 모든 문서 중 질문과 가장 관련성 높은 순서로 정렬
    best_chunks = rerank_results(query, all_chunks, embedder, top_k=6)

    # 5. 프롬프트 주입용 텍스트 생성
    formatted_context = []
    for chunk in best_chunks:
        source = chunk.get("source", "Unknown")
        text = chunk.get("text", "").strip()
        formatted_context.append(f"[{source}]\n{text}")

    if not formatted_context:
        return "No relevant documents found."

    return "\n\n".join(formatted_context)