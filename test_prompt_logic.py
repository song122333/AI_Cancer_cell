import os
import json
import numpy as np
import faiss
from langchain_upstage import UpstageEmbeddings, ChatUpstage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

# 1. 환경 설정
load_dotenv()

# [수정] 로컬 경로 설정 (내 컴퓨터 기준)
# 현재 이 파이썬 파일이 있는 폴더를 기준으로 'data' 폴더를 찾습니다.
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_PATH = os.path.join(CURRENT_DIR, "data")

# 파일 경로 지정
CORPUS_PATH = os.path.join(BASE_PATH, "ewha_corpus_final.jsonl")
INDEX_PATH = os.path.join(BASE_PATH, "ewha_index_final.faiss")

# 경로 확인용 출력 (실행 시 경로가 맞는지 눈으로 확인하세요)
print(f" 데이터 경로 확인: {BASE_PATH}")
if os.path.exists(CORPUS_PATH):
    print(" jsonl 파일 찾음!")
else:
    print(f" jsonl 파일 없음: {CORPUS_PATH}")

# 2. 모델 설정 (Solar Pro 2 사용)
llm = ChatUpstage(model="solar-pro2", temperature=0)
embedding_model = UpstageEmbeddings(model="solar-embedding-1-large")

# 3. V2 프롬프트 (부정문 논리 강화)
EWHA_AGENT_STRICT_V2 = """[SYSTEM]
당신은 이화여자대학교 학칙을 매우 정확하게 알고 있는 조교입니다.
아래에 주어진 학칙 발췌문(context)을 근거로 객관식 문제에 답해야 합니다.

[CONTEXT]
{context}

[QUESTION]
{question}

[OPTIONS]
{options}

[INSTRUCTION]
당신은 답변을 내리기 전에 반드시 아래의 [THOUGHT PROCESS]를 순서대로 수행해야 합니다.

[THOUGHT PROCESS]
1. **질문 유형 분석:** 질문이 '옳은 것'을 묻는지, '옳지 않은(틀린/제외한/아닌/잘못된) 것'을 묻는지 명확히 정의하세요.
   - 부정 표현이 있다면 **"부정형 질문"**이라고 명시하세요.
   
2. **선택지 검증:** 각 선택지 (A)~(D)를 하나씩 읽고, 위 [CONTEXT]의 내용과 일치하는지 O/X로 판별하세요.
   - (A): [내용] -> 문맥과 일치함(O) / 일치하지 않음(X) / 근거 없음(?) (근거: 제0조 0항)
   
3. **정답 도출:**
   - 긍정형 질문이면: 문맥과 **일치하는(O)** 선택지를 고르세요.
   - 부정형 질문이면: 다음 우선순위에 따라 정답을 선택하세요.
     1. 문맥에 명시된 내용과 **정면으로 배치되거나 틀린(X)** 내용 (최우선)
     2. 문맥에 없더라도, 학칙의 다른 조건(졸업 요건 등)과 **논리적으로 양립할 수 없는** 내용
     3. 문맥에 전혀 언급되지 않아 알 수 없는(?) 내용

[OUTPUT FORMAT]
위 [THOUGHT PROCESS]의 내용을 한국어로 요약해서 설명한 뒤, 마지막 줄에 정답을 출력하세요.

**[ANSWER]: (X) 선택지내용**
"""

# 4. 검색 및 RAG 함수 클래스
class SimpleRAG:
    def __init__(self):
        print("Loading Corpus and Index...")
        # Corpus 로드
        self.docs = []
        with open(CORPUS_PATH, "r", encoding="utf-8") as f:
            for line in f:
                self.docs.append(json.loads(line))
        
        # FAISS 인덱스 로드
        if not os.path.exists(INDEX_PATH):
            raise FileNotFoundError("FAISS 인덱스가 없습니다. 임베딩 과정을 먼저 수행하세요.")
        # [수정 후] 이 부분 전체를 복사해서 덮어쓰세요
        # FAISS 인덱스 로드 (한글 경로 우회 트릭)
        if not os.path.exists(INDEX_PATH):
            raise FileNotFoundError("FAISS 인덱스가 없습니다. 임베딩 과정을 먼저 수행하세요.")
        
        # 1. 현재 작업 위치 저장
        current_cwd = os.getcwd()
        
        try:
            # 2. 데이터 폴더로 이동 (파이썬은 한글 경로 이동 가능)
            os.chdir(os.path.dirname(INDEX_PATH))
            
            # 3. 파일명만 가지고 로드 (한글 경로가 안 들어가므로 에러 안 남)
            self.index = faiss.read_index(os.path.basename(INDEX_PATH))
            print(f"✅ Loaded {len(self.docs)} docs and {self.index.ntotal} vectors.")
            
        finally:
            # 4. 원래 위치로 복귀
            os.chdir(current_cwd)
        print(f"✅ Loaded {len(self.docs)} docs and {self.index.ntotal} vectors.")

    def retrieve(self, query, top_k=3):
        # 쿼리 임베딩
        q_vec = embedding_model.embed_query(query)
        q_vec = np.array(q_vec, dtype="float32").reshape(1, -1)
        
        # 정규화 (L2) - 학습때 정규화 했으면 여기서도 필수
        faiss.normalize_L2(q_vec)
        
        # 검색
        scores, idxs = self.index.search(q_vec, top_k)
        
        retrieved_texts = []
        print(f"\n🔍 [Retrieval] Top {top_k} Documents:")
        for rank, idx in enumerate(idxs[0]):
            doc = self.docs[idx]
            print(f"   {rank+1}. [{doc['section']}] (Score: {scores[0][rank]:.4f})")
            retrieved_texts.append(doc['text'])
            
        return "\n\n".join(retrieved_texts)

    def answer(self, question, options):
        # 1. 검색
        context = self.retrieve(question)
        
        # 2. 생성
        chain = ChatPromptTemplate.from_template(EWHA_AGENT_STRICT_V2) | llm | StrOutputParser()
        
        print("\n🤖 [Generation] Thinking...")
        response = chain.invoke({
            "context": context,
            "question": question,
            "options": options
        })
        return response

# 5. 테스트 실행
def run_3_questions_test():
    rag = SimpleRAG()
    
    # 테스트할 문제들
    questions = [
        {
            "qid": "Q6",
            "q": "1980학년도 이전 입학생에 대하여 적용하는 등급에 따른 성적점으로 잘못 연결된 것은 무엇인가?",
            "opt": "(A) 등급: A+, 성적점: 4\n(B) 등급: A-: 성적점: 3.5\n(C) 등급: B+, 성적점: 3\n(D) 등급: C, 성적점: 2"
        },
        {
            "qid": "Q8",
            "q": "복수전공 신청 자격에 해당하지 않는 것은?",
            "opt": "(A) 1학년을 마친 학생\n(B) 평균 평점이 2.50 이상인 학생\n(C) 졸업 직전 학기에 있는 학생\n(D) 재학생 신분인 경우"
        },
        {
            "qid": "Q20",
            "q": "재학 연한 초과로 제적당하지 않는 경우는?",
            "opt": "(A) 학사 편입\n(B) 복수 전공 중\n(C) 재입학 후 1년 이내\n(D) 휴학 중"
        }
    ]

    for item in questions:
        print("\n" + "="*60)
        print(f"📢 TEST {item['qid']}: {item['q']}")
        print(f"Options:\n{item['opt']}")
        print("-" * 60)
        
        try:
            result = rag.answer(item['q'], item['opt'])
            print("\n🎯 [Final Answer]")
            print(result)
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    run_3_questions_test()
