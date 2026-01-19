"""
Medical AI Advisor 실행 스크립트
- Interactive Mode (사용자와 실시간 대화)
- Pipeline: Router -> Retriever -> Solver
"""

import os
import sys
import time
from dotenv import load_dotenv

# 필요한 모듈 임포트 (우리가 수정한 파일들)
from src.retrieval.embeddings import Embedder
from src.retrieval.vector_store import FaissIndex
from src.retrieval.retriever import get_relevant_context
from src.llm.solver import run_multidisciplinary_debate
# 경고 메시지 제어
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
load_dotenv()

# ---------------------------------------------------
# 1. 데이터 경로 설정
# ---------------------------------------------------
# 실제 구축한 FAISS 인덱스 경로로 수정해주세요.
# 만약 파일이 없다면 None으로 처리되어 Wikipedia 검색만 수행합니다.

CLINICAL_INDEX_PATH = "data/Cancer_cell_merged.faiss"   # 가이드라인/임상 데이터
CLINICAL_META_PATH = "data/Cancer_cell_merged.jsonl"

LITERATURE_INDEX_PATH = "data/Cancer_cell_merged.faiss" # 논문 데이터
LITERATURE_META_PATH = "data/Cancer_cell_merged.jsonl"

# ---------------------------------------------------
# 2. 리소스 로딩 함수
# ---------------------------------------------------

def load_resources():
    print("\n[System] Initializing Medical AI Advisor...")
    
    # 1. 임베더 로딩 (시간이 조금 걸림)
    print("[System] Loading Embedder (BGE-M3)...")
    embedder = Embedder()

    # 2. Vector DB 로딩
    clinical_index = None
    if os.path.exists(CLINICAL_INDEX_PATH):
        print(f"[System] Loading Clinical Index from {CLINICAL_INDEX_PATH}...")
        clinical_index = FaissIndex(CLINICAL_INDEX_PATH, CLINICAL_META_PATH)
    else:
        print("[Warning] Clinical Index not found. Skipping...")

    literature_index = None
    if os.path.exists(LITERATURE_INDEX_PATH):
        print(f"[System] Loading Literature Index from {LITERATURE_INDEX_PATH}...")
        literature_index = FaissIndex(LITERATURE_INDEX_PATH, LITERATURE_META_PATH)
    else:
        print("[Warning] Literature Index not found. Skipping...")

    print("[System] Initialization Complete.\n")
    return embedder, clinical_index, literature_index

# ---------------------------------------------------
# 3. 메인 실행 루프
# ---------------------------------------------------

def main():
    embedder, clinical_index, literature_index = load_resources()

    print("="*70)
    print("   🏥 Multi-Agent Cancer Tumor Board AI (Debate Mode)")
    print("   (Mechanism vs Clinical vs Safety -> Final Verdict)")
    print("="*70)

    while True:
        try:
            user_query = input("\n🧑‍⚕️ User Query: ").strip()
            
            if not user_query:
                continue
            if user_query.lower() in ["q", "quit", "exit"]:
                break

            start_time = time.time()

            # 1. 검색 (Retrieval) - 모든 전문가가 공유할 Context
            print("   ↳ [Retriever] Searching clinical guidelines & papers...")
            context = get_relevant_context(
                query=user_query,
                embedder=embedder,
                clinical_index=clinical_index,
                literature_index=literature_index,
                use_wiki=True
            )

            # 2. 토론 실행 (Solver)
            # 여기서는 Router 없이 바로 Debate 함수를 호출합니다.
            debate_result = run_multidisciplinary_debate(user_query, context)

            elapsed = time.time() - start_time

            # 3. 결과 출력 (탭 형식처럼 구분해서 보여줌)
            print("\n" + "="*70)
            print(f"📋 **Tumor Board Report** (Generated in {elapsed:.2f}s)")
            print("="*70)

            # (A) 각 전문가 의견 요약 출력 (선택 사항 - 너무 길면 생략 가능)
            print("\n--- 🧬 [1. Mechanism Opinion] ---")
            print(debate_result["mechanism"][:500] + "...\n(See full logs for details)")

            print("\n--- 💊 [2. Clinical Opinion] ---")
            print(debate_result["clinical"][:500] + "...\n(See full logs for details)")

            print("\n--- ⚠️ [3. Safety Opinion] ---")
            print(debate_result["safety"][:500] + "...\n(See full logs for details)")

            # (B) 최종 결론 출력
            print("\n" + "*"*70)
            print("🎓 **CHIEF ONCOLOGIST FINAL VERDICT**")
            print("*"*70)
            print(debate_result["final_verdict"])
            print("*"*70)

        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"\n[Error] {e}")

if __name__ == "__main__":
    main()