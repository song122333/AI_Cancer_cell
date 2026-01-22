"""
Medical AI Advisor 실행 스크립트
<파이프라인>
입력: run.py에서 질문을 입력받음

검색: retriever.py가 FAISS DB에서 검색

토론: solver.py에서 프롬프트에 따라 세번 llm 호출
-Mechanism(기전): 약리 작용 분석
-Clinical(임상): 가이드 라인 적용
-Safety(안전): 위험 요소 경고

종합: prompts.py와 solver.py에서 3명의 의견을 하나로 조율

출력: run.py에서 프롬프트의 output가이드라인을 따라서 최종결과 출력

"""

import os
import sys
import time
from dotenv import load_dotenv

from src.retrieval.embeddings import SolarEmbedder
from src.retrieval.vector_store import FaissIndex
from src.retrieval.retriever import get_relevant_context
from src.llm.solver import run_multidisciplinary_debate

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
load_dotenv()

# ---------------------------------------------------
# 1. 데이터 경로 설정 (단일 경로 통합)
# ---------------------------------------------------
INDEX_PATH = "data/Cancer_cell_merged.faiss"
META_PATH = "data/Cancer_cell_merged.jsonl"

# ---------------------------------------------------
# 2. 리소스 로딩 함수
# ---------------------------------------------------
def load_resources():
    print("\n[System] Initializing Medical AI Advisor...")
    
    # 1. 임베더 로딩
    print("[System] Loading Embedder (Solar)...")
    try:
        embedder = SolarEmbedder()
    except Exception as e:
        print(f"[Critical Error] Embedder 로딩 실패: {e}")
        sys.exit(1)

    # 2. Vector DB 로딩
    vector_db = None
    
    if os.path.exists(INDEX_PATH) and os.path.exists(META_PATH):
        print(f"[System] Loading Vector DB form {INDEX_PATH}...")
        try:
            vector_db = FaissIndex(INDEX_PATH, META_PATH)
        except Exception as e:
            print(f"[Error] DB 로딩 중 오류 발생: {e}")
            vector_db = None
    else:
        print(f"[Warning] DB 파일을 찾을 수 없습니다 ({INDEX_PATH}). 검색 기능 없이 실행됩니다.")

    print("[System] Initialization Complete.\n")
    
    return embedder, vector_db

# ---------------------------------------------------
# 3. 메인 실행 루프
# ---------------------------------------------------

def main():
    embedder, vector_db= load_resources()

    print("="*70)
    print("    Multi-Agent Cancer Tumor Board AI (Debate Mode)")
    print("   (Mechanism vs Clinical vs Safety -> Final Verdict)")
    print("="*70)

    while True:
        try:
            user_query = input("\nUser Query: ").strip()
            
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
                vector_db=vector_db,
                use_wiki=True
            )

            # 2. 토론 실행 (Solver)
            print("   ↳ [Solver] Running multidisciplinary debate...")
            debate_result = run_multidisciplinary_debate(user_query, context)

            elapsed = time.time() - start_time

            # 3. 결과 출력
            print("\n" + "="*70)
            print(f"<Tumor Board Report>")
            print("="*70)

            #각 전문가 의견 요약 출력
            print("\n[1. Mechanism Opinion]")
            print(debate_result["mechanism"][:500] + "...\n(See full logs for details)")

            print("\n[2. Clinical Opinion]")
            print(debate_result["clinical"][:500] + "...\n(See full logs for details)")

            print("\n[3. Safety Opinion]")
            print(debate_result["safety"][:500] + "...\n(See full logs for details)")

            # 최종 결론 출력
            print("\n" + "*"*70)
            print("[CHIEF ONCOLOGIST FINAL VERDICT]")
            print("*"*70)
            print(debate_result["final_verdict"])
            print("*"*70)

        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"\n[Error] {e}")

if __name__ == "__main__":
    main()