# evaluate_testset.py
"""
testset.csv + 2_final.csv 평가 스크립트

- testset.csv: gold 정답 포함 (prompts, answers)
- 2_final.csv: run.py가 생성한 예측 (qid, your_answer)

출력:
- 전체 정확도
- Ewha / MMLU 별 정확도
- (옵션) MMLU 도메인별 정확도
"""
import re
from collections import defaultdict
import pandas as pd
from src.parsing.parse_testset import load_examples_from_csv
from src.routing.router import route_source, classify_domain

TESTSET_PATH = "data/testset.csv"
PRED_PATH = "2_final.csv"
# TESTSET_PATH = "data/private_test.csv"
# PRED_PATH = "2_final_private.csv"


def normalize_choice(s):
    """
    '(D)', 'D', '**[ANSWER]: (C) ...**' 등에서
    선택지 문자 하나(A~J)만 안전하게 뽑아내는 유틸.
    """
    if not isinstance(s, str):
        return None

    s = s.strip().upper()
    # 괄호 안의 문자 우선
    m = re.search(r"\(([A-J])\)", s)
    if m:
        return m.group(1)

    # 그냥 문자 하나만 있는 경우
    m2 = re.search(r"\b([A-J])\b", s)
    if m2:
        return m2.group(1)

    return None


def main():
    print("Loading testset & predictions...")

    # 1) gold 포함 Example 리스트
    # (load_examples_from_csv 함수는 외부 모듈에 정의되어 있어야 함)
    examples = load_examples_from_csv(TESTSET_PATH)

    # 2) 예측 파일 (run.py 결과)
    pred_df = pd.read_csv(PRED_PATH)
    # qid -> predicted choice 매핑 딕셔너리
    pred_map = {
        row["qid"]: normalize_choice(row["your_answer"])
        for _, row in pred_df.iterrows()
    }

    total = 0
    correct = 0
    # 틀린 문제 정보를 저장할 리스트
    wrong_examples = [] 

    # 소스별(Ewha / MMLU) 통계
    src_stats = {
        "ewha": {"total": 0, "correct": 0},
        "mmlu": {"total": 0, "correct": 0},
    }

    # 도메인별 통계 (MMLU용)
    domain_stats = defaultdict(lambda: {"total": 0, "correct": 0})

    missing_preds = []

    for ex in examples:
        gold = normalize_choice(ex.gold)
        pred = pred_map.get(ex.qid, None)

        if gold is None:
            continue

        if pred is None:
            missing_preds.append(ex.qid)
            src = route_source(ex)
            src_stats[src]["total"] += 1
            total += 1
            continue

        total += 1
        src = route_source(ex)
        src_stats[src]["total"] += 1

        if pred == gold:
            correct += 1
            src_stats[src]["correct"] += 1
        else:
            # 틀린 문제 정보 저장
            wrong_examples.append({
                "qid": ex.qid,
                "source": src,
                "prompt_snippet": ex.raw.split('\n')[0][:50] + "...", 
                "gold": gold,
                "pred": pred,
            })


        # MMLU인 경우 도메인 단위 통계도 집계
        if src == "mmlu":
            domain = classify_domain(ex)
            domain_stats[domain]["total"] += 1
            if pred == gold:
                domain_stats[domain]["correct"] += 1

    # -------------------- 결과 출력 --------------------

    if total == 0:
        print("평가할 샘플이 없습니다. testset/gold를 확인하세요.")
        return

    overall_acc = correct / total * 100.0

    print("\n================ EVALUATION RESULT ================")
    print(f"전체 샘플 수: {total}")
    print(f"전체 정답 수: {correct}")
    print(f"➡ 전체 정확도: {overall_acc:.2f}%")

    # 틀린 문제 상세 출력 
    if wrong_examples:
        print("\n================ FAILED EXAMPLES (틀린 문제) ================")
        print(f"총 {len(wrong_examples)}개 문제의 오답을 분석합니다:")
        
        # DataFrame으로 만들어서 깔끔하게 출력
        df_wrong = pd.DataFrame(wrong_examples)
        
        # Ewha와 MMLU로 나누어 정렬
        df_ewha_wrong = df_wrong[df_wrong['source'] == 'ewha']
        df_mmlu_wrong = df_wrong[df_wrong['source'] == 'mmlu']
        
        print("\n--- Ewha 학칙 오답 ---")
        if not df_ewha_wrong.empty:
            print(df_ewha_wrong.to_markdown(index=False, numalign="left", stralign="left"))
        else:
            print("Ewha 섹션에서 오답 없음!")

        print("\n--- MMLU 외부 KB 오답 ---")
        if not df_mmlu_wrong.empty:
            print(df_mmlu_wrong.to_markdown(index=False, numalign="left", stralign="left"))
        else:
            print("MMLU 섹션에서 오답 없음!")
        
        print("====================================================")


    print("\n[소스별(Ewha / MMLU) 정확도]")
    for src, st in src_stats.items():
        if st["total"] == 0:
            print(f"- {src}: 샘플 없음")
            continue
        acc = st["correct"] / st["total"] * 100.0
        print(f"- {src}: {st['correct']} / {st['total']}  ({acc:.2f}%)")

    print("\n[MMLU 도메인별 정확도]")
    if not domain_stats:
        print("- MMLU 샘플 없음")
    else:
        for d, st in domain_stats.items():
            if st["total"] == 0:
                continue
            acc = st["correct"] / st["total"] * 100.0
            print(f"- {d}: {st['correct']} / {st['total']}  ({acc:.2f}%)")

    if missing_preds:
        print(f"\n⚠ 예측이 비어 있는 qid 개수: {len(missing_preds)}")
        print(f"  예시 일부: {missing_preds[:5]}")

    print("\n===================================================\n")


if __name__ == "__main__":
    main()