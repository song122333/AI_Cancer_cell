import re
from dataclasses import dataclass
from typing import Dict, Optional, List

import pandas as pd


@dataclass
class Example:
    """하나의 객관식 문제를 담는 구조체."""
    qid: str
    raw: str          # QUESTION + 선택지 전체 원문
    question: str
    options: Dict[str, str]
    gold: Optional[str] = None   # 정답 (있으면), 예: "D"


# "QUESTION25)" 이런 부분 찾기
QID_PATTERN = re.compile(r"(QUESTION\d+)", re.IGNORECASE)
# "(A) something" 형태의 선택지 찾기
OPTION_PATTERN = re.compile(r"^\(([A-Z])\)\s*(.*)")


def parse_prompt(text: str, gold: Optional[str] = None) -> Example:
    """
    testset.csv의 'prompts' 한 줄을 받아서 Example로 변환.
    """
    raw = text.strip()
    lines = raw.splitlines()

    # 1) QID 추출
    first_line = lines[0]
    qid_match = QID_PATTERN.search(first_line)
    qid = qid_match.group(1) if qid_match else "UNKNOWN"

    # 2) 질문 문장만 추출 (QUESTIONxx) 제거)
    question = re.sub(r"QUESTION\d+\)\s*", "", first_line, flags=re.IGNORECASE).strip()

    # 3) 선택지 파싱
    options: Dict[str, str] = {}
    for line in lines[1:]:
        m = OPTION_PATTERN.match(line.strip())
        if not m:
            continue
        key = m.group(1)      # "A" ~ "D"
        val = m.group(2).strip()
        options[key] = val

    return Example(
        qid=qid,
        raw=raw,
        question=question,
        options=options,
        gold=gold,
    )


def load_examples_from_csv(path: str) -> List[Example]:
    """
    testset.csv 파일을 읽어서 Example 리스트로 변환.
    """
    df = pd.read_csv(path)
    has_gold = "answers" in df.columns

    examples: List[Example] = []

    for _, row in df.iterrows():
        prompt = row["prompts"]
        gold: Optional[str] = None

        if has_gold and isinstance(row["answers"], str):
            # "(D)" → "D"
            gold = row["answers"].strip().strip("()")

        ex = parse_prompt(prompt, gold)
        examples.append(ex)

    return examples