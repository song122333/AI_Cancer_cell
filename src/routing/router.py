import re
from typing import Literal

# 프로젝트에서 사용하는 LLM 호출 함수 import (기존 유지)
try:
    from src.llm.solver import call_solar
except ImportError:
    call_solar = None  # 테스트용

# ==========================================
# 1. 키워드 정의 (암/면역학 도메인 특화)
# ==========================================

# 1차 필터링: 암/면역학 관련 질문인지 확인하는 키워드
TARGET_KEYWORDS = [
    "암", "종양", "cancer", "tumor", "oncology",
    "면역", "immune", "T세포", "T cell", "T-cell",
    "PD-1", "PD-L1", "CTLA-4", "TIM-3", "LAG-3",
    "탈진", "exhaustion", "항암", "drug", "treatment",
    "억제제", "inhibitor", "바이오마커", "biomarker"
]

# 2차 분류: 세부 의도별 키워드 (LLM 실패 시 Fallback용)
INTENT_KEYWORDS = {
    # 기전/기초과학 (Mechanism) -> 면역학자 페르소나
    "mechanism": [
        "기전", "원리", "경로", "pathway", "signaling", "expression",
        "발현", "분자", "molecular", "receptor", "ligand", "interaction",
        "toxy", "nfat", "전사인자", "mechanism"
    ],
    # 임상/약물추천 (Clinical) -> 종양내과 의사 페르소나
    "clinical": [
        "약물", "치료제", "drug", "medicine", "approval", "fda",
        "승인", "임상", "clinical trial", "efficacy", "효능", "효과",
        "생존율", "survival", "dose", "용량", "1차 치료", "first-line"
    ],
    # 안전성/부작용 (Safety) -> 안전성 검토자 페르소나
    "safety": [
        "부작용", "독성", "side effect", "adverse", "toxicity",
        "safety", "risk", "위험", "합병증", "cytokine storm",
        "상호작용", "interaction", "contraindication", "금기"
    ]
}

# ==========================================
# 2. LLM 기반 의도 분류 (Few-Shot Prompt 수정)
# ==========================================

def llm_classify_intent(query: str) -> str:
    """
    LLM을 사용하여 질문의 의도를 [mechanism, clinical, safety] 중 하나로 분류
    """
    if not call_solar:
        return None

    FEW_SHOT = """
### EXAMPLES ###

[Mechanism Example 1]
Query: "PD-1과 PD-L1이 결합하면 T세포 내에서 어떤 신호 전달 경로가 차단되나요?"
Intent: mechanism

[Mechanism Example 2]
Query: "Explain the role of TOX transcription factor in T-cell exhaustion."
Intent: mechanism


[Clinical Example 1]
Query: "현재 FDA 승인을 받은 흑색종 치료용 면역관문억제제 목록을 알려줘."
Intent: clinical

[Clinical Example 2]
Query: "What is the recommended dosage of Pembrolizumab for NSCLC patients?"
Intent: clinical


[Safety Example 1]
Query: "CTLA-4 억제제 투여 시 발생할 수 있는 자가면역성 대장염 증상은?"
Intent: safety

[Safety Example 2]
Query: "Is there any drug interaction between corticosteroids and immunotherapy?"
Intent: safety
"""

    prompt = f"""
You are an intent classifier for an AI Oncologist System.
Classify the user's query into one of three categories:
1. mechanism: Questions about biological pathways, molecular interactions, or basic immunology.
2. clinical: Questions about drug recommendations, FDA approvals, clinical trials, or efficacy.
3. safety: Questions about side effects, toxicity, drug interactions, or risks.

Use the examples as reference.

{FEW_SHOT}

### TASK ###
Classify the following query:

Query:
{query}

Respond with only one category name in lowercase (mechanism, clinical, or safety).
"""

    try:
        output = call_solar(prompt).strip().lower()
        for category in ["mechanism", "clinical", "safety"]:
            if category in output:
                return category
        return None
    except Exception as e:
        print(f"LLM Classification Error: {e}")
        return None


# ==========================================
# 3. 라우팅 로직 (메인 함수)
# ==========================================

def is_relevant_query(text: str) -> bool:
    """
    1차 필터: 이 질문이 우리 시스템(암/면역)과 관련이 있는가?
    """
    text_lower = text.lower()
    # 키워드 기반 필터링 (필요시 LLM으로 교체 가능하나 속도를 위해 키워드 추천)
    return any(kw in text_lower for kw in TARGET_KEYWORDS)


def route_intent(query: str) -> Literal["mechanism", "clinical", "safety", "general"]:
    """
    사용자 질문을 분석하여 적절한 전문가 에이전트에게 라우팅합니다.
    
    Returns:
        - mechanism: 면역학자 (기전 설명)
        - clinical: 의사 (약물 추천)
        - safety: 안전성 검토자 (부작용 경고)
        - general: 관련 없음 또는 일반 대화
    """
    
    # 1. 도메인 관련성 체크
    if not is_relevant_query(query):
        return "general"

    # 2. LLM 기반 의도 분류 시도
    intent = llm_classify_intent(query)
    if intent:
        return intent

    # 3. Fallback: 키워드 기반 분류 (LLM 실패 시)
    query_lower = query.lower()
    scores = {k: 0 for k in INTENT_KEYWORDS.keys()}
    
    for category, keywords in INTENT_KEYWORDS.items():
        for kw in keywords:
            if kw in query_lower:
                scores[category] += 1
    
    # 점수가 가장 높은 카테고리 반환
    best_intent = max(scores, key=scores.get)
    
    # 키워드 매칭이 전혀 없으면 기본값은 'clinical' (가장 일반적인 질문으로 가정)
    if scores[best_intent] == 0:
        return "clinical"
        
    return best_intent

# ==========================================
# 테스트 코드
# ==========================================
if __name__ == "__main__":
    test_queries = [
        "안녕하세요, 점심 뭐 먹을까요?", # -> general
        "PD-1 억제제가 암세포를 공격하는 원리가 뭐야?", # -> mechanism
        "폐암 환자에게 쓸 수 있는 FDA 승인 약물 추천해줘", # -> clinical
        "옵디보랑 여보이 같이 쓰면 부작용 심해?", # -> safety
    ]
    
    print("--- Routing Test ---")
    for q in test_queries:
        result = route_intent(q)
        print(f"Q: {q[:20]}... -> Route: {result}")