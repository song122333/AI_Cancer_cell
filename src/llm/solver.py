# src/llm/solver.py

import os
import requests
import yaml
from typing import Dict
from dotenv import load_dotenv

from src.llm.prompts import AGENT_PROMPTS

load_dotenv()

# ------------------------------------------------
# 1. Upstage API 설정
# ------------------------------------------------

CONFIG_PATH = "configs.yaml"

if os.path.exists(CONFIG_PATH):
    with open("configs.yaml", "r", encoding="utf-8") as f:
        _cfg = yaml.safe_load(f)
else:
    _cfg = {
        "upstage": {
            "base_url": "https://api.upstage.ai/v1/chat/completions",
            "model_name": "solar-pro2",
        }
    }

UPSTAGE_API_KEY = os.getenv("UPSTAGE_API_KEY")
USER_AGENT = os.getenv("USER_AGENT", "medical-ai-advisor/1.0")
BASE_URL = _cfg["upstage"]["base_url"]
MODEL_NAME = _cfg["upstage"]["model_name"]


def call_solar(prompt: str, temperature: float = 0.1, max_tokens: int = 1500) -> str:
    """
    Upstage Chat 모델 호출 래퍼
    """
    if UPSTAGE_API_KEY is None:
        raise RuntimeError("UPSTAGE_API_KEY is not set. Check your .env file.")

    headers = {
        "Authorization": f"Bearer {UPSTAGE_API_KEY}",
        "Content-Type": "application/json",
        "User-Agent": USER_AGENT,
    }

    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    try:
        resp = requests.post(BASE_URL, headers=headers, json=payload, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        content = data["choices"][0]["message"]["content"]
        return content
    except Exception as e:
        return f"[API Error] {str(e)}"


# ------------------------------------------------
# 2. Debate Logic (토론 시뮬레이션)
# ------------------------------------------------

def run_multidisciplinary_debate(
    query: str,
    context: str
) -> Dict[str, str]:
    """
    3명의 전문가(기전, 임상, 안전성)가 각자 리포트를 작성하고,
    마지막에 의장(Moderator)이 이를 종합하여 최종 답변을 생성합니다.
    """
    
    reports = {}
    
    # 1. 전문가 3인 의견 청취
    specialists = ["mechanism", "clinical", "safety"]
    
    print(f"\n   [Debate] Convening Tumor Board for: '{query}'")

    for role in specialists:
        print(f"   [Debate] Consulting {role.capitalize()} Specialist...")
        
        # 각 전문가용 프롬프트 로드
        prompt_template = AGENT_PROMPTS.get(role)
        if not prompt_template:
            reports[role] = "Error: Prompt template not found."
            continue

        final_prompt = prompt_template.format(context=context, question=query)
        
        # LLM 호출
        response = call_solar(final_prompt, temperature=0.1, max_tokens=1000)
        reports[role] = response

    # 2. 의장(Moderator) 종합
    print(f"   [Debate] Chief Oncologist is synthesizing the final verdict...")
    
    moderator_template = AGENT_PROMPTS.get("moderator")
    if moderator_template:
        moderator_prompt = moderator_template.format(
            question=query,
            mechanism_report=reports.get("mechanism", ""),
            clinical_report=reports.get("clinical", ""),
            safety_report=reports.get("safety", "")
        )
        final_verdict = call_solar(moderator_prompt, temperature=0.1, max_tokens=1500)
    else:
        final_verdict = "Error: Moderator prompt not found."

    # 결과 반환
    return {
        "mechanism": reports.get("mechanism", ""),
        "clinical": reports.get("clinical", ""),
        "safety": reports.get("safety", ""),
        "final_verdict": final_verdict
    }