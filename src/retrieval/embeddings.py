# # src/retrieval/embeddings.py
# """
# BGE-M3 기반 임베딩 생성기
# - query/text 임베딩 모두 지원
# - retrieval.py와 vector_store.py에서 공통적으로 사용
# """

# from typing import List
# from transformers import AutoTokenizer, AutoModel
# import torch
# import numpy as np


# class Embedder:
#     """
#     BGE-M3 임베딩 모델 래퍼
#     - encode(): 여러 문장을 임베딩하여 numpy array로 반환
#     """

#     def __init__(self, model_name="BAAI/bge-m3"):
#         self.device = "cuda" if torch.cuda.is_available() else "cpu"

#         # 토크나이저 + 모델 로딩
#         self.tokenizer = AutoTokenizer.from_pretrained(model_name)
#         self.model = AutoModel.from_pretrained(model_name).to(self.device)

#     @torch.no_grad()
#     def encode(self, texts: List[str]) -> np.ndarray:
#         """
#         texts: List[str] → 2D numpy array (N x D)
#         """

#         if isinstance(texts, str):
#             texts = [texts]

#         inputs = self.tokenizer(
#             texts,
#             return_tensors="pt",
#             padding=True,
#             truncation=True,
#             max_length=256,
#         ).to(self.device)

#         outputs = self.model(**inputs)
#         # BGE-M3의 CLS 토큰 사용
#         dense = outputs.last_hidden_state[:, 0]

#         # L2 normalize
#         dense = dense / dense.norm(dim=1, keepdim=True)
#         return dense.cpu().numpy()

# src/retrieval/embeddings.py
"""
Upstage Solar Embedding 기반 임베딩 래퍼

- 규정: 오픈소스 임베딩(BGE 등) 사용 금지
- 여기서는 UpstageEmbeddings(model="solar-embedding-1-large")만 사용
- 이 모듈은 "질문 쿼리" 임베딩용으로만 쓰인다.
"""

import os
from typing import List

import numpy as np
from dotenv import load_dotenv
from langchain_upstage import UpstageEmbeddings


class Embedder:
    """
    Upstage 임베딩 모델 래퍼

    - encode(texts: List[str]) -> np.ndarray (N, D)
    - 주 사용처: RAG에서 질문 쿼리 임베딩 생성
    """

    def __init__(self, model_name: str = "solar-embedding-1-large"):
        # .env 로부터 UPSTAGE_API_KEY 로드 (안 되어 있으면 에러)
        load_dotenv()
        api_key = os.getenv("UPSTAGE_API_KEY")
        if not api_key:
            raise ValueError(
                "UPSTAGE_API_KEY 가 설정되어 있지 않습니다. "
                ".env 파일 또는 환경변수에 UPSTAGE_API_KEY를 넣어주세요."
            )

        self.model = UpstageEmbeddings(model=model_name)
        self.dim = None  # 필요하면 나중에 차원 체크용으로 사용

    def encode(self, texts: List[str]) -> np.ndarray:
        """
        texts: List[str] 또는 단일 str
        반환: (N, D) numpy 배열 (L2 정규화된 벡터)

        여기서는 쿼리 임베딩으로만 쓰이므로 embed_query()를 사용한다.
        """

        if isinstance(texts, str):
            texts = [texts]

        vectors = []
        for t in texts:
            vec = self.model.embed_query(t)  # 쿼리용 타워
            vectors.append(vec)

        arr = np.array(vectors, dtype="float32")  # (N, D)

        # L2 정규화 → FAISS IndexFlatIP와 함께 코사인 유사도처럼 사용
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        arr = arr / np.clip(norms, 1e-12, None)

        if self.dim is None:
            self.dim = arr.shape[1]
        elif self.dim != arr.shape[1]:
            raise ValueError(
                f"임베딩 차원 불일치: 기존 dim={self.dim}, 새 dim={arr.shape[1]}"
            )

        return arr
