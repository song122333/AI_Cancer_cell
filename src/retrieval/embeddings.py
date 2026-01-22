# # src/retrieval/embeddings.py

import os
import numpy as np
from typing import List, Union
from dotenv import load_dotenv
from langchain_upstage import UpstageEmbeddings

class SolarEmbedder:
    """
    Upstage Solar Embedding Wrapper (Optimized)
    """

    def __init__(self, model_name: str = "solar-embedding-1-large"):
        load_dotenv()
        api_key = os.getenv("UPSTAGE_API_KEY")
        
        if not api_key:
            raise ValueError("UPSTAGE_API_KEY가 설정되지 않았습니다. .env 파일을 확인해주세요.")

        # API Key
        self.model = UpstageEmbeddings(
            model=model_name, 
            upstage_api_key=api_key
        )

    def encode(self, texts: Union[str, List[str]]) -> np.ndarray:
        """
        texts: 질문(Query) 텍스트 또는 텍스트 리스트
        반환: (N, D) L2 Normalized Numpy Array
        """
        # 1. 단일 문자열을 리스트로 통일
        if isinstance(texts, str):
            texts = [texts]

        # 2. 빈 텍스트 필터링
        valid_texts = [t for t in texts if t.strip()]
        if not valid_texts:
            return np.array([])

        try:
            # 3. 임베딩 생성
            # RAG 질문용이므로 embed_query 사용
            vectors = [self.model.embed_query(t) for t in valid_texts]

            # 4. Numpy 변환
            arr = np.array(vectors, dtype="float32")

            # 5. L2 정규화
            norms = np.linalg.norm(arr, axis=1, keepdims=True)
            return arr / np.clip(norms, 1e-12, None)

        except Exception as e:
            print(f"[Embedding Error] 변환 중 오류 발생: {e}")
            return np.array([])