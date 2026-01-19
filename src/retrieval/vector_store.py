# src/retrieval/vector_store.py
# FAISS 인덱스 로더 
"""
FAISS 기반 dense vector store
- 이미 만들어진 index.faiss를 읽어서 검색만 수행
- search()는 검색 결과의 (score, id) 반환
"""

import faiss
import numpy as np
import json
from typing import List, Dict


class FaissIndex:
    """
    FAISS index + metadata(jsonl) 로더
    """

    def __init__(self, index_path: str, meta_path: str):
        self.index_path = index_path
        self.meta_path = meta_path

        # 실제 벡터 인덱스 로드
        self.index = faiss.read_index(index_path)

        # 메타데이터 로드: id → {"text":..., "domain":...}
        self.metadata = []
        with open(meta_path, "r", encoding="utf-8") as f:
            for line in f:
                self.metadata.append(json.loads(line))

        print(f"[FAISS] index 로드 완료: {index_path}")
        print(f"[META]  {len(self.metadata)}개 로드 완료")

    def search(self, query_emb: np.ndarray, top_k=5) -> List[Dict]:
        """
        query_emb: (1, D) 임베딩 벡터
        return: [{ "text":..., "score":.., **meta }, ...]
        """

        if query_emb.ndim == 1:
            query_emb = query_emb.reshape(1, -1)

        scores, idxs = self.index.search(query_emb, top_k)

        results = []
        for score, idx in zip(scores[0], idxs[0]):
            meta = self.metadata[idx]
            results.append({
                "score": float(score),
                "text": meta["text"],
                **meta
            })

        return results
