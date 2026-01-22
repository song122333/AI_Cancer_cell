# src/retrieval/vector_store.py

import faiss
import numpy as np
import json
import os
from typing import List, Dict

class FaissIndex:
    """
    FAISS index + metadata(jsonl) 로더
    - 역할: .faiss 인덱스와 .jsonl 메타데이터를 연결하여 검색 수행
    """

    def __init__(self, index_path: str, meta_path: str):
        self.index_path = index_path
        self.meta_path = meta_path

        # 1. 파일 존재 확인 (디버깅 효율성)
        if not os.path.exists(index_path) or not os.path.exists(meta_path):
            raise FileNotFoundError(f"DB 파일을 찾을 수 없습니다: {index_path} 또는 {meta_path}")

        # 2. FAISS 인덱스 메모리 매핑 로드 (속도 최적화)
        # read_index 대신 read_index를 쓰되, 필요시 mmap 옵션을 고려할 수 있음. 
        # 여기서는 기본 read_index 유지
        self.index = faiss.read_index(index_path)

        # 3. 메타데이터 로드
        self.metadata = []
        with open(meta_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip(): # 빈 줄 에러 방지
                    self.metadata.append(json.loads(line))

        print(f"[FAISS] Index 로드 완료 (Dim: {self.index.d}, Docs: {self.index.ntotal})")
        print(f"[META]  {len(self.metadata)}개 메타데이터 로드 완료")

    def search(self, query_emb: np.ndarray, top_k=5) -> List[Dict]:
        """
        벡터 검색 수행
        """
        # [효율화 1] 데이터 타입 강제 변환 (FAISS는 float32만 처리 가능)
        # float64가 들어오면 내부 변환하느라 느려지거나 에러날 수 있음
        if query_emb.dtype != np.float32:
            query_emb = query_emb.astype(np.float32)

        # 1차원 -> 2차원 변환
        if query_emb.ndim == 1:
            query_emb = query_emb.reshape(1, -1)

        # [효율화 2] 차원 검사 (Crash 방지)
        if query_emb.shape[1] != self.index.d:
            raise ValueError(f"차원 불일치: Index({self.index.d}) vs Query({query_emb.shape[1]})")

        # 검색 수행
        scores, idxs = self.index.search(query_emb, top_k)

        results = []
        # [효율화 3] 유효하지 않은 인덱스(-1) 필터링
        # FAISS는 이웃을 못 찾으면 -1을 반환하는데, 이를 그대로 metadata[-1]로 읽으면 데이터가 꼬임
        for score, idx in zip(scores[0], idxs[0]):
            if idx == -1 or idx >= len(self.metadata):
                continue

            meta = self.metadata[idx]
            results.append({
                "score": float(score),
                "text": meta.get("text", ""), # key error 방지
                **meta
            })

        return results
