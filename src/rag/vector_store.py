"""Vector store management with Pinecone"""
import time
from typing import List, Dict, Any, Optional
from pinecone import Pinecone, ServerlessSpec
from src.config.settings import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


class VectorStore:
    """
    Pinecone 벡터 스토어 관리

    공통 작업: 데이터 수집 및 전처리 팀
    역할:
    - 판결문 벡터화 및 저장 (OpenAI text-embedding-3-small, 1536차원)
    - 유사 판례 검색
    - 벡터 인덱스 관리
    """

    def __init__(self):
        """
        Pinecone 클라이언트 초기화

        - OpenAI text-embedding-3-small 모델 (1536 차원) 사용
        - Pinecone Inference API 활용 (임베딩 자동 생성)
        """
        self.client = Pinecone(api_key=settings.pinecone_api_key)
        self.index_name = settings.pinecone_index_name
        self.dimension = 1536  # OpenAI text-embedding-3-small

        # 인덱스 연결 (존재하는 경우)
        if self.index_name in self.client.list_indexes().names():
            self.index = self.client.Index(self.index_name)
        else:
            self.index = None

    def search_similar_cases(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
        include_metadata: bool = True
    ) -> List[Dict[str, Any]]:
        """
        유사 판례 검색

        Args:
            query_embedding: 쿼리 벡터 (1536 차원)
            top_k: 반환할 결과 수
            filters: 검색 필터 조건 (예: {"crime_type": "사기", "year": {"$gte": 2020}})
            include_metadata: 메타데이터 포함 여부

        Returns:
            유사 판례 리스트
            [
                {
                    "id": "case_123",
                    "score": 0.95,
                    "metadata": {
                        "case_name": "사기 사건",
                        "crime_type": "사기",
                        "sentence": "징역 2년",
                        "facts": "...",
                        "reasoning": "...",
                        "year": 2023
                    }
                },
                ...
            ]

        """
        if not self.index:
            logger.error(f"Index '{self.index_name}' not found")
            raise ValueError(f"Index '{self.index_name}' not found. Create index first.")

        # 검색 시작 로깅 및 시간 측정
        logger.debug(f"Searching similar cases (top_k={top_k}, filters={filters})")
        start_time = time.time()

        # Pinecone 검색
        results = self.index.query(
            vector=query_embedding,
            top_k=top_k,
            filter=filters,
            include_metadata=include_metadata
        )

        # 검색 완료 로깅
        duration = time.time() - start_time
        logger.debug(f"Search completed (duration={duration:.3f}s, results={len(results.matches)})")

        # 결과 포맷팅
        similar_cases = []
        for match in results.matches:
            case = {
                "id": match.id,
                "score": match.score,
            }
            if include_metadata and match.metadata:
                case["metadata"] = match.metadata

            similar_cases.append(case)

        return similar_cases

    async def search_similar_cases_async(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
        include_metadata: bool = True
    ) -> List[Dict[str, Any]]:
        """
        비동기 유사 판례 검색 (asyncio.to_thread 사용)
        """
        import asyncio
        return await asyncio.to_thread(
            self.search_similar_cases,
            query_embedding,
            top_k,
            filters,
            include_metadata
        )

    def upsert_judgments(
        self,
        judgments: List[Dict[str, Any]],
        batch_size: int = 100
    ):
        """
        판결문 벡터 저장

        Args:
            judgments: 판결문 리스트
                [
                    {
                        "id": "case_123",
                        "embedding": [0.1, 0.2, ...],  # 1536 차원 벡터
                        "metadata": {
                            "case_name": "사기 사건",
                            "crime_type": "사기",
                            "sentence": "징역 2년",
                            "facts": "...",
                            "reasoning": "...",
                            "year": 2023
                        }
                    },
                    ...
                ]
            batch_size: 배치 크기 (Pinecone 권장: 100)
        """
        if not self.index:
            logger.error(f"Index '{self.index_name}' not found for upsert operation")
            raise ValueError(f"Index '{self.index_name}' not found. Create index first.")

        logger.info(f"Starting upsert of {len(judgments)} judgments (batch_size={batch_size})")

        # 배치 단위로 업로드
        for i in range(0, len(judgments), batch_size):
            batch = judgments[i:i + batch_size]

            # Pinecone 형식으로 변환
            vectors = [
                {
                    "id": judgment["id"],
                    "values": judgment["embedding"],
                    "metadata": judgment.get("metadata", {})
                }
                for judgment in batch
            ]

            # 업서트
            self.index.upsert(vectors=vectors)
            logger.debug(f"Upserted batch {i//batch_size + 1} ({len(vectors)} vectors)")

        logger.info(f"Successfully upserted {len(judgments)} judgments")

    def delete_index(self):
        """인덱스 삭제"""
        if self.index_name in self.client.list_indexes().names():
            logger.warning(f"Deleting index '{self.index_name}'")
            self.client.delete_index(self.index_name)
            self.index = None
            logger.info(f"Index '{self.index_name}' deleted successfully")
        else:
            logger.warning(f"Index '{self.index_name}' does not exist, skipping deletion")

    def create_index(
        self,
        dimension: int = 1536,
        metric: str = "cosine",
        cloud: str = "aws",
        region: str = "us-east-1"
    ):
        """
        새 인덱스 생성

        Args:
            dimension: 벡터 차원 (기본: 1536, OpenAI text-embedding-3-small)
            metric: 거리 측정 방법 ("cosine", "euclidean", "dotproduct")
            cloud: 클라우드 제공자 ("aws", "gcp", "azure")
            region: 리전
        """
        # 기존 인덱스 확인
        if self.index_name in self.client.list_indexes().names():
            logger.info(f"Index '{self.index_name}' already exists, connecting to it")
            self.index = self.client.Index(self.index_name)
            return

        # Serverless 스펙으로 인덱스 생성
        logger.info(f"Creating new index '{self.index_name}' (dimension={dimension}, metric={metric}, cloud={cloud}, region={region})")
        self.client.create_index(
            name=self.index_name,
            dimension=dimension,
            metric=metric,
            spec=ServerlessSpec(
                cloud=cloud,
                region=region
            )
        )

        # 인덱스 연결
        self.index = self.client.Index(self.index_name)
        logger.info(f"Index '{self.index_name}' created and connected successfully")
