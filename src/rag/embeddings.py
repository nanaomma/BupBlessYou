"""Text embedding utilities"""
from typing import List
from openai import OpenAI
from src.config.settings import settings


class EmbeddingGenerator:
    """
    텍스트 임베딩 생성

    공통 작업: 데이터 수집 및 전처리 팀
    역할:
    - 판결문 텍스트를 벡터로 변환
    - OpenAI text-embedding-3-small 사용 (1536 차원)
    """

    def __init__(self, model_name: str = "text-embedding-3-small"):
        """
        OpenAI embedding 모델 초기화

        Args:
            model_name: OpenAI embedding 모델명 (기본: text-embedding-3-small)
        """
        self.client = OpenAI(api_key=settings.openai_api_key)
        self.model_name = model_name
        self.dimension = 1536  # text-embedding-3-small 차원

    def encode(self, texts: List[str]) -> List[List[float]]:
        """
        텍스트를 벡터로 변환 (배치 처리)

        Args:
            texts: 임베딩할 텍스트 리스트

        Returns:
            벡터 리스트 (각 벡터는 1536 차원)
        """
        # OpenAI API는 자동으로 배치 처리
        response = self.client.embeddings.create(
            model=self.model_name,
            input=texts,
            encoding_format="float"
        )

        # 임베딩 추출
        embeddings = [data.embedding for data in response.data]
        return embeddings

    def encode_query(self, query: str) -> List[float]:
        """
        검색 쿼리를 벡터로 변환

        Args:
            query: 검색 쿼리

        Returns:
            벡터 (1536 차원)
        """
        response = self.client.embeddings.create(
            model=self.model_name,
            input=[query],
            encoding_format="float"
        )

        return response.data[0].embedding
