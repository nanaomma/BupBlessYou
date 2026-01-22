"""Application settings and configuration"""
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional
from pydantic import model_validator, Field


class Settings(BaseSettings):
    """Application configuration settings"""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        case_sensitive=False,
        populate_by_name=True,  # alias와 필드 이름 모두 허용
    )

    # Application
    app_name: str = "BupBlessYou"
    environment: str = "development"  # development, staging, production
    debug: bool = True
    log_level: str = "INFO"  # DEBUG, INFO, WARNING, ERROR, CRITICAL

    # Debug Configuration
    debug_graph_execution: bool = True  # Log graph node execution
    debug_agent_state: bool = True      # Log agent state changes
    debug_llm_calls: bool = False       # Log LLM API calls (verbose)
    debug_checkpoint: bool = True       # Log checkpoint operations
    log_to_file: bool = True            # Enable file logging in debug mode

    # LangSmith Configuration (LLM Tracing & Monitoring)
    langsmith_api_key: Optional[str] = None
    langsmith_project: str = "bupblessyou"  # Project name in LangSmith
    langsmith_tracing: bool = False         # Enable LangSmith tracing
    langsmith_endpoint: str = "https://api.smith.langchain.com"


    # Observability & Evaluation (Optional)
    # Phoenix Configuration (Open-source LLM Observability)
    phoenix_enabled: bool = False  # Enable Phoenix tracing
    phoenix_collector_endpoint: str = "http://localhost:6006"  # Phoenix collector URL (local/cloud)
    phoenix_space_id: Optional[str] = None  # Phoenix Cloud space ID (e.g., "U3BhY2U6...")
    phoenix_api_key: Optional[str] = None  # Phoenix Cloud API key (only for cloud version)

    # RAGAS Configuration (RAG Quality Evaluation)
    ragas_enabled: bool = False  # Enable RAGAS evaluation (requires OpenAI API key)

    # LLM API Keys
    openai_api_key: str
    upstage_api_key: str

    # LLM Model Selection
    default_llm_provider: str = "openai"  # "openai" or "upstage"
    openai_model: str = "gpt-4o-mini"
    upstage_model: str = "solar-pro"

    # Database (AWS RDS)
    # Option 1: 직접 DATABASE_URL 설정 (우선순위 높음)
    database_url: Optional[str] = None

    # Option 2: 개별 파라미터로 자동 구성 (database_url이 없을 때)
    db_host: Optional[str] = None
    db_port: int = 5432
    db_name: Optional[str] = None
    db_user: Optional[str] = None
    db_password: Optional[str] = None

    # Vector Database (Pinecone)
    # Common Pinecone (for judgments and criminal law)
    # pinecone_api_key가 없으면 law_pinecone_api_key를 사용 (하위 호환성)
    pinecone_api_key: Optional[str] = None  # 공통 Pinecone API 키 (환경 변수: PINECONE_API_KEY)
    pinecone_environment: Optional[str] = None
    pinecone_index_name: str = "bupblessyou-judgments"  # 공통 Pinecone 인덱스 이름 (환경 변수: PINECONE_INDEX_NAME)
    
    # Law statutes (legacy - optional, 공통 Pinecone 사용 시 불필요)
    law_pinecone_api_key: Optional[str] = None
    law_pinecone_environment: Optional[str] = None
    law_pinecone_index_name: Optional[str] = None
    law_pinecone_namespace: Optional[str] = None

    # Sentencing guidelines
    sentence_pinecone_api_key: str
    sentence_pinecone_environment: str
    sentence_pinecone_index_name: str
    sentence_pinecone_namespace: str

    embedding_provider: str = Field(default="openai")
    embedding_model: str = Field(default="text-embedding-3-small")
    embedding_dimension: int = Field(default=1536)

    # Legal API (Optional)
    legal_api_key: Optional[str] = None

    @model_validator(mode='after')
    def assemble_database_url(self) -> 'Settings':
        # DATABASE_URL이 직접 설정되지 않았으면 개별 파라미터로 구성
        if not self.database_url:
            if self.db_host and self.db_user and self.db_password and self.db_name:
                self.database_url = f"postgresql://{self.db_user}:{self.db_password}@{self.db_host}:{self.db_port}/{self.db_name}"
            else:
                # Fallback for local development if DB settings are missing
                self.database_url = "sqlite:///./local_dev.db"

        # pinecone_api_key가 없으면 law_pinecone_api_key를 사용 (하위 호환성)
        if not self.pinecone_api_key and self.law_pinecone_api_key:
            self.pinecone_api_key = self.law_pinecone_api_key

        return self



# Global settings instance
settings = Settings()
