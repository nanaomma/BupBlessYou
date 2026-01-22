"""Database connection management"""
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from src.config.settings import settings

# Create database engine (using psycopg driver)
# Convert postgresql:// to postgresql+psycopg:// for psycopg3 driver
sync_database_url = settings.database_url.replace(
    "postgresql://",
    "postgresql+psycopg://"
)

engine = create_engine(
    sync_database_url,
    pool_pre_ping=True,
    pool_size=10,
    max_overflow=20
)

# Create session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base class for models
Base = declarative_base()


def get_db():
    """Dependency to get database session"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# === Async Database Connection (for async operations) ===

# Convert DATABASE_URL to async format
# postgresql:// -> postgresql+psycopg://
async_database_url = settings.database_url.replace(
    "postgresql://",
    "postgresql+psycopg://"
)

# Create async database engine
async_engine = create_async_engine(
    async_database_url,
    pool_pre_ping=True,
    pool_size=10,
    max_overflow=20,
    echo=False,
)

# Create async session factory
async_session_maker = async_sessionmaker(
    bind=async_engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autocommit=False,
    autoflush=False,
)
