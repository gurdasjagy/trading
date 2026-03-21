"""
Database connection and session management using SQLAlchemy async.

Dependencies: aiosqlite (for SQLite async support), sqlalchemy[asyncio]
Install with: pip install sqlalchemy aiosqlite
"""

from contextlib import asynccontextmanager
from typing import AsyncGenerator, Optional

from loguru import logger
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from .models import Base


class DatabaseManager:
    """Manages database connections and provides session factory."""

    _instance: Optional["DatabaseManager"] = None

    def __init__(self, database_url: str = "sqlite+aiosqlite:///./data/trading_bot.db"):
        # Convert plain sqlite:/// to sqlite+aiosqlite:/// if needed
        if database_url.startswith("sqlite:///") and "+aiosqlite" not in database_url:
            database_url = database_url.replace("sqlite:///", "sqlite+aiosqlite:///")

        self._engine = create_async_engine(database_url, echo=False)
        self._session_factory = async_sessionmaker(
            self._engine, class_=AsyncSession, expire_on_commit=False
        )

    @classmethod
    def get_instance(cls, database_url: Optional[str] = None) -> "DatabaseManager":
        if cls._instance is None:
            cls._instance = cls(database_url or "sqlite+aiosqlite:///./data/trading_bot.db")
        return cls._instance

    async def create_tables(self) -> None:
        """Create all tables if they don't exist."""
        async with self._engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        logger.info("Database tables created/verified")

    @asynccontextmanager
    async def session(self) -> AsyncGenerator[AsyncSession, None]:
        """Provide a transactional session context."""
        async with self._session_factory() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise

    async def close(self) -> None:
        await self._engine.dispose()


async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """Dependency-injection helper that yields a database session."""
    db = DatabaseManager.get_instance()
    async with db.session() as session:
        yield session
