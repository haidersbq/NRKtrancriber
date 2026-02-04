"""
Database storage for transcription results.

Uses SQLite for local storage with async support.
"""

import asyncio
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

from sqlalchemy import (
    Column,
    DateTime,
    Float,
    Integer,
    String,
    Text,
    create_engine,
    select,
)
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

logger = logging.getLogger(__name__)


class Base(DeclarativeBase):
    """Base class for SQLAlchemy models."""
    pass


class Transcription(Base):
    """Model for storing transcription records."""

    __tablename__ = "transcriptions"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    channel_id: Mapped[str] = mapped_column(String(100), index=True)
    audio_file: Mapped[str] = mapped_column(String(500))
    audio_start_time: Mapped[datetime] = mapped_column(DateTime, index=True)
    audio_end_time: Mapped[datetime] = mapped_column(DateTime)
    text: Mapped[str] = mapped_column(Text)
    segments_json: Mapped[str] = mapped_column(Text)  # JSON-encoded segments
    language: Mapped[str] = mapped_column(String(10))
    language_probability: Mapped[float] = mapped_column(Float)
    duration_seconds: Mapped[float] = mapped_column(Float)
    processing_time_seconds: Mapped[float] = mapped_column(Float)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class TranscriptionDatabase:
    """
    Async database interface for storing and querying transcriptions.
    """

    def __init__(self, database_path: Path):
        """
        Initialize the database.

        Args:
            database_path: Path to the SQLite database file
        """
        self.database_path = Path(database_path)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)

        # Create async engine
        self._engine = create_async_engine(
            f"sqlite+aiosqlite:///{self.database_path}",
            echo=False,
        )
        self._session_factory = async_sessionmaker(
            self._engine,
            class_=AsyncSession,
            expire_on_commit=False,
        )
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize the database schema."""
        if self._initialized:
            return

        async with self._engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

        self._initialized = True
        logger.info(f"Database initialized at {self.database_path}")

    async def save_transcription(self, result) -> int:
        """
        Save a transcription result to the database.

        Args:
            result: TranscriptionResult object

        Returns:
            The ID of the saved record
        """
        if not self._initialized:
            await self.initialize()

        async with self._session_factory() as session:
            transcription = Transcription(
                channel_id=result.channel_id,
                audio_file=str(result.audio_file),
                audio_start_time=result.audio_start_time,
                audio_end_time=result.audio_end_time,
                text=result.text,
                segments_json=json.dumps(
                    [s.to_dict() for s in result.segments],
                    ensure_ascii=False,
                ),
                language=result.language,
                language_probability=result.language_probability,
                duration_seconds=result.duration_seconds,
                processing_time_seconds=result.processing_time_seconds,
            )

            session.add(transcription)
            await session.commit()

            logger.debug(f"Saved transcription {transcription.id}")
            return transcription.id

    async def get_transcription(self, transcription_id: int) -> Optional[Transcription]:
        """Get a transcription by ID."""
        if not self._initialized:
            await self.initialize()

        async with self._session_factory() as session:
            result = await session.get(Transcription, transcription_id)
            return result

    async def get_transcriptions_by_channel(
        self,
        channel_id: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 100,
    ) -> list[Transcription]:
        """
        Get transcriptions for a specific channel.

        Args:
            channel_id: The channel ID to filter by
            start_time: Optional start time filter
            end_time: Optional end time filter
            limit: Maximum number of results

        Returns:
            List of Transcription records
        """
        if not self._initialized:
            await self.initialize()

        async with self._session_factory() as session:
            query = select(Transcription).where(
                Transcription.channel_id == channel_id
            )

            if start_time:
                query = query.where(Transcription.audio_start_time >= start_time)
            if end_time:
                query = query.where(Transcription.audio_end_time <= end_time)

            query = query.order_by(Transcription.audio_start_time.desc()).limit(limit)

            result = await session.execute(query)
            return list(result.scalars().all())

    async def search_transcriptions(
        self,
        query: str,
        channel_id: Optional[str] = None,
        limit: int = 50,
    ) -> list[Transcription]:
        """
        Search transcriptions by text content.

        Args:
            query: Search query string
            channel_id: Optional channel filter
            limit: Maximum results

        Returns:
            List of matching Transcription records
        """
        if not self._initialized:
            await self.initialize()

        async with self._session_factory() as session:
            stmt = select(Transcription).where(
                Transcription.text.contains(query)
            )

            if channel_id:
                stmt = stmt.where(Transcription.channel_id == channel_id)

            stmt = stmt.order_by(Transcription.audio_start_time.desc()).limit(limit)

            result = await session.execute(stmt)
            return list(result.scalars().all())

    async def get_recent_transcriptions(
        self,
        limit: int = 100,
    ) -> list[Transcription]:
        """
        Get recent transcriptions across all channels.

        Args:
            limit: Maximum number of results

        Returns:
            List of Transcription records ordered by most recent first
        """
        if not self._initialized:
            await self.initialize()

        async with self._session_factory() as session:
            query = select(Transcription).order_by(
                Transcription.audio_start_time.desc()
            ).limit(limit)

            result = await session.execute(query)
            return list(result.scalars().all())

    async def get_latest_transcription(
        self,
        channel_id: Optional[str] = None,
    ) -> Optional[Transcription]:
        """Get the most recent transcription."""
        if not self._initialized:
            await self.initialize()

        async with self._session_factory() as session:
            stmt = select(Transcription)

            if channel_id:
                stmt = stmt.where(Transcription.channel_id == channel_id)

            stmt = stmt.order_by(Transcription.audio_start_time.desc()).limit(1)

            result = await session.execute(stmt)
            return result.scalar_one_or_none()

    async def get_statistics(self, channel_id: Optional[str] = None) -> dict:
        """Get transcription statistics."""
        if not self._initialized:
            await self.initialize()

        from sqlalchemy import func

        async with self._session_factory() as session:
            base_query = select(
                func.count(Transcription.id).label("count"),
                func.sum(Transcription.duration_seconds).label("total_duration"),
                func.sum(Transcription.processing_time_seconds).label("total_processing"),
            )

            if channel_id:
                base_query = base_query.where(Transcription.channel_id == channel_id)

            result = await session.execute(base_query)
            row = result.one()

            return {
                "total_transcriptions": row.count or 0,
                "total_audio_duration_seconds": row.total_duration or 0,
                "total_processing_time_seconds": row.total_processing or 0,
            }

    async def delete_old_transcriptions(
        self,
        older_than: datetime,
        channel_id: Optional[str] = None,
    ) -> int:
        """
        Delete transcriptions older than a specified date.

        Returns:
            Number of deleted records
        """
        if not self._initialized:
            await self.initialize()

        from sqlalchemy import delete

        async with self._session_factory() as session:
            stmt = delete(Transcription).where(
                Transcription.audio_start_time < older_than
            )

            if channel_id:
                stmt = stmt.where(Transcription.channel_id == channel_id)

            result = await session.execute(stmt)
            await session.commit()

            deleted_count = result.rowcount
            logger.info(f"Deleted {deleted_count} old transcriptions")
            return deleted_count

    async def close(self) -> None:
        """Close the database connection."""
        await self._engine.dispose()
        logger.info("Database connection closed")
