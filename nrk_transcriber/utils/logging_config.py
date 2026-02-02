"""
Logging configuration for NRK Transcriber.
"""

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.logging import RichHandler


def setup_logging(
    level: str = "INFO",
    log_file: Optional[Path] = None,
    rich_output: bool = True,
) -> logging.Logger:
    """
    Configure logging for the application.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional file to write logs to
        rich_output: Use rich formatting for console output

    Returns:
        Configured root logger
    """
    log_level = getattr(logging, level.upper(), logging.INFO)

    # Create formatters
    file_formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    # Remove existing handlers
    root_logger.handlers.clear()

    # Console handler
    if rich_output:
        console_handler = RichHandler(
            console=Console(stderr=True),
            show_time=True,
            show_path=False,
            rich_tracebacks=True,
            tracebacks_show_locals=True,
        )
        console_handler.setLevel(log_level)
    else:
        console_handler = logging.StreamHandler(sys.stderr)
        console_handler.setFormatter(file_formatter)
        console_handler.setLevel(log_level)

    root_logger.addHandler(console_handler)

    # File handler (optional)
    if log_file:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(file_formatter)
        file_handler.setLevel(log_level)
        root_logger.addHandler(file_handler)

    # Reduce noise from third-party libraries
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("asyncio").setLevel(logging.WARNING)

    return root_logger


class TranscriptionStats:
    """Track and display transcription statistics."""

    def __init__(self):
        self.start_time = datetime.now()
        self.chunks_processed = 0
        self.total_audio_seconds = 0.0
        self.total_processing_seconds = 0.0
        self.errors = 0

    def add_chunk(self, audio_duration: float, processing_time: float) -> None:
        """Record a processed chunk."""
        self.chunks_processed += 1
        self.total_audio_seconds += audio_duration
        self.total_processing_seconds += processing_time

    def add_error(self) -> None:
        """Record an error."""
        self.errors += 1

    @property
    def runtime_seconds(self) -> float:
        """Total runtime in seconds."""
        return (datetime.now() - self.start_time).total_seconds()

    @property
    def realtime_factor(self) -> float:
        """Processing speed relative to realtime."""
        if self.total_processing_seconds == 0:
            return 0.0
        return self.total_audio_seconds / self.total_processing_seconds

    def get_summary(self) -> dict:
        """Get statistics summary."""
        return {
            "runtime_seconds": self.runtime_seconds,
            "chunks_processed": self.chunks_processed,
            "total_audio_seconds": self.total_audio_seconds,
            "total_processing_seconds": self.total_processing_seconds,
            "realtime_factor": self.realtime_factor,
            "errors": self.errors,
        }

    def __str__(self) -> str:
        """Format statistics as string."""
        return (
            f"Chunks: {self.chunks_processed} | "
            f"Audio: {self.total_audio_seconds:.1f}s | "
            f"Processing: {self.total_processing_seconds:.1f}s | "
            f"RTF: {self.realtime_factor:.2f}x | "
            f"Errors: {self.errors}"
        )
