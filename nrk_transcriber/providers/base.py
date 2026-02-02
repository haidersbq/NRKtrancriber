"""
Base classes for media providers.

All broadcaster-specific providers inherit from BaseProvider.
"""

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class MediaProgram:
    """Unified metadata for media content across all providers."""

    program_id: str
    title: str
    description: str
    duration_seconds: int
    audio_url: str
    provider: str  # e.g., "nrk", "bbc", "youtube"
    series_title: Optional[str] = None
    image_url: Optional[str] = None
    published_at: Optional[str] = None
    language: Optional[str] = None
    original_url: Optional[str] = None
    episode_number: Optional[str] = None
    # Parsed from URL
    start_time_seconds: Optional[int] = None


@dataclass
class ParsedURL:
    """Result of URL parsing."""

    provider: str
    content_type: str  # "live", "on_demand", "podcast", "video", "direct"
    content_id: str
    original_url: str
    series_id: Optional[str] = None
    timestamp_seconds: Optional[int] = None


class BaseProvider(ABC):
    """
    Abstract base class for all media providers.

    Each broadcaster (NRK, BBC, YouTube, etc.) implements this interface
    to provide a unified way to fetch and transcribe content.
    """

    PROVIDER_ID: str = ""  # e.g., "nrk", "bbc"
    PROVIDER_NAME: str = ""  # e.g., "NRK", "BBC Sounds"
    SUPPORTED_DOMAINS: list[str] = []  # e.g., ["radio.nrk.no", "nrk.no"]
    DEFAULT_LANGUAGE: Optional[str] = None  # e.g., "no", "en"

    @classmethod
    @abstractmethod
    def can_handle(cls, url: str) -> bool:
        """
        Check if this provider can handle the given URL.

        Args:
            url: The URL to check

        Returns:
            True if this provider can handle the URL
        """
        pass

    @abstractmethod
    def parse_url(self, url: str) -> ParsedURL:
        """
        Parse a URL and extract content identifiers.

        Args:
            url: The URL to parse

        Returns:
            ParsedURL with extracted information
        """
        pass

    @abstractmethod
    def get_program(self, url_or_id: str) -> MediaProgram:
        """
        Get program metadata and audio URL.

        Args:
            url_or_id: Either a full URL or a content ID

        Returns:
            MediaProgram with metadata and stream URL
        """
        pass

    def get_live_stream_url(self, channel_id: str) -> str:
        """
        Get URL for a live stream.

        Args:
            channel_id: The channel identifier

        Returns:
            Stream URL

        Raises:
            NotImplementedError if provider doesn't support live streams
        """
        raise NotImplementedError(
            f"{self.PROVIDER_NAME} does not support live streams"
        )

    def search(self, query: str, limit: int = 10) -> list[MediaProgram]:
        """
        Search for content.

        Args:
            query: Search query
            limit: Maximum results

        Returns:
            List of matching programs

        Raises:
            NotImplementedError if provider doesn't support search
        """
        raise NotImplementedError(
            f"{self.PROVIDER_NAME} does not support search"
        )

    @staticmethod
    def parse_timestamp_fragment(fragment: str) -> Optional[int]:
        """
        Parse timestamp from URL fragment (e.g., #t=14m19s).

        Args:
            fragment: URL fragment string

        Returns:
            Timestamp in seconds, or None if not found
        """
        if not fragment:
            return None

        # Match t=14m19s or t=859s or t=14:19
        patterns = [
            r't=(\d+)m(\d+)s',  # t=14m19s
            r't=(\d+)s',  # t=859s
            r't=(\d+):(\d+)',  # t=14:19
            r't=(\d+)',  # t=859
        ]

        for pattern in patterns:
            match = re.search(pattern, fragment)
            if match:
                groups = match.groups()
                if len(groups) == 2:
                    return int(groups[0]) * 60 + int(groups[1])
                else:
                    return int(groups[0])

        return None

    @staticmethod
    def parse_iso_duration(duration_str: str) -> int:
        """
        Parse ISO 8601 duration string to seconds.

        Args:
            duration_str: Duration string like "PT1H30M45S"

        Returns:
            Duration in seconds
        """
        if not duration_str:
            return 0

        match = re.match(
            r'PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?',
            duration_str
        )

        if not match:
            return 0

        hours = int(match.group(1) or 0)
        minutes = int(match.group(2) or 0)
        seconds = int(match.group(3) or 0)

        return hours * 3600 + minutes * 60 + seconds
