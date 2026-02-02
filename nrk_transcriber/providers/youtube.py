"""
YouTube provider using yt-dlp for metadata and audio extraction.

Supports YouTube videos, shorts, and playlists.
"""

import json
import logging
import re
import subprocess
from typing import Optional
from urllib.parse import urlparse, parse_qs

from .base import BaseProvider, MediaProgram, ParsedURL
from .registry import ProviderRegistry

logger = logging.getLogger(__name__)


@ProviderRegistry.register
class YouTubeProvider(BaseProvider):
    """
    Provider for YouTube content using yt-dlp.

    Supports:
    - Regular videos (youtube.com/watch?v=...)
    - Short URLs (youtu.be/...)
    - Shorts (youtube.com/shorts/...)
    - Timestamps in URL (?t=123 or &t=1m30s)
    """

    PROVIDER_ID = "youtube"
    PROVIDER_NAME = "YouTube"
    SUPPORTED_DOMAINS = ["youtube.com", "www.youtube.com", "youtu.be", "m.youtube.com"]
    DEFAULT_LANGUAGE = None  # YouTube has many languages

    @classmethod
    def can_handle(cls, url: str) -> bool:
        """Check if URL is from YouTube."""
        try:
            parsed = urlparse(url)
            domain = parsed.netloc.lower()
            return any(
                domain == d or domain.endswith(f".{d}")
                for d in cls.SUPPORTED_DOMAINS
            )
        except Exception:
            return False

    def parse_url(self, url: str) -> ParsedURL:
        """
        Parse a YouTube URL to extract video ID and timestamp.

        Supports:
        - youtube.com/watch?v=VIDEO_ID
        - youtu.be/VIDEO_ID
        - youtube.com/shorts/VIDEO_ID
        - Timestamps: ?t=123, &t=1m30s, #t=90
        """
        parsed = urlparse(url)
        video_id = None
        timestamp = None

        # Extract video ID based on URL format
        if "youtu.be" in parsed.netloc:
            # Short URL: youtu.be/VIDEO_ID
            video_id = parsed.path.strip("/").split("/")[0]
        elif "shorts" in parsed.path:
            # Shorts: youtube.com/shorts/VIDEO_ID
            parts = parsed.path.strip("/").split("/")
            if "shorts" in parts:
                idx = parts.index("shorts")
                if idx + 1 < len(parts):
                    video_id = parts[idx + 1]
        else:
            # Regular: youtube.com/watch?v=VIDEO_ID
            query = parse_qs(parsed.query)
            if "v" in query:
                video_id = query["v"][0]

        # Extract timestamp from query string (?t=123 or &t=1m30s)
        query = parse_qs(parsed.query)
        if "t" in query:
            timestamp = self._parse_youtube_timestamp(query["t"][0])

        # Also check fragment (#t=90)
        if not timestamp and parsed.fragment:
            timestamp = self.parse_timestamp_fragment(parsed.fragment)

        return ParsedURL(
            provider=self.PROVIDER_ID,
            content_type="video",
            content_id=video_id or "",
            original_url=url,
            timestamp_seconds=timestamp,
        )

    def get_program(self, url_or_id: str) -> MediaProgram:
        """
        Get video metadata using yt-dlp.

        Args:
            url_or_id: YouTube URL or video ID

        Returns:
            MediaProgram with metadata and video URL
        """
        # Ensure we have a full URL
        if not url_or_id.startswith("http"):
            url = f"https://www.youtube.com/watch?v={url_or_id}"
        else:
            url = url_or_id

        # Parse URL for timestamp
        parsed = self.parse_url(url)

        # Get metadata using yt-dlp
        try:
            result = subprocess.run(
                [
                    "yt-dlp",
                    "--dump-json",
                    "--no-download",
                    "--no-warnings",
                    url,
                ],
                capture_output=True,
                text=True,
                timeout=60,
            )

            if result.returncode != 0:
                error = result.stderr.strip() or "Unknown error"
                raise ValueError(f"yt-dlp failed: {error}")

            data = json.loads(result.stdout)

        except subprocess.TimeoutExpired:
            raise ValueError("Timeout fetching YouTube metadata")
        except json.JSONDecodeError:
            raise ValueError("Failed to parse YouTube metadata")
        except FileNotFoundError:
            raise ValueError(
                "yt-dlp not found. Install with: pip install yt-dlp"
            )

        # Extract language from automatic captions or audio
        language = None
        if data.get("language"):
            language = data["language"]
        elif data.get("automatic_captions"):
            # Use first available caption language as hint
            captions = data["automatic_captions"]
            if captions:
                language = list(captions.keys())[0]

        return MediaProgram(
            program_id=data.get("id", parsed.content_id),
            title=data.get("title", "Unknown"),
            series_title=data.get("playlist_title") or data.get("channel"),
            description=data.get("description", "")[:500],  # Truncate long descriptions
            duration_seconds=int(data.get("duration", 0)),
            audio_url=url,  # yt-dlp will handle actual extraction
            provider=self.PROVIDER_ID,
            image_url=data.get("thumbnail"),
            published_at=data.get("upload_date"),
            language=language,
            original_url=url,
            start_time_seconds=parsed.timestamp_seconds,
        )

    def _parse_youtube_timestamp(self, timestamp: str) -> int:
        """
        Parse YouTube timestamp formats.

        Supports:
        - Pure seconds: "123"
        - Minutes and seconds: "1m30s", "1:30"
        - Hours, minutes, seconds: "1h2m30s", "1:02:30"
        """
        # Pure number (seconds)
        if timestamp.isdigit():
            return int(timestamp)

        # Format: 1h2m30s
        match = re.match(
            r'(?:(\d+)h)?(?:(\d+)m)?(?:(\d+)s)?',
            timestamp
        )
        if match and any(match.groups()):
            hours = int(match.group(1) or 0)
            minutes = int(match.group(2) or 0)
            seconds = int(match.group(3) or 0)
            return hours * 3600 + minutes * 60 + seconds

        # Format: 1:30 or 1:02:30
        parts = timestamp.split(":")
        if len(parts) == 2:
            return int(parts[0]) * 60 + int(parts[1])
        elif len(parts) == 3:
            return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])

        return 0
