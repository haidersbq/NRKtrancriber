"""
Direct URL provider for handling direct audio file URLs.

Supports MP3, WAV, HLS (m3u8), and other audio formats
without requiring any API integration.
"""

import hashlib
import logging
import subprocess
from pathlib import Path
from urllib.parse import urlparse

from .base import BaseProvider, MediaProgram, ParsedURL
from .registry import ProviderRegistry

logger = logging.getLogger(__name__)

# Audio file extensions we can handle directly
AUDIO_EXTENSIONS = {
    ".mp3", ".wav", ".aac", ".ogg", ".flac", ".m4a",
    ".wma", ".opus", ".webm", ".mp4", ".mkv", ".avi",
}

# Streaming formats
STREAM_EXTENSIONS = {".m3u8", ".m3u", ".pls"}


@ProviderRegistry.register
class DirectURLProvider(BaseProvider):
    """
    Provider for direct audio/video URLs.

    Handles any URL that points directly to an audio file
    or HLS stream, without needing a specific API.
    """

    PROVIDER_ID = "direct"
    PROVIDER_NAME = "Direct URL"
    SUPPORTED_DOMAINS = []  # Matches based on file extension, not domain
    DEFAULT_LANGUAGE = None

    @classmethod
    def can_handle(cls, url: str) -> bool:
        """
        Check if URL points to a direct audio/video file.

        Returns True for URLs ending in known audio/video extensions
        or HLS manifests.
        """
        try:
            parsed = urlparse(url)
            path_lower = parsed.path.lower()

            # Check for audio file extensions
            for ext in AUDIO_EXTENSIONS | STREAM_EXTENSIONS:
                if path_lower.endswith(ext):
                    return True

            # Check for common streaming patterns
            if "manifest" in path_lower or "playlist" in path_lower:
                return True

            return False
        except Exception:
            return False

    def parse_url(self, url: str) -> ParsedURL:
        """Parse a direct URL."""
        parsed = urlparse(url)
        path = Path(parsed.path)

        # Determine content type
        ext = path.suffix.lower()
        if ext in STREAM_EXTENSIONS:
            content_type = "live"
        else:
            content_type = "direct"

        # Generate a content ID from the URL
        content_id = hashlib.md5(url.encode()).hexdigest()[:12]

        return ParsedURL(
            provider=self.PROVIDER_ID,
            content_type=content_type,
            content_id=content_id,
            original_url=url,
            timestamp_seconds=self.parse_timestamp_fragment(parsed.fragment),
        )

    def get_program(self, url_or_id: str) -> MediaProgram:
        """
        Get program info for a direct URL.

        Since there's no API, we extract what we can from the URL
        and optionally probe the file for duration.
        """
        url = url_or_id
        parsed = urlparse(url)
        path = Path(parsed.path)

        # Try to get a title from the filename
        title = path.stem or "Audio"
        # Clean up common URL patterns
        title = title.replace("-", " ").replace("_", " ").title()

        # Generate unique ID
        content_id = hashlib.md5(url.encode()).hexdigest()[:12]

        # Try to get duration via ffprobe
        duration = self._probe_duration(url)

        # Parse timestamp from fragment
        timestamp = self.parse_timestamp_fragment(parsed.fragment)

        return MediaProgram(
            program_id=content_id,
            title=title,
            description=f"Direct audio from {parsed.netloc}",
            duration_seconds=duration,
            audio_url=url,
            provider=self.PROVIDER_ID,
            original_url=url,
            start_time_seconds=timestamp,
        )

    def _probe_duration(self, url: str) -> int:
        """
        Try to get duration using ffprobe.

        Returns 0 if probing fails (will be determined during download).
        """
        try:
            result = subprocess.run(
                [
                    "ffprobe",
                    "-v", "quiet",
                    "-show_entries", "format=duration",
                    "-of", "default=noprint_wrappers=1:nokey=1",
                    url,
                ],
                capture_output=True,
                text=True,
                timeout=30,
            )

            if result.returncode == 0 and result.stdout.strip():
                return int(float(result.stdout.strip()))
        except subprocess.TimeoutExpired:
            logger.debug(f"Timeout probing duration for {url}")
        except Exception as e:
            logger.debug(f"Could not probe duration: {e}")

        return 0
