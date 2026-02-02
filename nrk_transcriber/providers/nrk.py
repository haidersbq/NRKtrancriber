"""
NRK (Norwegian Broadcasting Corporation) provider.

Supports NRK Radio on-demand content, podcasts, and live streams.
"""

import logging
from typing import Optional
from urllib.parse import urlparse

import requests

from .base import BaseProvider, MediaProgram, ParsedURL
from .registry import ProviderRegistry

logger = logging.getLogger(__name__)

NRK_PSAPI_BASE = "https://psapi.nrk.no"
NRK_RADIO_API = "https://psapi.nrk.no/radio"


@ProviderRegistry.register
class NRKProvider(BaseProvider):
    """
    Provider for NRK (Norwegian Broadcasting Corporation).

    Supports:
    - On-demand radio programs
    - Podcasts
    - Live radio streams
    """

    PROVIDER_ID = "nrk"
    PROVIDER_NAME = "NRK"
    SUPPORTED_DOMAINS = ["radio.nrk.no", "tv.nrk.no", "nrk.no"]
    DEFAULT_LANGUAGE = "no"

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "MediaTranscriber/1.0",
            "Accept": "application/json",
        })

    @classmethod
    def can_handle(cls, url: str) -> bool:
        """Check if URL is from NRK."""
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
        Parse an NRK URL to extract program information.

        Supports URLs like:
        - https://radio.nrk.no/serie/distriktsprogram-telemark/sesong/202602/DKTE01002126
        - https://radio.nrk.no/podkast/norgesglasset/l_f3c8d1a1-...
        - https://radio.nrk.no/direkte/p1
        """
        parsed = urlparse(url)
        path_parts = parsed.path.strip("/").split("/")

        content_type = "on_demand"
        content_id = ""
        series_id = None

        # Live stream: /direkte/p1
        if "direkte" in path_parts:
            content_type = "live"
            idx = path_parts.index("direkte")
            if idx + 1 < len(path_parts):
                content_id = path_parts[idx + 1]

        # Series/program: /serie/<series>/sesong/<season>/<program_id>
        elif "serie" in path_parts:
            content_type = "on_demand"
            idx = path_parts.index("serie")
            if idx + 1 < len(path_parts):
                series_id = path_parts[idx + 1]
            content_id = path_parts[-1]

        # Podcast: /podkast/<podcast>/<episode_id>
        elif "podkast" in path_parts:
            content_type = "podcast"
            idx = path_parts.index("podkast")
            if idx + 1 < len(path_parts):
                series_id = path_parts[idx + 1]
            if idx + 2 < len(path_parts):
                content_id = path_parts[idx + 2]

        return ParsedURL(
            provider=self.PROVIDER_ID,
            content_type=content_type,
            content_id=content_id,
            original_url=url,
            series_id=series_id,
            timestamp_seconds=self.parse_timestamp_fragment(parsed.fragment),
        )

    def get_program(self, url_or_id: str) -> MediaProgram:
        """
        Get program information from a URL or program ID.

        Args:
            url_or_id: Either a full NRK URL or a program ID

        Returns:
            MediaProgram with metadata and audio URL
        """
        # Parse URL if given
        timestamp = None
        if url_or_id.startswith("http"):
            parsed = self.parse_url(url_or_id)
            program_id = parsed.content_id
            timestamp = parsed.timestamp_seconds
            original_url = url_or_id
        else:
            program_id = url_or_id
            original_url = None

        if not program_id:
            raise ValueError(f"Could not extract program ID from: {url_or_id}")

        # Get manifest for stream URL
        manifest = self._get_playback_manifest(program_id)
        audio_url = self._extract_audio_url(manifest)

        if not audio_url:
            raise ValueError(f"Could not find audio URL for program: {program_id}")

        # Get program metadata
        try:
            metadata = self._get_program_metadata(program_id)
        except Exception as e:
            logger.warning(f"Could not fetch metadata: {e}")
            metadata = {}

        # Extract relevant fields
        titles = metadata.get("titles", {})
        duration = metadata.get("duration", "PT0S")

        # Handle duration - could be string, dict, or number
        duration_seconds = self._parse_duration_field(duration)

        # Get image URL
        image_url = None
        images = metadata.get("image", [])
        if images and isinstance(images, list):
            image_url = images[0].get("url")

        return MediaProgram(
            program_id=program_id,
            title=titles.get("title", program_id),
            series_title=titles.get("subtitle"),
            description=metadata.get("description", ""),
            duration_seconds=duration_seconds,
            audio_url=audio_url,
            provider=self.PROVIDER_ID,
            image_url=image_url,
            published_at=metadata.get("availability", {}).get("onDemand", {}).get("from"),
            language=self.DEFAULT_LANGUAGE,
            original_url=original_url,
            start_time_seconds=timestamp,
        )

    def get_live_stream_url(self, channel_id: str) -> str:
        """Get URL for a live NRK radio stream."""
        manifest = self._get_playback_manifest(f"nrk-{channel_id}")
        url = self._extract_audio_url(manifest)
        if not url:
            raise ValueError(f"Could not find stream URL for channel: {channel_id}")
        return url

    def search(self, query: str, limit: int = 10) -> list[MediaProgram]:
        """Search for NRK programs."""
        url = f"{NRK_RADIO_API}/search"
        params = {"q": query, "take": limit}

        response = self.session.get(url, params=params)
        response.raise_for_status()

        data = response.json()
        results = []

        for hit in data.get("hits", []):
            try:
                program = self.get_program(hit.get("id", ""))
                results.append(program)
            except Exception as e:
                logger.debug(f"Could not fetch program from search result: {e}")

        return results

    def _get_playback_manifest(self, program_id: str) -> dict:
        """Get the playback manifest for a program."""
        url = f"{NRK_PSAPI_BASE}/playback/manifest/program/{program_id}"

        logger.debug(f"Fetching manifest: {url}")
        response = self.session.get(url)
        response.raise_for_status()

        return response.json()

    def _get_program_metadata(self, program_id: str) -> dict:
        """Get detailed metadata for a program."""
        url = f"{NRK_RADIO_API}/catalog/programs/{program_id}"

        logger.debug(f"Fetching program metadata: {url}")
        response = self.session.get(url)
        response.raise_for_status()

        return response.json()

    def _extract_audio_url(self, manifest: dict) -> Optional[str]:
        """
        Extract the best audio URL from a playback manifest.

        Prefers HLS, falls back to progressive download.
        """
        playable = manifest.get("playable", {})

        # Try to get HLS stream
        assets = playable.get("assets", [])
        for asset in assets:
            if asset.get("format") == "HLS":
                return asset.get("url")

        # Fall back to any available URL
        for asset in assets:
            url = asset.get("url")
            if url:
                return url

        # Check for direct audio link
        if "resolve" in playable:
            return playable["resolve"].get("url")

        return None

    def _parse_duration_field(self, duration) -> int:
        """Parse duration from various formats."""
        if isinstance(duration, dict):
            return duration.get("seconds", 0)
        elif isinstance(duration, str):
            return self.parse_iso_duration(duration)
        elif isinstance(duration, (int, float)):
            return int(duration)
        return 0
