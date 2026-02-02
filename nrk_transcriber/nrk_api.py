"""
NRK API client for fetching program metadata and stream URLs.

Supports both live streams and on-demand content.
"""

import logging
import re
from dataclasses import dataclass
from typing import Optional
from urllib.parse import urlparse, parse_qs

import requests

logger = logging.getLogger(__name__)

NRK_PSAPI_BASE = "https://psapi.nrk.no"
NRK_RADIO_API = "https://psapi.nrk.no/radio"


@dataclass
class NRKProgram:
    """Metadata for an NRK program/episode."""

    program_id: str
    title: str
    series_title: Optional[str]
    description: str
    duration_seconds: int
    audio_url: str
    image_url: Optional[str] = None
    published_at: Optional[str] = None
    episode_number: Optional[str] = None


class NRKApiClient:
    """
    Client for NRK's public API.

    Fetches program metadata and stream URLs for on-demand content.
    """

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "NRKTranscriber/1.0",
            "Accept": "application/json",
        })

    def parse_nrk_url(self, url: str) -> dict:
        """
        Parse an NRK radio URL to extract program information.

        Supports URLs like:
        - https://radio.nrk.no/serie/distriktsprogram-telemark/sesong/202602/DKTE01002126
        - https://radio.nrk.no/podkast/norgesglasset/l_f3c8d1a1-...
        - https://radio.nrk.no/direkte/p1

        Returns:
            Dict with 'type' (live/program/podcast), 'id', and optional 'timestamp'
        """
        parsed = urlparse(url)
        path_parts = parsed.path.strip("/").split("/")

        result = {
            "url": url,
            "type": None,
            "id": None,
            "series_id": None,
            "timestamp_seconds": None,
        }

        # Parse timestamp from fragment (#t=14m19s)
        if parsed.fragment:
            timestamp_match = re.search(r't=(\d+)m(\d+)s', parsed.fragment)
            if timestamp_match:
                minutes = int(timestamp_match.group(1))
                seconds = int(timestamp_match.group(2))
                result["timestamp_seconds"] = minutes * 60 + seconds

        # Live stream: /direkte/p1
        if "direkte" in path_parts:
            result["type"] = "live"
            idx = path_parts.index("direkte")
            if idx + 1 < len(path_parts):
                result["id"] = path_parts[idx + 1]

        # Series/program: /serie/<series>/sesong/<season>/<program_id>
        elif "serie" in path_parts:
            result["type"] = "program"
            idx = path_parts.index("serie")
            if idx + 1 < len(path_parts):
                result["series_id"] = path_parts[idx + 1]
            # Program ID is usually the last part
            result["id"] = path_parts[-1]

        # Podcast: /podkast/<podcast>/<episode_id>
        elif "podkast" in path_parts:
            result["type"] = "podcast"
            idx = path_parts.index("podkast")
            if idx + 1 < len(path_parts):
                result["series_id"] = path_parts[idx + 1]
            if idx + 2 < len(path_parts):
                result["id"] = path_parts[idx + 2]

        return result

    def get_playback_manifest(self, program_id: str) -> dict:
        """
        Get the playback manifest for a program.

        This contains the actual stream URLs.
        """
        url = f"{NRK_PSAPI_BASE}/playback/manifest/program/{program_id}"

        logger.debug(f"Fetching manifest: {url}")
        response = self.session.get(url)
        response.raise_for_status()

        return response.json()

    def get_program_metadata(self, program_id: str) -> dict:
        """Get detailed metadata for a program."""
        url = f"{NRK_RADIO_API}/catalog/programs/{program_id}"

        logger.debug(f"Fetching program metadata: {url}")
        response = self.session.get(url)
        response.raise_for_status()

        return response.json()

    def get_podcast_episode(self, podcast_id: str, episode_id: str) -> dict:
        """Get metadata for a podcast episode."""
        url = f"{NRK_PSAPI_BASE}/radio/catalog/podcast/{podcast_id}/episodes/{episode_id}"

        logger.debug(f"Fetching podcast episode: {url}")
        response = self.session.get(url)
        response.raise_for_status()

        return response.json()

    def extract_audio_url(self, manifest: dict) -> Optional[str]:
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

    def get_program(self, url_or_id: str) -> NRKProgram:
        """
        Get program information from a URL or program ID.

        Args:
            url_or_id: Either a full NRK URL or a program ID

        Returns:
            NRKProgram with metadata and audio URL
        """
        # Parse URL if given
        if url_or_id.startswith("http"):
            parsed = self.parse_nrk_url(url_or_id)
            program_id = parsed["id"]
            timestamp = parsed.get("timestamp_seconds")
        else:
            program_id = url_or_id
            timestamp = None

        if not program_id:
            raise ValueError(f"Could not extract program ID from: {url_or_id}")

        # Get manifest for stream URL
        manifest = self.get_playback_manifest(program_id)
        audio_url = self.extract_audio_url(manifest)

        if not audio_url:
            raise ValueError(f"Could not find audio URL for program: {program_id}")

        # Get program metadata
        try:
            metadata = self.get_program_metadata(program_id)
        except Exception as e:
            logger.warning(f"Could not fetch metadata: {e}")
            metadata = {}

        # Extract relevant fields
        titles = metadata.get("titles", {})
        duration = metadata.get("duration", "PT0S")

        # Handle duration - could be string or dict
        if isinstance(duration, dict):
            # Some NRK APIs return {"seconds": 1234}
            duration_seconds = duration.get("seconds", 0)
        elif isinstance(duration, str):
            # Parse ISO 8601 duration (PT30M45S -> seconds)
            duration_seconds = self._parse_duration(duration)
        elif isinstance(duration, (int, float)):
            duration_seconds = int(duration)
        else:
            duration_seconds = 0

        return NRKProgram(
            program_id=program_id,
            title=titles.get("title", program_id),
            series_title=titles.get("subtitle"),
            description=metadata.get("description", ""),
            duration_seconds=duration_seconds,
            audio_url=audio_url,
            image_url=metadata.get("image", [{}])[0].get("url") if metadata.get("image") else None,
            published_at=metadata.get("availability", {}).get("onDemand", {}).get("from"),
        )

    def _parse_duration(self, duration_str: str) -> int:
        """Parse ISO 8601 duration string to seconds."""
        if not duration_str:
            return 0

        # Match PT1H30M45S format
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

    def search_programs(self, query: str, limit: int = 10) -> list[dict]:
        """
        Search for programs by keyword.

        Args:
            query: Search query
            limit: Maximum results

        Returns:
            List of program metadata dicts
        """
        url = f"{NRK_RADIO_API}/search"
        params = {
            "q": query,
            "take": limit,
        }

        response = self.session.get(url, params=params)
        response.raise_for_status()

        data = response.json()
        return data.get("hits", [])
