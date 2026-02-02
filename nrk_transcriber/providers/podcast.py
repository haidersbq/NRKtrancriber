"""
Podcast provider for RSS feeds.

Supports standard podcast RSS feeds and direct episode URLs.
"""

import hashlib
import logging
import re
from typing import Optional
from urllib.parse import urlparse

import requests

from .base import BaseProvider, MediaProgram, ParsedURL
from .registry import ProviderRegistry

logger = logging.getLogger(__name__)


def _parse_itunes_duration(duration_str: str) -> int:
    """
    Parse iTunes duration format to seconds.

    Formats:
    - "3600" (seconds)
    - "60:00" (MM:SS)
    - "1:00:00" (HH:MM:SS)
    """
    if not duration_str:
        return 0

    # Pure seconds
    if duration_str.isdigit():
        return int(duration_str)

    # MM:SS or HH:MM:SS
    parts = duration_str.split(":")
    try:
        if len(parts) == 2:
            return int(parts[0]) * 60 + int(parts[1])
        elif len(parts) == 3:
            return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
    except ValueError:
        pass

    return 0


@ProviderRegistry.register
class PodcastProvider(BaseProvider):
    """
    Provider for podcast RSS feeds.

    Supports:
    - Standard RSS/Atom podcast feeds
    - Direct episode audio URLs
    - Apple Podcasts links (extracts RSS feed)
    """

    PROVIDER_ID = "podcast"
    PROVIDER_NAME = "Podcast"
    SUPPORTED_DOMAINS = []  # Matches based on content, not domain
    DEFAULT_LANGUAGE = None

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "MediaTranscriber/1.0 (Podcast Fetcher)",
        })

    @classmethod
    def can_handle(cls, url: str) -> bool:
        """
        Check if URL is a podcast feed or episode.

        Returns True for:
        - URLs ending in .rss, .xml, /feed
        - URLs containing 'feed' or 'rss' in path
        - Apple Podcasts URLs
        - Spotify podcast URLs (limited support)
        """
        try:
            parsed = urlparse(url)
            path_lower = parsed.path.lower()

            # Common feed extensions
            if path_lower.endswith(('.rss', '.xml')):
                return True

            # Common feed patterns
            if '/feed' in path_lower or '/rss' in path_lower:
                return True

            # Apple Podcasts
            if 'podcasts.apple.com' in parsed.netloc:
                return True

            # Feed in domain
            if 'feeds.' in parsed.netloc or 'feed.' in parsed.netloc:
                return True

            return False
        except Exception:
            return False

    def parse_url(self, url: str) -> ParsedURL:
        """Parse a podcast URL."""
        parsed = urlparse(url)

        # Generate ID from URL
        content_id = hashlib.md5(url.encode()).hexdigest()[:12]

        # Determine type
        if 'podcasts.apple.com' in parsed.netloc:
            content_type = "apple_podcast"
        elif '/episode' in parsed.path.lower():
            content_type = "episode"
        else:
            content_type = "feed"

        return ParsedURL(
            provider=self.PROVIDER_ID,
            content_type=content_type,
            content_id=content_id,
            original_url=url,
            timestamp_seconds=self.parse_timestamp_fragment(parsed.fragment),
        )

    def get_program(self, url_or_id: str) -> MediaProgram:
        """
        Get podcast episode metadata.

        For RSS feeds, returns the latest episode.
        For Apple Podcasts, attempts to find the RSS feed.

        Args:
            url_or_id: Podcast feed URL or episode URL

        Returns:
            MediaProgram with episode metadata
        """
        url = url_or_id
        parsed = self.parse_url(url)

        # Handle Apple Podcasts URLs
        if parsed.content_type == "apple_podcast":
            feed_url = self._resolve_apple_podcast(url)
            if feed_url:
                url = feed_url
            else:
                raise ValueError(
                    "Could not resolve Apple Podcast to RSS feed. "
                    "Try using the direct RSS feed URL."
                )

        # Fetch and parse the feed
        try:
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            feed_content = response.text
        except Exception as e:
            raise ValueError(f"Failed to fetch podcast feed: {e}")

        # Parse RSS/Atom feed
        return self._parse_rss_feed(feed_content, url)

    def _parse_rss_feed(self, content: str, feed_url: str) -> MediaProgram:
        """
        Parse RSS feed and extract latest episode.

        Uses simple regex parsing to avoid feedparser dependency.
        """
        # Try to extract channel info
        channel_title = self._extract_tag(content, "title")
        channel_language = self._extract_tag(content, "language")

        # Find all items (episodes)
        items = re.findall(
            r'<item[^>]*>(.*?)</item>',
            content,
            re.DOTALL | re.IGNORECASE
        )

        if not items:
            raise ValueError("No episodes found in podcast feed")

        # Get the first (latest) item
        item = items[0]

        # Extract episode details
        title = self._extract_tag(item, "title") or "Unknown Episode"
        description = self._extract_tag(item, "description") or ""
        pub_date = self._extract_tag(item, "pubDate")

        # Get audio URL from enclosure
        audio_url = self._extract_enclosure_url(item)
        if not audio_url:
            raise ValueError("No audio URL found in episode")

        # Get duration (iTunes format)
        duration_str = self._extract_tag(item, "itunes:duration")
        duration = _parse_itunes_duration(duration_str) if duration_str else 0

        # Get episode GUID or generate from URL
        guid = self._extract_tag(item, "guid")
        episode_id = guid or hashlib.md5(audio_url.encode()).hexdigest()[:12]

        # Clean up description (remove HTML)
        description = re.sub(r'<[^>]+>', '', description)
        description = description[:500] if description else ""

        return MediaProgram(
            program_id=episode_id,
            title=title,
            series_title=channel_title,
            description=description,
            duration_seconds=duration,
            audio_url=audio_url,
            provider=self.PROVIDER_ID,
            published_at=pub_date,
            language=channel_language,
            original_url=feed_url,
        )

    def _extract_tag(self, content: str, tag: str) -> Optional[str]:
        """Extract content from an XML tag."""
        # Handle CDATA
        pattern = rf'<{tag}[^>]*>(?:<!\[CDATA\[)?(.*?)(?:\]\]>)?</{tag}>'
        match = re.search(pattern, content, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()
        return None

    def _extract_enclosure_url(self, item: str) -> Optional[str]:
        """Extract audio URL from enclosure tag."""
        # Standard enclosure
        match = re.search(
            r'<enclosure[^>]+url=["\']([^"\']+)["\']',
            item,
            re.IGNORECASE
        )
        if match:
            return match.group(1)

        # Media content
        match = re.search(
            r'<media:content[^>]+url=["\']([^"\']+)["\']',
            item,
            re.IGNORECASE
        )
        if match:
            return match.group(1)

        return None

    def _resolve_apple_podcast(self, url: str) -> Optional[str]:
        """
        Attempt to resolve an Apple Podcasts URL to RSS feed.

        Uses iTunes Search API to find the feed URL.
        """
        # Extract podcast ID from URL
        # Format: podcasts.apple.com/us/podcast/name/id123456789
        match = re.search(r'/id(\d+)', url)
        if not match:
            return None

        podcast_id = match.group(1)

        try:
            # Use iTunes API to get feed URL
            api_url = f"https://itunes.apple.com/lookup?id={podcast_id}&entity=podcast"
            response = self.session.get(api_url, timeout=10)
            response.raise_for_status()
            data = response.json()

            results = data.get("results", [])
            if results:
                return results[0].get("feedUrl")
        except Exception as e:
            logger.warning(f"Could not resolve Apple Podcast: {e}")

        return None

    def get_episodes(self, feed_url: str, limit: int = 10) -> list[MediaProgram]:
        """
        Get multiple episodes from a feed.

        Args:
            feed_url: RSS feed URL
            limit: Maximum episodes to return

        Returns:
            List of MediaProgram objects
        """
        try:
            response = self.session.get(feed_url, timeout=30)
            response.raise_for_status()
            content = response.text
        except Exception as e:
            raise ValueError(f"Failed to fetch podcast feed: {e}")

        # Extract channel info
        channel_title = self._extract_tag(content, "title")
        channel_language = self._extract_tag(content, "language")

        # Find all items
        items = re.findall(
            r'<item[^>]*>(.*?)</item>',
            content,
            re.DOTALL | re.IGNORECASE
        )

        episodes = []
        for item in items[:limit]:
            try:
                title = self._extract_tag(item, "title") or "Unknown"
                audio_url = self._extract_enclosure_url(item)
                if not audio_url:
                    continue

                duration_str = self._extract_tag(item, "itunes:duration")
                duration = _parse_itunes_duration(duration_str) if duration_str else 0

                guid = self._extract_tag(item, "guid")
                episode_id = guid or hashlib.md5(audio_url.encode()).hexdigest()[:12]

                episodes.append(MediaProgram(
                    program_id=episode_id,
                    title=title,
                    series_title=channel_title,
                    description=self._extract_tag(item, "description") or "",
                    duration_seconds=duration,
                    audio_url=audio_url,
                    provider=self.PROVIDER_ID,
                    language=channel_language,
                    original_url=feed_url,
                ))
            except Exception as e:
                logger.debug(f"Error parsing episode: {e}")
                continue

        return episodes
