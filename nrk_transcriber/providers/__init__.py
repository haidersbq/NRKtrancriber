"""
Media providers for various broadcasters and content sources.

Each provider implements the BaseProvider interface to provide
a unified way to fetch and transcribe content from different sources.

Supported providers:
- NRK: Norwegian Broadcasting Corporation (radio, podcasts)
- Direct: Direct audio URLs (MP3, WAV, HLS, etc.)
- YouTube: YouTube videos via yt-dlp
- Podcast: Generic podcast RSS feeds

Coming soon:
- BBC: BBC Sounds
- SVT: Swedish Television
- DR: Danish Broadcasting
"""

from .base import BaseProvider, MediaProgram, ParsedURL
from .registry import ProviderRegistry

# Import providers to register them
from . import nrk
from . import direct
from . import youtube
from . import podcast

__all__ = [
    "BaseProvider",
    "MediaProgram",
    "ParsedURL",
    "ProviderRegistry",
]


def get_provider(url_or_name: str) -> BaseProvider:
    """
    Get a provider for a URL or by name.

    Args:
        url_or_name: Either a URL (auto-detects provider) or provider ID

    Returns:
        An instantiated provider

    Raises:
        ValueError if no provider found
    """
    # Check if it's a URL
    if url_or_name.startswith("http"):
        provider = ProviderRegistry.detect_provider(url_or_name)
        if provider:
            return provider
        raise ValueError(
            f"Could not detect provider for URL: {url_or_name}. "
            f"Try specifying --provider explicitly."
        )

    # Otherwise treat as provider ID
    return ProviderRegistry.get_provider(url_or_name)
