"""Tests for configuration module."""

import pytest
from pathlib import Path

from nrk_transcriber.config import Config, ChannelConfig


def test_channel_config():
    """Test ChannelConfig dataclass."""
    channel = ChannelConfig(
        channel_id="nrk_p1",
        name="NRK P1",
        stream_url="https://example.com/stream",
        description="Test channel",
        category="mainstream",
        language="nb",
    )

    assert channel.channel_id == "nrk_p1"
    assert channel.name == "NRK P1"
    assert channel.language == "nb"


def test_config_load():
    """Test loading configuration."""
    config = Config.load()

    # Should have some channels loaded
    assert len(config.channels) > 0

    # Check default transcription settings
    assert config.transcription.model in ["tiny", "base", "small", "medium", "large"]
    assert config.transcription.language == "no"

    # Check stream settings
    assert config.stream.chunk_duration_seconds > 0
    assert config.stream.sample_rate == 16000


def test_config_get_channel():
    """Test getting a channel by ID."""
    config = Config.load()

    # Get existing channel
    channel = config.get_channel("nrk_p1")
    assert channel is not None
    assert channel.name == "NRK P1"

    # Get non-existing channel
    channel = config.get_channel("nonexistent")
    assert channel is None


def test_config_list_channels():
    """Test listing channels."""
    config = Config.load()

    # List all channels
    all_channels = config.list_channels()
    assert len(all_channels) > 0

    # List by category
    news_channels = config.list_channels(category="news")
    assert all(c.category == "news" for c in news_channels)
