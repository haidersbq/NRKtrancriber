"""
Configuration management for NRK Radio Transcriber.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml
from dotenv import load_dotenv


@dataclass
class ChannelConfig:
    """Configuration for a single NRK channel."""

    channel_id: str
    name: str
    stream_url: str
    hls_url: Optional[str] = None
    description: str = ""
    category: str = "general"
    language: str = "nb"
    region: Optional[str] = None


@dataclass
class TranscriptionConfig:
    """Configuration for transcription settings."""

    model: str = "medium"  # tiny, base, small, medium, large
    language: str = "no"  # Norwegian
    device: str = "auto"  # auto, cpu, cuda
    compute_type: str = "float16"  # float16, int8, float32
    beam_size: int = 5
    best_of: int = 5
    temperature: float = 0.0
    compression_ratio_threshold: float = 2.4
    log_prob_threshold: float = -1.0
    no_speech_threshold: float = 0.6
    condition_on_previous_text: bool = True
    initial_prompt: Optional[str] = None
    word_timestamps: bool = True
    vad_filter: bool = True  # Voice Activity Detection


@dataclass
class StreamConfig:
    """Configuration for stream capture."""

    chunk_duration_seconds: int = 30
    overlap_seconds: int = 2
    audio_format: str = "wav"  # wav for better quality with Whisper
    sample_rate: int = 16000  # Whisper expects 16kHz
    channels: int = 1  # Mono
    buffer_size: int = 4096
    reconnect_attempts: int = 5
    reconnect_delay_seconds: int = 5


@dataclass
class StorageConfig:
    """Configuration for storage settings."""

    output_dir: Path = field(default_factory=lambda: Path("output"))
    audio_dir: Path = field(default_factory=lambda: Path("output/audio"))
    transcripts_dir: Path = field(default_factory=lambda: Path("output/transcripts"))
    database_path: Path = field(default_factory=lambda: Path("output/transcriptions.db"))
    keep_audio_files: bool = False
    max_audio_age_hours: int = 24
    export_formats: list = field(default_factory=lambda: ["txt", "json", "srt"])


@dataclass
class Config:
    """Main configuration class."""

    channels: dict[str, ChannelConfig] = field(default_factory=dict)
    transcription: TranscriptionConfig = field(default_factory=TranscriptionConfig)
    stream: StreamConfig = field(default_factory=StreamConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)

    @classmethod
    def load(cls, config_dir: Optional[Path] = None) -> "Config":
        """Load configuration from files and environment."""
        load_dotenv()

        if config_dir is None:
            config_dir = Path(__file__).parent.parent / "config"

        config = cls()

        # Load channels configuration
        channels_file = config_dir / "channels.yaml"
        if channels_file.exists():
            with open(channels_file, "r", encoding="utf-8") as f:
                channels_data = yaml.safe_load(f)

            # Load main channels
            for channel_id, channel_data in channels_data.get("channels", {}).items():
                config.channels[channel_id] = ChannelConfig(
                    channel_id=channel_id,
                    name=channel_data.get("name", channel_id),
                    stream_url=channel_data.get("stream_url", ""),
                    hls_url=channel_data.get("hls_url"),
                    description=channel_data.get("description", ""),
                    category=channel_data.get("category", "general"),
                    language=channel_data.get("language", "nb"),
                )

            # Load regional channels
            for channel_id, channel_data in channels_data.get("regional", {}).items():
                config.channels[channel_id] = ChannelConfig(
                    channel_id=channel_id,
                    name=channel_data.get("name", channel_id),
                    stream_url=channel_data.get("stream_url", ""),
                    region=channel_data.get("region"),
                    category="regional",
                )

            # Apply defaults
            defaults = channels_data.get("defaults", {})
            if defaults:
                config.stream.chunk_duration_seconds = defaults.get(
                    "chunk_duration_seconds",
                    config.stream.chunk_duration_seconds
                )
                config.stream.overlap_seconds = defaults.get(
                    "overlap_seconds",
                    config.stream.overlap_seconds
                )
                config.transcription.model = defaults.get(
                    "whisper_model",
                    config.transcription.model
                )
                config.transcription.language = defaults.get(
                    "language",
                    config.transcription.language
                )

        # Override from environment variables
        config._load_env_overrides()

        # Ensure directories exist
        config.storage.output_dir.mkdir(parents=True, exist_ok=True)
        config.storage.audio_dir.mkdir(parents=True, exist_ok=True)
        config.storage.transcripts_dir.mkdir(parents=True, exist_ok=True)

        return config

    def _load_env_overrides(self) -> None:
        """Load configuration overrides from environment variables."""
        if model := os.getenv("NRK_WHISPER_MODEL"):
            self.transcription.model = model

        if device := os.getenv("NRK_DEVICE"):
            self.transcription.device = device

        if output_dir := os.getenv("NRK_OUTPUT_DIR"):
            self.storage.output_dir = Path(output_dir)
            self.storage.audio_dir = Path(output_dir) / "audio"
            self.storage.transcripts_dir = Path(output_dir) / "transcripts"
            self.storage.database_path = Path(output_dir) / "transcriptions.db"

        if chunk_duration := os.getenv("NRK_CHUNK_DURATION"):
            self.stream.chunk_duration_seconds = int(chunk_duration)

    def get_channel(self, channel_id: str) -> Optional[ChannelConfig]:
        """Get a channel configuration by ID."""
        return self.channels.get(channel_id)

    def list_channels(self, category: Optional[str] = None) -> list[ChannelConfig]:
        """List all channels, optionally filtered by category."""
        channels = list(self.channels.values())
        if category:
            channels = [c for c in channels if c.category == category]
        return channels
