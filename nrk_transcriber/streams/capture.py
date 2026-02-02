"""
Stream capture module for NRK radio streams.

Uses ffmpeg to capture audio streams and convert them to the format
expected by Whisper (16kHz mono WAV).
"""

import asyncio
import logging
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import AsyncIterator, Optional
import shutil

logger = logging.getLogger(__name__)


@dataclass
class AudioChunk:
    """Represents a chunk of captured audio."""

    channel_id: str
    file_path: Path
    start_time: datetime
    end_time: datetime
    duration_seconds: float
    sample_rate: int = 16000
    channels: int = 1
    format: str = "wav"

    @property
    def filename(self) -> str:
        """Generate a filename for this chunk."""
        timestamp = self.start_time.strftime("%Y%m%d_%H%M%S")
        return f"{self.channel_id}_{timestamp}.{self.format}"


@dataclass
class StreamCapture:
    """
    Captures audio from NRK radio streams using ffmpeg.

    Supports both direct MP3 streams and HLS (m3u8) streams.
    Converts audio to 16kHz mono WAV for Whisper compatibility.
    """

    channel_id: str
    stream_url: str
    output_dir: Path
    chunk_duration_seconds: int = 30
    overlap_seconds: int = 2
    sample_rate: int = 16000
    reconnect_attempts: int = 5
    reconnect_delay_seconds: int = 5

    _process: Optional[subprocess.Popen] = field(default=None, init=False, repr=False)
    _running: bool = field(default=False, init=False)
    _ffmpeg_path: str = field(default="ffmpeg", init=False)

    def __post_init__(self):
        """Validate configuration and check ffmpeg availability."""
        self.output_dir = Path(self.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Check if ffmpeg is available
        if not self._check_ffmpeg():
            raise RuntimeError(
                "ffmpeg not found. Please install ffmpeg to use stream capture."
            )

    def _check_ffmpeg(self) -> bool:
        """Check if ffmpeg is available on the system."""
        return shutil.which("ffmpeg") is not None

    def _build_ffmpeg_command(self, output_path: Path, duration: int) -> list[str]:
        """Build the ffmpeg command for stream capture."""
        cmd = [
            self._ffmpeg_path,
            "-y",  # Overwrite output files
            "-reconnect", "1",  # Enable reconnection
            "-reconnect_streamed", "1",
            "-reconnect_delay_max", str(self.reconnect_delay_seconds),
            "-i", self.stream_url,
            "-t", str(duration),  # Duration
            "-vn",  # No video
            "-acodec", "pcm_s16le",  # 16-bit PCM
            "-ar", str(self.sample_rate),  # Sample rate
            "-ac", "1",  # Mono
            "-f", "wav",  # Output format
            str(output_path),
        ]
        return cmd

    async def capture_chunk(self, duration: Optional[int] = None) -> AudioChunk:
        """
        Capture a single audio chunk from the stream.

        Args:
            duration: Duration in seconds (defaults to chunk_duration_seconds)

        Returns:
            AudioChunk with the captured audio
        """
        if duration is None:
            duration = self.chunk_duration_seconds

        start_time = datetime.now()
        timestamp = start_time.strftime("%Y%m%d_%H%M%S")
        output_path = self.output_dir / f"{self.channel_id}_{timestamp}.wav"

        cmd = self._build_ffmpeg_command(output_path, duration)

        logger.info(f"Capturing {duration}s of audio from {self.channel_id}")
        logger.debug(f"ffmpeg command: {' '.join(cmd)}")

        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        stdout, stderr = await process.communicate()

        if process.returncode != 0:
            error_msg = stderr.decode() if stderr else "Unknown error"
            logger.error(f"ffmpeg error: {error_msg}")
            raise RuntimeError(f"Failed to capture audio: {error_msg}")

        end_time = datetime.now()
        actual_duration = (end_time - start_time).total_seconds()

        chunk = AudioChunk(
            channel_id=self.channel_id,
            file_path=output_path,
            start_time=start_time,
            end_time=end_time,
            duration_seconds=actual_duration,
            sample_rate=self.sample_rate,
        )

        logger.info(f"Captured audio chunk: {output_path}")
        return chunk

    async def capture_continuous(
        self,
        max_chunks: Optional[int] = None,
        callback=None,
    ) -> AsyncIterator[AudioChunk]:
        """
        Continuously capture audio chunks from the stream.

        Args:
            max_chunks: Maximum number of chunks to capture (None for infinite)
            callback: Optional async callback function called with each chunk

        Yields:
            AudioChunk objects as they are captured
        """
        self._running = True
        chunk_count = 0

        logger.info(f"Starting continuous capture for {self.channel_id}")

        while self._running:
            if max_chunks is not None and chunk_count >= max_chunks:
                logger.info(f"Reached max chunks ({max_chunks}), stopping capture")
                break

            try:
                # Account for overlap in timing
                effective_duration = self.chunk_duration_seconds - self.overlap_seconds
                chunk = await self.capture_chunk(duration=effective_duration)
                chunk_count += 1

                if callback:
                    await callback(chunk)

                yield chunk

            except Exception as e:
                logger.error(f"Error capturing chunk: {e}")
                await self._handle_capture_error()

        self._running = False
        logger.info(f"Capture stopped after {chunk_count} chunks")

    async def _handle_capture_error(self) -> None:
        """Handle capture errors with exponential backoff."""
        for attempt in range(self.reconnect_attempts):
            delay = self.reconnect_delay_seconds * (2 ** attempt)
            logger.warning(f"Reconnecting in {delay}s (attempt {attempt + 1}/{self.reconnect_attempts})")
            await asyncio.sleep(delay)

            try:
                # Test connection by capturing a short sample
                test_chunk = await self.capture_chunk(duration=2)
                if test_chunk.file_path.exists():
                    test_chunk.file_path.unlink()  # Clean up test file
                    logger.info("Reconnection successful")
                    return
            except Exception:
                continue

        raise RuntimeError(f"Failed to reconnect after {self.reconnect_attempts} attempts")

    def stop(self) -> None:
        """Stop continuous capture."""
        logger.info(f"Stopping capture for {self.channel_id}")
        self._running = False

        if self._process and self._process.poll() is None:
            self._process.terminate()
            try:
                self._process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._process.kill()


class MultiStreamCapture:
    """
    Capture multiple NRK radio streams simultaneously.
    """

    def __init__(
        self,
        channels: dict[str, str],  # channel_id -> stream_url
        output_dir: Path,
        chunk_duration_seconds: int = 30,
    ):
        self.captures: dict[str, StreamCapture] = {}

        for channel_id, stream_url in channels.items():
            channel_output_dir = Path(output_dir) / channel_id
            self.captures[channel_id] = StreamCapture(
                channel_id=channel_id,
                stream_url=stream_url,
                output_dir=channel_output_dir,
                chunk_duration_seconds=chunk_duration_seconds,
            )

    async def capture_all(
        self,
        max_chunks: Optional[int] = None,
        callback=None,
    ) -> AsyncIterator[AudioChunk]:
        """
        Capture from all channels simultaneously.

        Yields:
            AudioChunk objects from all channels as they are captured
        """
        async def capture_channel(channel_id: str, capture: StreamCapture):
            async for chunk in capture.capture_continuous(
                max_chunks=max_chunks,
                callback=callback,
            ):
                yield chunk

        # Create tasks for all channels
        tasks = [
            capture_channel(channel_id, capture)
            for channel_id, capture in self.captures.items()
        ]

        # Merge all iterators
        for task in asyncio.as_completed([asyncio.create_task(t.__anext__()) for t in tasks]):
            try:
                chunk = await task
                yield chunk
            except StopAsyncIteration:
                continue

    def stop_all(self) -> None:
        """Stop all captures."""
        for capture in self.captures.values():
            capture.stop()
