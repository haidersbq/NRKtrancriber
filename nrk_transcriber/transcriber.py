"""
Main NRK Radio Transcriber orchestrator.

Coordinates stream capture, transcription, and storage.
"""

import asyncio
import logging
import signal
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

from .config import Config, ChannelConfig
from .streams import StreamCapture, AudioChunk
from .transcription import WhisperTranscriber, TranscriptionResult
from .storage import TranscriptionDatabase, TranscriptionExporter
from .utils.logging_config import TranscriptionStats

logger = logging.getLogger(__name__)


class NRKTranscriber:
    """
    Main orchestrator for NRK radio transcription.

    Handles the complete pipeline:
    1. Stream capture from NRK radio
    2. Transcription using Whisper
    3. Storage and export of results
    """

    def __init__(
        self,
        config: Optional[Config] = None,
        on_transcription: Optional[Callable[[TranscriptionResult], None]] = None,
    ):
        """
        Initialize the transcriber.

        Args:
            config: Configuration object (loads default if not provided)
            on_transcription: Callback function called with each transcription result
        """
        self.config = config or Config.load()
        self.on_transcription = on_transcription

        # Initialize components
        self._transcriber: Optional[WhisperTranscriber] = None
        self._database: Optional[TranscriptionDatabase] = None
        self._exporter: Optional[TranscriptionExporter] = None
        self._captures: dict[str, StreamCapture] = {}

        # State
        self._running = False
        self._stats = TranscriptionStats()
        self._shutdown_event = asyncio.Event()

    async def initialize(self) -> None:
        """Initialize all components."""
        logger.info("Initializing NRK Transcriber")

        # Initialize database
        self._database = TranscriptionDatabase(self.config.storage.database_path)
        await self._database.initialize()

        # Initialize exporter
        self._exporter = TranscriptionExporter(self.config.storage.transcripts_dir)

        # Initialize transcriber (lazy loading of model)
        self._transcriber = WhisperTranscriber(
            model=self.config.transcription.model,
            language=self.config.transcription.language,
            device=self.config.transcription.device,
            compute_type=self.config.transcription.compute_type,
            beam_size=self.config.transcription.beam_size,
            word_timestamps=self.config.transcription.word_timestamps,
            vad_filter=self.config.transcription.vad_filter,
        )

        logger.info("Initialization complete")

    async def _process_chunk(self, chunk: AudioChunk) -> Optional[TranscriptionResult]:
        """
        Process a single audio chunk through transcription.

        Args:
            chunk: The audio chunk to process

        Returns:
            TranscriptionResult if successful, None otherwise
        """
        try:
            logger.info(f"Transcribing {chunk.channel_id}: {chunk.file_path.name}")

            result = await self._transcriber.transcribe(
                audio_path=chunk.file_path,
                channel_id=chunk.channel_id,
                audio_start_time=chunk.start_time,
                audio_end_time=chunk.end_time,
            )

            # Update statistics
            self._stats.add_chunk(
                result.duration_seconds,
                result.processing_time_seconds,
            )

            # Save to database
            if self._database:
                await self._database.save_transcription(result)

            # Export to files
            if self._exporter and self.config.storage.export_formats:
                self._exporter.export_all(
                    result,
                    formats=self.config.storage.export_formats,
                )

            # Cleanup audio file if configured
            if not self.config.storage.keep_audio_files and chunk.file_path.exists():
                chunk.file_path.unlink()
                logger.debug(f"Deleted audio file: {chunk.file_path}")

            # Call callback if provided
            if self.on_transcription:
                self.on_transcription(result)

            logger.info(
                f"Transcribed {chunk.channel_id}: "
                f"{result.duration_seconds:.1f}s audio in "
                f"{result.processing_time_seconds:.1f}s "
                f"({result.duration_seconds / result.processing_time_seconds:.1f}x realtime)"
            )

            return result

        except Exception as e:
            self._stats.add_error()
            logger.error(f"Error processing chunk {chunk.file_path}: {e}")
            return None

    async def transcribe_channel(
        self,
        channel_id: str,
        duration_minutes: Optional[float] = None,
        max_chunks: Optional[int] = None,
    ) -> list[TranscriptionResult]:
        """
        Transcribe a single NRK radio channel.

        Args:
            channel_id: ID of the channel to transcribe
            duration_minutes: Maximum duration to transcribe (None for continuous)
            max_chunks: Maximum number of chunks to process

        Returns:
            List of transcription results
        """
        channel = self.config.get_channel(channel_id)
        if not channel:
            raise ValueError(f"Unknown channel: {channel_id}")

        if not self._transcriber:
            await self.initialize()

        # Load the model before starting capture
        self._transcriber.load_model()

        # Create stream capture
        capture = StreamCapture(
            channel_id=channel_id,
            stream_url=channel.stream_url,
            output_dir=self.config.storage.audio_dir / channel_id,
            chunk_duration_seconds=self.config.stream.chunk_duration_seconds,
            overlap_seconds=self.config.stream.overlap_seconds,
            sample_rate=self.config.stream.sample_rate,
        )
        self._captures[channel_id] = capture

        results = []
        self._running = True

        logger.info(f"Starting transcription of {channel.name}")

        # Calculate max chunks from duration if specified
        if duration_minutes and not max_chunks:
            chunk_duration = self.config.stream.chunk_duration_seconds
            max_chunks = int((duration_minutes * 60) / chunk_duration) + 1

        try:
            async for chunk in capture.capture_continuous(max_chunks=max_chunks):
                if not self._running or self._shutdown_event.is_set():
                    break

                result = await self._process_chunk(chunk)
                if result:
                    results.append(result)

        except asyncio.CancelledError:
            logger.info("Transcription cancelled")
        finally:
            capture.stop()
            del self._captures[channel_id]

        logger.info(f"Transcription complete. Processed {len(results)} chunks")
        return results

    async def transcribe_multiple_channels(
        self,
        channel_ids: list[str],
        duration_minutes: Optional[float] = None,
    ) -> dict[str, list[TranscriptionResult]]:
        """
        Transcribe multiple channels simultaneously.

        Args:
            channel_ids: List of channel IDs to transcribe
            duration_minutes: Maximum duration for each channel

        Returns:
            Dictionary mapping channel_id to list of results
        """
        if not self._transcriber:
            await self.initialize()

        self._transcriber.load_model()
        self._running = True

        results: dict[str, list[TranscriptionResult]] = {
            channel_id: [] for channel_id in channel_ids
        }

        async def process_channel(channel_id: str):
            channel_results = await self.transcribe_channel(
                channel_id,
                duration_minutes=duration_minutes,
            )
            results[channel_id] = channel_results

        tasks = [
            asyncio.create_task(process_channel(channel_id))
            for channel_id in channel_ids
        ]

        try:
            await asyncio.gather(*tasks)
        except asyncio.CancelledError:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)

        return results

    async def transcribe_file(
        self,
        audio_path: Path,
        channel_id: str = "file",
        language: Optional[str] = None,
    ) -> TranscriptionResult:
        """
        Transcribe a local audio file.

        Args:
            audio_path: Path to the audio file
            channel_id: Channel ID to associate with the transcription
            language: Optional language override (e.g., 'no', 'en', 'sv')

        Returns:
            TranscriptionResult
        """
        if not self._transcriber:
            await self.initialize()

        # Override language if specified
        if language and self._transcriber:
            self._transcriber.language = language

        self._transcriber.load_model()

        logger.info(f"Transcribing file: {audio_path}")

        result = await self._transcriber.transcribe(
            audio_path=audio_path,
            channel_id=channel_id,
        )

        # Save and export
        if self._database:
            await self._database.save_transcription(result)

        if self._exporter:
            self._exporter.export_all(result)

        return result

    def stop(self) -> None:
        """Stop all active transcriptions."""
        logger.info("Stopping transcriber")
        self._running = False
        self._shutdown_event.set()

        for capture in self._captures.values():
            capture.stop()

    async def shutdown(self) -> None:
        """Cleanup and shutdown."""
        self.stop()

        if self._transcriber:
            self._transcriber.unload_model()

        if self._database:
            await self._database.close()

        logger.info(f"Final stats: {self._stats}")

    def get_statistics(self) -> dict:
        """Get current transcription statistics."""
        return self._stats.get_summary()

    def list_channels(self, category: Optional[str] = None) -> list[ChannelConfig]:
        """List available channels."""
        return self.config.list_channels(category)


async def run_transcriber(
    channel_ids: list[str],
    duration_minutes: Optional[float] = None,
    config: Optional[Config] = None,
) -> dict[str, list[TranscriptionResult]]:
    """
    Convenience function to run the transcriber.

    Args:
        channel_ids: List of channel IDs to transcribe
        duration_minutes: Maximum duration per channel
        config: Optional configuration

    Returns:
        Dictionary of results by channel
    """
    transcriber = NRKTranscriber(config=config)

    # Setup signal handlers for graceful shutdown
    loop = asyncio.get_event_loop()

    def signal_handler():
        logger.info("Received shutdown signal")
        transcriber.stop()

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, signal_handler)

    try:
        await transcriber.initialize()

        if len(channel_ids) == 1:
            results = await transcriber.transcribe_channel(
                channel_ids[0],
                duration_minutes=duration_minutes,
            )
            return {channel_ids[0]: results}
        else:
            return await transcriber.transcribe_multiple_channels(
                channel_ids,
                duration_minutes=duration_minutes,
            )
    finally:
        await transcriber.shutdown()
