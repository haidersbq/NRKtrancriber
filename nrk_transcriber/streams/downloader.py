"""
Download on-demand audio content from various sources.

Handles HLS streams, direct audio files, and YouTube via yt-dlp.
"""

import asyncio
import logging
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

# YouTube domains for yt-dlp routing
YOUTUBE_DOMAINS = {"youtube.com", "www.youtube.com", "youtu.be", "m.youtube.com"}


@dataclass
class DownloadedAudio:
    """Result of a download operation."""

    program_id: str
    file_path: Path
    duration_seconds: float
    title: str
    download_time_seconds: float


class NRKDownloader:
    """
    Download audio from NRK on-demand content.

    Uses ffmpeg to handle HLS streams and convert to WAV.
    """

    def __init__(
        self,
        output_dir: Path,
        sample_rate: int = 16000,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.sample_rate = sample_rate

        if not shutil.which("ffmpeg"):
            raise RuntimeError("ffmpeg not found. Please install ffmpeg.")

    def _build_ffmpeg_command(
        self,
        audio_url: str,
        output_path: Path,
        start_time: Optional[int] = None,
        duration: Optional[int] = None,
    ) -> list[str]:
        """Build ffmpeg command for downloading and converting audio."""
        cmd = [
            "ffmpeg",
            "-y",  # Overwrite
            "-i", audio_url,
        ]

        # Add start time if specified
        if start_time:
            cmd.extend(["-ss", str(start_time)])

        # Add duration if specified
        if duration:
            cmd.extend(["-t", str(duration)])

        # Output format settings for Whisper
        cmd.extend([
            "-vn",  # No video
            "-acodec", "pcm_s16le",  # 16-bit PCM
            "-ar", str(self.sample_rate),  # Sample rate
            "-ac", "1",  # Mono
            "-f", "wav",
            str(output_path),
        ])

        return cmd

    def _is_youtube_url(self, url: str) -> bool:
        """Check if URL is from YouTube."""
        try:
            parsed = urlparse(url)
            domain = parsed.netloc.lower()
            return any(
                domain == d or domain.endswith(f".{d}")
                for d in YOUTUBE_DOMAINS
            )
        except Exception:
            return False

    async def download(
        self,
        audio_url: str,
        program_id: str,
        title: str = "",
        start_time: Optional[int] = None,
        duration: Optional[int] = None,
    ) -> DownloadedAudio:
        """
        Download audio from any supported source.

        Automatically routes YouTube URLs to yt-dlp, others to ffmpeg.

        Args:
            audio_url: URL to the audio stream (HLS, direct, or YouTube)
            program_id: Program ID for naming
            title: Program title
            start_time: Start offset in seconds
            duration: Duration to download in seconds (None for full)

        Returns:
            DownloadedAudio with file path and metadata
        """
        # Route YouTube URLs to yt-dlp
        if self._is_youtube_url(audio_url):
            return await self._download_with_ytdlp(
                audio_url, program_id, title, start_time, duration
            )

        # Use ffmpeg for everything else
        return await self._download_with_ffmpeg(
            audio_url, program_id, title, start_time, duration
        )

    async def _download_with_ffmpeg(
        self,
        audio_url: str,
        program_id: str,
        title: str = "",
        start_time: Optional[int] = None,
        duration: Optional[int] = None,
    ) -> DownloadedAudio:
        """Download audio using ffmpeg (for HLS, direct URLs)."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = self.output_dir / f"{program_id}_{timestamp}.wav"

        logger.info(f"Downloading {program_id}: {title}")
        if start_time:
            logger.info(f"  Starting at {start_time}s")
        if duration:
            logger.info(f"  Duration: {duration}s")

        cmd = self._build_ffmpeg_command(
            audio_url,
            output_path,
            start_time=start_time,
            duration=duration,
        )

        logger.debug(f"ffmpeg command: {' '.join(cmd)}")

        start = datetime.now()

        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        stdout, stderr = await process.communicate()

        download_time = (datetime.now() - start).total_seconds()

        if process.returncode != 0:
            error_msg = stderr.decode() if stderr else "Unknown error"
            logger.error(f"Download failed: {error_msg}")
            raise RuntimeError(f"Failed to download audio: {error_msg}")

        # Get actual duration from the downloaded file
        actual_duration = await self._get_audio_duration(output_path)

        logger.info(f"Downloaded {actual_duration:.1f}s of audio in {download_time:.1f}s")

        return DownloadedAudio(
            program_id=program_id,
            file_path=output_path,
            duration_seconds=actual_duration,
            title=title,
            download_time_seconds=download_time,
        )

    async def _download_with_ytdlp(
        self,
        url: str,
        program_id: str,
        title: str = "",
        start_time: Optional[int] = None,
        duration: Optional[int] = None,
    ) -> DownloadedAudio:
        """
        Download audio from YouTube using yt-dlp.

        Args:
            url: YouTube URL
            program_id: Video ID for naming
            title: Video title
            start_time: Start offset in seconds (applied post-download)
            duration: Duration to extract in seconds (applied post-download)

        Returns:
            DownloadedAudio with file path and metadata
        """
        if not shutil.which("yt-dlp"):
            raise RuntimeError("yt-dlp not found. Install with: pip install yt-dlp")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        temp_path = self.output_dir / f"{program_id}_{timestamp}_temp.%(ext)s"
        output_path = self.output_dir / f"{program_id}_{timestamp}.wav"

        logger.info(f"Downloading from YouTube: {title or program_id}")

        # First, download audio with yt-dlp
        ytdlp_cmd = [
            "yt-dlp",
            "-x",  # Extract audio only
            "--audio-format", "wav",
            "--audio-quality", "0",  # Best quality
            "-o", str(temp_path),
            "--no-playlist",  # Don't download playlists
            "--no-warnings",
            url,
        ]

        logger.debug(f"yt-dlp command: {' '.join(ytdlp_cmd)}")

        start = datetime.now()

        process = await asyncio.create_subprocess_exec(
            *ytdlp_cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        stdout, stderr = await process.communicate()

        if process.returncode != 0:
            error_msg = stderr.decode() if stderr else "Unknown error"
            logger.error(f"yt-dlp failed: {error_msg}")
            raise RuntimeError(f"Failed to download from YouTube: {error_msg}")

        # Find the downloaded file (yt-dlp replaces %(ext)s)
        temp_files = list(self.output_dir.glob(f"{program_id}_{timestamp}_temp.*"))
        if not temp_files:
            raise RuntimeError("yt-dlp did not create output file")
        downloaded_file = temp_files[0]

        # Convert to proper format and apply trimming if needed
        if start_time or duration or downloaded_file.suffix != ".wav":
            ffmpeg_cmd = [
                "ffmpeg",
                "-y",
                "-i", str(downloaded_file),
            ]

            if start_time:
                ffmpeg_cmd.extend(["-ss", str(start_time)])
                logger.info(f"  Starting at {start_time}s")

            if duration:
                ffmpeg_cmd.extend(["-t", str(duration)])
                logger.info(f"  Duration: {duration}s")

            ffmpeg_cmd.extend([
                "-vn",
                "-acodec", "pcm_s16le",
                "-ar", str(self.sample_rate),
                "-ac", "1",
                "-f", "wav",
                str(output_path),
            ])

            process = await asyncio.create_subprocess_exec(
                *ffmpeg_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            await process.communicate()

            # Clean up temp file
            downloaded_file.unlink(missing_ok=True)
        else:
            # Just rename if no processing needed
            downloaded_file.rename(output_path)

        download_time = (datetime.now() - start).total_seconds()

        # Get actual duration
        actual_duration = await self._get_audio_duration(output_path)

        logger.info(f"Downloaded {actual_duration:.1f}s of audio in {download_time:.1f}s")

        return DownloadedAudio(
            program_id=program_id,
            file_path=output_path,
            duration_seconds=actual_duration,
            title=title,
            download_time_seconds=download_time,
        )

    async def _get_audio_duration(self, file_path: Path) -> float:
        """Get the duration of an audio file using ffprobe."""
        cmd = [
            "ffprobe",
            "-v", "quiet",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(file_path),
        ]

        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        stdout, _ = await process.communicate()

        try:
            return float(stdout.decode().strip())
        except ValueError:
            return 0.0

    async def download_segments(
        self,
        audio_url: str,
        program_id: str,
        title: str = "",
        segment_duration: int = 300,  # 5 minutes
        total_duration: Optional[int] = None,
    ) -> list[DownloadedAudio]:
        """
        Download audio in segments for easier processing.

        Useful for long programs to process incrementally.

        Args:
            audio_url: URL to the audio
            program_id: Program ID
            title: Program title
            segment_duration: Duration of each segment in seconds
            total_duration: Total duration to download (None for full)

        Returns:
            List of DownloadedAudio segments
        """
        segments = []
        offset = 0

        while True:
            # Check if we've reached the total duration
            if total_duration and offset >= total_duration:
                break

            # Calculate segment duration
            remaining = total_duration - offset if total_duration else None
            seg_duration = min(segment_duration, remaining) if remaining else segment_duration

            try:
                segment = await self.download(
                    audio_url=audio_url,
                    program_id=f"{program_id}_seg{len(segments):03d}",
                    title=f"{title} (segment {len(segments) + 1})",
                    start_time=offset,
                    duration=seg_duration,
                )
                segments.append(segment)

                # If we got less than requested, we've reached the end
                if segment.duration_seconds < seg_duration - 1:
                    break

                offset += segment_duration

            except Exception as e:
                logger.error(f"Error downloading segment at {offset}s: {e}")
                break

        return segments
