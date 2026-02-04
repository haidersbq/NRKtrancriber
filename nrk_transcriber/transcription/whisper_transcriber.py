"""
Whisper-based transcription module for NRK radio.

Supports both OpenAI Whisper and faster-whisper (CTranslate2-based).
Optimized for Norwegian language transcription.
"""

import asyncio
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Union
import json

# Prevent OpenMP crash when multiple libraries (torch, ctranslate2) each bundle libiomp5
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

logger = logging.getLogger(__name__)


@dataclass
class TranscriptionWord:
    """A single transcribed word with timing."""

    word: str
    start: float
    end: float
    probability: float = 1.0


@dataclass
class TranscriptionSegment:
    """A segment of transcribed text with timing."""

    id: int
    text: str
    start: float
    end: float
    words: list[TranscriptionWord] = field(default_factory=list)
    avg_logprob: float = 0.0
    compression_ratio: float = 0.0
    no_speech_prob: float = 0.0
    speaker: Optional[str] = None

    def to_dict(self) -> dict:
        """Convert segment to dictionary."""
        d = {
            "id": self.id,
            "text": self.text,
            "start": self.start,
            "end": self.end,
            "words": [
                {"word": w.word, "start": w.start, "end": w.end, "probability": w.probability}
                for w in self.words
            ],
            "avg_logprob": self.avg_logprob,
            "compression_ratio": self.compression_ratio,
            "no_speech_prob": self.no_speech_prob,
        }
        if self.speaker is not None:
            d["speaker"] = self.speaker
        return d


@dataclass
class TranscriptionResult:
    """Result of a transcription operation."""

    channel_id: str
    audio_file: Path
    audio_start_time: datetime
    audio_end_time: datetime
    text: str
    segments: list[TranscriptionSegment]
    language: str
    language_probability: float
    duration_seconds: float
    processing_time_seconds: float

    def to_dict(self) -> dict:
        """Convert result to dictionary."""
        return {
            "channel_id": self.channel_id,
            "audio_file": str(self.audio_file),
            "audio_start_time": self.audio_start_time.isoformat(),
            "audio_end_time": self.audio_end_time.isoformat(),
            "text": self.text,
            "segments": [s.to_dict() for s in self.segments],
            "language": self.language,
            "language_probability": self.language_probability,
            "duration_seconds": self.duration_seconds,
            "processing_time_seconds": self.processing_time_seconds,
        }

    def to_json(self, indent: int = 2) -> str:
        """Convert result to JSON string."""
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    @property
    def has_speakers(self) -> bool:
        """Check if any segments have speaker labels."""
        return any(s.speaker is not None for s in self.segments)

    def to_speaker_text(self) -> str:
        """Format text with speaker labels, grouping consecutive segments by speaker."""
        if not self.has_speakers:
            return self.text

        lines = []
        current_speaker = None

        for segment in self.segments:
            speaker = segment.speaker or "Unknown"
            if speaker != current_speaker:
                current_speaker = speaker
                lines.append(f"\n[{speaker}]")
            lines.append(segment.text.strip())

        return "\n".join(lines).strip()

    def to_srt(self, offset_time: Optional[datetime] = None) -> str:
        """
        Convert result to SRT subtitle format.

        Args:
            offset_time: If provided, timestamps will be relative to this time
        """
        lines = []

        for i, segment in enumerate(self.segments, 1):
            start = self._format_srt_time(segment.start)
            end = self._format_srt_time(segment.end)

            lines.append(str(i))
            lines.append(f"{start} --> {end}")
            text = segment.text.strip()
            if segment.speaker:
                text = f"[{segment.speaker}] {text}"
            lines.append(text)
            lines.append("")

        return "\n".join(lines)

    def to_vtt(self) -> str:
        """Convert result to WebVTT subtitle format."""
        lines = ["WEBVTT", ""]

        for segment in self.segments:
            start = self._format_vtt_time(segment.start)
            end = self._format_vtt_time(segment.end)

            lines.append(f"{start} --> {end}")
            text = segment.text.strip()
            if segment.speaker:
                text = f"<v {segment.speaker}>{text}"
            lines.append(text)
            lines.append("")

        return "\n".join(lines)

    @staticmethod
    def _format_srt_time(seconds: float) -> str:
        """Format seconds as SRT timestamp (HH:MM:SS,mmm)."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millis = int((seconds % 1) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"

    @staticmethod
    def _format_vtt_time(seconds: float) -> str:
        """Format seconds as VTT timestamp (HH:MM:SS.mmm)."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millis = int((seconds % 1) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"


class WhisperTranscriber:
    """
    Transcription engine using Whisper models.

    Supports both OpenAI Whisper and faster-whisper backends.
    """

    def __init__(
        self,
        model: str = "medium",
        language: str = "no",
        device: str = "auto",
        compute_type: str = "float16",
        beam_size: int = 5,
        best_of: int = 5,
        temperature: float = 0.0,
        word_timestamps: bool = True,
        vad_filter: bool = True,
        use_faster_whisper: bool = True,
        initial_prompt: Optional[str] = None,
    ):
        """
        Initialize the Whisper transcriber.

        Args:
            model: Whisper model size (tiny, base, small, medium, large, large-v2, large-v3)
            language: Language code for transcription (no = Norwegian)
            device: Device to use (auto, cpu, cuda)
            compute_type: Computation type (float16, int8, float32)
            beam_size: Beam size for beam search
            best_of: Number of candidates when sampling
            temperature: Sampling temperature
            word_timestamps: Enable word-level timestamps
            vad_filter: Enable voice activity detection filter
            use_faster_whisper: Use faster-whisper backend (recommended)
            initial_prompt: Initial prompt to guide transcription
        """
        self.model_name = model
        self.language = language
        self.device = device
        self.compute_type = compute_type
        self.beam_size = beam_size
        self.best_of = best_of
        self.temperature = temperature
        self.word_timestamps = word_timestamps
        self.vad_filter = vad_filter
        self.use_faster_whisper = use_faster_whisper
        self.initial_prompt = initial_prompt or self._get_norwegian_prompt()

        self._model = None
        self._backend = None  # "faster-whisper" or "openai-whisper"
        self._loaded = False

    def _get_norwegian_prompt(self) -> str:
        """Get an initial prompt optimized for Norwegian content."""
        return (
            "Dette er en transkripsjon av norsk radio. "
            "Programmet kan inneholde nyheter, debatter, intervjuer og musikk."
        )

    def _resolve_device(self) -> str:
        """Resolve the device to use."""
        if self.device != "auto":
            return self.device

        try:
            import torch
            if torch.cuda.is_available():
                return "cuda"
        except ImportError:
            pass

        return "cpu"

    def load_model(self) -> None:
        """Load the Whisper model."""
        if self._loaded:
            return

        device = self._resolve_device()
        logger.info(f"Loading Whisper model '{self.model_name}' on {device}")

        if self.use_faster_whisper:
            self._load_faster_whisper(device)
        else:
            self._load_openai_whisper(device)

        self._loaded = True
        logger.info("Model loaded successfully")

    def _load_faster_whisper(self, device: str) -> None:
        """Load the faster-whisper model."""
        try:
            from faster_whisper import WhisperModel

            # Adjust compute type for CPU
            compute_type = self.compute_type
            if device == "cpu" and compute_type == "float16":
                compute_type = "int8"
                logger.info("Switching to int8 for CPU inference")

            self._model = WhisperModel(
                self.model_name,
                device=device,
                compute_type=compute_type,
            )
            self._backend = "faster-whisper"

        except ImportError:
            logger.warning("faster-whisper not available, falling back to OpenAI Whisper")
            self._load_openai_whisper(device)

    def _load_openai_whisper(self, device: str) -> None:
        """Load the OpenAI Whisper model."""
        import whisper

        self._model = whisper.load_model(self.model_name, device=device)
        self._backend = "openai-whisper"

    async def transcribe(
        self,
        audio_path: Union[str, Path],
        channel_id: str = "unknown",
        audio_start_time: Optional[datetime] = None,
        audio_end_time: Optional[datetime] = None,
        diarize: bool = False,
        diarizer=None,
    ) -> TranscriptionResult:
        """
        Transcribe an audio file.

        Args:
            audio_path: Path to the audio file
            channel_id: ID of the channel this audio is from
            audio_start_time: When the audio recording started
            audio_end_time: When the audio recording ended
            diarize: Whether to run speaker diarization
            diarizer: Optional pre-loaded SpeakerDiarizer instance

        Returns:
            TranscriptionResult with the transcribed text and metadata
        """
        if not self._loaded:
            self.load_model()

        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        start_time = datetime.now()

        # Run transcription in thread pool to not block async loop
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            self._transcribe_sync,
            str(audio_path),
        )

        # Run diarization if requested
        speaker_segments = None
        if diarize:
            speaker_segments = await self._run_diarization(
                audio_path, loop, diarizer
            )

        processing_time = (datetime.now() - start_time).total_seconds()

        # Build transcription result
        if audio_start_time is None:
            audio_start_time = datetime.now()
        if audio_end_time is None:
            audio_end_time = audio_start_time + timedelta(seconds=result["duration"])

        # Assign speakers to segments if diarization was run
        if speaker_segments is not None:
            from .diarizer import assign_speakers
            result["segments"] = assign_speakers(
                result["segments"], speaker_segments
            )

        segments = []
        for i, seg in enumerate(result["segments"]):
            words = []
            if "words" in seg:
                for w in seg["words"]:
                    words.append(TranscriptionWord(
                        word=w.get("word", ""),
                        start=w.get("start", 0),
                        end=w.get("end", 0),
                        probability=w.get("probability", 1.0),
                    ))

            segments.append(TranscriptionSegment(
                id=i,
                text=seg.get("text", ""),
                start=seg.get("start", 0),
                end=seg.get("end", 0),
                words=words,
                avg_logprob=seg.get("avg_logprob", 0),
                compression_ratio=seg.get("compression_ratio", 0),
                no_speech_prob=seg.get("no_speech_prob", 0),
                speaker=seg.get("speaker"),
            ))

        return TranscriptionResult(
            channel_id=channel_id,
            audio_file=audio_path,
            audio_start_time=audio_start_time,
            audio_end_time=audio_end_time,
            text=result["text"],
            segments=segments,
            language=result.get("language", self.language),
            language_probability=result.get("language_probability", 1.0),
            duration_seconds=result["duration"],
            processing_time_seconds=processing_time,
        )

    async def _run_diarization(self, audio_path, loop, diarizer=None):
        """Run speaker diarization on audio file."""
        from .diarizer import SpeakerDiarizer

        if diarizer is None:
            diarizer = SpeakerDiarizer()

        if not diarizer._loaded:
            await loop.run_in_executor(None, diarizer.load_model)

        speaker_segments = await loop.run_in_executor(
            None, diarizer.diarize, audio_path
        )
        return speaker_segments

    def _transcribe_sync(self, audio_path: str) -> dict:
        """Synchronous transcription method."""
        if self._backend == "faster-whisper":
            return self._transcribe_faster_whisper(audio_path)
        else:
            return self._transcribe_openai_whisper(audio_path)

    def _transcribe_faster_whisper(self, audio_path: str) -> dict:
        """Transcribe using faster-whisper."""
        segments_iter, info = self._model.transcribe(
            audio_path,
            language=self.language,
            beam_size=self.beam_size,
            best_of=self.best_of,
            temperature=self.temperature,
            word_timestamps=self.word_timestamps,
            vad_filter=self.vad_filter,
            initial_prompt=self.initial_prompt,
        )

        segments = []
        full_text = []

        for segment in segments_iter:
            seg_dict = {
                "start": segment.start,
                "end": segment.end,
                "text": segment.text,
                "avg_logprob": segment.avg_logprob,
                "compression_ratio": segment.compression_ratio,
                "no_speech_prob": segment.no_speech_prob,
            }

            if segment.words:
                seg_dict["words"] = [
                    {
                        "word": word.word,
                        "start": word.start,
                        "end": word.end,
                        "probability": word.probability,
                    }
                    for word in segment.words
                ]

            segments.append(seg_dict)
            full_text.append(segment.text)

        return {
            "text": " ".join(full_text).strip(),
            "segments": segments,
            "language": info.language,
            "language_probability": info.language_probability,
            "duration": info.duration,
        }

    def _transcribe_openai_whisper(self, audio_path: str) -> dict:
        """Transcribe using OpenAI Whisper."""
        result = self._model.transcribe(
            audio_path,
            language=self.language,
            beam_size=self.beam_size,
            best_of=self.best_of,
            temperature=self.temperature,
            word_timestamps=self.word_timestamps,
            initial_prompt=self.initial_prompt,
        )

        # Get audio duration
        import whisper
        audio = whisper.load_audio(audio_path)
        duration = len(audio) / whisper.audio.SAMPLE_RATE

        segments = []
        for seg in result["segments"]:
            seg_dict = {
                "start": seg["start"],
                "end": seg["end"],
                "text": seg["text"],
                "avg_logprob": seg.get("avg_logprob", 0),
                "compression_ratio": seg.get("compression_ratio", 0),
                "no_speech_prob": seg.get("no_speech_prob", 0),
            }

            if "words" in seg:
                seg_dict["words"] = seg["words"]

            segments.append(seg_dict)

        return {
            "text": result["text"],
            "segments": segments,
            "language": result.get("language", self.language),
            "language_probability": 1.0,
            "duration": duration,
        }

    def unload_model(self) -> None:
        """Unload the model to free memory."""
        if self._model is not None:
            del self._model
            self._model = None
            self._loaded = False

            # Try to free GPU memory
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass

            logger.info("Model unloaded")
