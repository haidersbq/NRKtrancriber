"""
Speaker diarization module using pyannote.audio.

Identifies "who spoke when" in audio files and assigns speaker labels
to transcription segments.
"""

import logging
from pathlib import Path
from typing import Optional, Union

logger = logging.getLogger(__name__)


class SpeakerSegment:
    """A segment of audio attributed to a specific speaker."""

    __slots__ = ("speaker", "start", "end")

    def __init__(self, speaker: str, start: float, end: float):
        self.speaker = speaker
        self.start = start
        self.end = end

    def __repr__(self) -> str:
        return f"SpeakerSegment({self.speaker}, {self.start:.1f}-{self.end:.1f})"


class SpeakerDiarizer:
    """
    Speaker diarization using pyannote.audio.

    Requires a HuggingFace token with access to pyannote models.
    Set HF_TOKEN environment variable or pass token directly.

    Usage:
        diarizer = SpeakerDiarizer(hf_token="hf_...")
        diarizer.load_model()
        segments = diarizer.diarize("/path/to/audio.wav")
    """

    def __init__(
        self,
        hf_token: Optional[str] = None,
        device: str = "auto",
        min_speakers: Optional[int] = None,
        max_speakers: Optional[int] = None,
    ):
        """
        Initialize the diarizer.

        Args:
            hf_token: HuggingFace token for accessing pyannote models.
                      Falls back to HF_TOKEN environment variable.
            device: Device to use (auto, cpu, cuda)
            min_speakers: Minimum expected number of speakers
            max_speakers: Maximum expected number of speakers
        """
        self.hf_token = hf_token
        self.device = device
        self.min_speakers = min_speakers
        self.max_speakers = max_speakers
        self._pipeline = None
        self._loaded = False

    def _resolve_token(self) -> str:
        """Resolve the HuggingFace token."""
        if self.hf_token:
            return self.hf_token

        import os
        token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
        if not token:
            raise ValueError(
                "HuggingFace token required for pyannote.audio. "
                "Set HF_TOKEN environment variable or pass hf_token parameter. "
                "Get a token at https://huggingface.co/settings/tokens "
                "and accept the model terms at https://huggingface.co/pyannote/speaker-diarization-3.1"
            )
        return token

    def _resolve_device(self) -> str:
        """Resolve the device to use."""
        if self.device != "auto":
            return self.device

        try:
            import torch
            if torch.cuda.is_available():
                return "cuda"
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
        except ImportError:
            pass

        return "cpu"

    def load_model(self) -> None:
        """Load the pyannote diarization pipeline."""
        if self._loaded:
            return

        try:
            from pyannote.audio import Pipeline
        except ImportError:
            raise ImportError(
                "pyannote.audio is required for speaker diarization. "
                "Install it with: pip install 'nrk-transcriber[diarize]'"
            )

        import torch

        token = self._resolve_token()
        device = self._resolve_device()

        logger.info(f"Loading pyannote diarization pipeline on {device}")

        self._pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=token,
        )

        if device != "cpu":
            self._pipeline.to(torch.device(device))

        self._loaded = True
        logger.info("Diarization pipeline loaded")

    def diarize(self, audio_path: Union[str, Path]) -> list[SpeakerSegment]:
        """
        Run speaker diarization on an audio file.

        Args:
            audio_path: Path to the audio file (WAV recommended)

        Returns:
            List of SpeakerSegment with speaker labels and timestamps
        """
        if not self._loaded:
            self.load_model()

        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        logger.info(f"Running diarization on {audio_path.name}")

        # Build diarization parameters
        params = {}
        if self.min_speakers is not None:
            params["min_speakers"] = self.min_speakers
        if self.max_speakers is not None:
            params["max_speakers"] = self.max_speakers

        diarization = self._pipeline(str(audio_path), **params)

        # Convert pyannote output to our SpeakerSegment format
        segments = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            segments.append(SpeakerSegment(
                speaker=speaker,
                start=turn.start,
                end=turn.end,
            ))

        # Sort by start time
        segments.sort(key=lambda s: s.start)

        logger.info(f"Diarization complete: {len(segments)} segments, "
                     f"{len(set(s.speaker for s in segments))} speakers")

        return segments

    def unload_model(self) -> None:
        """Unload the model to free memory."""
        if self._pipeline is not None:
            del self._pipeline
            self._pipeline = None
            self._loaded = False

            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass

            logger.info("Diarization model unloaded")


def assign_speakers(
    transcription_segments: list[dict],
    speaker_segments: list[SpeakerSegment],
) -> list[dict]:
    """
    Assign speaker labels to transcription segments based on timing overlap.

    For each transcription segment, finds the speaker who spoke for the
    longest duration during that segment's time window.

    Args:
        transcription_segments: List of segment dicts with 'start' and 'end' keys
        speaker_segments: List of SpeakerSegment from diarization

    Returns:
        The same segment dicts with 'speaker' key added
    """
    for seg in transcription_segments:
        seg_start = seg.get("start", 0)
        seg_end = seg.get("end", 0)

        if seg_start == seg_end:
            seg["speaker"] = None
            continue

        # Find which speaker has most overlap with this segment
        speaker_overlap: dict[str, float] = {}

        for spk_seg in speaker_segments:
            # Calculate overlap
            overlap_start = max(seg_start, spk_seg.start)
            overlap_end = min(seg_end, spk_seg.end)
            overlap = max(0, overlap_end - overlap_start)

            if overlap > 0:
                speaker_overlap[spk_seg.speaker] = (
                    speaker_overlap.get(spk_seg.speaker, 0) + overlap
                )

        if speaker_overlap:
            seg["speaker"] = max(speaker_overlap, key=speaker_overlap.get)
        else:
            seg["speaker"] = None

    return transcription_segments
