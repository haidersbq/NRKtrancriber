"""
Speaker diarization module using pyannote.audio.

Identifies "who spoke when" in audio files and assigns speaker labels
to transcription segments.
"""

import logging
import os
from pathlib import Path
from typing import Optional, Union

# Prevent OpenMP crash when multiple libraries (torch, ctranslate2) each bundle libiomp5
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

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

    def validate_prerequisites(self) -> None:
        """Validate that all prerequisites for diarization are met.

        Call this BEFORE starting a long transcription to fail fast
        instead of wasting 15+ minutes only to crash at the end.

        Raises:
            ImportError: If pyannote.audio is not installed
            ValueError: If HuggingFace token is not available
            RuntimeError: If torch/numpy versions are incompatible
        """
        # 1. Check pyannote.audio is installed
        try:
            import pyannote.audio  # noqa: F401
        except ImportError:
            raise ImportError(
                "pyannote.audio is required for speaker diarization. "
                "Install it with: pip install 'nrk-transcriber[diarize]'"
            )

        # 2. Check HuggingFace token is available
        self._resolve_token()

        # 3. Check torch/numpy compatibility
        self._check_torch_numpy_compat()

    @staticmethod
    def _login_hf(token: str) -> None:
        """Log in to HuggingFace Hub so the token is available globally.

        This avoids passing ``token`` or ``use_auth_token`` as a keyword
        argument to ``Pipeline.from_pretrained``, which breaks when the
        installed pyannote.audio and huggingface_hub versions disagree on
        the parameter name.
        """
        from huggingface_hub import login
        login(token=token, add_to_git_credential=False)

    def _check_torch_numpy_compat(self) -> None:
        """Check that torch and numpy versions are compatible."""
        try:
            import numpy as np
            np_major = int(np.__version__.split(".")[0])
            if np_major >= 2:
                import torch
                torch_version = tuple(int(x) for x in torch.__version__.split(".")[:2])
                if torch_version < (2, 4):
                    raise RuntimeError(
                        f"torch {torch.__version__} is incompatible with numpy {np.__version__}. "
                        f"Fix with: pip install 'numpy<2'\n"
                        f"(PyTorch < 2.4 requires NumPy 1.x)"
                    )
        except ImportError:
            pass

    def load_model(self) -> None:
        """Load the pyannote diarization pipeline."""
        if self._loaded:
            return

        self._check_torch_numpy_compat()

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

        # Log in globally so Pipeline.from_pretrained picks up the token
        # without us having to guess the correct keyword argument name.
        self._login_hf(token)

        self._pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
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


def check_diarization_ready() -> tuple[bool, str]:
    """Check if diarization prerequisites are met.

    Returns:
        (ok, message) tuple. ok=True if ready, False with error message if not.
    """
    try:
        diarizer = SpeakerDiarizer()
        diarizer.validate_prerequisites()
        return True, "Diarization prerequisites OK"
    except ImportError as e:
        return False, str(e)
    except ValueError as e:
        return False, str(e)
    except RuntimeError as e:
        return False, str(e)


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
