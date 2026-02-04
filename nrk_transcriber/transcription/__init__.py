"""Transcription modules."""

from .whisper_transcriber import WhisperTranscriber, TranscriptionResult, TranscriptionSegment

__all__ = ["WhisperTranscriber", "TranscriptionResult", "TranscriptionSegment"]

# Lazy import - diarizer requires optional pyannote.audio dependency
def get_diarizer(**kwargs):
    """Get a SpeakerDiarizer instance. Requires pyannote.audio."""
    from .diarizer import SpeakerDiarizer
    return SpeakerDiarizer(**kwargs)
