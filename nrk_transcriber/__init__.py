"""
NRK Radio Transcriber

A tool to capture and transcribe NRK radio streams using Whisper.
"""

__version__ = "0.1.0"
__author__ = "NRK Transcriber Contributors"

from .config import Config, ChannelConfig
from .transcriber import NRKTranscriber

__all__ = ["Config", "ChannelConfig", "NRKTranscriber", "__version__"]
