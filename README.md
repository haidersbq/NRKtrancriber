# NRK Radio Transcriber

A Python tool to capture and transcribe NRK (Norwegian Broadcasting Corporation) radio streams in real-time using OpenAI's Whisper speech recognition.

## Features

- **Live Stream Capture**: Capture audio from any NRK radio channel in real-time
- **Accurate Transcription**: Uses Whisper for high-quality Norwegian speech recognition
- **Multiple Channels**: Support for all major NRK radio channels (P1, P2, P3, Nyheter, etc.)
- **Multiple Output Formats**: Export transcriptions as TXT, JSON, SRT, VTT, or CSV
- **Database Storage**: SQLite database for storing and searching transcriptions
- **CLI Interface**: Easy-to-use command-line interface
- **Async Support**: Efficient async processing for multiple channels
- **GPU Acceleration**: Optional CUDA support for faster transcription

## Installation

### Prerequisites

- Python 3.10 or higher
- FFmpeg (required for audio stream capture)
- CUDA (optional, for GPU acceleration)

### Install FFmpeg

**Ubuntu/Debian:**
```bash
sudo apt update
sudo apt install ffmpeg
```

**macOS (Homebrew):**
```bash
brew install ffmpeg
```

**Windows:**
Download from [ffmpeg.org](https://ffmpeg.org/download.html) and add to PATH.

### Install the Package

```bash
# Clone the repository
git clone https://github.com/haidersbq/NRKtrancriber.git
cd NRKtrancriber

# Install with pip
pip install -e .

# Or install with GPU support
pip install -e ".[gpu]"
```

## Quick Start

### List Available Channels

```bash
nrk-transcriber channels
```

### Transcribe a Live Stream

```bash
# Transcribe NRK P1 for 5 minutes
nrk-transcriber transcribe nrk_p1 --duration 5

# Transcribe NRK Nyheter (news) continuously
nrk-transcriber transcribe nrk_nyheter

# Use a larger model for better accuracy
nrk-transcriber transcribe nrk_p2 --model large-v3 --duration 10
```

### Transcribe a Local Audio File

```bash
nrk-transcriber file recording.mp3 --format txt json srt
```

### Monitor Multiple Channels

```bash
nrk-transcriber monitor nrk_p1 nrk_nyheter nrk_sport --duration 30
```

### View Transcription History

```bash
# Show recent transcriptions
nrk-transcriber history

# Search for specific content
nrk-transcriber history --search "statsminister"

# Filter by channel
nrk-transcriber history --channel nrk_nyheter
```

### View Statistics

```bash
nrk-transcriber stats
```

## Available Channels

| Channel ID | Name | Description |
|------------|------|-------------|
| `nrk_p1` | NRK P1 | Norway's main radio channel |
| `nrk_p2` | NRK P2 | Culture and classical music |
| `nrk_p3` | NRK P3 | Youth and contemporary music |
| `nrk_p1_pluss` | NRK P1+ | Adult contemporary |
| `nrk_p13` | NRK P13 | Alternative and indie music |
| `nrk_mp3` | NRK mP3 | Electronic and dance music |
| `nrk_klassisk` | NRK Klassisk | Classical music 24/7 |
| `nrk_jazz` | NRK Jazz | Jazz music 24/7 |
| `nrk_folkemusikk` | NRK Folkemusikk | Norwegian folk music |
| `nrk_sport` | NRK Sport | Sports coverage |
| `nrk_nyheter` | NRK Nyheter | 24/7 News |
| `nrk_sapmi` | NRK Sápmi | Sami language programming |
| `nrk_super` | NRK Super | Children's programming |

## Configuration

### Environment Variables

```bash
# Whisper model (tiny, base, small, medium, large, large-v2, large-v3)
export NRK_WHISPER_MODEL=medium

# Device (auto, cpu, cuda)
export NRK_DEVICE=auto

# Output directory
export NRK_OUTPUT_DIR=/path/to/output

# Chunk duration in seconds
export NRK_CHUNK_DURATION=30
```

### Configuration File

Edit `config/channels.yaml` to customize channel settings and defaults.

## Python API

```python
import asyncio
from nrk_transcriber import NRKTranscriber, Config

async def main():
    # Initialize transcriber
    transcriber = NRKTranscriber()
    await transcriber.initialize()

    # Transcribe a channel for 5 minutes
    results = await transcriber.transcribe_channel(
        "nrk_nyheter",
        duration_minutes=5
    )

    # Print transcriptions
    for result in results:
        print(f"[{result.audio_start_time}] {result.text}")

    await transcriber.shutdown()

asyncio.run(main())
```

### Callback for Real-time Processing

```python
async def main():
    def on_transcription(result):
        print(f"New transcription: {result.text[:100]}...")

    transcriber = NRKTranscriber(on_transcription=on_transcription)
    await transcriber.initialize()

    # This will call the callback for each chunk
    await transcriber.transcribe_channel("nrk_p1", duration_minutes=10)

asyncio.run(main())
```

## Whisper Models

| Model | Size | Speed | Accuracy | VRAM |
|-------|------|-------|----------|------|
| tiny | 39M | Fastest | Lower | ~1GB |
| base | 74M | Fast | Good | ~1GB |
| small | 244M | Medium | Better | ~2GB |
| medium | 769M | Slow | Great | ~5GB |
| large-v3 | 1.5B | Slowest | Best | ~10GB |

For Norwegian, the `medium` model provides a good balance of speed and accuracy. Use `large-v3` for the highest quality.

## Output Formats

### TXT
Plain text transcription with metadata header.

### JSON
Full transcription data including word-level timestamps and confidence scores.

### SRT/VTT
Subtitle formats compatible with video players.

### CSV
Tabular format for analysis and data processing.

## Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Format code
black nrk_transcriber
ruff check nrk_transcriber

# Type checking
mypy nrk_transcriber
```

## License

MIT License - see LICENSE file for details.

## Acknowledgments

- [OpenAI Whisper](https://github.com/openai/whisper) for the speech recognition model
- [faster-whisper](https://github.com/guillaumekln/faster-whisper) for the optimized inference engine
- NRK for providing public radio streams
