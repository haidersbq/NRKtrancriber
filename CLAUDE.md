# Media Transcriber

Tool for capturing and transcribing media content from multiple sources using Whisper speech-to-text.

**Supported Providers:**
- **NRK** - Norwegian Broadcasting Corporation (radio, podcasts)
- **YouTube** - YouTube videos via yt-dlp
- **Podcast** - Any podcast RSS feed
- **Direct URLs** - Any MP3, WAV, HLS stream, or audio file

## Quick Start

```bash
# Activate environment
cd ~/NRKtrancriber
source venv/bin/activate

# Transcribe NRK content (auto-detected)
nrk-transcriber download "https://radio.nrk.no/serie/SERIES/sesong/SEASON/EPISODE#t=XmYs" --duration 5 --model small

# Transcribe YouTube video
nrk-transcriber download "https://www.youtube.com/watch?v=VIDEO_ID" --model small

# Transcribe podcast episode (RSS feed)
nrk-transcriber download "https://feeds.example.com/podcast.rss" --model small

# Transcribe direct audio URL
nrk-transcriber download "https://example.com/audio.mp3" --model small

# List available providers
nrk-transcriber providers
```

## CLI Commands

### Download & Transcribe (Any Source)
```bash
# NRK radio (auto-detected)
nrk-transcriber download "https://radio.nrk.no/serie/..."

# With start time (from URL fragment)
nrk-transcriber download "https://radio.nrk.no/serie/...#t=14m19s"

# Limit duration (minutes)
nrk-transcriber download "URL#t=14m19s" --duration 3

# YouTube video (requires yt-dlp)
nrk-transcriber download "https://www.youtube.com/watch?v=VIDEO_ID"
nrk-transcriber download "https://youtu.be/VIDEO_ID?t=90" --duration 5

# Podcast RSS feed (transcribes latest episode)
nrk-transcriber download "https://feeds.example.com/podcast.rss"

# Apple Podcasts (auto-resolves to RSS)
nrk-transcriber download "https://podcasts.apple.com/podcast/id123456789"

# Direct audio file
nrk-transcriber download "https://example.com/audio.mp3"

# Force specific provider
nrk-transcriber download "URL" --provider direct

# Choose model (tiny/small/medium/large/large-v3)
nrk-transcriber download "URL" --model small

# Override language detection
nrk-transcriber download "URL" --language en

# Enable speaker diarization (identify who spoke when)
nrk-transcriber download "URL" --diarize --model small
```

### Live Stream Transcription
```bash
nrk-transcriber transcribe p1 --duration 5  # 5 minutes of NRK P1
nrk-transcriber transcribe p2 --duration 10 --model small
```

### List Providers
```bash
nrk-transcriber providers
```

### Other Commands
```bash
# List NRK channels
nrk-transcriber channels

# Transcribe local audio file
nrk-transcriber file audio.wav --model small

# Transcribe with speaker identification
nrk-transcriber file audio.wav --model small --diarize

# View transcription history
nrk-transcriber history --limit 10

# Search transcriptions
nrk-transcriber history --search "keyword"

# View statistics
nrk-transcriber stats

# Monitor multiple channels
nrk-transcriber monitor p1 p2 --duration 5
```

## Project Structure

```
nrk_transcriber/
├── cli.py                 # Click CLI interface
├── transcriber.py         # Main transcriber orchestration
├── config.py              # Configuration management
├── providers/             # Media provider abstraction
│   ├── base.py            # BaseProvider, MediaProgram classes
│   ├── registry.py        # Provider auto-detection
│   ├── nrk.py             # NRK provider
│   ├── youtube.py         # YouTube provider (yt-dlp)
│   ├── podcast.py         # Podcast RSS provider
│   └── direct.py          # Direct URL provider
├── streams/
│   ├── capture.py         # Live stream capture via ffmpeg
│   └── downloader.py      # On-demand content download (ffmpeg + yt-dlp)
├── transcription/
│   ├── whisper_transcriber.py  # Whisper integration (faster-whisper)
│   └── diarizer.py             # Speaker diarization (pyannote.audio)
├── storage/
│   ├── database.py        # SQLite storage
│   └── exporter.py        # Export to TXT/JSON/SRT/VTT/CSV
└── utils/
    └── logging_config.py  # Logging setup
```

## Key Files

| File | Purpose |
|------|---------|
| `cli.py` | Entry point, defines `download`, `transcribe`, `providers` commands |
| `providers/base.py` | BaseProvider interface, MediaProgram dataclass |
| `providers/nrk.py` | NRK-specific API client |
| `providers/youtube.py` | YouTube via yt-dlp |
| `providers/podcast.py` | Podcast RSS feed parser |
| `providers/direct.py` | Direct audio URL handler |
| `whisper_transcriber.py` | Loads Whisper models, transcribes audio |
| `diarizer.py` | Speaker diarization via pyannote.audio |
| `downloader.py` | Downloads audio (ffmpeg + yt-dlp) |
| `config/channels.yaml` | NRK radio channel configurations |

## NRK URL Format

```
https://radio.nrk.no/serie/{series}/sesong/{season}/{episode}#t={time}
```

Example:
```
https://radio.nrk.no/serie/distriktsprogram-telemark/sesong/202602/DKTE01002126#t=14m19s
```

The `#t=14m19s` fragment specifies start time (14 minutes 19 seconds).

## Whisper Models

| Model | Speed | Quality | Use Case |
|-------|-------|---------|----------|
| tiny | ~32x realtime | Basic | Quick testing |
| small | ~6x realtime | Good | Balanced speed/quality |
| medium | ~2x realtime | Great | Default, good for Norwegian |
| large-v3 | ~1x realtime | Best | Maximum accuracy |

For Norwegian content, `small` or `medium` recommended.

## Speaker Diarization

The `--diarize` flag enables speaker identification — labeling "who spoke when" in the transcript.

### Setup

1. Install the diarization dependencies:
   ```bash
   pip install -e ".[diarize]"
   ```

2. Get a HuggingFace token (free) at https://huggingface.co/settings/tokens

3. Accept the pyannote model terms at https://huggingface.co/pyannote/speaker-diarization-3.1

4. Set your token:
   ```bash
   export HF_TOKEN="hf_your_token_here"
   ```

### Usage

```bash
# Transcribe with speaker labels
nrk-transcriber download "URL" --diarize --model small

# Local file with diarization
nrk-transcriber file interview.wav --diarize --model small
```

### Output with speakers

Plain text output groups consecutive segments by speaker:
```
[SPEAKER_00]
Velkommen til programmet. I dag skal vi snakke om...

[SPEAKER_01]
Takk for invitasjonen. Ja, det er et viktig tema...

[SPEAKER_00]
Kan du fortelle oss mer om bakgrunnen?
```

SRT/VTT subtitle files include speaker labels per segment.

## Output

Transcripts saved to `output/transcripts/{episode}/{date}/` in:
- `.txt` - Plain text
- `.json` - With timestamps and metadata
- `.srt` - Subtitle format

Audio cached in `output/audio/`.

Database at `output/transcriptions.db`.

## Dependencies

- **faster-whisper**: Primary transcription engine (CTranslate2-based)
- **ffmpeg**: Required for audio capture/conversion (install via `brew install ffmpeg`)
- **yt-dlp**: Required for YouTube support (`pip install yt-dlp`)
- **pyannote.audio**: Required for speaker diarization (`pip install pyannote.audio`)
- **Python 3.10+**: Required

### Optional Extras
```bash
pip install -e ".[youtube]"   # YouTube support (yt-dlp)
pip install -e ".[diarize]"   # Speaker diarization (pyannote.audio + torch)
pip install -e ".[all]"       # All optional features
```

## Development

```bash
# Install with dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Format code
black nrk_transcriber/
ruff check nrk_transcriber/
```

## Common Issues

1. **llvmlite build fails**: Use faster-whisper only (openai-whisper is optional)
2. **Command not found**: Activate venv first (`source venv/bin/activate`)
3. **Slow transcription**: Use `--model small` instead of medium/large

## Adding New Providers

To add a new broadcaster/source, create a provider in `providers/`:

```python
# providers/youtube.py
from .base import BaseProvider, MediaProgram, ParsedURL
from .registry import ProviderRegistry

@ProviderRegistry.register
class YouTubeProvider(BaseProvider):
    PROVIDER_ID = "youtube"
    PROVIDER_NAME = "YouTube"
    SUPPORTED_DOMAINS = ["youtube.com", "youtu.be"]
    DEFAULT_LANGUAGE = None

    @classmethod
    def can_handle(cls, url: str) -> bool:
        # Check if URL matches this provider
        ...

    def parse_url(self, url: str) -> ParsedURL:
        # Extract video ID, timestamps, etc.
        ...

    def get_program(self, url_or_id: str) -> MediaProgram:
        # Fetch metadata and return MediaProgram
        ...
```

Then import it in `providers/__init__.py`:
```python
from . import youtube  # Registers automatically via decorator
```

## Future Improvements

- [x] YouTube provider (via yt-dlp)
- [x] Podcast RSS provider
- [ ] BBC Sounds provider
- [ ] SVT/DR Nordic broadcasters
- [ ] `--end-time` option for precise segment selection
- [ ] Search command across transcripts
- [x] Speaker diarization (via pyannote.audio)
- [ ] Auto-translate to English
