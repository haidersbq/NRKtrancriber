# NRK Transcriber

Tool for capturing and transcribing NRK (Norwegian Broadcasting Corporation) radio content using Whisper speech-to-text.

## Quick Start

```bash
# Activate environment
cd ~/NRKtrancriber
source venv/bin/activate

# Transcribe on-demand content with timestamp and duration
nrk-transcriber download "https://radio.nrk.no/serie/SERIES/sesong/SEASON/EPISODE#t=XmYs" --duration 5 --model small
```

## CLI Commands

### Download & Transcribe On-Demand Content
```bash
# Basic usage
nrk-transcriber download "NRK_URL"

# With start time (from URL fragment)
nrk-transcriber download "https://radio.nrk.no/serie/...#t=14m19s"

# Limit duration (minutes)
nrk-transcriber download "URL#t=14m19s" --duration 3

# Choose model (tiny/small/medium/large/large-v3)
nrk-transcriber download "URL" --model small
```

### Live Stream Capture
```bash
nrk-transcriber capture p1 --duration 300  # 5 minutes of NRK P1
```

## Project Structure

```
nrk_transcriber/
├── cli.py                 # Click CLI interface
├── nrk_api.py             # NRK API client (fetches metadata, stream URLs)
├── transcriber.py         # Main transcriber orchestration
├── config.py              # Configuration management
├── streams/
│   ├── capture.py         # Live stream capture via ffmpeg
│   └── downloader.py      # On-demand content download
├── transcription/
│   └── whisper_transcriber.py  # Whisper integration (faster-whisper)
├── storage/
│   ├── database.py        # SQLite storage
│   └── exporter.py        # Export to TXT/JSON/SRT/VTT/CSV
└── utils/
    └── logging_config.py  # Logging setup
```

## Key Files

| File | Purpose |
|------|---------|
| `cli.py` | Entry point, defines `download` and `capture` commands |
| `nrk_api.py` | Parses NRK URLs, fetches program info and HLS stream URLs |
| `whisper_transcriber.py` | Loads Whisper models, transcribes audio |
| `downloader.py` | Downloads audio segments via ffmpeg |
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
- **Python 3.10+**: Required

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

## Future Improvements

- [ ] `--end-time` option for precise segment selection
- [ ] Search command across transcripts
- [ ] Speaker diarization
- [ ] NRK TV support
- [ ] Auto-translate to English
