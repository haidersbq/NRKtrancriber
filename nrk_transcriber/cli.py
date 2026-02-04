"""
Command-line interface for Media Transcriber.

Supports multiple media providers including NRK, direct URLs, and more.
"""

import os

# Must be set before any library imports — ctranslate2 (faster-whisper) and torch
# (pyannote) each bundle their own copy of libiomp5 (OpenMP). When both load in the
# same process, macOS aborts unless this flag is set.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import asyncio
import sys
from pathlib import Path
from typing import Optional

import click
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

from rich.prompt import Prompt, Confirm

from . import __version__
from .config import Config
from .transcriber import NRKTranscriber, run_transcriber
from .utils.logging_config import setup_logging
from .providers import ProviderRegistry, get_provider

console = Console()


def _check_diarization_prerequisites() -> bool:
    """Check diarization prerequisites and print helpful errors. Returns True if OK."""
    from .transcription.diarizer import check_diarization_ready

    ok, message = check_diarization_ready()
    if not ok:
        console.print(f"\n[red]Diarization cannot proceed:[/red] {message}")
        if "HF_TOKEN" in message or "HuggingFace token" in message:
            console.print(
                "\n[yellow]To fix:[/yellow]\n"
                "  1. Get a free token at https://huggingface.co/settings/tokens\n"
                "  2. Accept model terms at https://huggingface.co/pyannote/speaker-diarization-3.1\n"
                "  3. Run: export HF_TOKEN=\"hf_your_token_here\"\n"
                "  4. Then retry your command."
            )
        elif "pyannote" in message:
            console.print(
                "\n[yellow]To fix:[/yellow] pip install 'nrk-transcriber[diarize]'"
            )
        elif "numpy" in message.lower() or "torch" in message.lower():
            console.print(
                "\n[yellow]To fix:[/yellow] pip install 'numpy<2'"
            )
        return False
    return True

MODELS = {
    "1": "tiny",
    "2": "small",
    "3": "medium",
    "4": "large-v3",
}


def _interactive_mode(ctx):
    """Interactive mode - prompts the user step by step."""
    config = ctx.obj["config"]

    # Welcome
    console.print()
    console.print(Panel(
        "[bold]Media Transcriber[/bold]\n"
        "Transcribe audio from URLs, files, or live streams.\n"
        "Supports NRK, YouTube, podcasts, and direct audio links.",
        border_style="blue",
    ))

    # Step 1: Get URL or file path
    console.print()
    source = Prompt.ask(
        "[bold cyan]Paste a URL or file path[/bold cyan]"
    ).strip()

    if not source:
        console.print("[red]No input provided.[/red]")
        return

    # Check if it's a local file
    source_path = Path(source)
    is_local_file = source_path.exists() and source_path.is_file()

    if is_local_file:
        console.print(f"[dim]Local file detected: {source_path.name}[/dim]")
    else:
        # Try to detect provider
        api = ProviderRegistry.detect_provider(source)
        if api:
            console.print(f"[dim]Detected provider: {api.PROVIDER_NAME}[/dim]")
        else:
            console.print("[yellow]Could not auto-detect provider. Will try as direct URL.[/yellow]")

    # Step 2: Model selection
    console.print()
    console.print("[bold cyan]Choose model:[/bold cyan]")
    console.print("  [dim]1[/dim] tiny    - fastest, lower quality")
    console.print("  [dim]2[/dim] small   - good balance")
    console.print("  [dim]3[/dim] medium  - high quality (default)")
    console.print("  [dim]4[/dim] large   - best quality, slowest")
    model_choice = Prompt.ask(
        "  Select",
        choices=["1", "2", "3", "4"],
        default="3",
        show_choices=False,
    )
    model = MODELS[model_choice]
    console.print(f"[dim]  Using model: {model}[/dim]")

    # Step 3: Duration (only for URLs, not local files)
    duration_minutes = None
    if not is_local_file:
        console.print()
        dur_input = Prompt.ask(
            "[bold cyan]Duration in minutes[/bold cyan] [dim](Enter for full)[/dim]",
            default="",
            show_default=False,
        )
        if dur_input.strip():
            try:
                duration_minutes = int(dur_input.strip())
            except ValueError:
                console.print("[yellow]Invalid number, transcribing full content.[/yellow]")

    # Step 4: Speaker diarization
    console.print()
    diarize = Confirm.ask(
        "[bold cyan]Identify different speakers?[/bold cyan]",
        default=False,
    )

    if diarize and not _check_diarization_prerequisites():
        return

    # Step 5: Language override
    console.print()
    lang_input = Prompt.ask(
        "[bold cyan]Language[/bold cyan] [dim](Enter for auto, e.g. 'no', 'en')[/dim]",
        default="",
        show_default=False,
    )
    language = lang_input.strip() if lang_input.strip() else None

    # Summary
    console.print()
    summary_lines = [f"[bold]Source:[/bold] {source}"]
    summary_lines.append(f"[bold]Model:[/bold] {model}")
    if duration_minutes:
        summary_lines.append(f"[bold]Duration:[/bold] {duration_minutes} min")
    if diarize:
        summary_lines.append(f"[bold]Speakers:[/bold] enabled")
    if language:
        summary_lines.append(f"[bold]Language:[/bold] {language}")
    console.print(Panel("\n".join(summary_lines), title="Ready to transcribe", border_style="green"))

    console.print()
    if not Confirm.ask("[bold]Start transcription?[/bold]", default=True):
        console.print("[yellow]Cancelled.[/yellow]")
        return

    # Run transcription
    config.transcription.model = model

    if is_local_file:
        _run_local_transcription(config, source_path, model, diarize, language)
    else:
        _run_url_transcription(config, source, model, duration_minutes, diarize, language)


def _run_local_transcription(config, audio_file, model, diarize, language):
    """Run transcription on a local file (called from interactive mode)."""
    async def run():
        transcriber = NRKTranscriber(config=config)
        try:
            await transcriber.initialize()
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                TimeElapsedColumn(),
                console=console,
            ) as progress:
                desc = "Transcribing + diarizing..." if diarize else "Transcribing..."
                progress.add_task(desc, total=None)
                result = await transcriber.transcribe_file(
                    audio_file, diarize=diarize,
                    language=language,
                )

            console.print("\n[green]━━━ Transcription ━━━[/green]")
            if result.has_speakers:
                console.print(result.to_speaker_text())
            else:
                console.print(result.text)

            console.print(f"\n[bold]Language:[/bold] {result.language} ({result.language_probability:.1%})")
            console.print(f"[bold]Duration:[/bold] {result.duration_seconds:.1f}s")
            console.print(f"[bold]Processing time:[/bold] {result.processing_time_seconds:.1f}s")
            console.print(f"[bold]Speed:[/bold] {result.duration_seconds / result.processing_time_seconds:.1f}x realtime")
        finally:
            await transcriber.shutdown()

    asyncio.run(run())


def _run_url_transcription(config, url, model, duration_minutes, diarize, language):
    """Run transcription on a URL (called from interactive mode)."""
    from .streams import NRKDownloader

    duration_seconds = duration_minutes * 60 if duration_minutes else None

    async def run():
        # Detect provider
        api = ProviderRegistry.detect_provider(url)
        if not api:
            # Fall back to direct provider
            api = ProviderRegistry.get_provider("direct")

        try:
            program = api.get_program(url)
        except Exception as e:
            console.print(f"[red]Error fetching program: {e}[/red]")
            return

        start_time = program.start_time_seconds or 0
        transcribe_language = language or program.language or api.DEFAULT_LANGUAGE

        # Download and transcribe
        downloader = NRKDownloader(
            output_dir=config.storage.audio_dir,
            sample_rate=config.stream.sample_rate,
        )
        transcriber = NRKTranscriber(config=config)
        await transcriber.initialize()

        try:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                TimeElapsedColumn(),
                console=console,
            ) as progress:
                task = progress.add_task("Downloading audio...", total=None)

                audio = await downloader.download(
                    audio_url=program.audio_url,
                    program_id=program.program_id,
                    title=program.title,
                    start_time=start_time if start_time else None,
                    duration=duration_seconds,
                )

                desc = "Transcribing + diarizing..." if diarize else "Transcribing..."
                progress.update(task, description=desc)

                result = await transcriber.transcribe_file(
                    audio.file_path,
                    channel_id=program.program_id,
                    language=transcribe_language,
                    diarize=diarize,
                )

            console.print("\n[green]━━━ Transcription ━━━[/green]")
            if result.has_speakers:
                console.print(result.to_speaker_text())
            else:
                console.print(result.text)

            console.print(f"\n[bold]Duration:[/bold] {result.duration_seconds:.1f}s")
            console.print(f"[bold]Processing time:[/bold] {result.processing_time_seconds:.1f}s")
            console.print(f"[bold]Speed:[/bold] {result.duration_seconds / result.processing_time_seconds:.1f}x realtime")

            # Cleanup audio
            if audio.file_path.exists():
                audio.file_path.unlink()

        except KeyboardInterrupt:
            console.print("\n[yellow]Cancelled[/yellow]")
        finally:
            await transcriber.shutdown()

    asyncio.run(run())


@click.group(invoke_without_command=True)
@click.version_option(version=__version__, prog_name="nrk-transcriber")
@click.option(
    "--config-dir",
    type=click.Path(exists=True, path_type=Path),
    help="Path to configuration directory",
)
@click.option(
    "--log-level",
    type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"]),
    default="INFO",
    help="Logging level",
)
@click.option(
    "--log-file",
    type=click.Path(path_type=Path),
    help="Log to file",
)
@click.pass_context
def cli(ctx, config_dir: Optional[Path], log_level: str, log_file: Optional[Path]):
    """
    Media Transcriber - Transcribe audio from NRK, YouTube, podcasts, and more.

    Run without arguments for interactive mode, or use a subcommand.
    """
    ctx.ensure_object(dict)

    # Setup logging
    setup_logging(level=log_level, log_file=log_file)

    # Load configuration
    ctx.obj["config"] = Config.load(config_dir)

    # If no subcommand given, launch interactive mode
    if ctx.invoked_subcommand is None:
        _interactive_mode(ctx)


@cli.command()
@click.pass_context
def channels(ctx):
    """List available NRK radio channels."""
    config = ctx.obj["config"]

    table = Table(title="Available NRK Radio Channels")
    table.add_column("ID", style="cyan")
    table.add_column("Name", style="green")
    table.add_column("Category", style="yellow")
    table.add_column("Language")
    table.add_column("Description")

    for channel in config.list_channels():
        table.add_row(
            channel.channel_id,
            channel.name,
            channel.category,
            channel.language,
            channel.description[:40] + "..." if len(channel.description) > 40 else channel.description,
        )

    console.print(table)


@cli.command()
@click.argument("channel_id")
@click.option(
    "--duration",
    "-d",
    type=float,
    help="Duration in minutes to transcribe",
)
@click.option(
    "--chunks",
    "-n",
    type=int,
    help="Number of chunks to process",
)
@click.option(
    "--model",
    "-m",
    type=click.Choice(["tiny", "base", "small", "medium", "large", "large-v2", "large-v3"]),
    default="medium",
    help="Whisper model size",
)
@click.option(
    "--output-dir",
    "-o",
    type=click.Path(path_type=Path),
    help="Output directory for transcriptions",
)
@click.option(
    "--keep-audio",
    is_flag=True,
    help="Keep audio files after transcription",
)
@click.pass_context
def transcribe(
    ctx,
    channel_id: str,
    duration: Optional[float],
    chunks: Optional[int],
    model: str,
    output_dir: Optional[Path],
    keep_audio: bool,
):
    """
    Transcribe a live NRK radio stream.

    CHANNEL_ID is the ID of the channel to transcribe (use 'channels' command to list).
    """
    config = ctx.obj["config"]

    # Validate channel
    channel = config.get_channel(channel_id)
    if not channel:
        console.print(f"[red]Error: Unknown channel '{channel_id}'[/red]")
        console.print("Use 'nrk-transcriber channels' to see available channels.")
        sys.exit(1)

    # Apply overrides
    config.transcription.model = model
    if output_dir:
        config.storage.output_dir = output_dir
        config.storage.transcripts_dir = output_dir / "transcripts"
        config.storage.audio_dir = output_dir / "audio"
    config.storage.keep_audio_files = keep_audio

    console.print(Panel(
        f"[bold]Channel:[/bold] {channel.name}\n"
        f"[bold]Model:[/bold] {model}\n"
        f"[bold]Duration:[/bold] {duration or 'continuous'} minutes\n"
        f"[bold]Output:[/bold] {config.storage.transcripts_dir}",
        title="NRK Radio Transcriber",
    ))

    def on_transcription(result):
        console.print(f"\n[green]━━━ {result.audio_start_time.strftime('%H:%M:%S')} ━━━[/green]")
        console.print(result.text)

    async def run():
        transcriber = NRKTranscriber(config=config, on_transcription=on_transcription)

        try:
            await transcriber.initialize()

            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                TimeElapsedColumn(),
                console=console,
            ) as progress:
                progress.add_task(f"Transcribing {channel.name}...", total=None)

                await transcriber.transcribe_channel(
                    channel_id,
                    duration_minutes=duration,
                    max_chunks=chunks,
                )

        except KeyboardInterrupt:
            console.print("\n[yellow]Stopping transcription...[/yellow]")
        finally:
            await transcriber.shutdown()
            stats = transcriber.get_statistics()
            console.print(Panel(
                f"[bold]Chunks processed:[/bold] {stats['chunks_processed']}\n"
                f"[bold]Audio transcribed:[/bold] {stats['total_audio_seconds']:.1f}s\n"
                f"[bold]Processing time:[/bold] {stats['total_processing_seconds']:.1f}s\n"
                f"[bold]Realtime factor:[/bold] {stats['realtime_factor']:.2f}x",
                title="Statistics",
            ))

    asyncio.run(run())


@cli.command()
@click.argument("audio_file", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--model",
    "-m",
    type=click.Choice(["tiny", "base", "small", "medium", "large", "large-v2", "large-v3"]),
    default="medium",
    help="Whisper model size",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(path_type=Path),
    help="Output file path",
)
@click.option(
    "--format",
    "-f",
    type=click.Choice(["txt", "json", "srt", "vtt"]),
    multiple=True,
    default=["txt", "json"],
    help="Output format(s)",
)
@click.option(
    "--diarize",
    is_flag=True,
    help="Enable speaker diarization (identify who spoke when). Requires pyannote.audio and HF_TOKEN.",
)
@click.pass_context
def file(ctx, audio_file: Path, model: str, output: Optional[Path], format: tuple, diarize: bool):
    """
    Transcribe a local audio file.

    Supports common audio formats: WAV, MP3, FLAC, etc.
    """
    config = ctx.obj["config"]
    config.transcription.model = model
    config.storage.export_formats = list(format)

    # Validate diarization prerequisites BEFORE transcribing
    if diarize and not _check_diarization_prerequisites():
        sys.exit(1)

    console.print(f"[bold]Transcribing:[/bold] {audio_file}")
    console.print(f"[bold]Model:[/bold] {model}")
    if diarize:
        console.print(f"[bold]Diarization:[/bold] enabled")

    async def run():
        transcriber = NRKTranscriber(config=config)

        try:
            await transcriber.initialize()

            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                TimeElapsedColumn(),
                console=console,
            ) as progress:
                desc = "Loading model and transcribing..."
                if diarize:
                    desc = "Loading models, transcribing + diarizing..."
                progress.add_task(desc, total=None)
                result = await transcriber.transcribe_file(audio_file, diarize=diarize)

            console.print("\n[green]━━━ Transcription ━━━[/green]")
            if result.has_speakers:
                console.print(result.to_speaker_text())
            else:
                console.print(result.text)

            console.print(f"\n[bold]Language:[/bold] {result.language} ({result.language_probability:.1%})")
            console.print(f"[bold]Duration:[/bold] {result.duration_seconds:.1f}s")
            console.print(f"[bold]Processing time:[/bold] {result.processing_time_seconds:.1f}s")

        finally:
            await transcriber.shutdown()

    asyncio.run(run())


@cli.command()
@click.option(
    "--channel",
    "-c",
    help="Filter by channel ID",
)
@click.option(
    "--search",
    "-s",
    help="Search transcriptions by text",
)
@click.option(
    "--limit",
    "-n",
    type=int,
    default=10,
    help="Number of results to show",
)
@click.pass_context
def history(ctx, channel: Optional[str], search: Optional[str], limit: int):
    """View transcription history."""
    config = ctx.obj["config"]

    async def run():
        from .storage import TranscriptionDatabase

        db = TranscriptionDatabase(config.storage.database_path)
        await db.initialize()

        if search:
            results = await db.search_transcriptions(search, channel_id=channel, limit=limit)
        elif channel:
            results = await db.get_transcriptions_by_channel(channel, limit=limit)
        else:
            results = await db.get_recent_transcriptions(limit=limit)

        if not results:
            console.print("[yellow]No transcriptions found.[/yellow]")
            return

        for record in results:
            console.print(Panel(
                record.text[:500] + "..." if len(record.text) > 500 else record.text,
                title=f"{record.channel_id} - {record.audio_start_time.strftime('%Y-%m-%d %H:%M:%S')}",
            ))

        await db.close()

    asyncio.run(run())


@cli.command()
@click.pass_context
def stats(ctx):
    """Show transcription statistics."""
    config = ctx.obj["config"]

    async def run():
        from .storage import TranscriptionDatabase

        db = TranscriptionDatabase(config.storage.database_path)
        await db.initialize()

        statistics = await db.get_statistics()

        table = Table(title="Transcription Statistics")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")

        table.add_row("Total transcriptions", str(statistics["total_transcriptions"]))
        table.add_row(
            "Total audio duration",
            f"{statistics['total_audio_duration_seconds'] / 3600:.1f} hours"
        )
        table.add_row(
            "Total processing time",
            f"{statistics['total_processing_time_seconds'] / 3600:.1f} hours"
        )

        if statistics["total_processing_time_seconds"] > 0:
            rtf = statistics["total_audio_duration_seconds"] / statistics["total_processing_time_seconds"]
            table.add_row("Average realtime factor", f"{rtf:.2f}x")

        console.print(table)

        await db.close()

    asyncio.run(run())


@cli.command()
@click.argument("channel_ids", nargs=-1, required=True)
@click.option(
    "--duration",
    "-d",
    type=float,
    required=True,
    help="Duration in minutes",
)
@click.pass_context
def monitor(ctx, channel_ids: tuple, duration: float):
    """
    Monitor and transcribe multiple channels.

    Useful for capturing specific time periods across multiple channels.
    """
    config = ctx.obj["config"]

    # Validate channels
    for channel_id in channel_ids:
        if not config.get_channel(channel_id):
            console.print(f"[red]Error: Unknown channel '{channel_id}'[/red]")
            sys.exit(1)

    console.print(f"[bold]Monitoring {len(channel_ids)} channels for {duration} minutes[/bold]")

    async def run():
        results = await run_transcriber(
            list(channel_ids),
            duration_minutes=duration,
            config=config,
        )

        for channel_id, channel_results in results.items():
            console.print(f"\n[green]{channel_id}:[/green] {len(channel_results)} transcriptions")

    asyncio.run(run())


@cli.command()
@click.pass_context
def providers(ctx):
    """List available media providers."""
    provider_list = ProviderRegistry.list_providers()

    table = Table(title="Available Media Providers")
    table.add_column("ID", style="cyan")
    table.add_column("Name", style="green")
    table.add_column("Domains")
    table.add_column("Language")

    for p in provider_list:
        domains = ", ".join(p["domains"][:3])
        if len(p["domains"]) > 3:
            domains += "..."
        table.add_row(
            p["id"],
            p["name"],
            domains or "(any audio URL)",
            p["language"] or "-",
        )

    console.print(table)
    console.print("\n[dim]Use --provider to force a specific provider[/dim]")


@cli.command()
@click.argument("url")
@click.option(
    "--provider",
    "-p",
    help="Force specific provider (auto-detected if not specified)",
)
@click.option(
    "--model",
    "-m",
    type=click.Choice(["tiny", "base", "small", "medium", "large", "large-v2", "large-v3"]),
    default="medium",
    help="Whisper model size",
)
@click.option(
    "--duration",
    "-d",
    type=int,
    help="Duration in minutes to transcribe (from start time)",
)
@click.option(
    "--output-dir",
    "-o",
    type=click.Path(path_type=Path),
    help="Output directory for transcriptions",
)
@click.option(
    "--keep-audio",
    is_flag=True,
    help="Keep audio files after transcription",
)
@click.option(
    "--language",
    "-l",
    help="Override language detection (e.g., 'en', 'no', 'sv')",
)
@click.option(
    "--diarize",
    is_flag=True,
    help="Enable speaker diarization (identify who spoke when). Requires pyannote.audio and HF_TOKEN.",
)
@click.pass_context
def download(
    ctx,
    url: str,
    provider: Optional[str],
    model: str,
    duration: Optional[int],
    output_dir: Optional[Path],
    keep_audio: bool,
    language: Optional[str],
    diarize: bool,
):
    """
    Download and transcribe media from any supported source.

    Supports NRK, direct audio URLs (MP3, WAV, HLS), and more.
    Provider is auto-detected from the URL.

    Examples:
      # NRK radio (auto-detected)
      transcriber download "https://radio.nrk.no/serie/..."

      # With timestamp and duration
      transcriber download "URL#t=14m19s" --duration 20

      # Direct audio file
      transcriber download "https://example.com/audio.mp3"

      # Force provider
      transcriber download "URL" --provider direct
    """
    from .streams import NRKDownloader

    config = ctx.obj["config"]
    config.transcription.model = model

    if output_dir:
        config.storage.output_dir = output_dir
        config.storage.transcripts_dir = output_dir / "transcripts"
        config.storage.audio_dir = output_dir / "audio"
    config.storage.keep_audio_files = keep_audio

    # Validate diarization prerequisites BEFORE downloading/transcribing
    if diarize and not _check_diarization_prerequisites():
        sys.exit(1)

    # Convert duration from minutes to seconds
    duration_seconds = duration * 60 if duration else None

    async def run():
        # Get provider (auto-detect or forced)
        console.print(f"[bold]Fetching program info...[/bold]")

        try:
            if provider:
                api = ProviderRegistry.get_provider(provider)
                console.print(f"[dim]Using provider: {api.PROVIDER_NAME}[/dim]")
            else:
                api = ProviderRegistry.detect_provider(url)
                if not api:
                    console.print(f"[red]Could not detect provider for URL.[/red]")
                    console.print("Use --provider to specify one. Available providers:")
                    for p in ProviderRegistry.list_providers():
                        console.print(f"  - {p['id']}: {p['name']}")
                    sys.exit(1)
                console.print(f"[dim]Detected provider: {api.PROVIDER_NAME}[/dim]")

            program = api.get_program(url)
        except Exception as e:
            console.print(f"[red]Error fetching program: {e}[/red]")
            sys.exit(1)

        # Handle timestamp offset
        start_time = program.start_time_seconds or 0

        # Calculate what we're transcribing
        if program.duration_seconds > 0:
            transcribe_duration = duration_seconds if duration_seconds else (program.duration_seconds - start_time)
        else:
            transcribe_duration = duration_seconds or 0
        transcribe_duration_str = f"{transcribe_duration // 60}m {transcribe_duration % 60}s" if transcribe_duration else "unknown"

        # Show program info
        info_lines = [
            f"[bold]Provider:[/bold] {api.PROVIDER_NAME}",
            f"[bold]Title:[/bold] {program.title}",
        ]
        if program.series_title:
            info_lines.append(f"[bold]Series:[/bold] {program.series_title}")
        if program.duration_seconds > 0:
            info_lines.append(f"[bold]Full duration:[/bold] {program.duration_seconds // 60}m {program.duration_seconds % 60}s")
        if start_time:
            info_lines.append(f"[bold]Start:[/bold] {start_time // 60}m {start_time % 60}s")
        if duration_seconds:
            info_lines.append(f"[bold]Transcribe:[/bold] {transcribe_duration_str}")
        info_lines.extend([
            f"[bold]Model:[/bold] {model}",
            f"[bold]Diarization:[/bold] {'enabled' if diarize else 'disabled'}",
            f"[bold]Output:[/bold] {config.storage.transcripts_dir}",
        ])

        console.print(Panel("\n".join(info_lines), title="Media Program"))

        # Download audio
        downloader = NRKDownloader(
            output_dir=config.storage.audio_dir,
            sample_rate=config.stream.sample_rate,
        )

        transcriber = NRKTranscriber(config=config)
        await transcriber.initialize()

        try:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                TimeElapsedColumn(),
                console=console,
            ) as progress:
                # Download
                task = progress.add_task("Downloading audio...", total=None)

                audio = await downloader.download(
                    audio_url=program.audio_url,
                    program_id=program.program_id,
                    title=program.title,
                    start_time=start_time if start_time else None,
                    duration=duration_seconds,
                )

                # Transcribe (with language override if specified)
                transcribe_language = language or program.language or api.DEFAULT_LANGUAGE

                if diarize:
                    progress.update(task, description="Transcribing + diarizing speakers...")
                else:
                    progress.update(task, description="Transcribing...")

                result = await transcriber.transcribe_file(
                    audio.file_path,
                    channel_id=program.program_id,
                    language=transcribe_language,
                    diarize=diarize,
                )

            # Print transcription
            console.print("\n[green]━━━ Transcription ━━━[/green]")
            if result.has_speakers:
                console.print(result.to_speaker_text())
            else:
                console.print(result.text)

            console.print(f"\n[bold]Duration:[/bold] {result.duration_seconds:.1f}s")
            console.print(f"[bold]Processing time:[/bold] {result.processing_time_seconds:.1f}s")
            console.print(f"[bold]Speed:[/bold] {result.duration_seconds / result.processing_time_seconds:.1f}x realtime")

            # Cleanup audio if not keeping
            if not keep_audio and audio.file_path.exists():
                audio.file_path.unlink()

        except KeyboardInterrupt:
            console.print("\n[yellow]Cancelled[/yellow]")
        finally:
            await transcriber.shutdown()

    asyncio.run(run())


def main():
    """Entry point."""
    cli(obj={})


if __name__ == "__main__":
    main()
