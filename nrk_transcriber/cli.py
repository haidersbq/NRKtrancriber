"""
Command-line interface for NRK Radio Transcriber.
"""

import asyncio
import sys
from pathlib import Path
from typing import Optional

import click
from rich.console import Console
from rich.table import Table
from rich.live import Live
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

from . import __version__
from .config import Config
from .transcriber import NRKTranscriber, run_transcriber
from .utils.logging_config import setup_logging

console = Console()


@click.group()
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
    NRK Radio Transcriber - Capture and transcribe NRK radio streams.

    Uses Whisper for accurate Norwegian speech recognition.
    """
    ctx.ensure_object(dict)

    # Setup logging
    setup_logging(level=log_level, log_file=log_file)

    # Load configuration
    ctx.obj["config"] = Config.load(config_dir)


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
@click.pass_context
def file(ctx, audio_file: Path, model: str, output: Optional[Path], format: tuple):
    """
    Transcribe a local audio file.

    Supports common audio formats: WAV, MP3, FLAC, etc.
    """
    config = ctx.obj["config"]
    config.transcription.model = model
    config.storage.export_formats = list(format)

    console.print(f"[bold]Transcribing:[/bold] {audio_file}")
    console.print(f"[bold]Model:[/bold] {model}")

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
                progress.add_task("Loading model and transcribing...", total=None)
                result = await transcriber.transcribe_file(audio_file)

            console.print("\n[green]━━━ Transcription ━━━[/green]")
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
            # Get all recent
            results = await db.get_transcriptions_by_channel("", limit=limit)

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
@click.argument("url")
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
    "--segment-duration",
    "-s",
    type=int,
    default=300,
    help="Process in segments of N seconds (default: 300 = 5 min)",
)
@click.pass_context
def download(
    ctx,
    url: str,
    model: str,
    duration: Optional[int],
    output_dir: Optional[Path],
    keep_audio: bool,
    segment_duration: int,
):
    """
    Download and transcribe on-demand NRK content.

    URL can be a full NRK radio/podcast URL like:
    https://radio.nrk.no/serie/distriktsprogram-telemark/sesong/202602/DKTE01002126

    Supports timestamps in URL (e.g., #t=14m19s) and --duration to limit length.

    Examples:
      # Transcribe 20 minutes starting at 14:19
      nrk-transcriber download "URL#t=14m19s" --duration 20

      # Quick transcribe with small model
      nrk-transcriber download "URL" --model small --duration 5
    """
    from .nrk_api import NRKApiClient
    from .streams import NRKDownloader

    config = ctx.obj["config"]
    config.transcription.model = model

    if output_dir:
        config.storage.output_dir = output_dir
        config.storage.transcripts_dir = output_dir / "transcripts"
        config.storage.audio_dir = output_dir / "audio"
    config.storage.keep_audio_files = keep_audio

    # Convert duration from minutes to seconds
    duration_seconds = duration * 60 if duration else None

    async def run():
        # Parse URL and get program info
        api = NRKApiClient()

        console.print(f"[bold]Fetching program info...[/bold]")

        try:
            parsed = api.parse_nrk_url(url)
            program = api.get_program(url)
        except Exception as e:
            console.print(f"[red]Error fetching program: {e}[/red]")
            sys.exit(1)

        # Handle timestamp offset
        start_time = parsed.get("timestamp_seconds", 0)

        # Calculate what we're transcribing
        transcribe_duration = duration_seconds if duration_seconds else (program.duration_seconds - start_time)
        transcribe_duration_str = f"{transcribe_duration // 60}m {transcribe_duration % 60}s"

        # Show program info
        info_lines = [
            f"[bold]Title:[/bold] {program.title}",
            f"[bold]Series:[/bold] {program.series_title or 'N/A'}",
            f"[bold]Full duration:[/bold] {program.duration_seconds // 60}m {program.duration_seconds % 60}s",
        ]
        if start_time:
            info_lines.append(f"[bold]Start:[/bold] {start_time // 60}m {start_time % 60}s")
        if duration_seconds:
            info_lines.append(f"[bold]Transcribe:[/bold] {transcribe_duration_str}")
        info_lines.extend([
            f"[bold]Model:[/bold] {model}",
            f"[bold]Output:[/bold] {config.storage.transcripts_dir}",
        ])

        console.print(Panel("\n".join(info_lines), title="NRK Program"))

        # Download audio
        downloader = NRKDownloader(
            output_dir=config.storage.audio_dir,
            sample_rate=config.stream.sample_rate,
        )

        transcriber = NRKTranscriber(config=config)
        await transcriber.initialize()

        all_text = []

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

                progress.update(task, description="Transcribing...")

                # Transcribe
                result = await transcriber.transcribe_file(
                    audio.file_path,
                    channel_id=program.program_id,
                )

                all_text.append(result.text)

            # Print transcription
            console.print("\n[green]━━━ Transcription ━━━[/green]")
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
