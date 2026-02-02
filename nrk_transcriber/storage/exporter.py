"""
Export transcriptions to various file formats.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class TranscriptionExporter:
    """
    Export transcription results to various file formats.

    Supports: txt, json, srt, vtt, csv
    """

    def __init__(self, output_dir: Path):
        """
        Initialize the exporter.

        Args:
            output_dir: Directory for exported files
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _generate_filename(
        self,
        channel_id: str,
        timestamp: datetime,
        extension: str,
    ) -> Path:
        """Generate a filename for the export."""
        date_str = timestamp.strftime("%Y%m%d")
        time_str = timestamp.strftime("%H%M%S")
        filename = f"{channel_id}_{date_str}_{time_str}.{extension}"

        # Create date-based subdirectory
        date_dir = self.output_dir / channel_id / date_str
        date_dir.mkdir(parents=True, exist_ok=True)

        return date_dir / filename

    def export_txt(self, result, filename: Optional[Path] = None) -> Path:
        """
        Export transcription as plain text.

        Args:
            result: TranscriptionResult object
            filename: Optional custom filename

        Returns:
            Path to the exported file
        """
        if filename is None:
            filename = self._generate_filename(
                result.channel_id,
                result.audio_start_time,
                "txt",
            )

        content = f"""NRK Radio Transcription
Channel: {result.channel_id}
Start: {result.audio_start_time.isoformat()}
End: {result.audio_end_time.isoformat()}
Duration: {result.duration_seconds:.1f} seconds
Language: {result.language}

---

{result.text}
"""

        filename.write_text(content, encoding="utf-8")
        logger.info(f"Exported TXT: {filename}")
        return filename

    def export_json(self, result, filename: Optional[Path] = None) -> Path:
        """
        Export transcription as JSON.

        Args:
            result: TranscriptionResult object
            filename: Optional custom filename

        Returns:
            Path to the exported file
        """
        if filename is None:
            filename = self._generate_filename(
                result.channel_id,
                result.audio_start_time,
                "json",
            )

        content = result.to_json(indent=2)
        filename.write_text(content, encoding="utf-8")
        logger.info(f"Exported JSON: {filename}")
        return filename

    def export_srt(self, result, filename: Optional[Path] = None) -> Path:
        """
        Export transcription as SRT subtitle file.

        Args:
            result: TranscriptionResult object
            filename: Optional custom filename

        Returns:
            Path to the exported file
        """
        if filename is None:
            filename = self._generate_filename(
                result.channel_id,
                result.audio_start_time,
                "srt",
            )

        content = result.to_srt()
        filename.write_text(content, encoding="utf-8")
        logger.info(f"Exported SRT: {filename}")
        return filename

    def export_vtt(self, result, filename: Optional[Path] = None) -> Path:
        """
        Export transcription as WebVTT subtitle file.

        Args:
            result: TranscriptionResult object
            filename: Optional custom filename

        Returns:
            Path to the exported file
        """
        if filename is None:
            filename = self._generate_filename(
                result.channel_id,
                result.audio_start_time,
                "vtt",
            )

        content = result.to_vtt()
        filename.write_text(content, encoding="utf-8")
        logger.info(f"Exported VTT: {filename}")
        return filename

    def export_csv(
        self,
        results: list,
        filename: Optional[Path] = None,
        include_segments: bool = False,
    ) -> Path:
        """
        Export multiple transcriptions as CSV.

        Args:
            results: List of TranscriptionResult objects
            filename: Optional custom filename
            include_segments: Include individual segments

        Returns:
            Path to the exported file
        """
        import csv

        if filename is None:
            now = datetime.now()
            filename = self.output_dir / f"transcriptions_{now.strftime('%Y%m%d_%H%M%S')}.csv"

        with open(filename, "w", newline="", encoding="utf-8") as f:
            if include_segments:
                fieldnames = [
                    "channel_id",
                    "audio_start_time",
                    "segment_id",
                    "segment_start",
                    "segment_end",
                    "text",
                ]
            else:
                fieldnames = [
                    "channel_id",
                    "audio_start_time",
                    "audio_end_time",
                    "duration_seconds",
                    "language",
                    "text",
                ]

            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for result in results:
                if include_segments:
                    for segment in result.segments:
                        writer.writerow({
                            "channel_id": result.channel_id,
                            "audio_start_time": result.audio_start_time.isoformat(),
                            "segment_id": segment.id,
                            "segment_start": segment.start,
                            "segment_end": segment.end,
                            "text": segment.text.strip(),
                        })
                else:
                    writer.writerow({
                        "channel_id": result.channel_id,
                        "audio_start_time": result.audio_start_time.isoformat(),
                        "audio_end_time": result.audio_end_time.isoformat(),
                        "duration_seconds": result.duration_seconds,
                        "language": result.language,
                        "text": result.text,
                    })

        logger.info(f"Exported CSV: {filename}")
        return filename

    def export_all(
        self,
        result,
        formats: Optional[list[str]] = None,
    ) -> dict[str, Path]:
        """
        Export transcription to multiple formats.

        Args:
            result: TranscriptionResult object
            formats: List of formats to export (default: txt, json, srt)

        Returns:
            Dictionary mapping format to exported file path
        """
        if formats is None:
            formats = ["txt", "json", "srt"]

        exported = {}

        for fmt in formats:
            if fmt == "txt":
                exported["txt"] = self.export_txt(result)
            elif fmt == "json":
                exported["json"] = self.export_json(result)
            elif fmt == "srt":
                exported["srt"] = self.export_srt(result)
            elif fmt == "vtt":
                exported["vtt"] = self.export_vtt(result)
            else:
                logger.warning(f"Unknown export format: {fmt}")

        return exported
