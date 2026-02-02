#!/usr/bin/env python3
"""
Example of monitoring multiple NRK radio channels simultaneously.

This script captures and transcribes from multiple channels at once,
useful for monitoring news coverage across different programs.
"""

import asyncio
from datetime import datetime
from nrk_transcriber import NRKTranscriber
from nrk_transcriber.utils.logging_config import setup_logging


async def main():
    setup_logging(level="INFO")

    # Track transcriptions by channel
    transcription_counts = {}

    def on_transcription(result):
        channel = result.channel_id
        transcription_counts[channel] = transcription_counts.get(channel, 0) + 1

        timestamp = result.audio_start_time.strftime('%H:%M:%S')
        preview = result.text[:100] + "..." if len(result.text) > 100 else result.text

        print(f"[{timestamp}] [{channel}] {preview}")

    transcriber = NRKTranscriber(on_transcription=on_transcription)

    try:
        await transcriber.initialize()

        # Monitor news and sports for 5 minutes
        channels = ["nrk_nyheter", "nrk_p1", "nrk_sport"]

        print(f"Monitoring {len(channels)} channels for 5 minutes...")
        print(f"Channels: {', '.join(channels)}")
        print()

        results = await transcriber.transcribe_multiple_channels(
            channels,
            duration_minutes=5,
        )

        # Print summary
        print("\n" + "=" * 60)
        print("Summary")
        print("=" * 60)

        for channel_id, channel_results in results.items():
            total_text = " ".join(r.text for r in channel_results)
            word_count = len(total_text.split())
            print(f"{channel_id}: {len(channel_results)} chunks, {word_count} words")

    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        await transcriber.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
