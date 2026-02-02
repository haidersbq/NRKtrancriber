#!/usr/bin/env python3
"""
Basic example of transcribing NRK radio.

This script demonstrates how to:
1. Initialize the transcriber
2. Capture and transcribe a live stream
3. Handle transcription results
"""

import asyncio
from nrk_transcriber import NRKTranscriber
from nrk_transcriber.utils.logging_config import setup_logging


async def main():
    # Setup logging
    setup_logging(level="INFO")

    # Create transcriber with callback
    def on_transcription(result):
        print(f"\n{'='*60}")
        print(f"Channel: {result.channel_id}")
        print(f"Time: {result.audio_start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Duration: {result.duration_seconds:.1f}s")
        print(f"Processing: {result.processing_time_seconds:.1f}s")
        print(f"{'='*60}")
        print(result.text)
        print()

    transcriber = NRKTranscriber(on_transcription=on_transcription)

    try:
        # Initialize (loads config, sets up database)
        await transcriber.initialize()

        # Show available channels
        print("Available channels:")
        for channel in transcriber.list_channels():
            print(f"  - {channel.channel_id}: {channel.name}")
        print()

        # Transcribe NRK Nyheter (news) for 2 minutes
        print("Starting transcription of NRK Nyheter for 2 minutes...")
        results = await transcriber.transcribe_channel(
            "nrk_nyheter",
            duration_minutes=2,
        )

        # Summary
        print(f"\nTranscribed {len(results)} chunks")
        stats = transcriber.get_statistics()
        print(f"Total audio: {stats['total_audio_seconds']:.1f}s")
        print(f"Total processing: {stats['total_processing_seconds']:.1f}s")
        print(f"Realtime factor: {stats['realtime_factor']:.2f}x")

    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        await transcriber.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
