#!/usr/bin/env python3
"""
Example of searching through stored transcriptions.

This script demonstrates how to:
1. Search for keywords in transcriptions
2. Filter by channel and time
3. Export search results
"""

import asyncio
from datetime import datetime, timedelta
from nrk_transcriber.config import Config
from nrk_transcriber.storage import TranscriptionDatabase


async def main():
    config = Config.load()
    db = TranscriptionDatabase(config.storage.database_path)
    await db.initialize()

    print("NRK Transcription Search")
    print("=" * 60)

    # Get statistics
    stats = await db.get_statistics()
    print(f"Total transcriptions: {stats['total_transcriptions']}")
    print(f"Total audio: {stats['total_audio_duration_seconds'] / 3600:.1f} hours")
    print()

    # Search example - change this to search for what you want
    search_term = "Norge"  # Search for "Norway" mentions

    print(f"Searching for: '{search_term}'")
    results = await db.search_transcriptions(search_term, limit=10)

    if not results:
        print("No results found.")
    else:
        print(f"Found {len(results)} results:\n")

        for record in results:
            print(f"Channel: {record.channel_id}")
            print(f"Time: {record.audio_start_time}")
            print(f"Text: {record.text[:200]}...")
            print("-" * 40)

    # Get recent transcriptions from a specific channel
    print("\nRecent NRK Nyheter transcriptions:")
    recent = await db.get_transcriptions_by_channel(
        "nrk_nyheter",
        limit=5,
    )

    for record in recent:
        print(f"  [{record.audio_start_time.strftime('%H:%M')}] {record.text[:80]}...")

    # Get latest transcription
    latest = await db.get_latest_transcription()
    if latest:
        print(f"\nMost recent transcription: {latest.channel_id} at {latest.audio_start_time}")

    await db.close()


if __name__ == "__main__":
    asyncio.run(main())
