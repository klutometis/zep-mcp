#!/usr/bin/env python3
"""Import OpenMemory export into Zep Cloud.

Usage:
    uv run python import-memories.py /tmp/openmemory-export.json

Options:
    --nuke      Delete user and recreate before importing (clean slate)
    --user-id   Zep user ID (default: from ZEP_USER_ID env var, or "default")
    --delay     Delay between API calls in seconds (default: 0.5)
    --resume    Skip the first N memories (resume after partial import)

Requires ZEP_API_KEY env var.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

from zep_cloud.client import Zep

MAX_RETRIES = 5
INITIAL_BACKOFF = 2.0  # seconds


def add_with_retry(client, user_id: str, content: str, index: int) -> str:
    """Add memory with exponential backoff on rate-limit/transient errors."""
    for attempt in range(MAX_RETRIES):
        try:
            result = client.graph.add(user_id=user_id, type="text", data=content)
            return result.uuid_
        except Exception as e:
            status = getattr(e, "status_code", None)
            body = getattr(e, "body", None)
            err = str(e)

            print(
                f"  ERROR [{index}] attempt {attempt + 1}/{MAX_RETRIES}: "
                f"status={status} body={body!r} "
                f"content=({len(content)} chars): {content[:80]!r}",
                file=sys.stderr,
            )

            # Retry on rate-limit (429) or server errors (5xx)
            if status in (429, 500, 502, 503, 504) or "content-length': '34'" in err:
                wait = INITIAL_BACKOFF * (2**attempt)
                print(f"  Retrying in {wait:.0f}s...", file=sys.stderr)
                time.sleep(wait)
                continue
            # Non-retryable error: bail
            raise
    raise RuntimeError(f"Failed after {MAX_RETRIES} retries")


def main():
    parser = argparse.ArgumentParser(description="Import memories into Zep Cloud")
    parser.add_argument("file", help="Path to OpenMemory export JSON")
    parser.add_argument("--nuke", action="store_true", help="Clear graph before import")
    parser.add_argument("--user-id", default=os.environ.get("ZEP_USER_ID", "default"))
    parser.add_argument("--delay", type=float, default=0.5)
    parser.add_argument("--resume", type=int, default=0, help="Skip first N memories")
    args = parser.parse_args()

    api_key = os.environ.get("ZEP_API_KEY")
    if not api_key:
        print("ZEP_API_KEY required", file=sys.stderr)
        sys.exit(1)

    client = Zep(api_key=api_key)

    with open(args.file) as f:
        memories = json.load(f)

    print(f"Loaded {len(memories)} memories from {args.file}")

    if args.nuke:
        print(f"Nuking user '{args.user_id}'...")
        try:
            client.user.delete(args.user_id)
        except Exception:
            pass
        client.user.add(user_id=args.user_id)
        print("Clean slate.")
    else:
        # Ensure user exists
        try:
            client.user.add(user_id=args.user_id)
        except Exception:
            pass

    if args.resume > 0:
        print(
            f"Resuming from index {args.resume}, skipping first {args.resume} memories"
        )
        memories = memories[args.resume :]

    succeeded = 0
    failed = 0
    skipped = 0
    start = time.time()

    for i, mem in enumerate(memories):
        content = mem.get("content", "")
        if not content.strip():
            skipped += 1
            continue
        try:
            add_with_retry(client, args.user_id, content, index=i + args.resume)
            succeeded += 1
        except Exception as e:
            failed += 1
            print(f"  FAIL [{i + args.resume}]: {e!s:.120s}", file=sys.stderr)

        if args.delay > 0:
            time.sleep(args.delay)

        if (i + 1) % 50 == 0:
            elapsed = time.time() - start
            rate = (i + 1) / elapsed
            eta = (len(memories) - i - 1) / rate if rate > 0 else 0
            print(
                f"  [{i + 1 + args.resume}/{len(memories) + args.resume}] "
                f"{succeeded} ok, {failed} fail, {skipped} skip, "
                f"{rate:.1f}/s, {elapsed:.0f}s elapsed, ~{eta:.0f}s remaining"
            )

    elapsed = time.time() - start
    print(
        f"\nDone: {succeeded} succeeded, {failed} failed, "
        f"{skipped} skipped, {elapsed:.0f}s total"
    )


if __name__ == "__main__":
    main()
