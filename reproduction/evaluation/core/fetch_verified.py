"""Bounded sequential downloads with exact size/hash verification and rate limiting."""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
import time
import urllib.request


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(url: str, destination: Path, expected_size: int | None = None,
          expected_sha: str | None = None, max_bytes: int = 40000000,
          rate: int = 2800000) -> dict:
    """Limit application-read throughput; cached unknown content is never trusted."""
    if expected_size and expected_size > max_bytes:
        raise ValueError("declared object exceeds per-object budget")
    if destination.exists() and expected_sha:
        actual = digest(destination)
        if actual == expected_sha and (expected_size is None or destination.stat().st_size == expected_size):
            return {"url": url, "bytes": destination.stat().st_size, "sha256": actual, "cache_reused": True}
        raise ValueError(f"existing destination differs from frozen source: {destination}")
    if destination.exists():
        raise ValueError("existing object requires recorded expected hash")
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".partial")
    request = urllib.request.Request(url, headers={"User-Agent": "ink-surface-support/1"})
    started = time.monotonic()
    count = 0
    h = hashlib.sha256()
    with urllib.request.urlopen(request, timeout=45) as response, partial.open("wb") as out:
        length = response.headers.get("Content-Length")
        if length and int(length) > max_bytes:
            raise ValueError("HTTP object exceeds per-object budget")
        if expected_size is not None and length and int(length) != expected_size:
            raise ValueError("HTTP length differs from source metadata")
        headers = {k: response.headers.get(k) for k in ["Content-Length", "ETag", "Last-Modified"]}
        while True:
            chunk = response.read(262144)
            if not chunk:
                break
            count += len(chunk)
            if count > max_bytes:
                raise ValueError("download exceeded budget")
            out.write(chunk)
            h.update(chunk)
            pause = count / rate - (time.monotonic() - started)
            if pause > 0:
                time.sleep(pause)
    if expected_size is not None and count != expected_size:
        raise ValueError("download size mismatch")
    if length and count != int(length):
        raise ValueError("truncated HTTP payload")
    actual = h.hexdigest()
    if expected_sha and expected_sha != actual:
        raise ValueError("download SHA-256 mismatch")
    os.replace(partial, destination)
    elapsed = time.monotonic() - started
    return {"url": url, "bytes": count, "sha256": actual, "cache_reused": False,
            "seconds": elapsed, "average_bytes_per_second": count / elapsed,
            "rate_limit_bytes_per_second": rate, "headers": headers}
