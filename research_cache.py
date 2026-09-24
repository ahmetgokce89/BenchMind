from __future__ import annotations

import hashlib
import json
import os
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from benchmark_data import normalize_research


CACHE_SCHEMA_VERSION = 2
SUPPORTED_CACHE_SCHEMA_VERSIONS = {1, CACHE_SCHEMA_VERSION}
DEFAULT_CACHE_DIR = Path(__file__).resolve().parent / ".benchmind" / "research_cache"


def _normalized_url(value: Any) -> str:
    raw = str(value or "").strip()
    if not raw:
        return ""
    parsed = urlsplit(raw if "://" in raw else f"https://{raw}")
    hostname = (parsed.hostname or "").lower()
    if hostname.startswith("www."):
        hostname = hostname[4:]
    try:
        parsed_port = parsed.port
    except ValueError:
        parsed_port = None
    port = f":{parsed_port}" if parsed_port else ""
    path = parsed.path.rstrip("/")
    return urlunsplit(("https", f"{hostname}{port}", path, parsed.query, ""))


def _normalized_companies(companies: list[dict[str, Any]]) -> list[dict[str, str]]:
    return [
        {
            "name": " ".join(str(company.get("name", "")).split()).casefold(),
            "website": _normalized_url(company.get("website", "")),
        }
        for company in companies
        if str(company.get("name", "")).strip() or str(company.get("website", "")).strip()
    ]


def research_cache_key(companies: list[dict[str, Any]]) -> str:
    payload = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "companies": _normalized_companies(companies),
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _cache_path(companies: list[dict[str, Any]], cache_dir: Path | None = None) -> Path:
    return (cache_dir or DEFAULT_CACHE_DIR) / f"{research_cache_key(companies)}.json"


def load_cached_research(
    companies: list[dict[str, Any]],
    *,
    cache_dir: Path | None = None,
) -> tuple[dict[str, Any], str] | None:
    path = _cache_path(companies, cache_dir)
    if not path.is_file():
        return None
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if payload.get("schema_version") not in SUPPORTED_CACHE_SCHEMA_VERSIONS:
            return None
        research = payload.get("research")
        saved_at = str(payload.get("saved_at", ""))
        if not isinstance(research, dict) or not saved_at:
            return None
        return normalize_research(research), saved_at
    except (OSError, ValueError, TypeError):
        return None


def load_cached_research_by_key(
    cache_key: str,
    *,
    cache_dir: Path | None = None,
) -> tuple[dict[str, Any], str] | None:
    """Load a cached benchmark from a URL-safe, validated cache identifier."""
    normalized_key = str(cache_key or "").strip().lower()
    if re.fullmatch(r"[0-9a-f]{64}", normalized_key) is None:
        return None
    path = (cache_dir or DEFAULT_CACHE_DIR) / f"{normalized_key}.json"
    if not path.is_file():
        return None
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if payload.get("schema_version") not in SUPPORTED_CACHE_SCHEMA_VERSIONS:
            return None
        research = payload.get("research")
        saved_at = str(payload.get("saved_at", ""))
        if not isinstance(research, dict) or not saved_at:
            return None
        return normalize_research(research), saved_at
    except (OSError, ValueError, TypeError):
        return None


def list_cached_research(
    *,
    cache_dir: Path | None = None,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """Return valid saved benchmarks, newest first, optionally capped."""
    if limit is not None and limit < 0:
        raise ValueError("limit cannot be negative")
    directory = cache_dir or DEFAULT_CACHE_DIR
    if not directory.is_dir():
        return []
    entries: list[dict[str, Any]] = []
    for path in directory.glob("*.json"):
        if re.fullmatch(r"[0-9a-f]{64}", path.stem) is None:
            continue
        cached = load_cached_research_by_key(path.stem, cache_dir=directory)
        if cached is None:
            continue
        research, saved_at = cached
        companies = [
            str(company.get("name", "")).strip()
            for company in research.get("companies", [])
            if str(company.get("name", "")).strip()
        ]
        entries.append(
            {
                "key": path.stem,
                "saved_at": saved_at,
                "target_company": research.get("target_company") or (companies[0] if companies else "Bilinmiyor"),
                "companies": companies,
                "research": research,
            }
        )
    sorted_entries = sorted(entries, key=lambda entry: entry["saved_at"], reverse=True)
    return sorted_entries[:limit] if limit is not None else sorted_entries


def save_cached_research(
    companies: list[dict[str, Any]],
    research: dict[str, Any],
    *,
    cache_dir: Path | None = None,
) -> str:
    directory = cache_dir or DEFAULT_CACHE_DIR
    directory.mkdir(parents=True, exist_ok=True)
    saved_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    payload = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "saved_at": saved_at,
        "companies": _normalized_companies(companies),
        "research": research,
    }
    path = _cache_path(companies, directory)
    temporary_path = path.with_suffix(f".tmp-{os.getpid()}-{uuid.uuid4().hex}")
    try:
        with temporary_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
        os.replace(temporary_path, path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()
    return saved_at
