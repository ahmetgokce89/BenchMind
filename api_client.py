from __future__ import annotations

import os
from typing import Any

import httpx


class BenchMindAPIError(RuntimeError):
    pass


def api_base_url() -> str:
    return os.getenv("BENCHMIND_API_URL", "").strip().rstrip("/")


def _headers() -> dict[str, str]:
    token = os.getenv("BENCHMIND_API_TOKEN", "").strip()
    return {"Authorization": f"Bearer {token}"} if token else {}


def _request(method: str, path: str, **kwargs: Any) -> dict[str, Any]:
    base_url = api_base_url()
    if not base_url:
        raise BenchMindAPIError("BENCHMIND_API_URL is not configured.")
    try:
        response = httpx.request(
            method,
            f"{base_url}{path}",
            headers=_headers(),
            timeout=httpx.Timeout(180.0, connect=5.0),
            **kwargs,
        )
        response.raise_for_status()
        return response.json()
    except httpx.HTTPStatusError as exc:
        try:
            detail = exc.response.json().get("detail", exc.response.text)
        except ValueError:
            detail = exc.response.text
        raise BenchMindAPIError(str(detail)) from exc
    except (httpx.HTTPError, ValueError) as exc:
        raise BenchMindAPIError(f"FastAPI backend'e ulaşılamadı: {exc}") from exc


def get_backend_config() -> dict[str, Any]:
    return _request("GET", "/v1/config")


def create_benchmark_job(
    companies: list[dict[str, str]], model: str
) -> dict[str, Any]:
    return _request(
        "POST",
        "/v1/benchmarks",
        json={"companies": companies, "model": model},
    )


def get_benchmark_job(job_id: str) -> dict[str, Any]:
    return _request("GET", f"/v1/benchmarks/{job_id}")


def ask_backend(
    question: str,
    research: dict[str, Any],
    collection_name: str,
    model: str,
) -> dict[str, Any]:
    return _request(
        "POST",
        "/v1/chat",
        json={
            "question": question,
            "research": research,
            "collection_name": collection_name,
            "model": model,
        },
    )
