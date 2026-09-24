from __future__ import annotations

import os
import secrets
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field

from openai_service import (
    answer_from_research,
    clean_company_inputs,
    default_model_for_provider,
    get_api_config,
    research_companies,
)
from rag_service import (
    INSUFFICIENT_RETRIEVAL_MESSAGE,
    build_research_index,
    query_research_index,
)
from research_cache import save_cached_research

load_dotenv()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class CompanyInput(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    website: str = Field(default="", max_length=500)


class BenchmarkRequest(BaseModel):
    companies: list[CompanyInput] = Field(min_length=2, max_length=4)
    model: str | None = Field(default=None, max_length=120)


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=1200)
    research: dict[str, Any]
    collection_name: str = Field(default="", max_length=120)
    model: str | None = Field(default=None, max_length=120)


class JobCreated(BaseModel):
    job_id: str
    status: str


app = FastAPI(
    title="BenchMind API",
    version="1.0.0",
    description="Evidence-aware competitive benchmark and grounded Q&A API.",
)
_bearer = HTTPBearer(auto_error=False)
_jobs: dict[str, dict[str, Any]] = {}
_jobs_lock = threading.Lock()
_executor = ThreadPoolExecutor(
    max_workers=max(1, int(os.getenv("BENCHMIND_JOB_WORKERS", "1"))),
    thread_name_prefix="benchmind-api",
)
_max_active_jobs = max(1, int(os.getenv("BENCHMIND_MAX_ACTIVE_JOBS", "2")))


def _authenticate(
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer),
) -> None:
    expected = os.getenv("BENCHMIND_API_TOKEN", "").strip()
    require_token = os.getenv("BENCHMIND_REQUIRE_API_TOKEN", "false").lower() in {
        "1",
        "true",
        "yes",
    }
    if not expected:
        if require_token:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="BENCHMIND_API_TOKEN is required but not configured.",
            )
        return
    supplied = credentials.credentials if credentials else ""
    if not secrets.compare_digest(supplied, expected):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API token.",
            headers={"WWW-Authenticate": "Bearer"},
        )


def _set_job(job_id: str, **changes: Any) -> None:
    with _jobs_lock:
        if job_id in _jobs:
            _jobs[job_id].update(changes)
            _jobs[job_id]["updated_at"] = _utc_now()


def _run_benchmark_job(
    job_id: str,
    companies: list[dict[str, str]],
    provider: str,
    api_key: str,
    model: str,
) -> None:
    try:
        _set_job(job_id, status="researching", message="Live Research çalışıyor.")
        result = research_companies(companies, api_key, model, provider)
        try:
            save_cached_research(companies, result)
        except (OSError, TypeError, ValueError):
            pass
        # Publish the benchmark as soon as research is complete. RAG indexing can
        # take considerably longer (model loading, crawling and embedding), and
        # the UI should not keep showing the demo snapshot while that background
        # work continues.
        _set_job(
            job_id,
            status="indexing",
            message="Benchmark hazır. Kaynaklar Vector RAG index'ine alınıyor.",
            result=result,
            research_completed_at=_utc_now(),
        )
        rag_stats = build_research_index(result).to_dict()
        _set_job(
            job_id,
            status="completed",
            message="Benchmark ve Vector RAG index'i hazır.",
            result=result,
            rag_stats=rag_stats,
            completed_at=_utc_now(),
        )
    except Exception as exc:
        _set_job(
            job_id,
            status="failed",
            message="Benchmark oluşturulamadı.",
            error=str(exc),
            completed_at=_utc_now(),
        )


@app.get("/health", tags=["system"])
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/v1/config", dependencies=[Depends(_authenticate)], tags=["system"])
def config() -> dict[str, Any]:
    provider, api_key = get_api_config()
    return {
        "provider": provider,
        "model": default_model_for_provider(provider),
        "configured": bool(provider and api_key),
    }


@app.post(
    "/v1/benchmarks",
    response_model=JobCreated,
    status_code=status.HTTP_202_ACCEPTED,
    dependencies=[Depends(_authenticate)],
    tags=["benchmarks"],
)
def create_benchmark(request: BenchmarkRequest) -> JobCreated:
    provider, api_key = get_api_config()
    if not provider or not api_key:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="No LLM provider API key is configured.",
        )
    try:
        companies = clean_company_inputs(
            [company.model_dump() for company in request.companies]
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    with _jobs_lock:
        active_jobs = sum(
            job["status"] in {"queued", "researching", "indexing"}
            for job in _jobs.values()
        )
        if active_jobs >= _max_active_jobs:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Benchmark queue is full. Try again after the active job finishes.",
            )
        job_id = uuid4().hex
        now = _utc_now()
        _jobs[job_id] = {
            "job_id": job_id,
            "status": "queued",
            "message": "Benchmark sıraya alındı.",
            "requested_companies": companies,
            "requested_target_company": companies[0]["name"],
            "provider": provider,
            "model": request.model or default_model_for_provider(provider),
            "created_at": now,
            "updated_at": now,
        }
    _executor.submit(
        _run_benchmark_job,
        job_id,
        companies,
        provider,
        api_key,
        request.model or default_model_for_provider(provider),
    )
    return JobCreated(job_id=job_id, status="queued")


@app.get(
    "/v1/benchmarks/{job_id}",
    dependencies=[Depends(_authenticate)],
    tags=["benchmarks"],
)
def get_benchmark(job_id: str) -> dict[str, Any]:
    with _jobs_lock:
        job = _jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Benchmark job not found.")
        return dict(job)


@app.post("/v1/chat", dependencies=[Depends(_authenticate)], tags=["chat"])
def chat(request: ChatRequest) -> dict[str, Any]:
    provider, api_key = get_api_config()
    if not provider or not api_key:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="No LLM provider API key is configured.",
        )
    chunks: list[dict[str, Any]] = []
    if request.collection_name:
        try:
            chunks = query_research_index(request.collection_name, request.question)
        except Exception:
            chunks = []
        if not chunks:
            return {"answer": INSUFFICIENT_RETRIEVAL_MESSAGE, "retrieved_sources": []}
    answer = answer_from_research(
        request.question,
        request.research,
        api_key,
        request.model or default_model_for_provider(provider),
        provider,
        retrieved_chunks=chunks,
    )
    unique_sources: dict[str, dict[str, Any]] = {}
    for chunk in chunks:
        url = str(chunk.get("url", ""))
        unique_sources.setdefault(
            url,
            {
                "source_id": chunk.get("source_id", ""),
                "title": chunk.get("title", "Kaynak"),
                "url": url,
                "distance": chunk.get("distance"),
                "similarity": chunk.get("similarity"),
            },
        )
    return {"answer": answer, "retrieved_sources": list(unique_sources.values())}
