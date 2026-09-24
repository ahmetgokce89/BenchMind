from __future__ import annotations

import hashlib
import io
import ipaddress
import os
import re
import socket
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from functools import lru_cache
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urldefrag, urljoin, urlparse, urlunparse
from urllib.robotparser import RobotFileParser

import chromadb
import httpx
from sentence_transformers import SentenceTransformer

try:
    from pypdf import PdfReader
except ImportError:  # PDF support remains optional in local development.
    PdfReader = None


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DB_PATH = BASE_DIR / ".benchmind" / "chroma"
EMBEDDING_MODEL = os.getenv(
    "RAG_EMBEDDING_MODEL",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
)
try:
    MIN_RETRIEVAL_SIMILARITY = min(
        1.0, max(0.0, float(os.getenv("RAG_MIN_SIMILARITY", "0.40")))
    )
except ValueError:
    MIN_RETRIEVAL_SIMILARITY = 0.40
INSUFFICIENT_RETRIEVAL_MESSAGE = (
    "Mevcut araştırma dataset'inde bu soruyu destekleyen yeterli retrieval kanıtı yok."
)
USER_AGENT = "BenchMindResearchBot/1.0"
SKIPPED_PATH_PARTS = {
    "login", "logout", "signin", "signup", "register", "account", "cart",
    "checkout", "cdn-cgi", "wp-admin", "privacy", "terms", "legal",
}
SKIPPED_SUFFIXES = {
    ".7z", ".avi", ".css", ".doc", ".docx", ".gif", ".gz", ".ico",
    ".jpeg", ".jpg", ".js", ".json", ".mov", ".mp3", ".mp4", ".png",
    ".ppt", ".pptx", ".rar", ".rss", ".svg", ".tar", ".webp", ".xml",
    ".zip",
}


@dataclass(frozen=True)
class SourceDocument:
    document_id: str
    source_id: str
    title: str
    url: str
    publisher: str
    source_type: str
    text: str


@dataclass(frozen=True)
class RagStats:
    collection_name: str = ""
    documents: int = 0
    chunks: int = 0
    embeddings: int = 0
    failed_urls: int = 0
    embedding_model: str = EMBEDDING_MODEL
    status: str = "not_indexed"
    cache_hit: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


ProgressCallback = Callable[[str, dict[str, Any]], None]


def _emit(progress: ProgressCallback | None, stage: str, **payload: Any) -> None:
    if progress is not None:
        progress(stage, payload)


class _HTMLTextExtractor(HTMLParser):
    BLOCK_TAGS = {
        "article", "blockquote", "br", "dd", "div", "dl", "dt", "footer",
        "h1", "h2", "h3", "h4", "h5", "h6", "header", "li", "main", "p",
        "section", "table", "td", "th", "tr",
    }
    HIDDEN_TAGS = {"script", "style", "svg", "noscript", "template"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.links: list[str] = []
        self.title_parts: list[str] = []
        self._hidden_depth = 0
        self._in_title = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.casefold()
        if tag in self.HIDDEN_TAGS:
            self._hidden_depth += 1
        if self._hidden_depth:
            return
        if tag in self.BLOCK_TAGS:
            self.parts.append("\n")
        if tag == "title":
            self._in_title = True
        if tag == "a":
            href = dict(attrs).get("href")
            if href:
                self.links.append(href)

    def handle_endtag(self, tag: str) -> None:
        tag = tag.casefold()
        if tag in self.HIDDEN_TAGS and self._hidden_depth:
            self._hidden_depth -= 1
            return
        if self._hidden_depth:
            return
        if tag in self.BLOCK_TAGS:
            self.parts.append("\n")
        if tag == "title":
            self._in_title = False

    def handle_data(self, data: str) -> None:
        if self._hidden_depth:
            return
        value = data.strip()
        if value:
            self.parts.append(value)
            if self._in_title:
                self.title_parts.append(value)

    @property
    def text(self) -> str:
        joined = " ".join(self.parts)
        joined = re.sub(r"[ \t\f\v]+", " ", joined)
        joined = re.sub(r"\s*\n\s*", "\n", joined)
        return re.sub(r"\n{3,}", "\n\n", joined).strip()

    @property
    def title(self) -> str:
        return " ".join(self.title_parts).strip()


def _db_path() -> Path:
    return Path(os.getenv("BENCHMIND_VECTOR_DB_PATH", str(DEFAULT_DB_PATH)))


@lru_cache(maxsize=1)
def get_chroma_client() -> chromadb.PersistentClient:
    path = _db_path()
    path.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(path))


@lru_cache(maxsize=1)
def get_embedding_model() -> SentenceTransformer:
    try:
        return SentenceTransformer(EMBEDDING_MODEL, local_files_only=True)
    except Exception:
        return SentenceTransformer(EMBEDDING_MODEL)


def _encode(texts: list[str]) -> list[list[float]]:
    return get_embedding_model().encode(
        texts,
        show_progress_bar=False,
        normalize_embeddings=True,
    ).tolist()


def normalize_url(value: str) -> str:
    value = value.strip()
    if not value:
        return ""
    parsed = urlparse(value if "://" in value else f"https://{value}")
    if parsed.scheme.casefold() not in {"http", "https"} or not parsed.hostname:
        return ""
    path = re.sub(r"/{2,}", "/", parsed.path or "/")
    normalized = urlunparse(
        (parsed.scheme.casefold(), parsed.netloc.casefold(), path, "", parsed.query, "")
    )
    return urldefrag(normalized)[0]


def _is_public_host(hostname: str) -> bool:
    try:
        addresses = socket.getaddrinfo(hostname, None)
    except socket.gaierror:
        return False
    for address in addresses:
        ip = ipaddress.ip_address(address[4][0])
        if (
            ip.is_private
            or ip.is_loopback
            or ip.is_link_local
            or ip.is_multicast
            or ip.is_reserved
            or ip.is_unspecified
        ):
            return False
    return bool(addresses)


def _is_crawlable_url(url: str, root_host: str) -> bool:
    parsed = urlparse(url)
    hostname = (parsed.hostname or "").casefold()
    if parsed.scheme not in {"http", "https"} or not hostname:
        return False
    domain = root_host.removeprefix("www.")
    if hostname != domain and not hostname.endswith(f".{domain}"):
        return False
    lowered_path = parsed.path.casefold()
    if any(part in lowered_path.split("/") for part in SKIPPED_PATH_PARTS):
        return False
    return not any(lowered_path.endswith(suffix) for suffix in SKIPPED_SUFFIXES)


def _robots_allowed(
    client: httpx.Client,
    url: str,
    cache: dict[str, RobotFileParser | None],
) -> bool:
    parsed = urlparse(url)
    origin = f"{parsed.scheme}://{parsed.netloc}"
    if origin not in cache:
        robots = RobotFileParser()
        robots.set_url(f"{origin}/robots.txt")
        try:
            response = client.get(robots.url)
            if response.status_code < 400:
                robots.parse(response.text.splitlines())
                cache[origin] = robots
            else:
                cache[origin] = None
        except httpx.HTTPError:
            cache[origin] = None
    rule = cache[origin]
    return True if rule is None else rule.can_fetch(USER_AGENT, url)


def _extract_pdf(content: bytes) -> str:
    if PdfReader is None:
        raise RuntimeError("PDF indeksleme için pypdf kurulu değil.")
    reader = PdfReader(io.BytesIO(content))
    return "\n\n".join((page.extract_text() or "").strip() for page in reader.pages).strip()


def _fetch_page(
    client: httpx.Client,
    url: str,
    max_bytes: int,
) -> tuple[str, str, list[str], str]:
    root_host = (urlparse(url).hostname or "").casefold()
    current_url = url
    response = None
    for _ in range(6):
        current_host = (urlparse(current_url).hostname or "").casefold()
        if not _is_crawlable_url(current_url, root_host) or not _is_public_host(current_host):
            raise RuntimeError("Güvenli olmayan veya farklı domaine yönlendirme engellendi.")
        response = client.get(current_url)
        if response.status_code not in {301, 302, 303, 307, 308}:
            break
        location = response.headers.get("location", "")
        current_url = normalize_url(urljoin(current_url, location))
        if not current_url:
            raise RuntimeError("Geçersiz yönlendirme URL'si.")
    if response is None or response.status_code in {301, 302, 303, 307, 308}:
        raise RuntimeError("Kaynak çok fazla kez yönlendirdi.")
    response.raise_for_status()
    if len(response.content) > max_bytes:
        raise RuntimeError("Kaynak boyut sınırını aşıyor.")
    final_url = normalize_url(str(response.url))
    content_type = response.headers.get("content-type", "").casefold()
    if "application/pdf" in content_type or final_url.casefold().endswith(".pdf"):
        return _extract_pdf(response.content), "PDF document", [], final_url
    if "text/html" not in content_type and "text/plain" not in content_type:
        raise RuntimeError(f"Desteklenmeyen içerik türü: {content_type or 'bilinmiyor'}")
    if "text/plain" in content_type:
        return response.text.strip(), final_url, [], final_url
    parser = _HTMLTextExtractor()
    parser.feed(response.text)
    links = [normalize_url(urljoin(final_url, item)) for item in parser.links]
    return parser.text, parser.title or final_url, [item for item in links if item], final_url


def crawl_research_sources(
    research: dict[str, Any],
    *,
    max_documents: int | None = None,
    max_pages_per_domain: int | None = None,
    max_depth: int = 2,
    progress: ProgressCallback | None = None,
) -> tuple[list[SourceDocument], int]:
    """Download public source pages and bounded same-domain child pages."""
    max_documents = max_documents or int(os.getenv("RAG_MAX_DOCUMENTS", "30"))
    max_pages_per_domain = max_pages_per_domain or int(
        os.getenv("RAG_MAX_PAGES_PER_DOMAIN", "8")
    )
    max_bytes = int(os.getenv("RAG_MAX_DOCUMENT_BYTES", "3000000"))
    deadline = time.monotonic() + int(os.getenv("RAG_MAX_CRAWL_SECONDS", "30"))
    workers = max(1, min(16, int(os.getenv("RAG_CRAWL_WORKERS", "8"))))
    queue: deque[tuple[str, str, str, str, str, int]] = deque()
    seen: set[str] = set()
    queued: set[str] = set()
    public_hosts: dict[str, bool] = {}
    per_domain: dict[str, int] = {}
    documents: list[SourceDocument] = []
    failures = 0

    for source in research.get("sources", []):
        url = normalize_url(str(source.get("url", "")))
        host = (urlparse(url).hostname or "").casefold()
        if url and url not in queued:
            queue.append(
                (
                    url,
                    str(source.get("id", "SOURCE")),
                    str(source.get("title", "")),
                    str(source.get("publisher", "")),
                    str(source.get("source_type", "Web source")),
                    0,
                )
            )
            queued.add(url)
            per_domain.setdefault(host, 0)

    for index, company in enumerate(research.get("companies", []), start=1):
        url = normalize_url(str(company.get("website", "")))
        if url and url not in queued:
            host = (urlparse(url).hostname or "").casefold()
            queue.append(
                (url, f"W{index:02d}", str(company.get("name", "")), str(company.get("name", "")), "Company website", 0)
            )
            queued.add(url)
            per_domain.setdefault(host, 0)

    timeout = httpx.Timeout(12.0, connect=5.0)
    headers = {"User-Agent": USER_AGENT, "Accept": "text/html,application/pdf,text/plain"}
    robots_cache: dict[str, RobotFileParser | None] = {}
    _emit(progress, "crawl_started", message=f"En fazla {max_documents} document taranıyor.")
    with (
        httpx.Client(timeout=timeout, follow_redirects=False, headers=headers) as client,
        ThreadPoolExecutor(max_workers=workers, thread_name_prefix="benchmind-crawl") as executor,
    ):
        while queue and len(documents) < max_documents:
            if time.monotonic() >= deadline:
                failures += len(queue)
                break
            jobs: dict[Any, tuple[str, str, str, str, str, int, str]] = {}
            scheduled_per_domain: dict[str, int] = {}
            while queue and len(jobs) < min(workers, max_documents - len(documents)):
                url, source_id, seed_title, publisher, source_type, depth = queue.popleft()
                if url in seen:
                    continue
                seen.add(url)
                root_host = (urlparse(url).hostname or "").casefold()
                already_scheduled = scheduled_per_domain.get(root_host, 0)
                if per_domain.get(root_host, 0) + already_scheduled >= max_pages_per_domain:
                    continue
                if root_host not in public_hosts:
                    public_hosts[root_host] = _is_public_host(root_host)
                if not public_hosts[root_host] or not _robots_allowed(client, url, robots_cache):
                    failures += 1
                    continue
                future = executor.submit(_fetch_page, client, url, max_bytes)
                jobs[future] = (url, source_id, seed_title, publisher, source_type, depth, root_host)
                scheduled_per_domain[root_host] = already_scheduled + 1

            if not jobs:
                continue
            for future in as_completed(jobs):
                url, source_id, seed_title, publisher, source_type, depth, root_host = jobs[future]
                try:
                    text, page_title, links, final_url = future.result()
                except (httpx.HTTPError, RuntimeError, ValueError):
                    failures += 1
                    _emit(progress, "document_failed", message=f"Alınamadı: {url}", url=url)
                    continue
                text = text.strip()
                if len(text) < 200:
                    failures += 1
                    continue
                per_domain[root_host] = per_domain.get(root_host, 0) + 1
                document_id = f"D{len(documents) + 1:04d}"
                documents.append(
                    SourceDocument(
                        document_id=document_id,
                        source_id=source_id,
                        title=(page_title or seed_title or final_url)[:500],
                        url=final_url,
                        publisher=publisher[:200],
                        source_type=source_type[:100],
                        text=text,
                    )
                )
                _emit(
                    progress,
                    "document_downloaded",
                    message=f"[{len(documents)}/{max_documents}] {urlparse(final_url).netloc} indirildi.",
                    documents=len(documents),
                    url=final_url,
                )
                if depth >= max_depth or len(documents) >= max_documents:
                    continue
                candidates = {
                    item for item in links
                    if item not in seen and item not in queued and _is_crawlable_url(item, root_host)
                }
                priority_terms = ("pricing", "features", "docs", "help", "security", "integration", "review", "product")
                ordered = sorted(
                    candidates,
                    key=lambda item: (not any(term in item.casefold() for term in priority_terms), len(item)),
                )
                remaining = max_pages_per_domain - per_domain.get(root_host, 0)
                for child_url in ordered[: max(0, remaining)]:
                    queue.append((child_url, source_id, seed_title, publisher, source_type, depth + 1))
                    queued.add(child_url)

    _emit(progress, "crawl_finished", message=f"{len(documents)} document indirildi; {failures} URL alınamadı.")
    return documents, failures


def chunk_text(text: str, *, chunk_size: int = 1200, overlap: int = 180) -> list[str]:
    if chunk_size < 200 or overlap < 0 or overlap >= chunk_size:
        raise ValueError("Geçersiz chunk_size/overlap ayarı.")
    cleaned = re.sub(r"[ \t\f\v]+", " ", text)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
    if not cleaned:
        return []
    chunks: list[str] = []
    start = 0
    while start < len(cleaned):
        hard_end = min(len(cleaned), start + chunk_size)
        end = hard_end
        if hard_end < len(cleaned):
            candidates = [
                cleaned.rfind("\n\n", start + chunk_size // 2, hard_end),
                cleaned.rfind(". ", start + chunk_size // 2, hard_end),
                cleaned.rfind(" ", start + chunk_size // 2, hard_end),
            ]
            boundary = max(candidates)
            if boundary > start:
                end = boundary + (2 if cleaned[boundary:boundary + 2] in {"\n\n", ". "} else 1)
        chunk = cleaned[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(cleaned):
            break
        next_start = max(start + 1, end - overlap)
        whitespace = cleaned.find(" ", next_start, min(end, next_start + 80))
        start = whitespace + 1 if whitespace >= 0 else next_start
    return chunks


def collection_name_for(research: dict[str, Any]) -> str:
    identity = "|".join(
        [
            str(research.get("target_company", "research")),
            str(research.get("research_date", "")),
            *(str(item.get("url", "")) for item in research.get("sources", [])),
        ]
    )
    return f"benchmind_{hashlib.sha256(identity.encode('utf-8')).hexdigest()[:20]}"


def index_documents(
    research: dict[str, Any],
    documents: list[SourceDocument],
    *,
    failed_urls: int = 0,
    client: Any | None = None,
    encoder: Callable[[list[str]], list[list[float]]] | None = None,
    progress: ProgressCallback | None = None,
) -> RagStats:
    if not documents:
        return RagStats(failed_urls=failed_urls, status="empty")
    client = client or get_chroma_client()
    encoder = encoder or _encode
    collection_name = collection_name_for(research)
    try:
        client.delete_collection(collection_name)
    except Exception:
        pass
    collection = client.create_collection(
        collection_name,
        metadata={"hnsw:space": "cosine"},
    )
    ids: list[str] = []
    texts: list[str] = []
    metadatas: list[dict[str, Any]] = []
    for document in documents:
        for chunk_index, chunk in enumerate(chunk_text(document.text)):
            ids.append(f"{document.document_id}-C{chunk_index + 1:04d}")
            texts.append(chunk)
            metadatas.append(
                {
                    "document_id": document.document_id,
                    "source_id": document.source_id,
                    "title": document.title,
                    "url": document.url,
                    "publisher": document.publisher,
                    "source_type": document.source_type,
                    "chunk_index": chunk_index,
                }
            )
    if not texts:
        return RagStats(failed_urls=failed_urls, status="empty")
    batch_size = 64
    total_batches = (len(texts) + batch_size - 1) // batch_size
    _emit(progress, "chunking_finished", message=f"{len(documents)} document -> {len(texts)} chunk.")
    for start in range(0, len(texts), batch_size):
        stop = start + batch_size
        batch_texts = texts[start:stop]
        collection.add(
            ids=ids[start:stop],
            documents=batch_texts,
            embeddings=encoder(batch_texts),
            metadatas=metadatas[start:stop],
        )
        _emit(
            progress,
            "embedding_batch",
            message=f"Embedding batch {start // batch_size + 1}/{total_batches} kaydedildi.",
        )
    stats = RagStats(
        collection_name=collection_name,
        documents=len(documents),
        chunks=len(texts),
        embeddings=collection.count(),
        failed_urls=failed_urls,
        status="ready",
    )
    collection.modify(
        metadata={
            "index_status": "ready",
            "documents": stats.documents,
            "chunks": stats.chunks,
            "embeddings": stats.embeddings,
            "failed_urls": stats.failed_urls,
            "embedding_model": stats.embedding_model,
        }
    )
    _emit(progress, "index_ready", message=f"Chroma index hazır: {stats.embeddings} vector.")
    return stats


def cached_research_index(
    research: dict[str, Any],
    *,
    client: Any | None = None,
) -> RagStats | None:
    client = client or get_chroma_client()
    collection_name = collection_name_for(research)
    try:
        collection = client.get_collection(collection_name)
    except Exception:
        return None
    metadata = collection.metadata or {}
    if metadata.get("index_status") != "ready" or metadata.get("embedding_model") != EMBEDDING_MODEL:
        return None
    count = collection.count()
    if count <= 0:
        return None
    return RagStats(
        collection_name=collection_name,
        documents=int(metadata.get("documents", 0)),
        chunks=int(metadata.get("chunks", count)),
        embeddings=count,
        failed_urls=int(metadata.get("failed_urls", 0)),
        embedding_model=str(metadata.get("embedding_model", EMBEDDING_MODEL)),
        status="ready",
        cache_hit=True,
    )


def build_research_index(
    research: dict[str, Any],
    *,
    progress: ProgressCallback | None = None,
    force: bool = False,
) -> RagStats:
    if not force:
        cached = cached_research_index(research)
        if cached is not None:
            _emit(progress, "cache_hit", message=f"Mevcut index kullanıldı: {cached.embeddings} vector.")
            return cached
    documents, failures = crawl_research_sources(research, progress=progress)
    return index_documents(research, documents, failed_urls=failures, progress=progress)


def query_research_index(
    collection_name: str,
    question: str,
    *,
    n_results: int = 5,
    min_similarity: float = MIN_RETRIEVAL_SIMILARITY,
) -> list[dict[str, Any]]:
    if not collection_name or not question.strip():
        return []
    if not 0.0 <= min_similarity <= 1.0:
        raise ValueError("min_similarity 0 ile 1 arasında olmalıdır.")
    collection = get_chroma_client().get_collection(collection_name)
    count = collection.count()
    if not count:
        return []
    results = collection.query(
        query_embeddings=_encode([question]),
        n_results=min(n_results, count),
        include=["documents", "metadatas", "distances"],
    )
    rows = []
    for chunk_id, text, metadata, distance in zip(
        results.get("ids", [[]])[0],
        results.get("documents", [[]])[0],
        results.get("metadatas", [[]])[0],
        results.get("distances", [[]])[0],
        strict=False,
    ):
        similarity = 1.0 - float(distance)
        if similarity < min_similarity:
            continue
        rows.append(
            {
                "chunk_id": chunk_id,
                "text": text,
                "distance": round(float(distance), 5),
                "similarity": round(similarity, 5),
                **metadata,
            }
        )
    return rows
