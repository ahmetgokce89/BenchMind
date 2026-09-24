# BenchMind AI

> Turn scattered competitor data into sourced product decisions.

BenchMind AI is a Streamlit portfolio app for evidence-aware competitive product benchmarking. The included demo compares Fathom with Fireflies.ai, Otter.ai, and MeetGeek. Claude Code with an Anthropic API key, the direct Anthropic API, or OpenAI unlocks live web research and grounded Q&A.

## Features

- verified demo snapshot with clickable sources
- live research through Claude Code WebSearch, the Anthropic API, or the OpenAI Responses API
- bounded same-domain crawling of public HTML and PDF evidence
- local MiniLM embeddings and a persistent Chroma vector index
- semantic top-k retrieval for grounded chat
- benchmark table with a transparent weighted score
- radar and overall-score charts
- SWOT analysis and strategic recommendations
- evidence labels: `Verified fact`, `Customer opinion`, and `AI inference`
- conflicting-evidence warnings
- chat grounded only in the currently loaded research
- UTF-8 CSV export

## Run locally

Python 3.12 is recommended to match the Community Cloud deployment environment.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
streamlit run app.py
```

## FastAPI backend

BenchMind can run with Streamlit as a server-side client of FastAPI. The backend
keeps LLM credentials out of the browser, runs long benchmarks as background
jobs, builds the Chroma index, and serves grounded chat.

```powershell
$env:BENCHMIND_API_TOKEN="local-development-token"
$env:BENCHMIND_REQUIRE_API_TOKEN="true"
uvicorn api.main:app --host 127.0.0.1 --port 8000
```

In a second terminal:

```powershell
$env:BENCHMIND_API_URL="http://127.0.0.1:8000"
$env:BENCHMIND_API_TOKEN="local-development-token"
streamlit run app.py
```

OpenAPI documentation is at `http://127.0.0.1:8000/docs`. The main endpoints
are `POST /v1/benchmarks`, `GET /v1/benchmarks/{job_id}`, and `POST /v1/chat`.

## Docker and AWS Lightsail

The production Compose stack contains FastAPI, Streamlit, and Caddy with
automatic HTTPS. Add `BENCHMIND_API_TOKEN`, `APP_DOMAIN`, and `API_DOMAIN` to
the server-side `.env`, then run:

```bash
mkdir -p .benchmind
docker compose up -d --build
```

See [`deploy/README.md`](deploy/README.md) for the Lightsail, DNS, firewall,
verification, update, and rollback procedure.

Add your real Anthropic API key to `.env`. The direct Anthropic API is the default runtime because it avoids Claude Code CLI startup overhead and enforces the web-search limit:

```dotenv
LLM_PROVIDER=anthropic
ANTHROPIC_API_KEY=your_api_key_here
LIVE_RESEARCH_MAX_SEARCHES=4
LIVE_RESEARCH_MAX_OUTPUT_TOKENS=10000
GEMINI_API_KEY=your_gemini_api_key_here
GEMINI_MODEL=gemini-3.8-flash
```

Alternatively, use `LLM_PROVIDER=openai`, `OPENAI_API_KEY`, and `OPENAI_MODEL`.

The demo and CSV export work without an API key. `Live Research'ü Başlat` and `BenchMind'a Sor` require API access. Claude Code calls have configurable per-run USD budget limits to keep usage predictable. Live Research uses at most four web searches by default; override `LIVE_RESEARCH_MAX_SEARCHES` when a different speed/coverage tradeoff is needed.

## Vector RAG pipeline

After Live Research returns its source list, the benchmark is displayed immediately while a background job downloads public source pages and bounded same-domain child pages. The crawler respects `robots.txt`, fetches pages concurrently, extracts HTML/PDF text, creates overlapping chunks, generates normalized `all-MiniLM-L6-v2` embeddings locally, and stores them in a persistent Chroma collection. An unchanged ready collection is reused instead of being rebuilt. Chat questions retrieve the most relevant chunks before the selected LLM writes a sourced answer.

The UI reports four separate values: successfully downloaded documents, chunks, stored embeddings/vectors, and failed URLs. A document means one unique fetched web page or PDF; it is not presented as an independent publisher or source domain.

Crawler limits are configurable in `.env`:

```dotenv
RAG_MAX_DOCUMENTS=30
RAG_MAX_PAGES_PER_DOMAIN=8
RAG_MAX_DOCUMENT_BYTES=3000000
RAG_MAX_CRAWL_SECONDS=30
RAG_CRAWL_WORKERS=8
RAG_EMBEDDING_MODEL=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
RAG_MIN_SIMILARITY=0.40
BENCHMIND_VECTOR_DB_PATH=.benchmind/chroma
```

Chat retrieval keeps only chunks with cosine similarity of at least `0.40` by
default. If no chunk clears the threshold, BenchMind returns an insufficient
evidence message without calling the answer model. Tune this value with a
representative retrieval evaluation set before treating it as calibrated.

The crawler accepts public HTTP(S) targets only, rejects private/local network addresses, stays on each seed domain, and limits redirects, depth, page count, and response size. Some review platforms intentionally block automated fetching; these are counted under failed URLs instead of being represented as indexed evidence.

Local Chroma data survives application restarts. Streamlit Community Cloud storage is ephemeral across redeployments, so production-grade durability requires a managed vector database or external persistent volume.

## Test

```powershell
python -m unittest -v
```

## Deploy to Streamlit Community Cloud

1. Push the project to GitHub.
2. Create a Community Cloud app with `app.py` as the entry point.
3. Select Python 3.12.
4. Add `ANTHROPIC_API_KEY` and optionally `ANTHROPIC_MODEL` in the app's Secrets settings. OpenAI credentials remain supported as an alternative.
5. Deploy and verify both the demo and live-research paths.

Do not commit `.env` or `.streamlit/secrets.toml`.

## Project structure

```text
app.py                      Streamlit interface
benchmark_data.py           normalization, scoring, tables, and CSV export
openai_service.py           live research and grounded chat
prompts.py                  centralized system/user prompts and prompt builders
rag_service.py              crawling, chunking, embeddings, Chroma, and retrieval
data/demo_benchmark.json    reviewed presentation snapshot
PROJECT_PREPARATION.md      product and technical brief
test_benchmark_data.py      core data-layer tests
```

## Evidence policy

Sources are prioritized in this order:

1. official product sites
2. official pricing pages and documentation
3. credible review platforms
4. community discussions

Public product information and aggregated review themes only; no private customer data is collected.
