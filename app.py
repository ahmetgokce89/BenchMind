from __future__ import annotations

import html
import time
from concurrent.futures import Future, ThreadPoolExecutor
from queue import Empty, Queue
from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv

from api_client import (
    BenchMindAPIError,
    api_base_url,
    ask_backend,
    create_benchmark_job,
    get_backend_config,
    get_benchmark_job,
)
from benchmark_data import (
    CRITERIA,
    benchmark_dataframe,
    calculate_confidence,
    export_csv,
    findings_dataframe,
    load_demo_data,
)
from openai_service import (
    LIVE_RESEARCH_MAX_SEARCHES,
    clean_company_inputs,
    company_name_from_url,
    company_url_from_name,
    default_model_for_provider,
    estimate_live_research_cost,
    get_api_config,
    research_companies,
    stream_answer_from_research,
)
from rag_service import (
    INSUFFICIENT_RETRIEVAL_MESSAGE,
    build_research_index,
    cached_research_index,
    query_research_index,
)
from research_cache import (
    load_cached_research,
    load_cached_research_by_key,
    list_cached_research,
    research_cache_key,
    save_cached_research,
)


APP_RELEASE = "2026.11.03-company-color-system"


def escape_markdown_dollars(value: Any) -> str:
    """Keep dollar-denominated prices from being parsed as inline math."""
    return str(value if value is not None else "").replace("$", r"\$")


@st.cache_resource
def get_rag_executor() -> ThreadPoolExecutor:
    return ThreadPoolExecutor(max_workers=1, thread_name_prefix="benchmind-rag")


load_dotenv()
st.set_page_config(page_title="BenchMind AI", page_icon="◫", layout="wide")
st.markdown(
    """
    <style>
    :root {color-scheme:light;--ink:#172033;--muted:#5f6b7a;--line:#dfe3ea;--accent:#2563eb;--accent-soft:#eff6ff;--canvas:#f6f7fb;--surface:#fff;--sidebar:#111827;--company-0:#2563eb;--company-0-soft:#eaf2ff;--company-0-border:#7aa7ef;--company-1:#0f8a83;--company-1-soft:#e8f7f4;--company-1-border:#7bc8bd;--company-2:#b7791f;--company-2-soft:#fff5df;--company-2-border:#e1b963;--company-3:#a85578;--company-3-soft:#fbeef4;--company-3-border:#d79ab5}
    html,body,[data-testid="stAppViewContainer"],.stApp{color-scheme:light!important;background:var(--canvas)!important;color:var(--ink)!important}
    [data-testid="stHeader"]{background:rgba(246,247,251,.94)!important}
    [data-testid="stMain"] p,[data-testid="stMain"] label,[data-testid="stMain"] li,[data-testid="stMain"] h1,[data-testid="stMain"] h2,[data-testid="stMain"] h3,[data-testid="stMain"] h4,[data-testid="stMain"] [data-testid="stCaptionContainer"]{color:var(--ink)!important}
    [data-testid="stSidebar"]{background:linear-gradient(180deg,#5b3fa3 0%,#3f2a73 100%)!important;border-right:1px solid #332159;box-shadow:8px 0 24px rgba(49,33,89,.20)}
    [data-testid="stSidebar"]>div{background:transparent!important}
    [data-testid="stSidebarResizer"]{background:#332159}
    [data-testid="stSidebarCollapsedControl"]{background:#fff!important;border:1px solid var(--line);border-radius:0 10px 10px 0;color:var(--ink)!important}
    [data-testid="stSidebar"] h1,[data-testid="stSidebar"] h2,[data-testid="stSidebar"] h3,[data-testid="stSidebar"] p,[data-testid="stSidebar"] span:not(.tag),[data-testid="stSidebar"] label,[data-testid="stSidebar"] [data-testid="stCaptionContainer"]{color:#f8f7ff!important}
    [data-testid="stSidebar"] hr{border-color:rgba(255,255,255,.24)}
    [data-testid="stSidebar"] [data-baseweb="input"],[data-testid="stSidebar"] [data-baseweb="input"]>div{background:#fff!important;border-color:#aeb8c6!important}
    [data-testid="stSidebar"] input{background:#fff!important;color:#172033!important;-webkit-text-fill-color:#172033!important;caret-color:#2563eb}
    [data-testid="stSidebar"] input::placeholder{color:#7b8798!important;-webkit-text-fill-color:#7b8798!important}
    [data-baseweb="input"],[data-baseweb="textarea"],[data-baseweb="select"]>div{background:#fff!important;color:var(--ink)!important;border-color:#cbd2dc!important}
    [data-baseweb="input"] input,[data-baseweb="textarea"] textarea{color:var(--ink)!important;-webkit-text-fill-color:var(--ink)!important}
    [data-baseweb="popover"],[role="listbox"],[role="option"]{color-scheme:light!important;background:#fff!important;color:var(--ink)!important}
    [role="option"]:hover,[role="option"][aria-selected="true"]{background:var(--accent-soft)!important}
    .stButton>button,.stDownloadButton>button{background:#fff!important;color:var(--ink)!important;border:1px solid #cbd2dc!important;box-shadow:0 1px 2px rgba(15,23,42,.04)}
    .stButton>button:hover,.stDownloadButton>button:hover{color:#1d4ed8!important;border-color:#2563eb!important;background:#f8fbff!important}
    .stButton>button[kind="primary"],button[data-testid="stBaseButton-primary"]{background:#2563eb!important;color:#fff!important;border-color:#2563eb!important}
    .stButton>button[kind="primary"]:hover,button[data-testid="stBaseButton-primary"]:hover{background:#1d4ed8!important;color:#fff!important;border-color:#1d4ed8!important}
    .stTabs [data-baseweb="tab-list"]{gap:.45rem;border-bottom:1px solid var(--line);padding-bottom:.55rem}
    .stTabs [data-baseweb="tab"]{color:var(--ink)!important;background:#fff!important;border:1px solid #cbd2dc!important;border-radius:9px!important;padding:.55rem .9rem!important}
    .stTabs [data-baseweb="tab"] *{color:var(--ink)!important}
    .stTabs [data-baseweb="tab"][aria-selected="true"]{color:#fff!important;background:#2563eb!important;border-color:#2563eb!important}
    .stTabs [data-baseweb="tab"][aria-selected="true"] *{color:#fff!important}
    .stTabs [data-baseweb="tab-highlight"]{display:none!important}
    .block-container{max-width:1450px;padding-top:2rem;padding-bottom:4rem}
    .hero{padding:2rem 2.2rem;border:1px solid var(--line);border-radius:22px;background:linear-gradient(135deg,#fff 0%,#f3f7ff 100%);box-shadow:0 12px 32px rgba(15,23,42,.06);margin-bottom:1.3rem}
    .eyebrow,.section-kicker{color:#1d4ed8;font-size:.74rem;font-weight:800;letter-spacing:.13em}
    .hero h1{color:var(--ink);font-size:2.5rem;line-height:1.08;margin:.55rem 0}
    .hero p{color:var(--muted);font-size:1.05rem;max-width:800px;margin:0}
    .status-line{display:flex;gap:.55rem;flex-wrap:wrap;margin-top:1.2rem}
    .status-pill{border:1px solid #d6dce5;border-radius:999px;padding:.35rem .7rem;color:#475569;background:#fff;font-size:.78rem;font-weight:650}
    .company-input-title{display:flex;align-items:center;gap:.5rem;font-size:.76rem;font-weight:850;letter-spacing:.08em;margin:.05rem 0 .55rem;text-transform:uppercase}
    .company-input-title::before{content:"";display:inline-block;width:.65rem;height:.65rem;border-radius:50%}
    .company-input-title-0{color:var(--company-0)}.company-input-title-0::before{background:var(--company-0)}
    .company-input-title-1{color:var(--company-1)}.company-input-title-1::before{background:var(--company-1)}
    .company-input-title-2{color:var(--company-2)}.company-input-title-2::before{background:var(--company-2)}
    .company-input-title-3{color:var(--company-3)}.company-input-title-3::before{background:var(--company-3)}
    .st-key-company_input_0,.st-key-company_input_1,.st-key-company_input_2,.st-key-company_input_3{border-radius:16px!important;padding:.9rem .9rem .35rem!important;min-height:225px;box-shadow:0 7px 18px rgba(15,23,42,.05)}
    .st-key-company_input_0{background:var(--company-0-soft)!important;border-color:var(--company-0-border)!important}
    .st-key-company_input_1{background:var(--company-1-soft)!important;border-color:var(--company-1-border)!important}
    .st-key-company_input_2{background:var(--company-2-soft)!important;border-color:var(--company-2-border)!important}
    .st-key-company_input_3{background:var(--company-3-soft)!important;border-color:var(--company-3-border)!important}
    .st-key-company_input_0 [data-baseweb="input"],.st-key-company_input_1 [data-baseweb="input"],.st-key-company_input_2 [data-baseweb="input"],.st-key-company_input_3 [data-baseweb="input"]{background:rgba(255,255,255,.78)!important}
    .card{border:1px solid var(--line);border-radius:14px;padding:1rem;background:#fff;min-height:120px;margin-bottom:.7rem}
    .interactive-card{transition:transform .2s ease,box-shadow .2s ease,border-color .2s ease,background-color .2s ease}
    .interactive-card:hover{transform:translateY(-3px);border-color:#93b4f5;box-shadow:0 12px 26px rgba(37,99,235,.10);background:#fbfdff}
    .recommendation{border:1px solid #c8b9e8;border-left:4px solid #6d4bb5;background:#f4f0fc}.recommendation h4{margin:.45rem 0 .35rem;color:var(--ink)}.recommendation p{color:var(--muted);margin:0}
    .recommendation-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:.85rem;margin:.4rem 0 1rem}
    .recommendation-grid .recommendation{height:100%;margin:0;min-height:170px}
    .recommendation-grid .recommendation:hover{background:#ebe4fa;border-color:#a790d4;border-left-color:#5b3fa3;box-shadow:0 12px 26px rgba(91,63,163,.14)}
    .profile-details{display:grid;gap:.65rem;margin:.2rem 0 .55rem}
    .profile-detail{border:1px solid var(--line);border-radius:11px;background:#fff;padding:.72rem .8rem}
    .profile-detail p{color:var(--ink)!important;margin:.42rem 0 0;line-height:1.48}
    .profile-label{display:inline-block;border-radius:999px;padding:.18rem .55rem;font-size:.72rem;font-weight:800;letter-spacing:.01em}
    .profile-pricing{color:#1e40af;background:#dbeafe}
    .profile-free{color:#166534;background:#dcfce7}
    .profile-customer{color:#6b21a8;background:#f3e8ff}
    .swot-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));grid-template-rows:repeat(2,minmax(0,1fr));gap:.9rem;margin:.4rem 0 1.15rem}
    .swot-panel{border:1px solid var(--line);border-radius:16px;padding:1.05rem 1.1rem;min-height:230px;background:#fff}
    .swot-panel h4{margin:0 0 .8rem;color:var(--ink);font-size:1.02rem}
    .swot-item{border-top:1px solid rgba(100,116,139,.16);padding:.75rem 0 0;margin-top:.75rem}
    .swot-item:first-of-type{border-top:0;padding-top:0;margin-top:0}
    .swot-item p{margin:.48rem 0 .28rem;color:#334155}
    .swot-strengths,.swot-weaknesses,.swot-opportunities,.swot-threats{background:#edf8f5;border-color:#a7d4c8}
    .tag{display:inline-block;border-radius:999px;padding:.18rem .52rem;font-size:.68rem;font-weight:750;margin-right:.3rem}
    .fact{color:#075985;background:#e0f2fe}.opinion{color:#9a3412;background:#ffedd5}.inference{color:#5b21b6;background:#ede9fe}
    .high{color:#991b1b;background:#fee2e2}.medium{color:#854d0e;background:#fef3c7}.low{color:#166534;background:#dcfce7}
    div[data-testid="stMetric"]{box-sizing:border-box;height:122px;border:1px solid #c9bee8;background:#f2effb!important;border-radius:15px;padding:.85rem 1rem;display:flex;flex-direction:column;justify-content:center;transition:transform .18s ease,background-color .18s ease,border-color .18s ease,box-shadow .18s ease}
    div[data-testid="stMetric"]:hover{transform:translateY(-3px);background:#e8e1f7!important;border-color:#9e8bce;box-shadow:0 10px 22px rgba(67,56,125,.14)}
    div[data-testid="stMetric"] *{color:var(--ink)}
    .small{color:#718096;font-size:.82rem}
    .benchmark-wrap{width:100%;overflow:visible;margin:.4rem 0 1rem}
    .benchmark-table{width:100%;border-collapse:separate;border-spacing:0;table-layout:fixed;font-size:.76rem;color:#334155}
    .benchmark-table th{background:#eef1f5;color:var(--ink);font-weight:750;text-align:left;padding:.7rem .55rem;border-top:1px solid var(--line);border-bottom:1px solid var(--line)}
    .benchmark-table th:first-child{border-left:1px solid #dce5ef;border-radius:10px 0 0 0}
    .benchmark-table th:last-child{border-right:1px solid #dce5ef;border-radius:0 10px 0 0}
    .benchmark-table td{padding:.72rem .55rem;vertical-align:top;line-height:1.42;background:#fff;border-bottom:1px solid #dce5ef;overflow-wrap:anywhere;word-break:normal;white-space:normal;transition:background-color .18s ease,box-shadow .18s ease}
    .benchmark-table td:first-child{border-left:1px solid #dce5ef;font-weight:750}
    .benchmark-table td:last-child{border-right:1px solid #dce5ef;text-align:center;font-weight:800;color:#1d4ed8}
    .benchmark-table tr:last-child td:first-child{border-radius:0 0 0 10px}
    .benchmark-table tr:last-child td:last-child{border-radius:0 0 10px 0}
    .benchmark-table a{color:#1d4ed8;text-decoration:none}
    .benchmark-table .company-row-0 td{background:var(--company-0-soft)}.benchmark-table .company-row-1 td{background:var(--company-1-soft)}
    .benchmark-table .company-row-2 td{background:var(--company-2-soft)}.benchmark-table .company-row-3 td{background:var(--company-3-soft)}
    .benchmark-table .company-row-0 td:first-child{border-left:4px solid var(--company-0)}.benchmark-table .company-row-1 td:first-child{border-left:4px solid var(--company-1)}
    .benchmark-table .company-row-2 td:first-child{border-left:4px solid var(--company-2)}.benchmark-table .company-row-3 td:first-child{border-left:4px solid var(--company-3)}
    .benchmark-table .company-row-0:hover td{background:#dce9fd}.benchmark-table .company-row-1:hover td{background:#d9f0eb}
    .benchmark-table .company-row-2:hover td{background:#fbeac6}.benchmark-table .company-row-3:hover td{background:#f5dfe9}
    .benchmark-table tbody tr:hover td{box-shadow:inset 0 1px rgba(37,99,235,.12),inset 0 -1px rgba(37,99,235,.12)}
    .company-dot{display:inline-block;width:.55rem;height:.55rem;border-radius:50%;margin-right:.38rem;vertical-align:.03rem}
    .company-row-0 .company-dot{background:var(--company-0)}.company-row-1 .company-dot{background:var(--company-1)}
    .company-row-2 .company-dot{background:var(--company-2)}.company-row-3 .company-dot{background:var(--company-3)}
    .st-key-profile_company_0,.st-key-profile_company_1,.st-key-profile_company_2,.st-key-profile_company_3{border-radius:15px;margin:.55rem 0;padding:.28rem .45rem .48rem}
    .st-key-profile_company_0{background:var(--company-0-soft);border:1px solid var(--company-0-border)}
    .st-key-profile_company_1{background:var(--company-1-soft);border:1px solid var(--company-1-border)}
    .st-key-profile_company_2{background:var(--company-2-soft);border:1px solid var(--company-2-border)}
    .st-key-profile_company_3{background:var(--company-3-soft);border:1px solid var(--company-3-border)}
    .st-key-profile_company_0 [data-testid="stExpander"],.st-key-profile_company_1 [data-testid="stExpander"],.st-key-profile_company_2 [data-testid="stExpander"],.st-key-profile_company_3 [data-testid="stExpander"]{background:transparent!important;border:0!important}
    .st-key-profile_company_0 .profile-detail,.st-key-profile_company_1 .profile-detail,.st-key-profile_company_2 .profile-detail,.st-key-profile_company_3 .profile-detail{background:rgba(255,255,255,.62)}
    .st-key-profile_company_0 [data-testid="stAlert"],.st-key-profile_company_1 [data-testid="stAlert"],.st-key-profile_company_2 [data-testid="stAlert"],.st-key-profile_company_3 [data-testid="stAlert"]{background:rgba(255,255,255,.62)!important;border-color:rgba(71,85,105,.22)!important}
    .evidence-table{font-size:.72rem}
    .evidence-table td:last-child{text-align:left;font-weight:400;color:#334155}
    @media (max-width:900px){.swot-grid,.recommendation-grid{grid-template-columns:1fr;grid-template-rows:auto}.st-key-company_input_0,.st-key-company_input_1,.st-key-company_input_2,.st-key-company_input_3{min-height:auto}}
    </style>
    """,
    unsafe_allow_html=True,
)


def init_state() -> None:
    if "research" not in st.session_state:
        benchmark_key = str(st.query_params.get("benchmark", "")).strip()
        cached = load_cached_research_by_key(benchmark_key) if benchmark_key else None
        if cached:
            research, saved_at = cached
            st.session_state.research = research
            st.session_state.research_mode = "Kayıtlı Live Research"
            st.session_state.active_saved_research = {
                "key": benchmark_key,
                "saved_at": saved_at,
            }
            st.session_state.resolved_company_inputs = [
                {
                    "name": company.get("name", ""),
                    "website": company.get("website", ""),
                }
                for company in research.get("companies", [])[:4]
            ]
        else:
            st.session_state.research = load_demo_data()
            st.session_state.research_mode = "Demo snapshot'ı"
            if benchmark_key and "benchmark" in st.query_params:
                del st.query_params["benchmark"]
    st.session_state.setdefault("chat_history", [])
    st.session_state.setdefault("pending_question", None)
    st.session_state.setdefault("confirm_live_research", False)
    st.session_state.setdefault("remote_job_id", None)
    st.session_state.setdefault("remote_job", None)
    st.session_state.setdefault("remote_job_result_loaded", None)
    st.session_state.setdefault("active_saved_research", None)
    st.session_state.setdefault("show_saved_research_dialog", False)
    st.session_state.setdefault(
        "rag_stats",
        {
            "collection_name": "", "documents": 0, "chunks": 0,
            "embeddings": 0, "failed_urls": 0, "status": "not_indexed",
        },
    )


def start_rag_index_job(research: dict[str, Any]) -> None:
    previous = st.session_state.get("rag_job")
    if isinstance(previous, Future) and not previous.done():
        previous.cancel()
    events: Queue = Queue()

    def report(stage: str, payload: dict[str, Any]) -> None:
        events.put(
            {
                "time": time.strftime("%H:%M:%S"),
                "stage": stage,
                "message": payload.get("message", stage),
            }
        )

    st.session_state.rag_events_queue = events
    st.session_state.rag_events = [
        {"time": time.strftime("%H:%M:%S"), "message": "RAG index arka planda başlatıldı."}
    ]
    st.session_state.rag_stats = {
        "collection_name": "", "documents": 0, "chunks": 0,
        "embeddings": 0, "failed_urls": 0, "status": "indexing",
    }
    st.session_state.rag_job = get_rag_executor().submit(
        build_research_index, research, progress=report
    )


def refresh_rag_index_job() -> bool:
    events = st.session_state.get("rag_events_queue")
    if isinstance(events, Queue):
        while True:
            try:
                event = events.get_nowait()
            except Empty:
                break
            st.session_state.setdefault("rag_events", []).append(event)
            st.session_state.rag_events = st.session_state.rag_events[-30:]
    job = st.session_state.get("rag_job")
    if not isinstance(job, Future) or not job.done():
        return False
    if st.session_state.rag_stats.get("status") != "indexing":
        return False
    try:
        st.session_state.rag_stats = job.result().to_dict()
    except Exception as exc:
        st.session_state.rag_stats = {
            "collection_name": "", "documents": 0, "chunks": 0,
            "embeddings": 0, "failed_urls": 0, "status": "failed",
            "error": str(exc),
        }
    return True


def render_rag_status_panel() -> None:
    refresh_rag_index_job()
    rag_stats = st.session_state.rag_stats
    st.markdown('<div class="section-kicker">VECTOR RAG INDEX</div>', unsafe_allow_html=True)
    if rag_stats.get("status") == "ready":
        for column, (label, value) in zip(st.columns(4), [
            ("İndirilen document", rag_stats.get("documents", 0)),
            ("Chunk", rag_stats.get("chunks", 0)),
            ("Embedding/vector", rag_stats.get("embeddings", 0)),
            ("Alınamayan URL", rag_stats.get("failed_urls", 0)),
        ], strict=False):
            column.metric(label, value)
        cache_note = " · cache kullanıldı" if rag_stats.get("cache_hit") else ""
        st.caption(
            f"Embedding modeli: {rag_stats.get('embedding_model', 'Bilinmiyor')} · "
            f"Kalıcı Chroma collection: {rag_stats.get('collection_name', '')}{cache_note}"
        )
    elif rag_stats.get("status") == "indexing":
        st.info("Benchmark hazır. Kaynaklar arka planda Vector RAG index'ine alınıyor.")
        events = st.session_state.get("rag_events", [])
        for event in events[-8:]:
            st.caption(f"{event.get('time', '')} · {event.get('message', '')}")
    elif rag_stats.get("status") == "failed":
        st.error(f"Araştırma tamamlandı ancak Vector RAG index oluşturulamadı: {rag_stats.get('error', 'Bilinmeyen hata')}")
    else:
        st.info("Demo snapshot'ı Vector RAG olarak indekslenmez. Gerçek web document, chunk ve embedding metrikleri Live Research sonrasında burada gösterilir.")


@st.fragment(run_every=1.0)
def render_running_rag_status() -> None:
    completed = refresh_rag_index_job()
    render_rag_status_panel()
    if completed:
        st.rerun(scope="app")


def load_demo_state() -> None:
    """Restore both the demo results and the company input widgets."""
    demo_data = load_demo_data()
    st.session_state.research = demo_data
    st.session_state.research_mode = "Demo snapshot'ı"
    st.session_state.chat_history = []
    st.session_state.pending_question = None
    st.session_state.remote_job_id = None
    st.session_state.remote_job = None
    st.session_state.remote_job_result_loaded = None
    st.session_state.active_saved_research = None
    st.session_state.show_saved_research_dialog = False
    st.session_state.rag_stats = {
        "collection_name": "", "documents": 0, "chunks": 0,
        "embeddings": 0, "failed_urls": 0, "status": "not_indexed",
    }
    previous = st.session_state.pop("rag_job", None)
    if isinstance(previous, Future) and not previous.done():
        previous.cancel()
    st.session_state.pop("rag_events_queue", None)
    st.session_state.pop("rag_events", None)
    st.session_state["saved_research_key"] = ""
    if "benchmark" in st.query_params:
        del st.query_params["benchmark"]
    for index, company in enumerate(demo_data.get("companies", [])[:4]):
        st.session_state[f"name_{index}"] = company.get("name", "")
        st.session_state[f"url_{index}"] = company.get("website", "")


def badge(label: str) -> str:
    css = {
        "Verified fact": "fact", "Customer opinion": "opinion", "AI inference": "inference",
        "High": "high", "Medium": "medium", "Low": "low",
    }.get(label, "inference")
    display_label = {
        "Verified fact": "Doğrulanmış bilgi (Verified fact)",
        "Customer opinion": "Müşteri görüşü (Customer opinion)",
        "AI inference": "AI çıkarımı (AI inference)",
        "High": "Yüksek",
        "Medium": "Orta",
        "Low": "Düşük",
    }.get(label, label)
    return f'<span class="tag {css}">{html.escape(display_label)}</span>'


def links(ids: list[str], data: dict[str, Any], as_html: bool = False) -> str:
    lookup = {item.get("id"): item for item in data.get("sources", [])}
    if as_html:
        output = [
            f'<a href="{html.escape(lookup[item_id]["url"], quote=True)}" target="_blank">{html.escape(item_id)}</a>'
            for item_id in ids if item_id in lookup
        ]
    else:
        output = [f"[{item_id}]({lookup[item_id]['url']})" for item_id in ids if item_id in lookup]
    return " · ".join(output) if output else "Bağlantılı kaynak yok"


def render_conflicts(data: dict[str, Any]) -> None:
    for conflict in data.get("conflicts", []):
        st.warning(
            f"**Çelişkili kanıt — {conflict.get('topic', 'Belirtilmemiş')}**\n\n"
            f"{conflict.get('description', '')}\n\nKaynaklar: {links(conflict.get('source_ids', []), data)}"
        )


def render_radar(data: dict[str, Any]) -> None:
    labels = [label for label, _ in CRITERIA.values()]
    colors = ["#2563eb", "#0f8a83", "#b7791f", "#a85578"]
    figure = go.Figure()
    for index, company in enumerate(data.get("companies", [])):
        values = [float(company.get("scores", {}).get(key, 0)) for key in CRITERIA]
        figure.add_trace(go.Scatterpolar(
            r=values + [values[0]], theta=labels + [labels[0]], fill="toself", opacity=.55,
            name=company.get("name", "Bilinmiyor"), line_color=colors[index % len(colors)],
        ))
    figure.update_layout(
        polar={"radialaxis":{"visible":True,"range":[0,10],"tickfont":{"size":10}}},
        template="simple_white", font={"color":"#172033"},
        margin={"l":45,"r":45,"t":35,"b":35}, height=470,
        legend={"orientation":"h","y":-.13}, paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(figure, width="stretch")


def render_bar(data: dict[str, Any]) -> None:
    frame = benchmark_dataframe(data).sort_values("Genel puan")
    figure = px.bar(
        frame, x="Genel puan", y="Şirket", orientation="h", text="Genel puan", color="Genel puan",
        color_continuous_scale=["#bfdbfe", "#2563eb"], range_x=[0,10],
    )
    figure.update_traces(textposition="outside")
    figure.update_layout(
        coloraxis_showscale=False, xaxis_title="Ağırlıklı puan / 10", yaxis_title="", height=390,
        template="simple_white", font={"color":"#172033"},
        margin={"l":20,"r":35,"t":25,"b":30}, paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(figure, width="stretch")


def render_benchmark_table(data: dict[str, Any]) -> None:
    headers = [
        ("Şirket", "10%"),
        ("Fiyatlandırma", "17%"),
        ("Ücretsiz plan", "18%"),
        ("Hedef müşteri", "17%"),
        ("Temel güçlü yön", "16%"),
        ("Temel sınırlama", "16%"),
        ("Genel puan", "6%"),
    ]
    header_html = "".join(
        f'<th style="width:{width}">{html.escape(label)}</th>' for label, width in headers
    )
    rows = []
    for index, company in enumerate(data.get("companies", [])):
        name = html.escape(str(company.get("name", "Bilinmiyor")))
        website = html.escape(str(company.get("website", "")), quote=True)
        name_with_dot = f'<span class="company-dot"></span>{name}'
        company_cell = f'<a href="{website}" target="_blank">{name_with_dot}</a>' if website else name_with_dot
        values = [
            company_cell,
            html.escape(str(company.get("pricing", "Bulunamadı"))),
            html.escape(str(company.get("free_plan", "Bulunamadı"))),
            html.escape(str(company.get("target_customer", "Bulunamadı"))),
            html.escape(str(company.get("key_strength", ""))),
            html.escape(str(company.get("key_limitation", ""))),
            f'{float(company.get("overall_score", 0)):.2f}',
        ]
        color_index = index % 4
        rows.append(
            f'<tr class="company-row-{color_index}">'
            + "".join(f"<td>{value}</td>" for value in values)
            + "</tr>"
        )
    table_html = (
        '<div class="benchmark-wrap"><table class="benchmark-table">'
        f"<thead><tr>{header_html}</tr></thead><tbody>{''.join(rows)}</tbody></table></div>"
    )
    st.markdown(table_html, unsafe_allow_html=True)


def render_evidence_table(frame: pd.DataFrame, data: dict[str, Any]) -> None:
    if frame.empty:
        st.info("Seçilen filtrelerle eşleşen kanıt bulunamadı.")
        return
    headers = [
        ("Şirket", "9%"),
        ("Kategori", "10%"),
        ("Kanıt türü", "13%"),
        ("Bulgu", "30%"),
        ("Kaynak ID'leri", "7%"),
        ("Kaynak URL'leri", "23%"),
        ("Araştırma tarihi", "8%"),
    ]
    header_html = "".join(
        f'<th style="width:{width}">{html.escape(label)}</th>' for label, width in headers
    )
    rows = []
    company_colors = {
        str(company.get("name", "")): index % 4
        for index, company in enumerate(data.get("companies", []))
    }
    for finding in frame.to_dict(orient="records"):
        raw_urls = str(finding.get("Kaynak URL'leri", ""))
        urls = [url.strip() for url in raw_urls.split("|") if url.strip()]
        url_cell = "<br>".join(
            f'<a href="{html.escape(url, quote=True)}" target="_blank">{html.escape(url)}</a>'
            for url in urls
        ) or "—"
        company_name = str(finding.get("Şirket", ""))
        color_index = company_colors.get(company_name)
        company_label = html.escape(company_name)
        if color_index is not None:
            company_label = f'<span class="company-dot"></span>{company_label}'
        values = [
            company_label,
            html.escape(str(finding.get("Kategori", ""))),
            html.escape(str(finding.get("Kanıt türü", ""))),
            html.escape(str(finding.get("Bulgu", ""))),
            html.escape(str(finding.get("Kaynak ID'leri", ""))),
            url_cell,
            html.escape(str(finding.get("Araştırma tarihi", ""))),
        ]
        row_class = f' class="company-row-{color_index}"' if color_index is not None else ""
        rows.append(f"<tr{row_class}>" + "".join(f"<td>{value}</td>" for value in values) + "</tr>")
    table_html = (
        '<div class="benchmark-wrap"><table class="benchmark-table evidence-table">'
        f"<thead><tr>{header_html}</tr></thead><tbody>{''.join(rows)}</tbody></table></div>"
    )
    st.markdown(table_html, unsafe_allow_html=True)


def render_swot(data: dict[str, Any]) -> None:
    labels = {"strengths":("Güçlü yönler","↗"), "weaknesses":("Zayıf yönler","↘"),
              "opportunities":("Fırsatlar","◎"), "threats":("Tehditler","△")}
    panels = []
    for key, (title, symbol) in labels.items():
        items = []
        for item in data.get("swot", {}).get(key, []):
            items.append(
                f'<div class="swot-item">{badge(item.get("evidence_type","AI inference"))}'
                f'<p>{html.escape(item.get("statement",""))}</p>'
                f'<span class="small">{links(item.get("source_ids",[]), data, as_html=True)}</span></div>'
            )
        panels.append(
            f'<section class="swot-panel swot-{key} interactive-card">'
            f'<h4>{symbol} {html.escape(title)}</h4>{"".join(items)}</section>'
        )
    st.markdown(f'<div class="swot-grid">{"".join(panels)}</div>', unsafe_allow_html=True)


def render_recommendations(data: dict[str, Any]) -> None:
    items = data.get("recommendations", [])
    cards = [
        f'<article class="card recommendation interactive-card">{badge(item.get("priority","Medium"))}{badge("AI inference")}'
        f'<h4>{html.escape(item.get("title","Öneri"))}</h4>'
        f'<p>{html.escape(item.get("statement",""))}</p>'
        f'<div class="small" style="margin-top:.6rem">Kanıt: {links(item.get("source_ids",[]),data,as_html=True)}</div></article>'
        for item in items
    ]
    st.markdown(f'<div class="recommendation-grid">{"".join(cards)}</div>', unsafe_allow_html=True)


def ask(
    question: str,
    data: dict[str, Any],
    api_key: str | None,
    model: str,
    provider: str | None,
) -> None:
    st.session_state.chat_history.append({"role":"user", "content":question})
    with st.chat_message("user"):
        st.markdown(question)
    retrieved_chunks: list[dict[str, Any]] = []
    if not api_key:
        answer = "API erişimi yapılandırılmamış. `ANTHROPIC_API_KEY` veya `OPENAI_API_KEY` değerini ekleyin."
    elif api_base_url():
        try:
            with st.spinner("FastAPI kanıt paketini kontrol ediyor..."):
                response = ask_backend(
                    question,
                    data,
                    str(st.session_state.get("rag_stats", {}).get("collection_name", "")),
                    model,
                )
            answer = str(response.get("answer", ""))
            retrieved_chunks = list(response.get("retrieved_sources", []))
            with st.chat_message("assistant"):
                st.markdown(answer)
        except BenchMindAPIError as exc:
            answer = f"Kaynaklara dayalı yanıt üretilemedi: {exc}"
            with st.chat_message("assistant"):
                st.error(answer)
    else:
        try:
            with st.spinner("Kanıt paketi kontrol ediliyor..."):
                rag_stats = st.session_state.get("rag_stats", {})
                if rag_stats.get("status") == "ready":
                    retrieved_chunks = query_research_index(
                        str(rag_stats.get("collection_name", "")), question
                    )
            with st.chat_message("assistant"):
                if rag_stats.get("status") == "ready" and not retrieved_chunks:
                    answer = INSUFFICIENT_RETRIEVAL_MESSAGE
                    st.markdown(answer)
                else:
                    answer = st.write_stream(
                        stream_answer_from_research(
                            question,
                            data,
                            api_key,
                            model,
                            provider or "OpenAI",
                            retrieved_chunks=retrieved_chunks,
                        )
                    )
        except Exception as exc:
            answer = f"Kaynaklara dayalı yanıt üretilemedi: {exc}"
            with st.chat_message("assistant"):
                st.error(answer)
    if not api_key:
        with st.chat_message("assistant"):
            st.markdown(answer)
    assistant_message: dict[str, Any] = {"role":"assistant", "content":answer}
    if retrieved_chunks and api_base_url():
        assistant_message["retrieved_sources"] = retrieved_chunks
    elif retrieved_chunks:
        unique_sources = {}
        for chunk in retrieved_chunks:
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
        assistant_message["retrieved_sources"] = list(unique_sources.values())
    st.session_state.chat_history.append(assistant_message)


def refresh_remote_benchmark_job() -> bool:
    job_id = st.session_state.get("remote_job_id")
    if not job_id:
        return False
    job = get_benchmark_job(str(job_id))
    st.session_state.remote_job = job

    changed = False
    status = str(job.get("status", ""))
    result = job.get("result")
    result_loaded = st.session_state.get("remote_job_result_loaded") == job_id

    # The API exposes the benchmark when research finishes, before the slower
    # crawl/embedding phase completes. Load it once and keep polling for final
    # RAG statistics without resetting the displayed benchmark or chat again.
    if status in {"indexing", "completed"} and isinstance(result, dict) and not result_loaded:
        st.session_state.research = result
        st.session_state.research_mode = "FastAPI Live Research"
        st.session_state.active_saved_research = None
        st.session_state.chat_history = []
        st.session_state.resolved_company_inputs = st.session_state.get(
            "remote_job_inputs", []
        )
        st.session_state.remote_job_result_loaded = job_id
        remote_inputs = st.session_state.get("remote_job_inputs", [])
        if remote_inputs:
            st.query_params["benchmark"] = research_cache_key(remote_inputs)
        st.session_state.rag_stats = (
            job.get("rag_stats", {})
            if status == "completed"
            else {
                "collection_name": "", "documents": 0, "chunks": 0,
                "embeddings": 0, "failed_urls": 0, "status": "indexing",
            }
        )
        changed = True

    if status == "completed":
        st.session_state.rag_stats = job.get("rag_stats", {})
        st.session_state.remote_job_id = None
        st.session_state.remote_job_result_loaded = None
        changed = True
    return changed


@st.fragment(run_every=2.0)
def render_remote_benchmark_job() -> None:
    if not st.session_state.get("remote_job_id"):
        return
    try:
        completed = refresh_remote_benchmark_job()
    except BenchMindAPIError as exc:
        st.error(f"FastAPI job durumu alınamadı: {exc}")
        return
    if completed:
        st.rerun(scope="app")
    job = st.session_state.get("remote_job") or {}
    if job.get("status") == "failed":
        st.error(f"Live Research başarısız oldu: {job.get('error', 'Bilinmeyen hata')}")
        st.session_state.remote_job_id = None
        return
    st.info(
        f"FastAPI job `{str(job.get('job_id', ''))[:8]}` · "
        f"{job.get('message', 'İşlem sürüyor.')}"
    )


def sync_company_name_from_url(index: int, default_name: str, default_url: str) -> None:
    name_key = f"name_{index}"
    url_key = f"url_{index}"
    current_name = st.session_state.get(name_key, "").strip()
    current_url = st.session_state.get(url_key, "").strip()
    if current_url != default_url and (not current_name or current_name == default_name):
        inferred_name = company_name_from_url(current_url)
        if inferred_name:
            st.session_state[name_key] = inferred_name


def sync_company_url_from_name(index: int, default_name: str, default_url: str) -> None:
    """Discard a stale demo URL when the corresponding company name changes."""
    name_key = f"name_{index}"
    url_key = f"url_{index}"
    current_name = st.session_state.get(name_key, "").strip()
    current_url = st.session_state.get(url_key, "").strip()
    if current_name == default_name or current_url != default_url:
        return
    inferred_url = company_url_from_name(current_name)
    st.session_state[url_key] = inferred_url
    if inferred_url:
        inferred_name = company_name_from_url(inferred_url)
        if inferred_name:
            st.session_state[name_key] = inferred_name


init_state()
refresh_rag_index_job()
backend_enabled = bool(api_base_url())
backend_error = ""
if backend_enabled:
    try:
        backend_config = get_backend_config()
        provider = backend_config.get("provider")
        api_key = "configured-in-fastapi" if backend_config.get("configured") else None
    except BenchMindAPIError as exc:
        provider, api_key = None, None
        backend_error = str(exc)
else:
    try:
        provider, api_key = get_api_config(st.secrets)
    except Exception:
        provider, api_key = get_api_config()

with st.sidebar:
    st.markdown("## BenchMind AI")
    st.caption("Görünür kanıt sınırlarıyla competitive intelligence.")
    st.caption(f"Sürüm: `{APP_RELEASE}`")
    st.divider()
    st.markdown("**Workspace**")
    st.caption(f"Mod: {st.session_state.research_mode}")
    api_label = f"FastAPI · {provider}" if backend_enabled and provider else provider
    st.caption(f"API: {api_label + ' bağlı' if api_key and api_label else 'Yapılandırılmamış'}")
    if backend_error:
        st.error(f"FastAPI bağlantısı kurulamadı: {backend_error}")
    rag_stats = st.session_state.rag_stats
    rag_label = {"ready": "Hazır", "indexing": "Hazırlanıyor"}.get(
        rag_stats.get("status"), "İndekslenmedi"
    )
    st.caption(
        f"Vector RAG: {rag_label} · {rag_stats.get('documents', 0)} document · "
        f"{rag_stats.get('chunks', 0)} chunk · {rag_stats.get('embeddings', 0)} embedding"
    )
    default_model = default_model_for_provider(provider)
    model_name = st.text_input(
        f"{provider or 'LLM'} model'i",
        default_model,
        key=f"llm_model_{(provider or 'none').lower()}",
        help="ANTHROPIC_MODEL veya OPENAI_MODEL ile de ayarlanabilir.",
    )
    if provider == "Claude Code":
        st.caption("Live Research Claude Code WebSearch kullanır ve çalışma başına varsayılan $1 bütçe sınırına sahiptir.")
    elif provider == "Anthropic":
        st.caption(
            f"Live Research en fazla {LIVE_RESEARCH_MAX_SEARCHES} Claude web search kullanır; "
            "token ve arama ücreti API bakiyenizden düşer."
        )
    elif provider == "Gemini":
        st.caption(
            "Live Research Gemini Google Search grounding kullanır; ücret seçili Google AI projesine yansır."
        )
    st.divider()
    st.markdown("**Kanıt politikası**")
    st.caption("Resmî ürün sayfaları → resmî dokümantasyon → review platform'ları → topluluk yorumları")
    st.markdown(f"{badge('Verified fact')} {badge('Customer opinion')} {badge('AI inference')}", unsafe_allow_html=True)
    st.divider()
    st.caption("Portföy MVP'si · Yalnızca herkese açık bilgiler")

data = st.session_state.research
st.markdown(
    f"""<div class="hero"><div class="eyebrow">COMPETITIVE PRODUCT INTELLIGENCE</div>
    <h1>BenchMind AI</h1><p>Dağınık rakip verilerini kaynaklara dayalı ürün kararlarına dönüştürün. Positioning'i,
    ürün yeteneklerini, müşteri sinyallerini ve stratejik fırsatları kanıt odaklı tek bir workspace'te karşılaştırın.</p>
    <div class="status-line"><span class="status-pill">{html.escape(st.session_state.research_mode)}</span>
    <span class="status-pill">Son araştırma: {html.escape(data.get('research_date','Bilinmiyor'))}</span>
    <span class="status-pill">Sektör: {html.escape(data.get('industry','Bilinmiyor'))}</span></div></div>""",
    unsafe_allow_html=True,
)

st.markdown('<div class="section-kicker">RESEARCH KURULUMU</div>', unsafe_allow_html=True)
st.subheader("Bir hedef şirket ve en fazla üç rakip seçin")
defaults = [("Fathom","https://www.fathom.ai/"), ("Fireflies.ai","https://fireflies.ai/"),
            ("Otter.ai","https://otter.ai/"), ("MeetGeek","https://meetgeek.ai/")]
resolved_widget_inputs = st.session_state.pop("resolved_company_inputs", None)
if resolved_widget_inputs:
    for index, company in enumerate(resolved_widget_inputs[:4]):
        st.session_state[f"name_{index}"] = company.get("name", "")
        st.session_state[f"url_{index}"] = company.get("website", "")
for index, (default_name, default_url) in enumerate(defaults):
    sync_company_name_from_url(index, default_name, default_url)
company_columns = st.columns(4)
inputs = []
for i, ((default_name, default_url), company_col) in enumerate(
    zip(defaults, company_columns, strict=False)
):
    role = "Hedef şirket" if i == 0 else f"Rakip {i}"
    name_key = f"name_{i}"
    url_key = f"url_{i}"
    st.session_state.setdefault(name_key, default_name)
    st.session_state.setdefault(url_key, default_url)
    with company_col:
        with st.container(key=f"company_input_{i}", border=True):
            st.markdown(
                f'<div class="company-input-title company-input-title-{i}">{role}</div>',
                unsafe_allow_html=True,
            )
            name = st.text_input(
                "Şirket adı",
                key=name_key,
                on_change=sync_company_url_from_name,
                args=(i, default_name, default_url),
                help="Şirket adı veya amazon.com.tr gibi bir domain girebilirsiniz.",
            )
            url = st.text_input(
                "Web sitesi (isteğe bağlı)",
                key=url_key,
                on_change=sync_company_name_from_url,
                args=(i, default_name, default_url),
                help="URL'yi değiştirdiğinizde varsayılan şirket adı otomatik olarak güncellenir.",
            )
    inputs.append({"name":name, "website":url})

try:
    resolved_inputs = clean_company_inputs(inputs)
except ValueError:
    resolved_inputs = inputs
cached_research = load_cached_research(resolved_inputs)
saved_research = list_cached_research(limit=3)
saved_by_key = {entry["key"]: entry for entry in saved_research}
saved_options = list(saved_by_key)


def saved_research_label(cache_key: str) -> str:
    entry = saved_by_key[cache_key]
    target_name = str(entry.get("target_company", "Bilinmiyor"))
    rivals = [name for name in entry.get("companies", []) if name != target_name]
    comparison = ", ".join(rivals) if rivals else "Rakip yok"
    saved_at = str(entry.get("saved_at", "")).replace("T", " ").replace("+00:00", " UTC")
    return f"{target_name} — {comparison} — {saved_at}"


left_button, cached_button, live_button, _ = st.columns([1,1.25,1.25,1.5])
with left_button:
    st.button("Demo'yu Yükle", width="stretch", on_click=load_demo_state)
with cached_button:
    cached_clicked = st.button(
        "Kayıtlı Sonucu Getir",
        width="stretch",
        disabled=not saved_options,
        help="Son üç Live Research sonucunu seçim penceresinde gösterir.",
    )
with live_button:
    live_clicked = st.button(
        "Live Research",
        type="primary",
        width="stretch",
        disabled=bool(st.session_state.get("remote_job_id")),
    )

if cached_research:
    cached_result, cached_at = cached_research
    cached_companies = ", ".join(
        str(company.get("name", "")).strip()
        for company in cached_result.get("companies", [])
        if str(company.get("name", "")).strip()
    )
    st.caption(
        "Bu şirket listesi için kayıtlı araştırma hazır · "
        f"{cached_at.replace('T', ' ').replace('+00:00', ' UTC')}"
        + (f" · {cached_companies}" if cached_companies else "")
    )

def dismiss_saved_research_dialog() -> None:
    st.session_state.show_saved_research_dialog = False


@st.dialog("Son 3 Live Research", on_dismiss=dismiss_saved_research_dialog)
def show_saved_research_dialog() -> None:
    st.caption("Yüklemek istediğiniz araştırmayı seçin. Benchmark tablosu seçiminize göre yenilenir.")
    selected_saved_key = st.radio(
        "Kayıtlı araştırmalar",
        saved_options,
        format_func=saved_research_label,
        key="saved_research_dialog_key",
    )
    selected_entry = saved_by_key[selected_saved_key]
    selected_companies = ", ".join(selected_entry.get("companies", []))
    st.info(
        f"Hedef: **{selected_entry.get('target_company', 'Bilinmiyor')}**"
        + (f" · Karşılaştırma: {selected_companies}" if selected_companies else "")
    )
    if st.button("Seçili Kaydı Yükle", type="primary", width="stretch"):
        result = selected_entry["research"]
        st.session_state.research = result
        st.session_state.research_mode = "Kayıtlı Live Research"
        st.session_state.active_saved_research = {
            "key": selected_saved_key,
            "saved_at": selected_entry.get("saved_at", ""),
        }
        st.session_state.chat_history = []
        st.session_state.resolved_company_inputs = [
            {"name": company.get("name", ""), "website": company.get("website", "")}
            for company in result.get("companies", [])[:4]
        ]
        st.query_params["benchmark"] = selected_saved_key
        if backend_enabled:
            cached_rag = cached_research_index(result)
            st.session_state.rag_stats = (
                cached_rag.to_dict()
                if cached_rag
                else {
                    "collection_name": "", "documents": 0, "chunks": 0,
                    "embeddings": 0, "failed_urls": 0, "status": "not_indexed",
                }
            )
        else:
            start_rag_index_job(result)
        st.session_state.show_saved_research_dialog = False
        st.rerun(scope="app")


if cached_clicked:
    st.session_state.show_saved_research_dialog = True
    active_saved = st.session_state.get("active_saved_research") or {}
    preferred_key = active_saved.get("key")
    if preferred_key not in saved_by_key:
        preferred_key = saved_options[0]
    st.session_state.saved_research_dialog_key = preferred_key

if st.session_state.show_saved_research_dialog:
    show_saved_research_dialog()

if live_clicked:
    if not api_key:
        st.error("API erişimi yapılandırılmamış. GEMINI_API_KEY, ANTHROPIC_API_KEY veya OPENAI_API_KEY değerini `.env` dosyasına ya da Streamlit Secrets'a ekleyin.")
    else:
        st.session_state.confirm_live_research = True


@st.dialog("Live Research maliyet onayı")
def confirm_and_run_live_research() -> None:
    st.write(f"**Hedef şirket:** {resolved_inputs[0]['name']}")
    st.caption(
        "Araştırılacak şirketler: "
        + ", ".join(company["name"] for company in resolved_inputs)
    )
    st.write(f"**Sağlayıcı:** {provider or 'OpenAI'}")
    st.write(f"**Model:** `{model_name}`")
    cost_estimate = estimate_live_research_cost(
        provider,
        model_name,
        LIVE_RESEARCH_MAX_SEARCHES,
    )
    if cost_estimate["known"]:
        low_usd = float(cost_estimate["low_usd"])
        high_usd = float(cost_estimate["high_usd"])
        if cost_estimate.get("is_budget_cap"):
            st.info(f"Çalışma bütçe sınırı: **${high_usd:.2f}**")
        else:
            st.info(
                f"Tahmini Live Research maliyeti: **${low_usd:.3f}–${high_usd:.3f}**"
            )
    else:
        search_cost = cost_estimate.get("search_cost_usd")
        if search_cost is None:
            st.warning("Bu model için maliyet tahmini bulunmuyor.")
        else:
            st.warning(
                "Bu modelin token fiyatı bilinmiyor. "
                f"En fazla {LIVE_RESEARCH_MAX_SEARCHES} arama için arama ücreti "
                f"**${float(search_cost):.2f}** olabilir; token maliyeti buna dahil değildir."
            )
    if cost_estimate.get("search_note"):
        st.caption(str(cost_estimate["search_note"]))
    st.caption(
        "Tahmin kullanım miktarına göre değişir; kesin tutar sağlayıcının faturasındaki tutardır. "
        f"Fiyat tarihi: {cost_estimate['pricing_as_of']}."
    )
    approve, cancel = st.columns(2)
    if cancel.button("Vazgeç", width="stretch"):
        st.session_state.confirm_live_research = False
        st.rerun()
    if approve.button("Onayla ve araştır", type="primary", width="stretch"):
        st.session_state.confirm_live_research = False
        if backend_enabled:
            try:
                job = create_benchmark_job(resolved_inputs, model_name)
            except BenchMindAPIError as exc:
                st.error(f"FastAPI benchmark job'ı başlatılamadı: {exc}")
            else:
                st.session_state.remote_job_id = job["job_id"]
                st.session_state.remote_job = job
                st.session_state.remote_job_inputs = resolved_inputs
                st.session_state.remote_job_result_loaded = None
                st.rerun()
            return
        try:
            with st.spinner("Resmî sayfalar, dokümantasyon, fiyatlandırma ve müşteri sinyalleri araştırılıyor..."):
                result = research_companies(resolved_inputs, api_key, model_name, provider or "OpenAI")
        except Exception as exc:
            st.error(f"Live Research başarısız oldu: {exc}")
        else:
            st.session_state.research = result
            st.session_state.research_mode = "Live Research"
            st.session_state.active_saved_research = None
            st.session_state.chat_history = []
            try:
                save_cached_research(resolved_inputs, result)
                st.query_params["benchmark"] = research_cache_key(resolved_inputs)
            except (OSError, TypeError, ValueError) as exc:
                st.warning(f"Araştırma tamamlandı ancak kayıtlı sonuç oluşturulamadı: {exc}")
            start_rag_index_job(result)
            st.session_state.resolved_company_inputs = resolved_inputs
            st.rerun()


if st.session_state.confirm_live_research:
    confirm_and_run_live_research()

render_remote_benchmark_job()

data = st.session_state.research
companies = data.get("companies", [])
leader = max(companies, key=lambda item:item.get("overall_score",0), default={})
target = next((item for item in companies if item.get("name") == data.get("target_company")), companies[0] if companies else {})
verified = sum(item.get("evidence_type") == "Verified fact" for item in data.get("findings", []))
confidence = data.get("confidence") or calculate_confidence(data)

active_names = ", ".join(
    str(company.get("name", "")).strip()
    for company in companies
    if str(company.get("name", "")).strip()
)
active_saved = st.session_state.get("active_saved_research")
active_saved_at = ""
if isinstance(active_saved, dict):
    active_saved_at = str(active_saved.get("saved_at", "")).replace(
        "T", " "
    ).replace("+00:00", " UTC")
st.info(
    f"Aktif benchmark: **{data.get('target_company', 'Bilinmiyor')}**"
    + (f" · Karşılaştırma: {active_names}" if active_names else "")
    + (f" · Kayıt zamanı: {active_saved_at}" if active_saved_at else "")
    + f" · Mod: {st.session_state.research_mode}"
)

if st.session_state.rag_stats.get("status") == "indexing":
    render_running_rag_status()
else:
    render_rag_status_panel()

st.divider()
for column, (label, value, delta) in zip(st.columns(5), [
    ("Karşılaştırılan ürün", len(companies), None), ("Hedef puanı", f"{target.get('overall_score',0):.2f}/10", None),
    ("Kategori lideri", leader.get("name","Bilinmiyor"), f"{leader.get('overall_score',0):.2f}/10"),
    ("Doğrulanmış bulgu", verified, f"{len(data.get('sources',[]))} kaynak"),
    ("Kanıt güveni", f"{confidence.get('score', 0)}/100", confidence.get("level", "Düşük")),
], strict=False):
    column.metric(label, value, delta)
st.caption(
    "Kanıt güveni model tahmini değildir. "
    + confidence.get("explanation", "Mevcut kanıt paketinden hesaplanır.")
)

usage = data.get("_usage", {})
if usage.get("provider") == "Gemini" and usage.get("token_cost_usd") is not None:
    free_total = float(usage["cost_with_free_search_usd"])
    paid_search_total = float(usage["cost_after_free_search_quota_usd"])
    search_queries = int(usage.get("search_queries", 0))
    st.success(
        f"Gerçek maliyet: **${free_total:.4f}** · {usage.get('total_tokens', 0):,} token · "
        f"{search_queries} arama"
    )
    if paid_search_total > free_total:
        st.caption(f"Search ücretsiz kotası aşılmışsa: ${paid_search_total:.4f}")

overview, evidence, chat = st.tabs(["Genel bakış", "Kanıtlar ve kaynaklar", "BenchMind'a Sor"])
with overview:
    st.markdown('<div class="section-kicker">YÖNETİCİ ÖZETİ</div>', unsafe_allow_html=True)
    st.subheader(f"Pazar {data.get('target_company','hedef şirket')} hakkında ne söylüyor?")
    st.write(data.get("executive_summary", "Yönetici özeti bulunmuyor."))
    render_conflicts(data)
    st.markdown("### Benchmark tablosu")
    render_benchmark_table(data)
    for company_index, company in enumerate(companies):
        color_index = company_index % 4
        with st.container(key=f"profile_company_{color_index}"):
            with st.expander(f"{company.get('name')} — ürün profili"):
                col1, col2 = st.columns([1.35,1])
                pricing = html.escape(str(company.get("pricing", "Bulunamadı")))
                free_plan = html.escape(str(company.get("free_plan", "Bulunamadı")))
                target_customer = html.escape(str(company.get("target_customer", "Bulunamadı")))
                col1.markdown(
                    f"""<div class="profile-details">
                    <div class="profile-detail"><span class="profile-label profile-pricing">Fiyatlandırma</span><p>{pricing}</p></div>
                    <div class="profile-detail"><span class="profile-label profile-free">Ücretsiz plan</span><p>{free_plan}</p></div>
                    <div class="profile-detail"><span class="profile-label profile-customer">Hedef müşteri</span><p>{target_customer}</p></div>
                    </div>""",
                    unsafe_allow_html=True,
                )
                col2.success(
                    f"**Güçlü yön**\n\n{company.get('key_strength', 'Güçlü yön kaydedilmemiş.')}"
                )
                col2.warning(
                    f"**Sınırlamalar**\n\n{company.get('key_limitation', 'Sınırlama kaydedilmemiş.')}"
                )
    st.markdown("### Competitive scorecard")
    chart1, chart2 = st.columns(2)
    with chart1:
        st.caption("Kategori profili")
        render_radar(data)
    with chart2:
        st.caption("Ağırlıklı genel puan")
        render_bar(data)
    st.caption("Puanlar kanıtlardan türetilen analitik değerlendirmelerdir; doğrulanmış ürün bilgileri değildir.")
    st.markdown("### SWOT analizi")
    render_swot(data)
    st.markdown("### Stratejik öneriler")
    render_recommendations(data)
    st.download_button("Benchmark CSV'yi İndir", export_csv(data),
                       f"benchmind_{data.get('target_company','benchmark').lower().replace(' ','_')}.csv", "text/csv")

with evidence:
    st.markdown('<div class="section-kicker">AUDIT TRAIL</div>', unsafe_allow_html=True)
    st.subheader("Benchmark'ın arkasındaki kanıtları inceleyin")
    finding_frame = findings_dataframe(data)
    filter1, filter2 = st.columns(2)
    company_values = ["Tümü"] + sorted(finding_frame["Şirket"].unique()) if not finding_frame.empty else ["Tümü"]
    type_values = ["Tümü"] + sorted(finding_frame["Kanıt türü"].unique()) if not finding_frame.empty else ["Tümü"]
    company_filter = filter1.selectbox("Şirket", company_values)
    type_filter = filter2.selectbox("Kanıt türü", type_values)
    filtered = finding_frame
    if company_filter != "Tümü": filtered = filtered[filtered["Şirket"] == company_filter]
    if type_filter != "Tümü": filtered = filtered[filtered["Kanıt türü"] == type_filter]
    render_evidence_table(filtered, data)
    st.markdown("### Kaynak listesi")
    sources = pd.DataFrame(data.get("sources", [])).rename(columns={"id":"ID","title":"Başlık","publisher":"Yayıncı","source_type":"Tür","url":"URL"})
    if sources.empty:
        st.info("Bu araştırma için kayıtlı kaynak bulunmuyor.")
    else:
        st.dataframe(sources[["ID","Başlık","Yayıncı","Tür","URL"]], width="stretch", hide_index=True,
                     column_config={"URL":st.column_config.LinkColumn("Kaynağı aç",display_text="Aç")})
    if data.get("conflicts"):
        st.markdown("### Çelişkili kanıtlar")
        render_conflicts(data)

with chat:
    st.markdown('<div class="section-kicker">KAYNAĞA DAYALI Q&A</div>', unsafe_allow_html=True)
    st.subheader("Mevcut kanıt paketine sorun")
    st.caption("BenchMind chat sırasında web'de arama yapmaz. Yüklenen kanıtlardan yanıt verir veya yeterli kanıt olmadığını bildirir.")
    target_name = data.get("target_company", "hedef şirket")
    suggestions = [f"{target_name} en belirgin competitive advantage'a hangi alanda sahip?",
                   "Hangi rakibin multilingual yaklaşımı en güçlü?",
                   "Hedef ürün ekibi bir sonraki adımda neye öncelik vermeli?"]
    for column, question in zip(st.columns(3), suggestions, strict=False):
        with column:
            if st.button(question, key=f"suggest_{question}", width="stretch"):
                st.session_state.pending_question = question
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("retrieved_sources"):
                with st.expander("Bu yanıt için retrieve edilen kaynaklar"):
                    for source in message["retrieved_sources"]:
                        title = source.get("title") or source.get("url") or "Kaynak"
                        st.markdown(
                            f"- [{source.get('source_id', 'Kaynak')}] "
                            f"[{title}]({source.get('url', '')}) · similarity: {source.get('similarity', '—')}"
                        )
    typed = st.chat_input("Mevcut benchmark hakkında bir soru sorun")
    active_question = typed or st.session_state.pop("pending_question", None)
    if active_question:
        ask(active_question, data, api_key, model_name, provider)
        st.rerun()
