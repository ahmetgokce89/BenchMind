from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from datetime import date
from collections.abc import Iterator
from typing import Any
from urllib.parse import urlparse

from anthropic import Anthropic
from dotenv import load_dotenv
from google import genai
from google.genai import types as genai_types
from openai import OpenAI

from benchmark_data import CRITERIA, compact_context, normalize_research
from prompts import (
    GEMINI_RESEARCH_INSTRUCTIONS,
    GROUNDED_CHAT_INSTRUCTIONS,
    build_gemini_search_prompt,
    build_gemini_structuring_prompt,
    build_grounded_chat_prompt,
    build_research_instructions,
    build_research_json_contract,
    build_research_prompt,
)


load_dotenv()

OPENAI_DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-5.6-sol")
ANTHROPIC_DEFAULT_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-5")
GEMINI_DEFAULT_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
CLAUDE_CODE_DEFAULT_MODEL = os.getenv("CLAUDE_CODE_MODEL", ANTHROPIC_DEFAULT_MODEL)
DEFAULT_MODEL = OPENAI_DEFAULT_MODEL
try:
    LIVE_RESEARCH_MAX_SEARCHES = max(1, int(os.getenv("LIVE_RESEARCH_MAX_SEARCHES", "4")))
except ValueError:
    LIVE_RESEARCH_MAX_SEARCHES = 4
try:
    LIVE_RESEARCH_MAX_OUTPUT_TOKENS = max(
        2000, int(os.getenv("LIVE_RESEARCH_MAX_OUTPUT_TOKENS", "10000"))
    )
except ValueError:
    LIVE_RESEARCH_MAX_OUTPUT_TOKENS = 10000

CHAT_MAX_OUTPUT_TOKENS = 500
LIVE_RESEARCH_PRICING_AS_OF = "2026-09-15"


def estimate_live_research_cost(
    provider: str | None,
    model: str,
    max_searches: int = LIVE_RESEARCH_MAX_SEARCHES,
) -> dict[str, Any]:
    """Estimate one research run; the API response remains the billing authority."""
    normalized_model = model.strip().casefold()
    provider_key = (provider or "OpenAI").casefold()
    gemini_free_search_note = (
        "Aylık ortak 5.000 Gemini 3.x search kotası aşılırsa sorgu başına $0.014 eklenir."
        if provider_key == "gemini"
        else None
    )
    search_cost = 0.0 if provider_key == "gemini" else max_searches * 0.01
    # A typical BenchMind run uses roughly 10K-30K input and 4K-10K output
    # tokens. Prices are USD per million tokens.
    token_prices: dict[tuple[str, str], tuple[float, float]] = {
        ("anthropic", "claude-sonnet-5"): (2.0, 10.0),
        ("anthropic", "claude-sonnet-4-6"): (3.0, 15.0),
        ("anthropic", "claude-sonnet-4-5"): (3.0, 15.0),
        ("anthropic", "claude-haiku-4-5-20251001"): (1.0, 5.0),
        ("openai", "gpt-5.6-sol"): (4.0, 20.0),
        ("openai", "gpt-5.6-terra"): (2.0, 12.0),
        ("openai", "gpt-5.6-luna"): (0.2, 1.2),
        ("gemini", "gemini-3.8-flash"): (0.75, 3.75),
        ("gemini", "gemini-3.5-flash-lite"): (0.3, 2.5),
        ("gemini", "gemini-3.1-flash-lite"): (0.25, 1.5),
    }
    prices = token_prices.get((provider_key, normalized_model))
    if provider_key == "claude code":
        budget = float(os.getenv("CLAUDE_CODE_RESEARCH_BUDGET_USD", "1.00"))
        return {
            "known": True,
            "low_usd": 0.0,
            "high_usd": budget,
            "search_cost_usd": None,
            "max_searches": max_searches,
            "pricing_as_of": LIVE_RESEARCH_PRICING_AS_OF,
            "is_budget_cap": True,
            "search_note": None,
        }
    if prices is None:
        return {
            "known": False,
            "search_cost_usd": search_cost,
            "max_searches": max_searches,
            "pricing_as_of": LIVE_RESEARCH_PRICING_AS_OF,
            "is_budget_cap": False,
            "search_note": gemini_free_search_note,
        }

    input_price, output_price = prices
    output_low, output_high = (6_000, 16_000) if provider_key == "gemini" else (4_000, 10_000)
    low = search_cost + (10_000 * input_price + output_low * output_price) / 1_000_000
    high = search_cost + (30_000 * input_price + output_high * output_price) / 1_000_000
    return {
        "known": True,
        "low_usd": low,
        "high_usd": high,
        "search_cost_usd": search_cost,
        "max_searches": max_searches,
        "pricing_as_of": LIVE_RESEARCH_PRICING_AS_OF,
        "is_budget_cap": False,
        "search_note": gemini_free_search_note,
    }


def company_name_from_url(website: str) -> str:
    value = website.strip()
    if not value:
        return ""
    parsed = urlparse(value if "://" in value else f"https://{value}")
    hostname = (parsed.hostname or "").lower().strip(".")
    parts = [part for part in hostname.split(".") if part]
    while parts and parts[0] in {"www", "app", "web", "go"}:
        parts.pop(0)
    if not parts:
        return ""
    brand = parts[0].replace("-", " ").replace("_", " ").strip()
    return brand.title()


def company_url_from_name(value: str) -> str:
    """Interpret a domain entered in the company-name field as its website."""
    raw = value.strip()
    if not raw or any(character.isspace() for character in raw):
        return ""
    parsed = urlparse(raw if "://" in raw else f"https://{raw}")
    hostname = (parsed.hostname or "").lower().strip(".")
    if parsed.scheme not in {"http", "https"} or "." not in hostname:
        return ""
    port = f":{parsed.port}" if parsed.port else ""
    path = parsed.path.rstrip("/")
    return f"https://{hostname}{port}{path}"


def _get_secret(streamlit_secrets: Any | None, name: str) -> str | None:
    value = os.getenv(name)
    if value:
        return value
    if streamlit_secrets is not None:
        try:
            return streamlit_secrets.get(name)
        except Exception:
            pass
    return None


def get_api_config(streamlit_secrets: Any | None = None) -> tuple[str | None, str | None]:
    """Return the selected provider and key without exposing the credential."""
    keys = {
        "Claude Code": _get_secret(streamlit_secrets, "ANTHROPIC_API_KEY"),
        "Anthropic": _get_secret(streamlit_secrets, "ANTHROPIC_API_KEY"),
        "OpenAI": _get_secret(streamlit_secrets, "OPENAI_API_KEY"),
        "Gemini": _get_secret(streamlit_secrets, "GEMINI_API_KEY"),
    }
    preferred = os.getenv("LLM_PROVIDER", "").strip().casefold()
    if streamlit_secrets is not None and not preferred:
        try:
            preferred = str(streamlit_secrets.get("LLM_PROVIDER", "")).strip().casefold()
        except Exception:
            pass

    provider_names = {
        "claude code": "Claude Code",
        "claude-code": "Claude Code",
        "claude_code": "Claude Code",
        "claude": "Claude Code",
        "anthropic": "Anthropic",
        "openai": "OpenAI",
        "gemini": "Gemini",
        "google": "Gemini",
    }
    selected = provider_names.get(preferred)
    if selected == "Claude Code" and keys[selected] and shutil.which("claude"):
        return selected, keys[selected]
    if selected and selected != "Claude Code" and keys[selected]:
        return selected, keys[selected]
    for provider in ("Gemini", "Anthropic", "OpenAI", "Claude Code"):
        if provider == "Claude Code" and not shutil.which("claude"):
            continue
        if keys[provider]:
            return provider, keys[provider]
    return None, None


def get_api_key(streamlit_secrets: Any | None = None) -> str | None:
    """Backward-compatible key helper."""
    return get_api_config(streamlit_secrets)[1]


def default_model_for_provider(provider: str | None) -> str:
    if provider == "Claude Code":
        return CLAUDE_CODE_DEFAULT_MODEL
    if provider == "Gemini":
        return GEMINI_DEFAULT_MODEL
    return ANTHROPIC_DEFAULT_MODEL if provider == "Anthropic" else OPENAI_DEFAULT_MODEL


def _anthropic_text(response: Any) -> str:
    return "".join(
        block.text for block in response.content if getattr(block, "type", None) == "text"
    ).strip()


def _parse_json_text(value: str) -> dict[str, Any]:
    cleaned = value.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, count=1, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned, count=1)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start >= 0 and end > start:
            return json.loads(cleaned[start:end + 1])
        raise


def _run_claude_code(
    instructions: str,
    prompt: str,
    model: str,
    *,
    schema: dict[str, Any] | None = None,
    web_search: bool = False,
) -> str:
    executable = shutil.which("claude")
    if not executable:
        raise RuntimeError("Claude Code CLI bulunamadı. VS Code'da Claude Code kurulumunu kontrol edin.")

    budget_name = "CLAUDE_CODE_RESEARCH_BUDGET_USD" if web_search else "CLAUDE_CODE_CHAT_BUDGET_USD"
    budget = os.getenv(budget_name, "1.00" if web_search else "0.10")
    command = [
        executable,
        "--print",
        "--bare",
        "--output-format",
        "json",
        "--model",
        model,
        "--effort",
        "low",
        "--max-budget-usd",
        budget,
        "--no-session-persistence",
        "--permission-mode",
        "dontAsk",
        "--system-prompt",
        instructions,
    ]
    if schema is not None:
        command.extend(["--json-schema", json.dumps(schema, ensure_ascii=False)])
    if web_search:
        command.extend(["--tools", "WebSearch", "--allowed-tools", "WebSearch"])
    else:
        command.extend(["--tools", ""])

    try:
        completed = subprocess.run(
            command,
            input=prompt,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=600 if web_search else 180,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError("Claude Code zaman aşımına uğradı; ağ bağlantısını kontrol edin.") from exc
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout or "Bilinmeyen Claude Code hatası").strip()
        raise RuntimeError(f"Claude Code çağrısı başarısız: {detail[:800]}")

    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError("Claude Code geçerli JSON yanıtı döndürmedi.") from exc
    if payload.get("is_error"):
        raise RuntimeError(str(payload.get("result") or "Claude Code isteği başarısız oldu."))
    structured = payload.get("structured_output")
    if schema is not None and structured is not None:
        return json.dumps(structured, ensure_ascii=False)
    return str(payload.get("result") or "").strip()


def _evidence_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "statement": {"type": "string"},
            "evidence_type": {
                "type": "string",
                "enum": ["Verified fact", "Customer opinion", "AI inference"],
            },
            "source_ids": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["statement", "evidence_type", "source_ids"],
    }


def _research_schema() -> dict[str, Any]:
    score_properties = {key: {"type": "number", "minimum": 0, "maximum": 10} for key in CRITERIA}
    company_schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "name": {"type": "string"},
            "website": {"type": "string"},
            "pricing": {"type": "string"},
            "free_plan": {"type": "string"},
            "target_customer": {"type": "string"},
            "key_strength": {"type": "string"},
            "key_limitation": {"type": "string"},
            "scores": {
                "type": "object",
                "additionalProperties": False,
                "properties": score_properties,
                "required": list(CRITERIA),
            },
        },
        "required": [
            "name",
            "website",
            "pricing",
            "free_plan",
            "target_customer",
            "key_strength",
            "key_limitation",
            "scores",
        ],
    }
    finding_schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "id": {"type": "string"},
            "company": {"type": "string"},
            "category": {"type": "string"},
            "statement": {"type": "string"},
            "evidence_type": {
                "type": "string",
                "enum": ["Verified fact", "Customer opinion", "AI inference"],
            },
            "source_ids": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["id", "company", "category", "statement", "evidence_type", "source_ids"],
    }
    source_schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "id": {"type": "string"},
            "title": {"type": "string"},
            "publisher": {"type": "string"},
            "source_type": {"type": "string"},
            "url": {"type": "string"},
        },
        "required": ["id", "title", "publisher", "source_type", "url"],
    }
    conflict_schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "topic": {"type": "string"},
            "description": {"type": "string"},
            "source_ids": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["topic", "description", "source_ids"],
    }
    recommendation_schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "title": {"type": "string"},
            "priority": {"type": "string", "enum": ["High", "Medium", "Low"]},
            "statement": {"type": "string"},
            "evidence_type": {"type": "string", "enum": ["AI inference"]},
            "source_ids": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["title", "priority", "statement", "evidence_type", "source_ids"],
    }
    swot_properties = {
        key: {"type": "array", "items": _evidence_schema(), "minItems": 1, "maxItems": 4}
        for key in ["strengths", "weaknesses", "opportunities", "threats"]
    }
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "research_date": {"type": "string"},
            "industry": {"type": "string"},
            "target_company": {"type": "string"},
            "executive_summary": {"type": "string"},
            "companies": {"type": "array", "items": company_schema, "minItems": 2, "maxItems": 4},
            "findings": {"type": "array", "items": finding_schema, "minItems": 6, "maxItems": 24},
            "conflicts": {"type": "array", "items": conflict_schema, "maxItems": 6},
            "swot": {
                "type": "object",
                "additionalProperties": False,
                "properties": swot_properties,
                "required": list(swot_properties),
            },
            "recommendations": {
                "type": "array",
                "items": recommendation_schema,
                "minItems": 3,
                "maxItems": 6,
            },
            "sources": {"type": "array", "items": source_schema, "minItems": 4, "maxItems": 30},
        },
        "required": [
            "research_date",
            "industry",
            "target_company",
            "executive_summary",
            "companies",
            "findings",
            "conflicts",
            "swot",
            "recommendations",
            "sources",
        ],
    }


def _run_anthropic_research(
    client: Any,
    *,
    model: str,
    instructions: str,
    prompt: str,
) -> str:
    tools = [
        {
            "type": "web_search_20260318",
            "name": "web_search",
            "max_uses": LIVE_RESEARCH_MAX_SEARCHES,
            "response_inclusion": "excluded",
        }
    ]
    messages: list[dict[str, Any]] = [
        {"role": "user", "content": f"{prompt}\n\n{build_research_json_contract(CRITERIA)}"}
    ]
    for _ in range(3):
        response = client.messages.create(
            model=model,
            max_tokens=LIVE_RESEARCH_MAX_OUTPUT_TOKENS,
            system=instructions,
            messages=messages,
            tools=tools,
            output_config={"effort": "low"},
        )
        output_text = _anthropic_text(response)
        if output_text:
            return output_text
        stop_reason = getattr(response, "stop_reason", None)
        if stop_reason == "pause_turn":
            messages.append({"role": "assistant", "content": response.content})
            continue
        block_types = [str(getattr(block, "type", "unknown")) for block in response.content]
        if stop_reason == "max_tokens":
            raise RuntimeError(
                f"Claude {LIVE_RESEARCH_MAX_OUTPUT_TOKENS} token sınırına ulaştı ve JSON yanıtını tamamlayamadı."
            )
        raise RuntimeError(
            f"Claude JSON text döndürmedi (stop_reason={stop_reason or 'bilinmiyor'}, "
            f"blocks={','.join(block_types) or 'yok'})."
        )
    raise RuntimeError("Claude web research üç kez pause_turn döndürdü; lütfen yeniden deneyin.")


def _gemini_grounding_context(response: Any) -> str:
    """Build a compact evidence package from Gemini text and grounding metadata."""
    text = (response.text or "").strip()
    candidate = response.candidates[0] if getattr(response, "candidates", None) else None
    metadata = getattr(candidate, "grounding_metadata", None)
    chunks = getattr(metadata, "grounding_chunks", None) or []
    sources: list[str] = []
    for index, chunk in enumerate(chunks, start=1):
        web = getattr(chunk, "web", None)
        uri = str(getattr(web, "uri", "") or "").strip()
        title = str(getattr(web, "title", "") or "").strip()
        if uri:
            sources.append(f"G{index:02d} | {title or 'Kaynak'} | {uri}")
    if not text or not sources:
        raise RuntimeError(
            "Gemini Google Search kaynaklı kanıt döndürmedi; ücretli Search kotasını ve model erişimini kontrol edin."
        )
    return f"ARAŞTIRMA NOTLARI:\n{text}\n\nGOOGLE SEARCH KAYNAKLARI:\n" + "\n".join(sources)


def _gemini_usage_summary(responses: list[Any], model: str) -> dict[str, Any]:
    prices = {
        "gemini-3.8-flash": (0.75, 3.75),
        "gemini-3.5-flash-lite": (0.3, 2.5),
        "gemini-3.1-flash-lite": (0.25, 1.5),
    }
    input_tokens = output_tokens = thinking_tokens = total_tokens = 0
    search_queries = 0
    for response in responses:
        usage = getattr(response, "usage_metadata", None)
        input_tokens += int(getattr(usage, "prompt_token_count", 0) or 0)
        output_tokens += int(getattr(usage, "candidates_token_count", 0) or 0)
        thinking_tokens += int(getattr(usage, "thoughts_token_count", 0) or 0)
        total_tokens += int(getattr(usage, "total_token_count", 0) or 0)
        for candidate in getattr(response, "candidates", None) or []:
            metadata = getattr(candidate, "grounding_metadata", None)
            search_queries += len(getattr(metadata, "web_search_queries", None) or [])

    rates = prices.get(model.strip().casefold())
    token_cost = None
    if rates:
        input_price, output_price = rates
        token_cost = (
            input_tokens * input_price
            + (output_tokens + thinking_tokens) * output_price
        ) / 1_000_000
    search_cost_after_free_quota = search_queries * 0.014
    return {
        "provider": "Gemini",
        "model": model,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "thinking_tokens": thinking_tokens,
        "total_tokens": total_tokens,
        "search_queries": search_queries,
        "token_cost_usd": token_cost,
        "cost_with_free_search_usd": token_cost,
        "cost_after_free_search_quota_usd": (
            token_cost + search_cost_after_free_quota if token_cost is not None else None
        ),
        "pricing_as_of": LIVE_RESEARCH_PRICING_AS_OF,
    }


def clean_company_inputs(companies: list[dict[str, str]]) -> list[dict[str, str]]:
    cleaned = []
    seen = set()
    for company in companies:
        name = re.sub(r"\s+", " ", company.get("name", "")).strip()
        website = company.get("website", "").strip()
        # Users commonly paste a domain into the field labelled "Hedef Şirket".
        # If a stale default URL belongs to another brand, the domain in the
        # edited name field is the fresher signal and must not silently collapse
        # back to the demo company.
        name_as_website = company_url_from_name(name)
        if name_as_website:
            name_brand = company_name_from_url(name_as_website)
            website_brand = company_name_from_url(website) if website else ""
            if not website or _company_identity_key(name_brand) != _company_identity_key(
                website_brand
            ):
                name = name_brand
                website = name_as_website
        inferred_name = company_name_from_url(website) if website else ""
        if inferred_name:
            entered_key = re.sub(r"[^a-z0-9]", "", name.casefold())
            inferred_key = re.sub(r"[^a-z0-9]", "", inferred_name.casefold())
            names_agree = bool(
                entered_key
                and inferred_key
                and (
                    entered_key.startswith(inferred_key)
                    or inferred_key.startswith(entered_key)
                )
            )
            if not names_agree:
                name = inferred_name
        if not name or name.casefold() in seen:
            continue
        seen.add(name.casefold())
        cleaned.append({"name": name[:100], "website": website[:300]})
    if not 2 <= len(cleaned) <= 4:
        raise ValueError("Bir hedef şirket ile birbirinden farklı bir ila üç rakip girin.")
    return cleaned


def _company_identity_key(value: Any) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value or "").casefold())


def _company_result_matches(expected: dict[str, str], actual: dict[str, Any]) -> bool:
    expected_name = _company_identity_key(expected.get("name"))
    actual_name = _company_identity_key(actual.get("name"))
    names_match = bool(
        expected_name
        and actual_name
        and (
            expected_name.startswith(actual_name)
            or actual_name.startswith(expected_name)
        )
    )
    expected_host = urlparse(
        expected.get("website", "")
        if "://" in expected.get("website", "")
        else f"https://{expected.get('website', '')}"
    ).hostname
    actual_website = str(actual.get("website", ""))
    actual_host = urlparse(
        actual_website if "://" in actual_website else f"https://{actual_website}"
    ).hostname
    hosts_match = bool(
        expected_host
        and actual_host
        and (
            expected_host.removeprefix("www.") == actual_host.removeprefix("www.")
            or expected_host.removeprefix("www.").endswith(
                f".{actual_host.removeprefix('www.')}"
            )
            or actual_host.removeprefix("www.").endswith(
                f".{expected_host.removeprefix('www.')}"
            )
        )
    )
    return names_match or hosts_match


def _align_research_companies(
    data: dict[str, Any],
    expected_companies: list[dict[str, str]],
) -> dict[str, Any]:
    actual_companies = data.get("companies", [])
    if not isinstance(actual_companies, list):
        raise RuntimeError("Araştırma yanıtındaki şirket listesi geçersiz.")

    remaining = [company for company in actual_companies if isinstance(company, dict)]
    aligned = []
    company_name_map: dict[str, str] = {}
    for expected in expected_companies:
        match_index = next(
            (
                index
                for index, actual in enumerate(remaining)
                if _company_result_matches(expected, actual)
            ),
            None,
        )
        if match_index is None:
            raise RuntimeError(
                "Araştırma yanıtı istenen şirketlerle eşleşmedi. "
                f"'{expected['name']}' sonucu bulunamadı; yanlış benchmark gösterilmedi."
            )
        company = dict(remaining.pop(match_index))
        original_name = str(company.get("name", ""))
        company["name"] = expected["name"]
        if expected.get("website"):
            company["website"] = expected["website"]
        company_name_map[original_name.casefold()] = expected["name"]
        aligned.append(company)

    normalized = dict(data)
    normalized["companies"] = aligned
    normalized["target_company"] = expected_companies[0]["name"]
    for finding in normalized.get("findings", []):
        if isinstance(finding, dict):
            original = str(finding.get("company", ""))
            finding["company"] = company_name_map.get(original.casefold(), original)
    return normalized


def research_companies(
    companies: list[dict[str, str]],
    api_key: str,
    model: str = DEFAULT_MODEL,
    provider: str = "OpenAI",
) -> dict[str, Any]:
    cleaned = clean_company_inputs(companies)
    usage_summary: dict[str, Any] | None = None
    instructions = build_research_instructions(LIVE_RESEARCH_MAX_SEARCHES)
    prompt = build_research_prompt(cleaned, CRITERIA, date.today().isoformat())

    if provider == "Claude Code":
        output_text = _run_claude_code(
            instructions,
            prompt,
            model,
            schema=_research_schema(),
            web_search=True,
        )
    elif provider == "Anthropic":
        client = Anthropic(api_key=api_key)
        output_text = _run_anthropic_research(
            client,
            model=model,
            instructions=instructions,
            prompt=prompt,
        )
    elif provider == "Gemini":
        client = genai.Client(api_key=api_key)
        evidence_response = client.models.generate_content(
            model=model,
            contents=build_gemini_search_prompt(prompt),
            config=genai_types.GenerateContentConfig(
                system_instruction=instructions,
                max_output_tokens=6000,
                tools=[genai_types.Tool(google_search=genai_types.GoogleSearch())],
                thinking_config=genai_types.ThinkingConfig(thinking_level="LOW"),
            ),
        )
        evidence = _gemini_grounding_context(evidence_response)
        response = client.models.generate_content(
            model=model,
            contents=build_gemini_structuring_prompt(
                evidence, build_research_json_contract(CRITERIA)
            ),
            config=genai_types.GenerateContentConfig(
                system_instruction=GEMINI_RESEARCH_INSTRUCTIONS,
                max_output_tokens=LIVE_RESEARCH_MAX_OUTPUT_TOKENS,
                response_mime_type="application/json",
                thinking_config=genai_types.ThinkingConfig(thinking_level="LOW"),
            ),
        )
        output_text = (response.text or "").strip()
        usage_summary = _gemini_usage_summary([evidence_response, response], model)
    else:
        client = OpenAI(api_key=api_key)
        response = client.responses.create(
            model=model,
            reasoning={"effort": "none"},
            store=False,
            instructions=instructions,
            input=prompt,
            tools=[{"type": "web_search"}],
            tool_choice="auto",
            include=["web_search_call.action.sources"],
            max_tool_calls=LIVE_RESEARCH_MAX_SEARCHES,
            max_output_tokens=LIVE_RESEARCH_MAX_OUTPUT_TOKENS,
            text={
                "format": {
                    "type": "json_schema",
                    "name": "competitive_benchmark",
                    "strict": True,
                    "schema": _research_schema(),
                }
            },
        )
        output_text = response.output_text or ""
    if not output_text:
        raise RuntimeError("Araştırma yanıtında structured output bulunamadı.")
    data = _parse_json_text(output_text)
    normalized = normalize_research(_align_research_companies(data, cleaned))
    if usage_summary:
        normalized["_usage"] = usage_summary
    return normalized


def answer_from_research(
    question: str,
    research: dict[str, Any],
    api_key: str,
    model: str = DEFAULT_MODEL,
    provider: str = "OpenAI",
    retrieved_chunks: list[dict[str, Any]] | None = None,
) -> str:
    cleaned_question = re.sub(r"\s+", " ", question).strip()[:1200]
    if not cleaned_question:
        raise ValueError("Mevcut araştırma hakkında bir soru girin.")

    instructions = GROUNDED_CHAT_INSTRUCTIONS
    prompt = build_grounded_chat_prompt(
        cleaned_question,
        compact_context(research),
        retrieved_chunks,
    )

    if provider == "Claude Code":
        answer = _run_claude_code(instructions, prompt, model)
    elif provider == "Anthropic":
        client = Anthropic(api_key=api_key)
        response = client.messages.create(
            model=model,
            max_tokens=CHAT_MAX_OUTPUT_TOKENS,
            system=instructions,
            messages=[{"role": "user", "content": prompt}],
            output_config={"effort": "low"},
        )
        answer = _anthropic_text(response)
    elif provider == "Gemini":
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model=model,
            contents=prompt,
            config=genai_types.GenerateContentConfig(
                system_instruction=instructions,
                max_output_tokens=CHAT_MAX_OUTPUT_TOKENS,
                thinking_config=genai_types.ThinkingConfig(thinking_level="LOW"),
            ),
        )
        answer = (response.text or "").strip()
    else:
        client = OpenAI(api_key=api_key)
        response = client.responses.create(
            model=model,
            reasoning={"effort": "none"},
            store=False,
            instructions=instructions,
            input=prompt,
            max_output_tokens=CHAT_MAX_OUTPUT_TOKENS,
        )
        answer = (response.output_text or "").strip()
    if not answer:
        return "Mevcut araştırma dataset'inde yeterli kanıt yok."
    return answer


def stream_answer_from_research(
    question: str,
    research: dict[str, Any],
    api_key: str,
    model: str = DEFAULT_MODEL,
    provider: str = "OpenAI",
    retrieved_chunks: list[dict[str, Any]] | None = None,
) -> Iterator[str]:
    """Stream a grounded chat answer while preserving the non-streaming API."""
    cleaned_question = re.sub(r"\s+", " ", question).strip()[:1200]
    if not cleaned_question:
        raise ValueError("Mevcut araştırma hakkında bir soru girin.")

    instructions = GROUNDED_CHAT_INSTRUCTIONS
    prompt = build_grounded_chat_prompt(
        cleaned_question,
        compact_context(research),
        retrieved_chunks,
    )

    emitted = False
    if provider == "Claude Code":
        text = _run_claude_code(instructions, prompt, model)
        if text:
            emitted = True
            yield text
    elif provider == "Anthropic":
        client = Anthropic(api_key=api_key)
        with client.messages.stream(
            model=model,
            max_tokens=CHAT_MAX_OUTPUT_TOKENS,
            system=instructions,
            messages=[{"role": "user", "content": prompt}],
            output_config={"effort": "low"},
        ) as stream:
            for text in stream.text_stream:
                if text:
                    emitted = True
                    yield text
    elif provider == "Gemini":
        client = genai.Client(api_key=api_key)
        stream = client.models.generate_content_stream(
            model=model,
            contents=prompt,
            config=genai_types.GenerateContentConfig(
                system_instruction=instructions,
                max_output_tokens=CHAT_MAX_OUTPUT_TOKENS,
                thinking_config=genai_types.ThinkingConfig(thinking_level="LOW"),
            ),
        )
        for chunk in stream:
            text = chunk.text or ""
            if text:
                emitted = True
                yield text
    else:
        client = OpenAI(api_key=api_key)
        with client.responses.stream(
            model=model,
            reasoning={"effort": "none"},
            store=False,
            instructions=instructions,
            input=prompt,
            max_output_tokens=CHAT_MAX_OUTPUT_TOKENS,
        ) as stream:
            for event in stream:
                if getattr(event, "type", "") == "response.output_text.delta":
                    delta = getattr(event, "delta", "")
                    if delta:
                        emitted = True
                        yield delta

    if not emitted:
        yield "Mevcut araştırma dataset'inde yeterli kanıt yok."
