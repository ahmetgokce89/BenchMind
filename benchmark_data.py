from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
DEMO_DATA_PATH = BASE_DIR / "data" / "demo_benchmark.json"

CRITERIA = {
    "pricing_value": ("Fiyatlandırma ve değer", 0.15),
    "platform_support": ("Kayıt ve platform desteği", 0.15),
    "transcription_languages": ("Transcription ve dil desteği", 0.15),
    "ai_summaries": ("AI özetleri ve action item'lar", 0.20),
    "integrations": ("Integration ve automation", 0.15),
    "team_security": ("Ekip, admin ve güvenlik", 0.10),
    "customer_sentiment": ("Müşteri görüşleri", 0.10),
}

EVIDENCE_LABELS = {
    "Verified fact": "Doğrulanmış bilgi (Verified fact)",
    "Customer opinion": "Müşteri görüşü (Customer opinion)",
    "AI inference": "AI çıkarımı (AI inference)",
}


def load_demo_data() -> dict[str, Any]:
    with DEMO_DATA_PATH.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    return normalize_research(data)


def normalize_research(data: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(data)
    normalized.setdefault("research_date", "Bilinmiyor")
    normalized.setdefault("industry", "AI toplantı asistanları")
    normalized.setdefault("executive_summary", "")
    normalized.setdefault("companies", [])
    normalized.setdefault("findings", [])
    normalized.setdefault("sources", [])
    normalized.setdefault("conflicts", [])
    normalized.setdefault("recommendations", [])
    normalized.setdefault(
        "swot",
        {"strengths": [], "weaknesses": [], "opportunities": [], "threats": []},
    )

    source_ids = {source.get("id") for source in normalized["sources"]}
    for finding in normalized["findings"]:
        finding.setdefault("company", "Pazar")
        finding.setdefault("category", "Genel")
        finding.setdefault("evidence_type", "AI inference")
        finding["source_ids"] = [
            source_id for source_id in finding.get("source_ids", []) if source_id in source_ids
        ]

    for company in normalized["companies"]:
        company.setdefault("website", "")
        company.setdefault("pricing", "Bulunamadı")
        company.setdefault("free_plan", "Bulunamadı")
        company.setdefault("target_customer", "Bulunamadı")
        company.setdefault("key_strength", "")
        company.setdefault("key_limitation", "")
        company.setdefault("scores", {})
        for criterion in CRITERIA:
            company["scores"].setdefault(criterion, 0.0)
        company["overall_score"] = calculate_overall_score(company["scores"])

    # Confidence is derived from the evidence package, never guessed by the LLM.
    normalized["confidence"] = calculate_confidence(normalized)

    return normalized


def calculate_overall_score(scores: dict[str, Any]) -> float:
    weighted_score = 0.0
    for key, (_, weight) in CRITERIA.items():
        try:
            value = min(10.0, max(0.0, float(scores.get(key, 0))))
        except (TypeError, ValueError):
            value = 0.0
        weighted_score += value * weight
    return round(weighted_score, 2)


def calculate_confidence(data: dict[str, Any]) -> dict[str, Any]:
    """Calculate an explainable evidence-confidence score from 0 to 100.

    This deliberately excludes source freshness until source publication dates are
    collected. Treating the research run date as a source date would overstate
    confidence.
    """
    findings = data.get("findings", [])
    sources = data.get("sources", [])
    conflicts = data.get("conflicts", [])
    if not findings:
        return {
            "score": 0,
            "level": "Düşük",
            "coverage": 0,
            "evidence_quality": 0,
            "agreement": 0,
            "explanation": "Güven puanı için kanıt bulunmuyor.",
        }

    source_ids = {source.get("id") for source in sources if source.get("id")}
    cited_findings = [
        finding
        for finding in findings
        if any(source_id in source_ids for source_id in finding.get("source_ids", []))
    ]
    citation_coverage = len(cited_findings) / len(findings)

    companies = [
        company.get("name") for company in data.get("companies", []) if company.get("name")
    ]
    company_breadth = []
    for company_name in companies:
        linked_ids = {
            source_id
            for finding in findings
            if finding.get("company") == company_name
            for source_id in finding.get("source_ids", [])
            if source_id in source_ids
        }
        # Two independent sources per company is the initial coverage target.
        company_breadth.append(min(len(linked_ids) / 2, 1.0))
    breadth_score = sum(company_breadth) / len(company_breadth) if company_breadth else 0.0
    coverage = round(100 * ((0.7 * citation_coverage) + (0.3 * breadth_score)))

    evidence_weights = {
        "Verified fact": 1.0,
        "Customer opinion": 0.75,
        "AI inference": 0.5,
    }
    evidence_quality = round(
        100
        * sum(evidence_weights.get(finding.get("evidence_type"), 0.5) for finding in findings)
        / len(findings)
    )

    # Each disclosed conflict lowers agreement, while keeping the penalty bounded.
    agreement = max(40, 100 - (15 * len(conflicts)))
    score = round((0.45 * coverage) + (0.35 * evidence_quality) + (0.20 * agreement))
    level = "Yüksek" if score >= 75 else "Orta" if score >= 50 else "Düşük"
    return {
        "score": score,
        "level": level,
        "coverage": coverage,
        "evidence_quality": evidence_quality,
        "agreement": agreement,
        "explanation": (
            f"Kaynak kapsamı {coverage}/100, kanıt kalitesi {evidence_quality}/100, "
            f"kaynak uyumu {agreement}/100. Kaynak güncelliği henüz puana dahil değil."
        ),
    }


def benchmark_dataframe(data: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for company in data.get("companies", []):
        row = {
            "Şirket": company.get("name", "Bilinmiyor"),
            "Website": company.get("website", ""),
            "Fiyatlandırma": company.get("pricing", "Bulunamadı"),
            "Ücretsiz plan": company.get("free_plan", "Bulunamadı"),
            "Hedef müşteri": company.get("target_customer", "Bulunamadı"),
            "Temel güçlü yön": company.get("key_strength", ""),
            "Temel sınırlama": company.get("key_limitation", ""),
            "Genel puan": company.get("overall_score", 0),
        }
        for key, (label, _) in CRITERIA.items():
            row[label] = company.get("scores", {}).get(key, 0)
        rows.append(row)
    return pd.DataFrame(rows)


def scores_dataframe(data: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for company in data.get("companies", []):
        for key, (label, weight) in CRITERIA.items():
            rows.append(
                {
                    "Şirket": company.get("name", "Bilinmiyor"),
                    "Kriter": label,
                    "Puan": float(company.get("scores", {}).get(key, 0)),
                    "Ağırlık": weight,
                }
            )
    return pd.DataFrame(rows)


def findings_dataframe(data: dict[str, Any]) -> pd.DataFrame:
    source_lookup = {source["id"]: source for source in data.get("sources", [])}
    rows = []
    company_order = {
        company.get("name"): index
        for index, company in enumerate(data.get("companies", []))
    }
    findings = sorted(
        enumerate(data.get("findings", [])),
        key=lambda item: (
            company_order.get(item[1].get("company"), len(company_order)),
            item[0],
        ),
    )
    for _, finding in findings:
        linked_sources = [
            source_lookup[source_id]
            for source_id in finding.get("source_ids", [])
            if source_id in source_lookup
        ]
        rows.append(
            {
                "Şirket": finding.get("company", "Pazar"),
                "Kategori": finding.get("category", "Genel"),
                "Kanıt türü": EVIDENCE_LABELS.get(
                    finding.get("evidence_type", "AI inference"),
                    finding.get("evidence_type", "AI inference"),
                ),
                "Bulgu": finding.get("statement", ""),
                "Kaynak ID'leri": ", ".join(source.get("id", "") for source in linked_sources),
                "Kaynak URL'leri": " | ".join(source.get("url", "") for source in linked_sources),
                "Araştırma tarihi": data.get("research_date", "Bilinmiyor"),
            }
        )
    return pd.DataFrame(rows)


def export_csv(data: dict[str, Any]) -> bytes:
    benchmark = benchmark_dataframe(data).copy()
    benchmark.insert(0, "Araştırma tarihi", data.get("research_date", "Bilinmiyor"))
    return benchmark.to_csv(index=False).encode("utf-8-sig")


def compact_context(data: dict[str, Any]) -> str:
    payload = {
        "research_date": data.get("research_date"),
        "industry": data.get("industry"),
        "executive_summary": data.get("executive_summary"),
        "companies": data.get("companies"),
        "findings": data.get("findings"),
        "sources": data.get("sources"),
        "conflicts": data.get("conflicts"),
        "swot": data.get("swot"),
        "recommendations": data.get("recommendations"),
    }
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
