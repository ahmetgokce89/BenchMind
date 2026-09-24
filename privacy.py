from __future__ import annotations

import re
from collections import Counter
from functools import lru_cache

import pandas as pd

try:
    from presidio_analyzer import AnalyzerEngine
    from presidio_anonymizer import AnonymizerEngine
except ImportError:
    AnalyzerEngine = None
    AnonymizerEngine = None


REGEX_PATTERNS = {
    "EMAIL_ADDRESS": re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
    "PHONE_NUMBER": re.compile(r"\b(?:\+?\d{1,2}[-.\s]?)?(?:\(?\d{3}\)?[-.\s]?)\d{3}[-.\s]?\d{4}\b"),
    "CREDIT_CARD": re.compile(r"\b(?:\d[ -]*?){13,16}\b"),
    "URL": re.compile(r"\bhttps?://[^\s]+"),
}


class RegexPrivacyEngine:
    mode = "regex"

    def analyze(self, text: str) -> list[dict]:
        hits = []
        for entity_type, pattern in REGEX_PATTERNS.items():
            for match in pattern.finditer(text):
                hits.append(
                    {
                        "entity_type": entity_type,
                        "start": match.start(),
                        "end": match.end(),
                        "score": 0.5,
                        "text": match.group(0),
                    }
                )
        return sorted(hits, key=lambda item: item["start"])

    def anonymize(self, text: str, hits: list[dict]) -> str:
        masked = text
        for hit in sorted(hits, key=lambda item: item["start"], reverse=True):
            masked = masked[: hit["start"]] + f"<{hit['entity_type']}>" + masked[hit["end"] :]
        return masked


class PresidioPrivacyEngine:
    mode = "presidio"

    def __init__(self) -> None:
        self.analyzer = AnalyzerEngine()
        self.anonymizer = AnonymizerEngine()

    def analyze(self, text: str) -> list[dict]:
        results = self.analyzer.analyze(text=text, language="en")
        hits = []
        for result in results:
            hits.append(
                {
                    "entity_type": result.entity_type,
                    "start": result.start,
                    "end": result.end,
                    "score": round(float(result.score), 3),
                    "text": text[result.start : result.end],
                }
            )
        return hits

    def anonymize(self, text: str, hits: list[dict]) -> str:
        results = self.analyzer.analyze(text=text, language="en")
        anonymized = self.anonymizer.anonymize(text=text, analyzer_results=results)
        return anonymized.text


@lru_cache(maxsize=1)
def get_privacy_engine():
    if AnalyzerEngine is not None and AnonymizerEngine is not None:
        try:
            return PresidioPrivacyEngine()
        except Exception:
            return RegexPrivacyEngine()
    return RegexPrivacyEngine()


def describe_privacy_engine() -> str:
    engine = get_privacy_engine()
    if engine.mode == "presidio":
        return "Presidio"
    return "Regex fallback (Presidio unavailable in the current environment)"


def anonymize_text(text: str) -> tuple[str, list[dict]]:
    engine = get_privacy_engine()
    hits = engine.analyze(text)
    anonymized_text = engine.anonymize(text, hits)
    return anonymized_text, hits


def anonymize_dataframe(df: pd.DataFrame, text_column: str) -> tuple[pd.DataFrame, dict]:
    processed = df.copy()
    anonymized_texts = []
    entity_labels = []
    entity_counter: Counter[str] = Counter()
    rows_with_pii = 0
    total_words = 0

    for text in processed[text_column].astype(str).tolist():
        anonymized_text, entities = anonymize_text(text)
        labels = [entity["entity_type"] for entity in entities]

        anonymized_texts.append(anonymized_text)
        entity_labels.append(", ".join(labels))
        entity_counter.update(labels)
        total_words += len(text.split())

        if entities:
            rows_with_pii += 1

    processed["anonymized_text"] = anonymized_texts
    processed["pii_entities"] = entity_labels
    processed["pii_entity_count"] = [0 if not labels else len(labels.split(", ")) for labels in entity_labels]

    stats = {
        "rows_with_pii": rows_with_pii,
        "total_pii_entities": int(sum(entity_counter.values())),
        "entity_counts": dict(entity_counter),
        "avg_words": total_words / len(processed) if len(processed) else 0.0,
    }
    return processed, stats
