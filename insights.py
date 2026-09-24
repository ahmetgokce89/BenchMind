from __future__ import annotations

import re
from collections import Counter

STOPWORDS = {
    "about",
    "after",
    "again",
    "also",
    "and",
    "are",
    "because",
    "been",
    "before",
    "being",
    "between",
    "both",
    "could",
    "does",
    "each",
    "from",
    "good",
    "great",
    "have",
    "just",
    "more",
    "much",
    "only",
    "really",
    "that",
    "their",
    "them",
    "there",
    "they",
    "this",
    "very",
    "what",
    "when",
    "with",
    "would",
    "your",
}


def extract_top_terms(texts: list[str], limit: int = 10) -> list[dict[str, int | str]]:
    counter: Counter[str] = Counter()
    for text in texts:
        for token in re.findall(r"[a-zA-Z][a-zA-Z']{2,}", text.lower()):
            if token in STOPWORDS:
                continue
            if token.startswith("<") and token.endswith(">"):
                continue
            counter[token] += 1

    return [{"term": term, "count": count} for term, count in counter.most_common(limit)]
