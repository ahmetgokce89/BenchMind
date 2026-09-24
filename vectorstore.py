from __future__ import annotations

import chromadb
import pandas as pd
from sentence_transformers import SentenceTransformer
import streamlit as st

_embedding_model = None


class KeywordCollection:
    def __init__(self, records: list[dict]) -> None:
        self.records = records


def get_embedding_model() -> SentenceTransformer:
    global _embedding_model
    if _embedding_model is None:
        try:
            _embedding_model = SentenceTransformer(
                "sentence-transformers/all-MiniLM-L6-v2",
                local_files_only=True,
            )
        except Exception:
            _embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return _embedding_model


@st.cache_resource
def get_chroma_client() -> chromadb.Client:
    return chromadb.Client()


def build_collection(
    chroma_client: chromadb.Client,
    collection_name: str,
    df: pd.DataFrame,
    text_column: str,
):
    try:
        try:
            chroma_client.delete_collection(collection_name)
        except Exception:
            pass

        collection = chroma_client.create_collection(collection_name)
        model = get_embedding_model()
        texts = df[text_column].astype(str).tolist()
        embeddings = model.encode(texts, show_progress_bar=False).tolist()

        collection.add(
            ids=df["review_id"].astype(str).tolist(),
            documents=texts,
            embeddings=embeddings,
            metadatas=[
                {
                    "review_id": row.review_id,
                    "sentiment_label": row.sentiment_label,
                    "pii_entities": row.pii_entities or "",
                    "pii_entity_count": int(row.pii_entity_count),
                    "source_score": str(row.source_score),
                }
                for row in df.itertuples(index=False)
            ],
        )
        return collection
    except Exception:
        return KeywordCollection(
            records=[
                {
                    "review_id": row.review_id,
                    "sentiment_label": row.sentiment_label,
                    "pii_entities": row.pii_entities or "",
                    "pii_entity_count": int(row.pii_entity_count),
                    "source_score": str(row.source_score),
                    "text": getattr(row, text_column),
                }
                for row in df.itertuples(index=False)
            ]
        )


def query_collection(collection, question: str, n_results: int = 5) -> list[dict]:
    if isinstance(collection, KeywordCollection):
        return query_keyword_collection(collection, question, n_results=n_results)

    model = get_embedding_model()
    query_embedding = model.encode([question], show_progress_bar=False).tolist()
    results = collection.query(query_embeddings=query_embedding, n_results=n_results)

    rows = []
    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    for document, metadata, distance in zip(documents, metadatas, distances, strict=False):
        rows.append(
            {
                "review_id": metadata.get("review_id", ""),
                "sentiment_label": metadata.get("sentiment_label", "Unknown"),
                "pii_entities": metadata.get("pii_entities", ""),
                "pii_entity_count": metadata.get("pii_entity_count", 0),
                "source_score": metadata.get("source_score", ""),
                "distance": round(float(distance), 4),
                "text": document,
            }
        )
    return rows


def query_keyword_collection(collection: KeywordCollection, question: str, n_results: int = 5) -> list[dict]:
    query_terms = [term.lower() for term in question.split() if len(term) > 2]

    def score(record: dict) -> tuple[int, int]:
        lowered = record["text"].lower()
        overlap = sum(lowered.count(term) for term in query_terms)
        return overlap, -len(record["text"])

    ranked_records = sorted(collection.records, key=score, reverse=True)
    top_records = ranked_records[:n_results]
    rows = []
    for record in top_records:
        rows.append(
            {
                "review_id": record["review_id"],
                "sentiment_label": record["sentiment_label"],
                "pii_entities": record["pii_entities"],
                "pii_entity_count": record["pii_entity_count"],
                "source_score": record["source_score"],
                "distance": None,
                "text": record["text"],
            }
        )
    return rows
