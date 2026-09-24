from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent


@dataclass(frozen=True)
class DatasetConfig:
    key: str
    label: str
    path: Path
    text_column: str
    score_column: str
    source_name: str
    description: str


DATASET_CATALOG = {
    "Amazon Fine Food Reviews": DatasetConfig(
        key="amazon_food",
        label="Amazon Fine Food Reviews",
        path=BASE_DIR / "data" / "amazon_reviews.csv",
        text_column="review",
        score_column="rating",
        source_name="Kaggle: snap/amazon-fine-food-reviews",
        description="Product reviews with 1-5 ratings.",
    ),
    "Google Play App Reviews": DatasetConfig(
        key="google_play",
        label="Google Play App Reviews",
        path=BASE_DIR / "data" / "market_reviews.csv",
        text_column="review",
        score_column="sentiment",
        source_name="Kaggle: lava18/google-play-store-apps",
        description="App reviews with sentiment labels.",
    ),
    "Womens Clothing Reviews": DatasetConfig(
        key="womens_clothing",
        label="Womens Clothing Reviews",
        path=BASE_DIR / "data" / "clothing_reviews.csv",
        text_column="review",
        score_column="rating",
        source_name="Kaggle: nicapotato/womens-ecommerce-clothing-reviews",
        description="Retail reviews with 1-5 ratings.",
    ),
}


def get_dataset_config(dataset_label: str) -> DatasetConfig:
    return DATASET_CATALOG[dataset_label]


def normalize_sentiment(value) -> str:
    if pd.isna(value):
        return "Unknown"
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"positive", "negative", "neutral"}:
            return lowered.capitalize()
        return "Unknown"

    numeric_value = float(value)
    if numeric_value >= 4:
        return "Positive"
    if numeric_value <= 2:
        return "Negative"
    return "Neutral"


def load_dataset(dataset_label: str, sample_size: int = 200, seed: int = 42) -> pd.DataFrame:
    config = get_dataset_config(dataset_label)
    frame = pd.read_csv(config.path).dropna(subset=[config.text_column]).copy()

    if sample_size and len(frame) > sample_size:
        frame = frame.sample(n=sample_size, random_state=seed).reset_index(drop=True)
    else:
        frame = frame.reset_index(drop=True)

    frame["review_text"] = frame[config.text_column].astype(str)
    frame["sentiment_label"] = frame[config.score_column].apply(normalize_sentiment)
    frame["source_score"] = frame[config.score_column].astype(str)
    frame["review_id"] = [f"{config.key}-{index + 1}" for index in range(len(frame))]

    return frame[["review_id", "review_text", "sentiment_label", "source_score"]]
