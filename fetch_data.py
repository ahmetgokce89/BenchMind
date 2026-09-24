"""
Kaggle'dan iki farklı müşteri yorum dataseti indirir.

Kaynak A: Amazon ürün yorumları (snap/amazon-fine-food-reviews)
Kaynak B: Kadın giyim e-ticaret yorumları (nicapotato/womens-ecommerce-clothing-reviews)

Çalıştır: python fetch_data.py
"""

import kagglehub
import pandas as pd
import os

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)


def fetch_amazon_reviews():
    print("Amazon yorumları indiriliyor...")
    path = kagglehub.dataset_download("snap/amazon-fine-food-reviews")
    df = pd.read_csv(os.path.join(path, "Reviews.csv"))
    df = df[["Text", "Score"]].dropna().head(500)
    df.columns = ["review", "rating"]
    out = os.path.join(DATA_DIR, "amazon_reviews.csv")
    df.to_csv(out, index=False)
    print(f"Kaydedildi: {out} ({len(df)} yorum)")
    return out


def fetch_clothing_reviews():
    print("Giyim yorumları indiriliyor...")
    path = kagglehub.dataset_download("nicapotato/womens-ecommerce-clothing-reviews")
    df = pd.read_csv(os.path.join(path, "Womens Clothing E-Commerce Reviews.csv"))
    df = df[["Review Text", "Rating"]].dropna().head(500)
    df.columns = ["review", "rating"]
    out = os.path.join(DATA_DIR, "clothing_reviews.csv")
    df.to_csv(out, index=False)
    print(f"Kaydedildi: {out} ({len(df)} yorum)")
    return out


def fetch_market_reviews():
    print("App Market (Google Play) yorumları indiriliyor...")
    path = kagglehub.dataset_download("lava18/google-play-store-apps")
    # Kaggle'daki bu veri setinin içinde kullanıcı yorumları da bulunmaktadır
    df = pd.read_csv(os.path.join(path, "googleplaystore_user_reviews.csv"))
    df = df[["Translated_Review", "Sentiment"]].dropna().head(500)
    df.columns = ["review", "sentiment"]
    out = os.path.join(DATA_DIR, "market_reviews.csv")
    df.to_csv(out, index=False)
    print(f"Kaydedildi: {out} ({len(df)} yorum)")
    return out


if __name__ == "__main__":
    fetch_amazon_reviews()
    fetch_clothing_reviews()
    fetch_market_reviews()
    print("\nHer iki CSV de 'data/' klasöründe hazır.")
    print("DualSense'i başlatmak için: streamlit run app.py")
