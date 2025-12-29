import ast
import pandas as pd

UNUSED_COLUMNS = [
    "homepage",
    "tagline",
    "spoken_languages",
    "keywords",
    "production_companies",
    "production_countries",
    "original_title",
    "overview",
]


def parse_genres(genres_raw):
    """Tür bilgisini (JSON string) Python liste/isim formatına çevir."""
    try:
        genres = ast.literal_eval(genres_raw)
        return [genre["name"] for genre in genres]
    except Exception:
        return []


def clean_movies(df: pd.DataFrame) -> pd.DataFrame:
    """Kullanılmayan sütunları at, eksikleri düzelt, türetilmiş alanları ekle."""
    df = df.drop(columns=[c for c in UNUSED_COLUMNS if c in df.columns], errors="ignore")

    # Süre eksiklerini medyan ile doldur; çıkış tarihi boş satırları at
    if "runtime" in df.columns:
        df["runtime"] = df["runtime"].fillna(df["runtime"].median())
    df = df.dropna(subset=["release_date"])

    # Çıkış tarihini datetime yap ve release_year ekle
    df["release_date"] = pd.to_datetime(df["release_date"], errors="coerce")
    df["release_year"] = df["release_date"].dt.year

    # Tür listesini ayrıştır
    if "genres" in df.columns:
        df["genre_list"] = df["genres"].apply(parse_genres)

    return df
