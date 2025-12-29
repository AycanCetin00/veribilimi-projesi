import pandas as pd
import numpy as np


def load_datasets(movies_path: str = "movies.csv", credits_path: str = "credits.csv"):
    """Ham film ve oyuncu/ekip verilerini oku."""
    movies = pd.read_csv(movies_path)
    credits = pd.read_csv(credits_path)
    return movies, credits


def sample_movies(movies: pd.DataFrame, sample_size: int = 1000, random_state=None):
    """Veri seti boyutunu asmayacak rastgele film örnekle."""
    sample_size = min(sample_size, len(movies))
    sampled = movies.sample(n=sample_size, random_state=random_state).reset_index(drop=True)
    return sampled
