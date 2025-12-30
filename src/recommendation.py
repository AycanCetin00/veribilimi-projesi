import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import euclidean_distances
import joblib


def recommend_similar_titles(title, movies_with_clusters, scaled_features, valid_idx, top_n=5):
    """Aynı kümedeki benzer filmleri, ölçekli özelliklere bakarak öner."""
    if title not in movies_with_clusters["title"].values:
        return pd.DataFrame()

    idx = movies_with_clusters[movies_with_clusters["title"] == title].index[0]
    if idx not in valid_idx:
        return pd.DataFrame()

    cluster_id = movies_with_clusters.loc[idx, "cluster"]
    candidate_idx = movies_with_clusters[movies_with_clusters["cluster"] == cluster_id].index
    feat_matrix = scaled_features[np.isin(valid_idx, candidate_idx)]
    target_vec = scaled_features[list(valid_idx).index(idx)]
    dists = euclidean_distances([target_vec], feat_matrix)[0]

    ranked = pd.DataFrame({"idx": candidate_idx, "dist": dists}).sort_values("dist")
    ranked = ranked[ranked["idx"] != idx].head(top_n)
    return movies_with_clusters.loc[ranked["idx"], ["title", "release_year", "popularity", "cluster"]]


def simulate_user_behavior(movies_with_clusters: pd.DataFrame, n_users=100, watch_range=(5, 20), seed=None):
    rng = np.random.default_rng(seed)
    user_watch_data = []
    movie_ids = movies_with_clusters["id"].dropna().values

    for user_id in range(1, n_users + 1):
        n_watches = rng.integers(watch_range[0], watch_range[1])
        chosen = rng.choice(movie_ids, min(n_watches, len(movie_ids)), replace=False)
        for movie_id in chosen:
            user_watch_data.append(
                {"user_id": user_id, "movie_id": int(movie_id), "rating": rng.uniform(3, 10)}
            )

    return pd.DataFrame(user_watch_data)


def build_user_item_matrix(user_behavior_df: pd.DataFrame):
    """Kullanıcı-film puan matrisini oluşturur."""
    return user_behavior_df.pivot_table(index="user_id", columns="movie_id", values="rating", fill_value=0)


def find_similar_users(user_item_matrix: pd.DataFrame, user_id, top_n=5):
    if user_id not in user_item_matrix.index:
        return []
    user_vector = user_item_matrix.loc[user_id]
    similarities = user_item_matrix.corrwith(user_vector, axis=1)
    similarities = similarities.drop(index=user_id, errors="ignore")
    similar_users = similarities[similarities > 0].sort_values(ascending=False).head(top_n)
    return similar_users.index.tolist()


def recommend_movies_collaborative(user_item_matrix, movies, user_id, top_n=5):
    if user_id not in user_item_matrix.index:
        return pd.DataFrame()
    similar_users = find_similar_users(user_item_matrix, user_id, top_n=10)
    if not similar_users:
        return pd.DataFrame()

    recommendations = user_item_matrix.loc[similar_users].sum(axis=0)
    user_watched = user_item_matrix.loc[user_id][user_item_matrix.loc[user_id] > 0].index
    recommendations = recommendations[~recommendations.index.isin(user_watched)]

    top_movie_ids = recommendations.nlargest(top_n).index.tolist()
    return movies[movies["id"].isin(top_movie_ids)][["id", "title", "release_year", "popularity", "cluster"]]


def recommend_movies_content_based(user_behavior_df, movies, user_id, top_n=5):
    user_movies = user_behavior_df[user_behavior_df["user_id"] == user_id]["movie_id"].values
    user_genres = set()

    for movie_id in user_movies:
        genres = movies[movies["id"] == movie_id]["genre_list"].values
        if len(genres) > 0:
            user_genres.update(genres[0])

    if not user_genres:
        return pd.DataFrame()

    candidates = movies[movies["genre_list"].apply(lambda x: bool(user_genres & set(x)))]
    candidates = candidates[~candidates["id"].isin(user_movies)]
    return candidates.nlargest(top_n, "popularity")[
        ["id", "title", "release_year", "popularity", "genre_list", "cluster"]
    ]


def recommend_movies_hybrid(user_behavior_df, user_item_matrix, movies, user_id, top_n=5, alpha=0.6):
    collab_recs = recommend_movies_collaborative(user_item_matrix, movies, user_id, top_n=top_n * 2)
    content_recs = recommend_movies_content_based(user_behavior_df, movies, user_id, top_n=top_n * 2)

    hybrid_scores = {}
    for _, row in collab_recs.iterrows():
        hybrid_scores[row["id"]] = hybrid_scores.get(row["id"], 0) + alpha

    for _, row in content_recs.iterrows():
        hybrid_scores[row["id"]] = hybrid_scores.get(row["id"], 0) + (1 - alpha)

    top_ids = sorted(hybrid_scores, key=hybrid_scores.get, reverse=True)[:top_n]
    return movies[movies["id"].isin(top_ids)][["id", "title", "release_year", "popularity", "genre_list", "cluster"]]


def save_recommendation_artifacts(user_behavior_df, user_item_matrix, movies_with_clusters, kmeans, scaler, path):
    joblib.dump(
        {
            "user_behavior_df": user_behavior_df,
            "user_item_matrix": user_item_matrix,
            "movies_with_clusters": movies_with_clusters,
            "kmeans": kmeans,
            "scaler": scaler,
        },
        path,
    )
