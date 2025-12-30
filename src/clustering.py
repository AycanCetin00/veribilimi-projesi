import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
import joblib


def prepare_clustering_data(movies: pd.DataFrame, feature_cols):
    """Sayısal özellikleri seç, sayıya çevir ve ölçekle."""
    clustering_df = movies[feature_cols].copy()
    for c in feature_cols:
        clustering_df[c] = pd.to_numeric(clustering_df[c], errors="coerce")
    clustering_df = clustering_df.dropna()
    valid_idx = clustering_df.index

    scaler = StandardScaler()
    scaled = scaler.fit_transform(clustering_df)
    return clustering_df, scaled, scaler, valid_idx


def search_k(scaled, k_range=range(1, 11)):
    inertias = []
    sil_scores = []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(scaled)
        inertias.append(km.inertia_)
        sil_scores.append(silhouette_score(scaled, km.labels_) if k >= 2 else np.nan)

    best_k = int(np.nanargmax(sil_scores) + 1)
    if best_k < 2:
        best_k = 3
    return best_k, inertias, sil_scores


def fit_kmeans(scaled, n_clusters: int):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(scaled)
    return kmeans, labels


def project_clusters(scaled, labels):
    pca = PCA(n_components=2, random_state=42)
    proj = pca.fit_transform(scaled)
    proj_df = pd.DataFrame(proj, columns=["dim1", "dim2"])
    proj_df["cluster"] = labels
    return pca, proj_df


def cluster_centers(kmeans: KMeans, scaler: StandardScaler, feature_cols):
    centers = pd.DataFrame(scaler.inverse_transform(kmeans.cluster_centers_), columns=feature_cols)
    return centers


def save_pipeline(scaler, kmeans, pca, path="kmeans_pipeline.joblib"):
    joblib.dump({"scaler": scaler, "kmeans": kmeans, "pca": pca}, path)
