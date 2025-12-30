import numpy as np
import matplotlib.pyplot as plt

from src.data_io import load_datasets, sample_movies
from src.preprocess import clean_movies
from src.visualize import (
    plot_basic_distributions,
    plot_correlation_heatmap,
    plot_k_diagnostics,
    plot_cluster_projection,
    plot_cluster_overview,
)
from src.clustering import (
    prepare_clustering_data,
    search_k,
    fit_kmeans,
    project_clusters,
    cluster_centers,
    save_pipeline as save_kmeans_pipeline,
)
from src.classification import (
    train_classifier,
    predict_popularity_by_title as predict_popularity_clf,
    save_pipeline as save_clf_pipeline,
)
from src.regression import (
    train_regressor,
    predict_popularity_by_title as predict_popularity_reg,
    save_pipeline as save_reg_pipeline,
)
from src.recommendation import (
    recommend_similar_titles,
    simulate_user_behavior,
    build_user_item_matrix,
    recommend_movies_collaborative,
    recommend_movies_content_based,
    recommend_movies_hybrid,
    save_recommendation_artifacts,
)
from src.stats import print_global_stats, print_cluster_stats, recommendation_success

from src.visualize import (
    confusion_matrix_clf,
    roc_curve_clf,
    pr_curve_clf,
    feature_importance_clf,
)

from src.visualize import (
    feature_importance_reg,
    predicted_vs_actual_reg,
    residuals_reg,
)


# Grafiklerde Türkçe karakter desteği
plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]


CLUSTER_FEATURES = ["budget", "popularity", "revenue", "runtime", "vote_average", "vote_count"]
MODEL_FEATURES = ["budget", "revenue", "runtime", "vote_average", "vote_count"]
PLOTS_DIR = "plots"


def main():
    # 1) Veriyi oku ve örnekle
    movies_full, credits = load_datasets()
    print(f"Orijinal film veri seti: {len(movies_full)} film")

    movies = sample_movies(movies_full, sample_size=1000, random_state=None)
    print(f"Rastgele örneklem: {len(movies)} film")
    print("Film veri seti (ilk 5):")
    print(movies.head())

    # 2) Ön işleme
    movies_clean = clean_movies(movies)
    print("\nTemizleme tamamlandı")
    print(f"Kalan sütunlar: {movies_clean.columns.tolist()}")
    print(f"Veri seti boyutu: {movies_clean.shape[0]} satır x {movies_clean.shape[1]} sütun")
    print("\nEksik veri sayıları:")
    print(movies_clean.isnull().sum())
    print("\nTarih dönüşüm örneği:")
    print(movies_clean[["release_date", "release_year"]].head())
    print("\nFilm türleri örnek:")
    print(movies_clean[["title", "genre_list"]].head())

    # 3) Görselleştirme
    plot_basic_distributions(movies_clean, save_dir=PLOTS_DIR)
    plot_correlation_heatmap(movies_clean, CLUSTER_FEATURES, save_dir=PLOTS_DIR)

    # 4) Kümeleme
    print("\n" + "=" * 80)
    print("KÜMELEME BAŞLIYOR")
    print("=" * 80)
    clustering_df, scaled, scaler, valid_idx = prepare_clustering_data(movies_clean, CLUSTER_FEATURES)
    print(f"Kümeleme için hazır film sayısı: {len(clustering_df)}")

    best_k, inertias, sil_scores = search_k(scaled, k_range=range(1, 11))
    plot_k_diagnostics(list(range(1, 11)), inertias, sil_scores, save_dir=PLOTS_DIR)
    print(f"Seçilen en iyi küme sayısı: k = {best_k}")

    kmeans, labels = fit_kmeans(scaled, best_k)
    movies_clean.loc[valid_idx, "cluster"] = labels
    centers_df = cluster_centers(kmeans, scaler, CLUSTER_FEATURES)
    print("\nKümelerin merkez değerleri:")
    print(centers_df.round(2))

    print("\nHer kümeden örnek filmler:")
    for cl in sorted(movies_clean["cluster"].dropna().unique()):
        print(f"\n--- GRUP {int(cl)} ---")
        print(
            movies_clean[movies_clean["cluster"] == cl]
            .nlargest(3, "popularity")[["title", "release_year", "budget", "popularity"]]
            .to_string(index=False)
        )

    pca, proj_df = project_clusters(scaled, labels)
    plot_cluster_projection(proj_df, labels, save_dir=PLOTS_DIR)
    plot_cluster_overview(movies_clean.loc[valid_idx], save_dir=PLOTS_DIR)
    save_kmeans_pipeline(scaler, kmeans, pca, "kmeans_pipeline.joblib")
    print("Model kaydedildi: kmeans_pipeline.joblib")

        # 5) Sınıflandırma
    print("\n" + "=" * 80)
    print("SINIFLANDIRMA (popüler film tahmini)")
    print("=" * 80)

    clf, clf_scaler, clf_metrics = train_classifier(
        movies_clean.loc[valid_idx],
        features=MODEL_FEATURES
    )

    print("Sınıflandırma raporu:")
    print(clf_metrics["report"])
    print(f"Doğruluk (accuracy): {clf_metrics['accuracy']:.3f}")

    if not np.isnan(clf_metrics["roc_auc"]):
        print(f"ROC AUC: {clf_metrics['roc_auc']:.3f}")

    # 🔹 Test verisini tekrar üret (grafikler için)
    X = movies_clean.loc[valid_idx, MODEL_FEATURES].dropna()
    y = (movies_clean.loc[X.index, "popularity"] >= clf_metrics["threshold"]).astype(int)

    X_scaled = clf_scaler.transform(X)
    y_pred = clf.predict(X_scaled)
    y_proba = clf.predict_proba(X_scaled)[:, 1]

    # 📊 GRAFİKLER
    feature_importance_clf(clf, MODEL_FEATURES, save_dir=PLOTS_DIR)
    confusion_matrix_clf(y, y_pred, save_dir=PLOTS_DIR)
    roc_curve_clf(y, y_proba, save_dir=PLOTS_DIR)
    pr_curve_clf(y, y_proba, save_dir=PLOTS_DIR)

    # 🔹 Örnek tahmin
    sample_title = movies_clean.loc[valid_idx, "title"].iloc[0]
    print("\nÖrnek tahmin:")
    print(
        sample_title,
        "->",
        predict_popularity_clf(sample_title, movies_clean, clf, clf_scaler, MODEL_FEATURES)
    )

    save_clf_pipeline(
        clf,
        clf_scaler,
        MODEL_FEATURES,
        clf_metrics["threshold"],
        "classification_pipeline.joblib"
    )

    print("Model kaydedildi: classification_pipeline.joblib")


        # 6) Regresyon
    print("\n" + "=" * 80)
    print("REGRESYON (popülerlik skoru tahmini)")
    print("=" * 80)

    reg, reg_scaler, reg_metrics = train_regressor(
        movies_clean.loc[valid_idx],
        features=MODEL_FEATURES
    )

    print(f"RMSE: {reg_metrics['rmse']:.3f}")
    print(f"R2: {reg_metrics['r2']:.3f}")

    # 🔹 Grafikler için veri hazırlığı
    X = movies_clean.loc[valid_idx, MODEL_FEATURES].dropna()
    y_true = movies_clean.loc[X.index, "popularity"]

    X_scaled = reg_scaler.transform(X)
    y_pred = reg.predict(X_scaled)

    # 📊 REGRESYON GRAFİKLERİ
    feature_importance_reg(reg, MODEL_FEATURES, save_dir=PLOTS_DIR)
    predicted_vs_actual_reg(y_true, y_pred, save_dir=PLOTS_DIR)
    residuals_reg(y_true, y_pred, save_dir=PLOTS_DIR)

    # 🔹 Örnek tahmin
    print("\nÖrnek tahmin:")
    print(
        sample_title,
        "->",
        predict_popularity_reg(sample_title, movies_clean, reg, reg_scaler, MODEL_FEATURES)
    )

    save_reg_pipeline(
        reg,
        reg_scaler,
        MODEL_FEATURES,
        "regression_pipeline.joblib"
    )

    print("Model kaydedildi: regression_pipeline.joblib")


    # 7) Basit içerik tabanlı öneriler (kümeler içinde)
    print("\n" + "=" * 80)
    print("BENZER FİLM ÖNERİSİ")
    print("=" * 80)
    print(recommend_similar_titles(sample_title, movies_clean, scaled, valid_idx, top_n=5))

    # 8) Kullanıcı simülasyonu ve öneriler
    print("\n" + "=" * 80)
    print("KULLANICI SİMÜLASYONU VE ÖNERİLER")
    print("=" * 80)
    user_behavior_df = simulate_user_behavior(movies_clean.loc[valid_idx], n_users=100, watch_range=(5, 20))
    user_item_matrix = build_user_item_matrix(user_behavior_df)
    print(f"Toplam kullanıcı: {user_behavior_df['user_id'].nunique()}")
    print(f"Toplam izleme kaydı: {len(user_behavior_df)}")
    print(f"Ortalama izleme: {len(user_behavior_df) / user_behavior_df['user_id'].nunique():.1f} film/kullanıcı")
    print(f"Kullanıcı-Film tablosu: {user_item_matrix.shape[0]} x {user_item_matrix.shape[1]}")

    test_users = [1, 5, 10, 25]
    for user_id in test_users:
        print(f"\n--- KULLANICI {user_id} ---")
        watched_titles = movies_clean[movies_clean["id"].isin(
            user_behavior_df[user_behavior_df["user_id"] == user_id]["movie_id"]
        )]["title"].tolist()
        print(f"İzlenen {len(watched_titles)} film: {watched_titles[:5]}" + ("..." if len(watched_titles) > 5 else ""))

        collab = recommend_movies_collaborative(user_item_matrix, movies_clean, user_id, top_n=3)
        print("\nYöntem 1: İşbirlikçi filtreleme")
        print(collab if not collab.empty else "Öneri bulunamadı")

        content = recommend_movies_content_based(user_behavior_df, movies_clean, user_id, top_n=3)
        print("\nYöntem 2: İçerik tabanlı")
        print(content if not content.empty else "Öneri bulunamadı")

        hybrid = recommend_movies_hybrid(user_behavior_df, user_item_matrix, movies_clean, user_id, top_n=3)
        print("\nYöntem 3: Hibrit")
        print(hybrid if not hybrid.empty else "Öneri bulunamadı")

    save_recommendation_artifacts(
        user_behavior_df, user_item_matrix, movies_clean, kmeans, scaler, "recommendation_system.joblib"
    )
    print("\nSistem kaydedildi: recommendation_system.joblib")

    # 9) İstatistiksel analizler
    print_global_stats(movies_clean.loc[valid_idx])
    print_cluster_stats(movies_clean.loc[valid_idx])

    rec_stats = recommendation_success(
        user_behavior_df,
        lambda uid: recommend_movies_collaborative(user_item_matrix, movies_clean, uid, top_n=5),
        lambda uid: recommend_movies_content_based(user_behavior_df, movies_clean, uid, top_n=5),
        lambda uid: recommend_movies_hybrid(user_behavior_df, user_item_matrix, movies_clean, uid, top_n=5),
        sample_user_ids=list(user_behavior_df["user_id"].unique())[:20],
    )
    print("\nÖneri sistemleri başarı oranı (20 kullanıcı):")
    for method, stats in rec_stats.items():
        rate = (stats["success"] / stats["total"] * 100) if stats["total"] else 0
        print(f"  {method}: {rate:.1f}% ({stats['success']}/{stats['total']})")

    print("\nProje tamamlandı!")


if __name__ == "__main__":
    main()
