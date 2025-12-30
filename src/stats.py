import pandas as pd


def print_global_stats(movies: pd.DataFrame):
    print("\n===== Genel İstatistikler =====")
    print(f"Toplam film: {len(movies)}")
    print(f"Ortalama bütçe: ${movies['budget'].mean():,.0f}")
    print(f"Ortalama gişe geliri: ${movies['revenue'].mean():,.0f}")
    print(f"Ortalama popülerlik: {movies['popularity'].mean():.2f}")
    print(f"Ortalama IMDb puanı: {movies['vote_average'].mean():.2f}/10")
    print(f"Ortalama süre: {movies['runtime'].mean():.0f} dk")


def print_cluster_stats(movies: pd.DataFrame):
    print("\n===== Küme Bazlı İstatistikler =====")
    for cl in sorted(movies["cluster"].dropna().unique()):
        cluster_data = movies[movies["cluster"] == cl]
        print(f"\n--- Grup {int(cl)} ({len(cluster_data)} film) ---")
        print(f"  Bütçe min/maks/ort: ${cluster_data['budget'].min():,.0f} / ${cluster_data['budget'].max():,.0f} / ${cluster_data['budget'].mean():,.0f}")
        print(f"  Gelir ort/maks: ${cluster_data['revenue'].mean():,.0f} / ${cluster_data['revenue'].max():,.0f}")
        print(f"  Popülerlik ort/std: {cluster_data['popularity'].mean():.2f} / {cluster_data['popularity'].std():.2f}")
        print(f"  IMDb ort: {cluster_data['vote_average'].mean():.2f}")
        print(f"  Ortalama süre: {cluster_data['runtime'].mean():.0f} dk")


def recommendation_success(user_behavior_df, recommend_collab, recommend_content, recommend_hybrid, sample_user_ids):
    stats = {
        "collaborative": {"success": 0, "total": 0},
        "content": {"success": 0, "total": 0},
        "hybrid": {"success": 0, "total": 0},
    }

    for user_id in sample_user_ids:
        stats["collaborative"]["total"] += 1
        stats["content"]["total"] += 1
        stats["hybrid"]["total"] += 1

        if not recommend_collab(user_id).empty:
            stats["collaborative"]["success"] += 1
        if not recommend_content(user_id).empty:
            stats["content"]["success"] += 1
        if not recommend_hybrid(user_id).empty:
            stats["hybrid"]["success"] += 1

    return stats
