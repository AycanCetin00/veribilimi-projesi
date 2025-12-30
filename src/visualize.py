import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve


def _save_or_show(fig, filename: str | None, save_dir: str | None):
    if save_dir and filename:
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        fig.savefig(Path(save_dir) / filename, bbox_inches="tight")
        plt.show()   # 👈 TAM OLARAK BURAYA
    else:
        plt.show()


def plot_basic_distributions(movies: pd.DataFrame, save_dir: str | None = None):
    """Tür sayıları, oy dağılımı ve popülerlik-gelir ilişkisini çizer."""
    # Tür dağılımı
    all_genres = movies["genre_list"].explode()
    genre_counts = all_genres.value_counts()

    fig, ax = plt.subplots(figsize=(12, 6))
    genre_counts.plot(kind="bar", color="steelblue", ax=ax)
    ax.set_title("Film Türlerinin Dağılımı")
    ax.set_xlabel("Film Türleri")
    ax.set_ylabel("Film Sayısı")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    _save_or_show(fig, "tur_dagilimi.png", save_dir)

    # Oy sayısı dağılımı
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(movies["vote_count"], bins=40, color="lightcoral", edgecolor="black")
    ax.set_title("Oy Sayısı Dağılımı")
    ax.set_xlabel("Oy Sayısı")
    ax.set_ylabel("Film Frekansı")
    ax.grid(axis="y", alpha=0.3)
    _save_or_show(fig, "oy_dagilimi.png", save_dir)

    # Popülerlik vs gişe geliri
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(movies["popularity"], movies["revenue"], alpha=0.5, s=50, color="darkgreen")
    ax.set_title("Film Popülerliği vs Gişe Geliri")
    ax.set_xlabel("Popülerlik")
    ax.set_ylabel("Gişe Geliri ($)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _save_or_show(fig, "populerlik_gise.png", save_dir)


def plot_correlation_heatmap(movies: pd.DataFrame, numeric_cols, save_dir: str | None = None):
    """Sayısal değişkenler arası korelasyon ısı haritası."""
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(movies[numeric_cols].corr(), annot=True, cmap="coolwarm", cbar_kws={"label": "Korelasyon"}, ax=ax)
    ax.set_title("Sayısal Özellikler Korelasyonları")
    fig.tight_layout()
    _save_or_show(fig, "korelasyon_heatmap.png", save_dir)


def plot_k_diagnostics(k_values, inertias, sil_scores, save_dir: str | None = None):
    """Dirsek yöntemi ve silhouette skorlarını çizer."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(k_values, inertias, "bo-", linewidth=2, markersize=8)
    axes[0].set_title("Dirsek (Elbow) Yöntemi")
    axes[0].set_xlabel("Küme Sayısı (k)")
    axes[0].set_ylabel("Inertia")
    axes[0].grid(True, alpha=0.3)
    for i, k in enumerate(k_values):
        axes[0].text(k, inertias[i], f"{inertias[i]:.0f}", ha="center", fontsize=8)

    valid_sil = [(k, score) for k, score in zip(k_values, sil_scores) if not np.isnan(score)]
    if valid_sil:
        k_vals = [v[0] for v in valid_sil]
        s_vals = [v[1] for v in valid_sil]
        axes[1].plot(k_vals, s_vals, "go-", linewidth=2, markersize=8)
        for k, score in valid_sil:
            axes[1].text(k, score, f"{score:.2f}", ha="center", fontsize=8)
    axes[1].set_title("Silhouette Skoru")
    axes[1].set_xlabel("Küme Sayısı (k)")
    axes[1].set_ylabel("Silhouette Skoru")
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    _save_or_show(fig, "k_secimi.png", save_dir)


def plot_cluster_projection(proj_df: pd.DataFrame, labels, save_dir: str | None = None):
    """PCA ile 2 boyutta küme dağılım grafiği."""
    fig, ax = plt.subplots(figsize=(10, 7))
    colors = ["#FF6B6B", "#4ECDC4", "#45B7D1", "#FFA07A", "#98D8C8"]
    for cluster in sorted(set(labels)):
        mask = proj_df["cluster"] == cluster
        ax.scatter(
            proj_df[mask]["dim1"],
            proj_df[mask]["dim2"],
            label=f"Grup {int(cluster)}",
            s=100,
            alpha=0.7,
            color=colors[int(cluster) % len(colors)],
        )

    ax.set_title("Filmler Kümelere Göre Dağılım")
    ax.set_xlabel("Boyut 1")
    ax.set_ylabel("Boyut 2")
    ax.legend(loc="best", fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _save_or_show(fig, "kume_pca.png", save_dir)


def plot_cluster_overview(movies_with_clusters: pd.DataFrame, save_dir: str | None = None):
    """Küme boyutları ve ortalama bütçeleri özetleyen çift grafik."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    cluster_sizes = movies_with_clusters["cluster"].value_counts().sort_index()
    axes[0].bar(cluster_sizes.index, cluster_sizes.values, color="skyblue", edgecolor="black")
    axes[0].set_title("Her Gruptaki Film Sayısı")
    axes[0].set_xlabel("Grup Numarası")
    axes[0].set_ylabel("Film Sayısı")
    axes[0].grid(True, alpha=0.3, axis="y")

    if "budget" in movies_with_clusters.columns:
        avg_budget_by_cluster = movies_with_clusters.groupby("cluster")["budget"].mean()
        axes[1].bar(avg_budget_by_cluster.index, avg_budget_by_cluster.values, color="coral", edgecolor="black")
        axes[1].set_title("Her Grubun Ortalama Bütçesi")
        axes[1].set_xlabel("Grup Numarası")
        axes[1].set_ylabel("Ortalama Bütçe ($)")
        axes[1].grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    _save_or_show(fig, "kume_ozetleri.png", save_dir)


def confusion_matrix_clf(y_true, y_pred, save_dir=None):
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Tahmin")
    ax.set_ylabel("Gerçek")
    fig.tight_layout()

    if save_dir:
        fig.savefig(f"{save_dir}/confusion_matrix_clf.png")
    plt.show()

def roc_curve_clf(y_true, y_proba, save_dir=None):
    fpr, tpr, _ = roc_curve(y_true, y_proba)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, label="ROC Curve")
    ax.plot([0, 1], [0, 1], "k--")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend()
    ax.grid(True)
    fig.tight_layout()

    if save_dir:
        fig.savefig(f"{save_dir}/roc_curve_clf.png")
    plt.show()

def pr_curve_clf(y_true, y_proba, save_dir=None):
    precision, recall, _ = precision_recall_curve(y_true, y_proba)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(recall, precision)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision–Recall Curve")
    ax.grid(True)
    fig.tight_layout()

    if save_dir:
        fig.savefig(f"{save_dir}/pr_curve_clf.png")
    plt.show()


def feature_importance_clf(model, feature_names, save_dir=None):
    import numpy as np
    import matplotlib.pyplot as plt

    importances = model.feature_importances_
    indices = np.argsort(importances)

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.barh(
        np.array(feature_names)[indices],
        importances[indices],
    )

    ax.set_title("Feature Importances - Classification")
    ax.set_xlabel("Önem Skoru")
    ax.grid(axis="x", alpha=0.3)

    fig.tight_layout()

    if save_dir:
        fig.savefig(f"{save_dir}/feature_importance_clf.png")

    plt.show()


def feature_importance_reg(model, feature_names, save_dir=None):
    import numpy as np
    import matplotlib.pyplot as plt

    importances = model.feature_importances_
    indices = np.argsort(importances)

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.barh(
        np.array(feature_names)[indices],
        importances[indices],
    )

    ax.set_title("Feature Importances - Regression")
    ax.set_xlabel("Önem Skoru")
    ax.grid(axis="x", alpha=0.3)

    fig.tight_layout()

    if save_dir:
        fig.savefig(f"{save_dir}/feature_importance_reg.png")

    plt.show()



def predicted_vs_actual_reg(y_true, y_pred, save_dir=None):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_true, y_pred, alpha=0.6)
    ax.plot([y_true.min(), y_true.max()],
            [y_true.min(), y_true.max()],
            "r--")
    ax.set_xlabel("Gerçek Değerler")
    ax.set_ylabel("Tahmin Edilen Değerler")
    ax.set_title("Gerçek vs Tahmin (Regresyon)")
    ax.grid(True)
    fig.tight_layout()

    if save_dir:
        fig.savefig(f"{save_dir}/predicted_vs_actual_reg.png")
    plt.show()


def residuals_reg(y_true, y_pred, save_dir=None):
    residuals = y_true - y_pred

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.hist(residuals, bins=30, edgecolor="black")
    ax.set_title("Residuals (Hata) Dağılımı")
    ax.set_xlabel("Hata (Gerçek - Tahmin)")
    ax.set_ylabel("Frekans")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if save_dir:
        fig.savefig(f"{save_dir}/residuals_reg.png")
    plt.show()

