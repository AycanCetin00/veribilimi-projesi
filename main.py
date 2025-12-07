import pandas as pd
import ast
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import numpy as np


# -------------------------------------------
# 1) VERİYİ OKUMA
# -------------------------------------------
movies = pd.read_csv("movies.csv")
credits = pd.read_csv("credits.csv")

print("Movies veri seti ilk 5 satır:")
print(movies.head())

print("\nCredits veri seti ilk 5 satır:")
print(credits.head())

# -------------------------------------------
# 2) GEREKSİZ SÜTUNLARI SİLME
# -------------------------------------------
movies_clean = movies.drop(columns=[
    "homepage",
    "tagline",
    "spoken_languages",
    "keywords",
    "production_companies",
    "production_countries",
    "original_title",
    "overview"
])

print("\nTemizlenmiş veri seti sütunları:")
print(movies_clean.columns)

# -------------------------------------------
# 3) EKSİK VERİLERİ DOLDURMA
# -------------------------------------------

# runtime eksiklerini median ile doldur
movies_clean["runtime"] = movies_clean["runtime"].fillna(movies_clean["runtime"].median())

# release_date eksik olan satırı sil
movies_clean = movies_clean.dropna(subset=["release_date"])

print("\nEksik veri sonrası kontrol:")
print(movies_clean.isnull().sum())

# -------------------------------------------
# 4) TARİHİ (release_date) YIL FORMATINA ÇEVİRME
# -------------------------------------------
movies_clean["release_date"] = pd.to_datetime(movies_clean["release_date"], errors="coerce")
movies_clean["release_year"] = movies_clean["release_date"].dt.year

print("\nTarih → Yıl dönüşümü örnek:")
print(movies_clean[["release_date", "release_year"]].head())

# -------------------------------------------
# 5) GENRE SÜTUNUNU DÜZENLEME
# -------------------------------------------
def extract_genres(g):
    try:
        g = ast.literal_eval(g)      # string -> Python list
        return [genre["name"] for genre in g]
    except:
        return []

movies_clean["genre_list"] = movies_clean["genres"].apply(extract_genres)

print("\nTür dönüşümü örnek:")
print(movies_clean[["genres", "genre_list"]].head())

# -------------------------------------------
# 6) GÖRSELLEŞTİRME
# -------------------------------------------

# GENRE DAĞILIMI
all_genres = movies_clean["genre_list"].explode()
genre_counts = all_genres.value_counts()

plt.figure(figsize=(12,6))
genre_counts.plot(kind="bar")
plt.title("Film Türlerinin Dağılımı")
plt.xlabel("Türler")
plt.ylabel("Film Sayısı")
plt.tight_layout()
plt.show()

# VOTE COUNT HISTOGRAM
plt.figure(figsize=(10,5))
plt.hist(movies_clean["vote_count"], bins=40, color="skyblue")
plt.title("Vote Count Dağılımı")
plt.xlabel("Vote Count")
plt.ylabel("Frekans")
plt.show()

# POPULARITY vs REVENUE SCATTER PLOT
plt.figure(figsize=(8,5))
plt.scatter(movies_clean["popularity"], movies_clean["revenue"], alpha=0.4)
plt.title("Popularity vs Revenue")
plt.xlabel("Popularity")
plt.ylabel("Revenue")
plt.show()

# KORELASYON MATRİSİ
numeric_cols = movies_clean[["budget", "popularity", "revenue", "runtime", "vote_average", "vote_count"]]

plt.figure(figsize=(10,6))
sns.heatmap(numeric_cols.corr(), annot=True, cmap="coolwarm")
plt.title("Korelasyon Matrisi")
plt.show()

# -------------------------------------------
# 7) KÜMELEME MODELİ (K-MEANS) - GÜNCELLENDİ
# -------------------------------------------

# Kullanılacak featureları sabitle
feature_cols = ["budget", "popularity", "revenue", "runtime", "vote_average", "vote_count"]

# Kopyala ve numeric zorla
clustering_df = movies_clean[feature_cols].copy()
for c in feature_cols:
    clustering_df[c] = pd.to_numeric(clustering_df[c], errors="coerce")

# Eksikleri kaldır (index korunur)
clustering_df = clustering_df.dropna()
valid_idx = clustering_df.index

# (İsteğe bağlı) Çok büyük çarpık değerler için log1p uygulayabilirsiniz:
# clustering_df[["budget","revenue"]] = np.log1p(clustering_df[["budget","revenue"]])

# Scale et
scaler = StandardScaler()
scaled = scaler.fit_transform(clustering_df)

# Elbow ve Silhouette ile uygun k ara (silhouette için k>=2)
from sklearn.metrics import silhouette_score
inertias = []
sil_scores = []
K_range = range(1, 11)
for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(scaled)
    inertias.append(km.inertia_)
    if k >= 2:
        sil_scores.append(silhouette_score(scaled, km.labels_))
    else:
        sil_scores.append(np.nan)

# Görseller
plt.figure(figsize=(12,4))
plt.subplot(1,2,1)
plt.plot(K_range, inertias, "bo-")
plt.title("Elbow")
plt.xlabel("k")
plt.grid(True)

plt.subplot(1,2,2)
plt.plot(K_range, sil_scores, "go-")
plt.title("Silhouette (k>=2 anlamlı)")
plt.xlabel("k")
plt.grid(True)
plt.tight_layout()
plt.show()

# Otomatik seçim: silhouette en yüksek k (>=2) veya fallback 3
best_k = int(np.nanargmax(sil_scores) + 1)  # nanargmax döndürür index, index k-1
if best_k < 2:
    best_k = 3

kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
labels = kmeans.fit_predict(scaled)

# Sonuçları orijinal veriyle hizala
movies_clean.loc[valid_idx, "cluster"] = labels

print(f"\nKümeleme tamamlandı. Seçilen k = {best_k}")
print(movies_clean["cluster"].value_counts().sort_index())

# Küme merkezleri (orijinal ölçeğe geri döndür)
centers_df = pd.DataFrame(
    scaler.inverse_transform(kmeans.cluster_centers_),
    columns=feature_cols
)
print("\nKüme Merkezleri (oranlar orijinal ölçeğe çevrildi):")
print(centers_df)

# Örnek: her kümedeki top 5 film
print("\nKüme örnekleri (her kümeden 5 film):")
for cl in sorted(movies_clean["cluster"].dropna().unique()):
    print(f"\nCluster {int(cl)}:")
    print(movies_clean[movies_clean["cluster"]==cl][["title","release_year","budget","revenue","popularity"]].head(5))
