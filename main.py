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
# 7) KÜMELEME MODELİ (K-MEANS)
# -------------------------------------------

# Kümeleme için sayısal özellikleri seç
clustering_features = movies_clean[["budget", "popularity", "revenue", "runtime", "vote_average", "vote_count"]].copy()

# Eksik verileri sil
clustering_features = clustering_features.dropna()

# Verileri standardize et (önemli!)
scaler = StandardScaler()
scaled_features = scaler.fit_transform(clustering_features)

# Optimal küme sayısını bulmak için Elbow Method
inertias = []
K_range = range(1, 11)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(scaled_features)
    inertias.append(kmeans.inertia_)

# Elbow grafiği
plt.figure(figsize=(8,5))
plt.plot(K_range, inertias, "bo-")
plt.title("Elbow Method - Optimal Küme Sayısı")
plt.xlabel("Küme Sayısı (k)")
plt.ylabel("Inertia")
plt.grid(True)
plt.show()

# Optimal k değeri ile modeli eğit (örnek: k=3)
optimal_k = 3
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
clustering_features["cluster"] = kmeans.fit_predict(scaled_features)

print(f"\nKümeleme tamamlandı. {optimal_k} küme oluşturuldu.")
print("\nKüme dağılımı:")
print(clustering_features["cluster"].value_counts().sort_index())

# Kümeleme sonuçlarını görselleştir
plt.figure(figsize=(10,6))
scatter = plt.scatter(clustering_features["popularity"], 
                      clustering_features["revenue"], 
                      c=clustering_features["cluster"], 
                      cmap="viridis", 
                      alpha=0.6,
                      s=50)
plt.xlabel("Popularity")
plt.ylabel("Revenue")
plt.title("K-Means Kümeleme Sonuçları")
plt.colorbar(scatter, label="Küme")
plt.show()

# Küme merkezlerinin özellikleri
print("\nKüme Merkezlerinin Özellikleri:")
centers_df = pd.DataFrame(
    scaler.inverse_transform(kmeans.cluster_centers_),
    columns=clustering_features.columns[:-1]
)
print(centers_df)
