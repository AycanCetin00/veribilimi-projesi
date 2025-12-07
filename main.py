import pandas as pd
import ast
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.decomposition import PCA
import joblib
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics import silhouette_score

# -------------------------------------------
# 1) VERİYİ OKUMA VE RASTGELE ÖRNEKLEME
# -------------------------------------------

# Tüm veriyi oku
movies_full = pd.read_csv("movies.csv")
credits = pd.read_csv("credits.csv")

print(f"Orijinal veri seti: {len(movies_full)} film")

# Rastgele 1000 film seç (her çalıştırmada farklı olacak)
np.random.seed(None)  # Her çalıştırmada farklı random state
sample_size = min(1000, len(movies_full))  # Eğer 1000'den az varsa hepsini al
movies = movies_full.sample(n=sample_size, random_state=None).reset_index(drop=True)

print(f"Seçilen örneklem: {len(movies)} film")
print("\nFilmler veri seti ilk 5 satır:")
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
print(f"Temizlenmiş veri seti boyutu: {movies_clean.shape}")

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

# Kullanılacak featureları sabitle
feature_cols = ["budget", "popularity", "revenue", "runtime", "vote_average", "vote_count"]

# Kopyala ve numeric zorla
clustering_df = movies_clean[feature_cols].copy()
for c in feature_cols:
    clustering_df[c] = pd.to_numeric(clustering_df[c], errors="coerce")

# Eksikleri kaldır (index korunur)
clustering_df = clustering_df.dropna()
valid_idx = clustering_df.index

print(f"\nKümeleme için kullanılacak film sayısı: {len(clustering_df)}")

# Scale et
scaler = StandardScaler()
scaled = scaler.fit_transform(clustering_df)

# Elbow ve Silhouette ile uygun k ara
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
plt.title("Elbow Method")
plt.xlabel("k (Küme Sayısı)")
plt.ylabel("Inertia")
plt.grid(True)

plt.subplot(1,2,2)
plt.plot(K_range, sil_scores, "go-")
plt.title("Silhouette Score")
plt.xlabel("k (Küme Sayısı)")
plt.ylabel("Silhouette Score")
plt.grid(True)
plt.tight_layout()
plt.show()

# Otomatik seçim: silhouette en yüksek k (>=2) veya fallback 3
best_k = int(np.nanargmax(sil_scores) + 1)
if best_k < 2:
    best_k = 3

kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
labels = kmeans.fit_predict(scaled)

# Sonuçları orijinal veriyle hizala
movies_clean.loc[valid_idx, "cluster"] = labels

print(f"\nKümeleme tamamlandı. Seçilen k = {best_k}")
print(movies_clean["cluster"].value_counts().sort_index())

# Küme merkezleri
centers_df = pd.DataFrame(
    scaler.inverse_transform(kmeans.cluster_centers_),
    columns=feature_cols
)
print("\nKüme Merkezleri:")
print(centers_df)

# Her kümedeki örnek filmler
print("\nKüme Örnekleri (Her kümeden 5 film):")
for cl in sorted(movies_clean["cluster"].dropna().unique()):
    print(f"\n--- Cluster {int(cl)} ---")
    print(movies_clean[movies_clean["cluster"]==cl][["title","release_year","budget","revenue","popularity"]].head(5))

# -------------------------
# 8) KÜMELERİ 2B GÖRSELLEŞTİRME (PCA)
# -------------------------

pca = PCA(n_components=2, random_state=42)
proj = pca.fit_transform(scaled)
proj_df = pd.DataFrame(proj, index=valid_idx, columns=["PC1", "PC2"])
proj_df["cluster"] = labels

plt.figure(figsize=(8,6))
sns.scatterplot(data=proj_df, x="PC1", y="PC2", hue="cluster", palette="tab10", s=50, alpha=0.7)
plt.title("PCA ile 2B Görselleştirme - Kümeler")
plt.legend(title="Cluster")
plt.show()

# -------------------------
# 9) KÜME PROFİLLERİ
# -------------------------
profile = clustering_df.copy()
profile["cluster"] = labels
print("\n=== CLUSTER PROFİLLERİ (ORTALAMA) ===")
print(profile.groupby("cluster").mean().round(2))

# Tür dağılımı
movies_with_clusters = movies_clean.loc[valid_idx].copy()
movies_with_clusters["cluster"] = labels
movies_with_clusters["genre_list"] = movies_with_clusters["genre_list"].apply(lambda g: g if isinstance(g, list) else [])
genre_by_cluster = (movies_with_clusters.explode("genre_list")
                    .groupby(["cluster","genre_list"])["title"].count()
                    .reset_index(name="count"))
print("\n=== CLUSTER'LAR İÇİN EN YAYGÜN TÜRLER ===")
print(genre_by_cluster.sort_values(["cluster","count"], ascending=[True, False]).groupby("cluster").head(5))

# -------------------------
# 10) KÜMELEME MODELİNİ KAYDETME
# -------------------------
joblib.dump({"scaler": scaler, "kmeans": kmeans, "pca": pca}, "kmeans_pipeline.joblib")
print("\n✓ Model kaydedildi: kmeans_pipeline.joblib")

# -------------------------
# 11) BASİT ÖNERİ FONKSİYONU
# -------------------------

def recommend_similar_titles(title, top_n=5):
    """Aynı kümeden benzer filmleri öner"""
    if title not in movies_with_clusters["title"].values:
        print(f"Film '{title}' bulunamadı!")
        return pd.DataFrame()
    
    idx = movies_with_clusters[movies_with_clusters["title"]==title].index[0]
    if idx not in valid_idx:
        return pd.DataFrame()
    
    cl = movies_with_clusters.loc[idx, "cluster"]
    cand_idx = movies_with_clusters[movies_with_clusters["cluster"]==cl].index
    feat_matrix = scaled[np.isin(valid_idx, cand_idx)]
    target_vec = scaled[list(valid_idx).index(idx)]
    dists = euclidean_distances([target_vec], feat_matrix)[0]
    ranked = pd.DataFrame({"idx": cand_idx, "dist": dists}).sort_values("dist")
    ranked = ranked[ranked["idx"] != idx].head(top_n)
    return movies_with_clusters.loc[ranked["idx"], ["title","release_year","popularity","cluster"]]

print("\n=== ÖRNEK: AYNI KÜMEDEN BENZERİ FILMLER ===")
sample_title = movies_with_clusters["title"].iloc[0]
print(f"Film: {sample_title}")
print(recommend_similar_titles(sample_title, top_n=5))

# -------------------------
# 12) GERÇEKÇİ KULLANICI VERİSİ OLUŞTURMA
# -------------------------

np.random.seed(None)  # Her çalıştırmada farklı veri
n_users = 100
n_watches_per_user = (5, 20)

user_watch_data = []
for user_id in range(1, n_users + 1):
    n_watches = np.random.randint(n_watches_per_user[0], n_watches_per_user[1])
    movie_ids = np.random.choice(
        movies_with_clusters["id"].dropna().values, 
        min(n_watches, len(movies_with_clusters)), 
        replace=False
    )
    for movie_id in movie_ids:
        user_watch_data.append({
            'user_id': user_id,
            'movie_id': int(movie_id),
            'rating': np.random.uniform(3, 10)
        })

user_behavior_df = pd.DataFrame(user_watch_data)
print(f"\n=== KULLANICI DAVRANIŞI VERİSİ ===")
print(f"Toplam Kullanıcı: {user_behavior_df['user_id'].nunique()}")
print(f"Toplam İzleme Kaydı: {len(user_behavior_df)}")
print(user_behavior_df.head(10))

# -------------------------
# 13) KULLANICI-FİLM MATRİSİ
# -------------------------

user_item_matrix = user_behavior_df.pivot_table(
    index='user_id',
    columns='movie_id',
    values='rating',
    fill_value=0
)

print(f"\nKullanıcı-Film Matrisi Boyutu: {user_item_matrix.shape}")

# -------------------------
# 14) İŞBİRLİKÇİ FİLTRELEME
# -------------------------

def find_similar_users(user_id, top_n=5):
    """Benzer kullanıcıları bul"""
    if user_id not in user_item_matrix.index:
        return []
    
    # Kullanıcının izleme vektörü (index = movie_id)
    user_vector = user_item_matrix.loc[user_id]
    # Satır bazlı korelasyon (kullanıcı-kullanıcı benzerliği) için axis=1 kullan
    similarities = user_item_matrix.corrwith(user_vector, axis=1)
    # Kendisini çıkar ve pozitif korelasyona göre sırala
    similarities = similarities.drop(index=user_id, errors="ignore")
    similar_users = similarities[similarities > 0].sort_values(ascending=False).head(top_n)
    return similar_users.index.tolist()

def recommend_movies_collaborative(user_id, top_n=5):
    """İşbirlikçi filtreleme ile öneriler"""
    if user_id not in user_item_matrix.index:
        return pd.DataFrame()
    
    similar_users = find_similar_users(user_id, top_n=10)
    
    if not similar_users:
        return pd.DataFrame()
    
    recommendations = user_item_matrix.loc[similar_users].sum(axis=0)
    user_watched = user_item_matrix.loc[user_id][user_item_matrix.loc[user_id] > 0].index
    recommendations = recommendations[~recommendations.index.isin(user_watched)]
    
    top_movie_ids = recommendations.nlargest(top_n).index.tolist()
    result = movies_with_clusters[movies_with_clusters['id'].isin(top_movie_ids)][
        ['id', 'title', 'release_year', 'popularity', 'cluster']
    ]
    return result

# -------------------------
# 15) İÇERİK TABANLI ÖNERI
# -------------------------

def recommend_movies_content_based(user_id, top_n=5):
    """İçerik tabanlı öneri (genre bazlı)"""
    user_movies = user_behavior_df[user_behavior_df['user_id'] == user_id]['movie_id'].values
    user_genres = set()
    
    for movie_id in user_movies:
        genres = movies_with_clusters[movies_with_clusters['id'] == movie_id]['genre_list'].values
        if len(genres) > 0:
            user_genres.update(genres[0])
    
    if not user_genres:
        return pd.DataFrame()
    
    candidates = movies_with_clusters[
        movies_with_clusters['genre_list'].apply(lambda x: bool(user_genres & set(x)))
    ]
    
    candidates = candidates[~candidates['id'].isin(user_movies)]
    result = candidates.nlargest(top_n, 'popularity')[
        ['id', 'title', 'release_year', 'popularity', 'genre_list', 'cluster']
    ]
    return result

# -------------------------
# 16) HİBRİT ÖNERI SISTEMI
# -------------------------

def recommend_movies_hybrid(user_id, top_n=5, alpha=0.6):
    """Hibrit öneri (işbirlikçi + içerik)"""
    collab_recs = recommend_movies_collaborative(user_id, top_n=top_n*2)
    content_recs = recommend_movies_content_based(user_id, top_n=top_n*2)
    
    hybrid_scores = {}
    
    for _, row in collab_recs.iterrows():
        movie_id = row['id']
        hybrid_scores[movie_id] = hybrid_scores.get(movie_id, 0) + alpha
    
    for _, row in content_recs.iterrows():
        movie_id = row['id']
        hybrid_scores[movie_id] = hybrid_scores.get(movie_id, 0) + (1 - alpha)
    
    top_ids = sorted(hybrid_scores, key=hybrid_scores.get, reverse=True)[:top_n]
    result = movies_with_clusters[movies_with_clusters['id'].isin(top_ids)][
        ['id', 'title', 'release_year', 'popularity', 'genre_list', 'cluster']
    ]
    return result

# -------------------------
# 17) ÖNERİLERİ TEST ET VE KARŞILAŞTIR
# -------------------------

test_users = [1, 5, 10, 25]

print("\n" + "="*100)
print("KULLANICI ÖNERİ SİSTEMİ TEST SONUÇLARI")
print("="*100)

for user_id in test_users:
    print(f"\n{'='*100}")
    print(f"KULLANICI {user_id}")
    print(f"{'='*100}")
    
    # İzleme geçmişi
    user_watched = user_behavior_df[user_behavior_df['user_id'] == user_id]
    watched_titles = movies_with_clusters[movies_with_clusters['id'].isin(user_watched['movie_id'])]['title'].tolist()
    print(f"\n📺 İzlediği Filmler ({len(watched_titles)} adet):")
    for i, title in enumerate(watched_titles[:5], 1):
        print(f"   {i}. {title}")
    if len(watched_titles) > 5:
        print(f"   ... ve {len(watched_titles)-5} film daha")
    
    # İşbirlikçi öneriler
    print(f"\n1️⃣  İŞBİRLİKÇİ FİLTRELEME ÖNERİLERİ:")
    collab = recommend_movies_collaborative(user_id, top_n=3)
    if not collab.empty:
        for i, (_, row) in enumerate(collab.iterrows(), 1):
            print(f"   {i}. {row['title']} ({int(row['release_year'])}) - Pop: {row['popularity']:.2f}")
    else:
        print("   Öneri bulunamadı.")
    
    # İçerik tabanlı öneriler
    print(f"\n2️⃣  İÇERİK TABANLI ÖNERİLER (GENRE):")
    content = recommend_movies_content_based(user_id, top_n=3)
    if not content.empty:
        for i, (_, row) in enumerate(content.iterrows(), 1):
            genres = ", ".join(row['genre_list'][:3])
            print(f"   {i}. {row['title']} ({int(row['release_year'])}) - Türler: {genres}")
    else:
        print("   Öneri bulunamadı.")
    
    # Hibrit öneriler
    print(f"\n3️⃣  HİBRİT ÖNERİLER:")
    hybrid = recommend_movies_hybrid(user_id, top_n=3)
    if not hybrid.empty:
        for i, (_, row) in enumerate(hybrid.iterrows(), 1):
            print(f"   {i}. {row['title']} ({int(row['release_year'])}) - Pop: {row['popularity']:.2f}")
    else:
        print("   Öneri bulunamadı.")

# -------------------------
# 18) SİSTEMİ KAYDETME
# -------------------------

joblib.dump({
    "user_behavior_df": user_behavior_df,
    "user_item_matrix": user_item_matrix,
    "movies_with_clusters": movies_with_clusters,
    "kmeans": kmeans,
    "scaler": scaler
}, "recommendation_system.joblib")

print("\n\n✓ Tüm sistem kaydedildi: recommendation_system.joblib")
