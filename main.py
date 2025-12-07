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

# Türkçe yazı desteği
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'DejaVu Sans']

# -------------------------------------------
# 1) VERİYİ OKUMA VE RASTGELE ÖRNEKLEME
# -------------------------------------------

# Tüm veriyi oku
movies_full = pd.read_csv("movies.csv")
credits = pd.read_csv("credits.csv")

print(f"📽️  Orijinal film veri seti: {len(movies_full)} film")

# Rastgele 1000 film seç (her çalıştırmada farklı olacak)
np.random.seed(None)
sample_size = min(1000, len(movies_full))
movies = movies_full.sample(n=sample_size, random_state=None).reset_index(drop=True)

print(f"✅ Seçilen rastgele örneklem: {len(movies)} film")
print("\n📋 Film veri seti (ilk 5 film):")
print(movies.head())

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

print("\n✂️  Temizleme işlemi tamamlandı")
print(f"Kalan sütunlar: {movies_clean.columns.tolist()}")
print(f"Veri seti boyutu: {movies_clean.shape[0]} satır × {movies_clean.shape[1]} sütun")

# -------------------------------------------
# 3) EKSİK VERİLERİ DOLDURMA
# -------------------------------------------

# runtime (film süresi) eksiklerini medyan ile doldur
movies_clean["runtime"] = movies_clean["runtime"].fillna(movies_clean["runtime"].median())

# release_date (çıkış tarihi) eksik olan satırları sil
movies_clean = movies_clean.dropna(subset=["release_date"])

print("\n🔧 Eksik verileri tamamlandı:")
print(movies_clean.isnull().sum())

# -------------------------------------------
# 4) TARİHİ (release_date) YIL FORMATINA ÇEVİRME
# -------------------------------------------
movies_clean["release_date"] = pd.to_datetime(movies_clean["release_date"], errors="coerce")
movies_clean["release_year"] = movies_clean["release_date"].dt.year

print("\n📅 Tarih → Yıl dönüşümü örnek:")
print(movies_clean[["release_date", "release_year"]].head())

# -------------------------------------------
# 5) GENRE (TÜR) SÜTUNUNU DÜZENLEME
# -------------------------------------------
def extract_genres(g):
    """Türleri string'den liste'ye çevir"""
    try:
        g = ast.literal_eval(g)
        return [genre["name"] for genre in g]
    except:
        return []

movies_clean["genre_list"] = movies_clean["genres"].apply(extract_genres)

print("\n🎬 Film türleri (Örnek):")
print(movies_clean[["title", "genre_list"]].head())

# -------------------------------------------
# 6) GÖRSELLEŞTİRME
# -------------------------------------------

# 📊 TÜRLERE GÖRE FİLM SAYISI
all_genres = movies_clean["genre_list"].explode()
genre_counts = all_genres.value_counts()

plt.figure(figsize=(12,6))
genre_counts.plot(kind="bar", color="steelblue")
plt.title("📊 Film Türlerinin Dağılımı (Toplam 1000 filmde hangi türler en yaygın?)", fontsize=14, fontweight='bold')
plt.xlabel("Film Türleri")
plt.ylabel("Film Sayısı")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 📈 OY SAYISININ DAĞILIMI
plt.figure(figsize=(10,5))
plt.hist(movies_clean["vote_count"], bins=40, color="lightcoral", edgecolor='black')
plt.title("📈 Filmlerin Aldığı Oy Sayısı Dağılımı\n(Kaç kişi filme oy vermiş?)", fontsize=14, fontweight='bold')
plt.xlabel("Oy Sayısı")
plt.ylabel("Film Frekansı (kaç film bu kadar oy aldı?)")
plt.grid(axis='y', alpha=0.3)
plt.show()

# 💰 POPÜLARİTE vs GELİR İLİŞKİSİ
plt.figure(figsize=(10,6))
plt.scatter(movies_clean["popularity"], movies_clean["revenue"], alpha=0.5, s=50, color="darkgreen")
plt.title("💰 Film Popülaritesi vs Gişe Geliri\n(Popüler filmler daha fazla gelir mi?)", fontsize=14, fontweight='bold')
plt.xlabel("Popülarite Puanı (0-100)")
plt.ylabel("Gişe Geliri ($)")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# 🔗 SAYISAL ÖZELLİKLER ARASINDA İLİŞKİ
numeric_cols = movies_clean[["budget", "popularity", "revenue", "runtime", "vote_average", "vote_count"]]

plt.figure(figsize=(10,8))
sns.heatmap(numeric_cols.corr(), annot=True, cmap="coolwarm", cbar_kws={'label': 'Korelasyon'})
plt.title("🔗 Sayısal Özellikler Arasındaki İlişkiler\n(Renkler ne kadar kuvvetli ilişki olduğunu gösterir)", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

# -------------------------------------------
# 7) KÜMELEME MODELİ (K-MEANS) - FİLMLERİ GRUPLAMA
# -------------------------------------------

print("\n" + "="*100)
print("🎯 ADIM 3: KÜMELEMEYİ (FİLMLERİ GRUPLAMA) BAŞLATIYORUZ")
print("="*100)

# Kullanılacak özellikleri seç
feature_cols = ["budget", "popularity", "revenue", "runtime", "vote_average", "vote_count"]

print(f"\n📍 Seçilen özellikler:")
for i, col in enumerate(feature_cols, 1):
    print(f"   {i}. {col}")

# Kopyala ve sayısal yap
clustering_df = movies_clean[feature_cols].copy()
for c in feature_cols:
    clustering_df[c] = pd.to_numeric(clustering_df[c], errors="coerce")

# Eksikleri kaldır
clustering_df = clustering_df.dropna()
valid_idx = clustering_df.index

print(f"\n✅ Kümeleme için hazır film sayısı: {len(clustering_df)} film")

# Verileri normalize et (0-1 arasına çevir)
scaler = StandardScaler()
scaled = scaler.fit_transform(clustering_df)

# En uygun küme sayısını bul (1 ile 10 arasında test et)
inertias = []
sil_scores = []
K_range = range(1, 11)

print("\n🔍 En uygun küme sayısını arıyor...")
for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(scaled)
    inertias.append(km.inertia_)
    if k >= 2:
        sil_scores.append(silhouette_score(scaled, km.labels_))
    else:
        sil_scores.append(np.nan)

# Grafikleri göster
plt.figure(figsize=(14,5))

# Elbow yöntemi
plt.subplot(1,2,1)
plt.plot(K_range, inertias, "bo-", linewidth=2, markersize=8)
plt.title("📉 Elbow Yöntemi\n(En uygun küme sayısını bulmak için)", fontsize=12, fontweight='bold')
plt.xlabel("Küme Sayısı (k)")
plt.ylabel("İç Hata (Inertia - Düşük olması iyi)")
plt.grid(True, alpha=0.3)
for i, k in enumerate(K_range):
    plt.text(k, inertias[i], f'{inertia:.0f}', ha='center', fontsize=8)

# Silhouette skoru
plt.subplot(1,2,2)
valid_sil = [(k, score) for k, score in zip(K_range, sil_scores) if not np.isnan(score)]
k_vals = [v[0] for v in valid_sil]
s_vals = [v[1] for v in valid_sil]
plt.plot(k_vals, s_vals, "go-", linewidth=2, markersize=8)
plt.title("⭐ Silhouette Skoru\n(Yüksek olması iyi kümelenme anlamına gelir)", fontsize=12, fontweight='bold')
plt.xlabel("Küme Sayısı (k)")
plt.ylabel("Silhouette Skoru (0 ile 1 arasında)")
plt.grid(True, alpha=0.3)
for k, score in valid_sil:
    plt.text(k, score, f'{score:.2f}', ha='center', fontsize=8)

plt.tight_layout()
plt.show()

# Otomatik seçim
best_k = int(np.nanargmax(sil_scores) + 1)
if best_k < 2:
    best_k = 3

print(f"\n✨ En iyi küme sayısı seçildi: k = {best_k}")

# Final kümeleme
kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
labels = kmeans.fit_predict(scaled)

movies_clean.loc[valid_idx, "cluster"] = labels

print(f"\n✅ Kümeleme tamamlandı!")
print(f"🎞️  Her kümede kaç film var?")
print(movies_clean["cluster"].value_counts().sort_index())

# Küme merkezlerini göster
centers_df = pd.DataFrame(
    scaler.inverse_transform(kmeans.cluster_centers_),
    columns=feature_cols
)
print(f"\n📊 Kümelerin Özellikleri (Merkez Değerler):")
print(centers_df.round(2))

# Her kümede örnek filmler
print(f"\n🎬 Her Kümeden Örnek Filmler:")
for cl in sorted(movies_clean["cluster"].dropna().unique()):
    print(f"\n--- GRUP {int(cl)} (TOPLAM {len(movies_clean[movies_clean['cluster']==cl])} FİLM) ---")
    print("Popüler filmler:")
    print(movies_clean[movies_clean["cluster"]==cl]
          .nlargest(3, 'popularity')[["title","release_year","budget","popularity"]]
          .to_string(index=False))

# -------------------------
# 8) KÜMELERI GÖRSEL OLARAK GÖSTER (2 Boyutlu)
# -------------------------

pca = PCA(n_components=2, random_state=42)
proj = pca.fit_transform(scaled)
proj_df = pd.DataFrame(proj, index=valid_idx, columns=["Boyut 1", "Boyut 2"])
proj_df["Küme"] = labels

plt.figure(figsize=(10,7))
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8']
for cluster in sorted(proj_df["Küme"].unique()):
    mask = proj_df["Küme"] == cluster
    plt.scatter(proj_df[mask]["Boyut 1"], 
               proj_df[mask]["Boyut 2"],
               label=f'Grup {int(cluster)}',
               s=100,
               alpha=0.7,
               color=colors[int(cluster) % len(colors)])

plt.title("🎞️  Filmler Kümelere Göre Gruplandırıldı\n(Her renk farklı bir film grubunu temsil eder)", 
         fontsize=14, fontweight='bold')
plt.xlabel("Boyut 1 (Temel Özellik)")
plt.ylabel("Boyut 2 (İkincil Özellik)")
plt.legend(loc='best', fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# -------------------------
# 9) KÜMELERIN ÖZELLİKLERİ
# -------------------------

profile = clustering_df.copy()
profile["Küme"] = labels

print("\n" + "="*100)
print("📊 HER KÜMENIN ORTALAMA ÖZELLİKLERİ")
print("="*100)
print(profile.groupby("Küme").mean().round(2))

# Tür dağılımı
movies_with_clusters = movies_clean.loc[valid_idx].copy()
movies_with_clusters["Küme"] = labels
movies_with_clusters["genre_list"] = movies_with_clusters["genre_list"].apply(lambda g: g if isinstance(g, list) else [])
genre_by_cluster = (movies_with_clusters.explode("genre_list")
                    .groupby(["Küme","genre_list"])["title"].count()
                    .reset_index(name="Film Sayısı"))

print("\n" + "="*100)
print("🎬 HER KÜMEDEKI EN YAYGIN TÜRLER")
print("="*100)
for cl in sorted(movies_with_clusters["Küme"].unique()):
    print(f"\n--- GRUP {int(cl)} ---")
    cluster_genres = genre_by_cluster[genre_by_cluster["Küme"] == cl].nlargest(5, "Film Sayısı")
    for idx, (_, row) in enumerate(cluster_genres.iterrows(), 1):
        print(f"   {idx}. {row['genre_list']}: {int(row['Film Sayısı'])} film")

# -------------------------
# 10) KÜMELEME MODELİNİ KAYDETME
# -------------------------
joblib.dump({"scaler": scaler, "kmeans": kmeans, "pca": pca}, "kmeans_pipeline.joblib")
print("\n✅ Model kaydedildi: kmeans_pipeline.joblib")

# -------------------------
# 11) BASİT ÖNERİ FONKSİYONU
# -------------------------

def recommend_similar_titles(title, top_n=5):
    """Aynı gruptaki benzer filmleri öner"""
    if title not in movies_with_clusters["title"].values:
        print(f"❌ Film '{title}' bulunamadı!")
        return pd.DataFrame()
    
    idx = movies_with_clusters[movies_with_clusters["title"]==title].index[0]
    if idx not in valid_idx:
        return pd.DataFrame()
    
    cl = movies_with_clusters.loc[idx, "Küme"]
    cand_idx = movies_with_clusters[movies_with_clusters["Küme"]==cl].index
    feat_matrix = scaled[np.isin(valid_idx, cand_idx)]
    target_vec = scaled[list(valid_idx).index(idx)]
    dists = euclidean_distances([target_vec], feat_matrix)[0]
    ranked = pd.DataFrame({"idx": cand_idx, "dist": dists}).sort_values("dist")
    ranked = ranked[ranked["idx"] != idx].head(top_n)
    return movies_with_clusters.loc[ranked["idx"], ["title","release_year","popularity","Küme"]]

print("\n" + "="*100)
print("💡 ÖRNEK: BİR FİLME BENZER FİLMLER ÖNERME")
print("="*100)
sample_title = movies_with_clusters["title"].iloc[0]
print(f"\n🎬 Seçilen film: {sample_title}")
print("\n📽️  Aynı gruptaki benzer filmler:")
print(recommend_similar_titles(sample_title, top_n=5))

# -------------------------
# 12) KULLANICI VERİSİ OLUŞTURMA (SİMÜLASYON)
# -------------------------

print("\n" + "="*100)
print("👥 ADIM 4: KURGUSAL KULLANICI VERİSİ OLUŞTURULUYOR")
print("="*100)

np.random.seed(None)
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
print(f"\n✅ Kullanıcı verileri hazır!")
print(f"👥 Toplam Kullanıcı: {user_behavior_df['user_id'].nunique()}")
print(f"🎬 Toplam İzleme Kaydı: {len(user_behavior_df)}")
print(f"📊 Ortalama her kullanıcı {len(user_behavior_df)/user_behavior_df['user_id'].nunique():.1f} film izlemiş")

# -------------------------
# 13) KULLANICI-FİLM TABLOSU OLUŞTUR
# -------------------------

user_item_matrix = user_behavior_df.pivot_table(
    index='user_id',
    columns='movie_id',
    values='rating',
    fill_value=0
)

print(f"\n📊 Kullanıcı-Film Tablosu: {user_item_matrix.shape[0]} kullanıcı × {user_item_matrix.shape[1]} film")

# -------------------------
# 14) İŞBİRLİKÇİ FİLTRELEME (Benzer Kullanıcı Bulma)
# -------------------------

def find_similar_users(user_id, top_n=5):
    """Benzer zevkleri olan kullanıcıları bul"""
    if user_id not in user_item_matrix.index:
        return []
    
    user_vector = user_item_matrix.loc[user_id]
    similarities = user_item_matrix.corrwith(user_vector, axis=1)
    similarities = similarities.drop(index=user_id, errors="ignore")
    similar_users = similarities[similarities > 0].sort_values(ascending=False).head(top_n)
    return similar_users.index.tolist()

def recommend_movies_collaborative(user_id, top_n=5):
    """İşbirlikçi Filtreleme: Benzer kullanıcıların izledikleri filmleri öner"""
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
        ['id', 'title', 'release_year', 'popularity', 'Küme']
    ]
    return result

# -------------------------
# 15) İÇERİK TABANLI ÖNERI (Benzer Tür Bulma)
# -------------------------

def recommend_movies_content_based(user_id, top_n=5):
    """İçerik Tabanlı: Kullanıcının izlediği türdeki diğer filmleri öner"""
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
        ['id', 'title', 'release_year', 'popularity', 'genre_list', 'Küme']
    ]
    return result

# -------------------------
# 16) HİBRİT ÖNERI (2 Yöntemi Birleştir)
# -------------------------

def recommend_movies_hybrid(user_id, top_n=5, alpha=0.6):
    """Hibrit: İşbirlikçi ve İçerik Tabanlı Önerileri Birleştir"""
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
        ['id', 'title', 'release_year', 'popularity', 'genre_list', 'Küme']
    ]
    return result

# -------------------------
# 17) ÖNERİLERİ TEST ET
# -------------------------

test_users = [1, 5, 10, 25]

print("\n" + "="*100)
print("🎯 ADIM 5: ÖNERİ SİSTEMİNİ TEST EDIYORUZ")
print("="*100)

for user_id in test_users:
    print(f"\n{'='*100}")
    print(f"👤 KULLANICI {user_id}")
    print(f"{'='*100}")
    
    # İzleme geçmişi
    user_watched = user_behavior_df[user_behavior_df['user_id'] == user_id]
    watched_titles = movies_with_clusters[movies_with_clusters['id'].isin(user_watched['movie_id'])]['title'].tolist()
    
    print(f"\n📺 Bu kullanıcı {len(watched_titles)} film izlemiş:")
    for i, title in enumerate(watched_titles[:5], 1):
        print(f"   {i}. {title}")
    if len(watched_titles) > 5:
        print(f"   ... ve {len(watched_titles)-5} film daha")
    
    # İşbirlikçi öneriler
    print(f"\n🔗 YÖNTEMİ 1: İŞBİRLİKÇİ FİLTRELEME")
    print("   (Benzer zevkteki kullanıcıların izlediği filmler)")
    collab = recommend_movies_collaborative(user_id, top_n=3)
    if not collab.empty:
        for i, (_, row) in enumerate(collab.iterrows(), 1):
            print(f"   {i}. {row['title']} ({int(row['release_year'])}) ⭐ {row['popularity']:.1f}")
    else:
        print("   ❌ Öneri bulunamadı")
    
    # İçerik tabanlı öneriler
    print(f"\n📂 YÖNTEMİ 2: İÇERİK TABANLI")
    print("   (Aynı türdeki popüler filmler)")
    content = recommend_movies_content_based(user_id, top_n=3)
    if not content.empty:
        for i, (_, row) in enumerate(content.iterrows(), 1):
            genres = ", ".join(row['genre_list'][:2])
            print(f"   {i}. {row['title']} ({int(row['release_year'])}) - Türler: {genres}")
    else:
        print("   ❌ Öneri bulunamadı")
    
    # Hibrit öneriler
    print(f"\n⚡ YÖNTEMİ 3: HİBRİT (Her ikisini birleştir)")
    hybrid = recommend_movies_hybrid(user_id, top_n=3)
    if not hybrid.empty:
        for i, (_, row) in enumerate(hybrid.iterrows(), 1):
            print(f"   {i}. {row['title']} ({int(row['release_year'])}) ⭐ {row['popularity']:.1f}")
    else:
        print("   ❌ Öneri bulunamadı")

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

print("\n\n✅ Sistem kaydedildi: recommendation_system.joblib")

# -----------------------------------------------
# 19) DETAYLI İSTATİSTİKSEL ANALİZ
# -----------------------------------------------

print("\n" + "="*100)
print("📊 ADIM 6: DETAYLI İSTATİSTİKLER")
print("="*100)

# Genel istatistikler
print("\n🎬 TÜJÜN FİLMLER HAKKINDA:")
print(f"   • Toplam Film: {len(movies_with_clusters)}")
print(f"   • Ortalama Bütçe: ${movies_with_clusters['budget'].mean():,.0f}")
print(f"   • Ortalama Gişe Geliri: ${movies_with_clusters['revenue'].mean():,.0f}")
print(f"   • Ortalama Popülarite Puanı: {movies_with_clusters['popularity'].mean():.2f}/100")
print(f"   • Ortalama IMDb Puanı: {movies_with_clusters['vote_average'].mean():.2f}/10")
print(f"   • Ortalama Film Süresi: {movies_with_clusters['runtime'].mean():.0f} dakika")

# Grup başına detaylı istatistikler
print("\n" + "="*100)
print("📈 HER GRUP (CLUSTER) HAKKINDA DETAYLAR")
print("="*100)

for cl in sorted(movies_with_clusters["Küme"].dropna().unique()):
    cluster_data = movies_with_clusters[movies_with_clusters["Küme"] == cl]
    print(f"\n{'─'*80}")
    print(f"📌 GRUP {int(cl)} ({len(cluster_data)} film)")
    print(f"{'─'*80}")
    print(f"   💰 Bütçe:")
    print(f"      • Minimum: ${cluster_data['budget'].min():,.0f}")
    print(f"      • Maksimum: ${cluster_data['budget'].max():,.0f}")
    print(f"      • Ortalama: ${cluster_data['budget'].mean():,.0f}")
    print(f"   💵 Gişe Geliri:")
    print(f"      • Ortalama: ${cluster_data['revenue'].mean():,.0f}")
    print(f"      • Maksimum: ${cluster_data['revenue'].max():,.0f}")
    print(f"   ⭐ Popülarite Puanı: Ort={cluster_data['popularity'].mean():.2f}, Std={cluster_data['popularity'].std():.2f}")
    print(f"   📊 IMDb Puanı: {cluster_data['vote_average'].mean():.2f}/10")
    print(f"   🕐 Ortalama Süre: {cluster_data['runtime'].mean():.0f} dakika")

# -----------------------------------------------
# 20) ÖNERİ SİSTEMİ BAŞARI ORANI
# -----------------------------------------------

print("\n" + "="*100)
print("📈 ÖNERİ SİSTEMİ BAŞARI ORANI")
print("="*100)

recommendation_stats = {
    'İşbirlikçi Filtreleme': {'başarı': 0, 'toplam': 0},
    'İçerik Tabanlı': {'başarı': 0, 'toplam': 0},
    'Hibrit': {'başarı': 0, 'toplam': 0}
}

all_test_users = list(user_behavior_df['user_id'].unique())[:20]

for user_id in all_test_users:
    collab = recommend_movies_collaborative(user_id, top_n=5)
    recommendation_stats['İşbirlikçi Filtreleme']['toplam'] += 1
    if not collab.empty:
        recommendation_stats['İşbirlikçi Filtreleme']['başarı'] += 1
    
    content = recommend_movies_content_based(user_id, top_n=5)
    recommendation_stats['İçerik Tabanlı']['toplam'] += 1
    if not content.empty:
        recommendation_stats['İçerik Tabanlı']['başarı'] += 1
    
    hybrid = recommend_movies_hybrid(user_id, top_n=5)
    recommendation_stats['Hibrit']['toplam'] += 1
    if not hybrid.empty:
        recommendation_stats['Hibrit']['başarı'] += 1

print("\n📊 20 kullanıcı ile test yapıldı. Sonuçlar:\n")
for method, stats in recommendation_stats.items():
    success_rate = (stats['başarı'] / stats['toplam'] * 100) if stats['toplam'] > 0 else 0
    print(f"   ✅ {method}: {success_rate:.1f}% başarı ({stats['başarı']}/{stats['toplam']})")

# -----------------------------------------------
# 21) GÖRSELLEŞTIRMELER
# -----------------------------------------------

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Grup büyüklükleri
cluster_sizes = movies_with_clusters['Küme'].value_counts().sort_index()
axes[0].bar(cluster_sizes.index, cluster_sizes.values, color='skyblue', edgecolor='black')
axes[0].set_title('📊 Her Gruptaki Film Sayısı\n(Gruplar deneli dağılmış mı?)', fontsize=12, fontweight='bold')
axes[0].set_xlabel('Grup Numarası')
axes[0].set_ylabel('Film Sayısı')
axes[0].grid(True, alpha=0.3, axis='y')

# Grup başına ortalama bütçe
avg_budget_by_cluster = movies_with_clusters.groupby('Küme')['budget'].mean()
axes[1].bar(avg_budget_by_cluster.index, avg_budget_by_cluster.values, color='coral', edgecolor='black')
axes[1].set_title('💰 Her Grubun Ortalama Bütçesi\n(Hangi gruplar daha pahalı?)', fontsize=12, fontweight='bold')
axes[1].set_xlabel('Grup Numarası')
axes[1].set_ylabel('Ortalama Bütçe ($)')
axes[1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.show()

# -----------------------------------------------
# 22) FINAL RAPOR
# -----------------------------------------------

print("\n" + "="*100)
print("📋 FİNAL RAPOR - FİLM ÖNERİ SİSTEMİ")
print("="*100)

report = f"""
🎯 PROJE NEYİ YAPIYOR?
   Filmler otomatik olarak benzer özelliklere göre gruplara ayrılıyor.
   Kullanıcılara da benzer zevkindeki kullanıcıların izlediği filmler öneriliyor.

📊 KULLANILAN VERİ:
   • Orijinal Kütüphane: {len(movies_full):,} film
   • Bu Çalıştırmada Kullanılan: {len(movies_with_clusters)} film (rastgele seçildi)
   • Özellikler: Bütçe, Popülarite, Gelir, Süre, IMDb Puanı, Oy Sayısı

🔬 YAPILAN İŞLEMLER:
   1️⃣  Veri Temizleme: Eksik verileri tamamla, türleri ayıkla
   2️⃣  Analiz: İstatistikler, korelasyonlar, görseller
   3️⃣  Kümeleme: K-Means algoritması ile {best_k} grup oluştur
   4️⃣  Öneriler: 3 yöntemle film önerileri ver
   5️⃣  Değerlendirme: Başarı oranlarını ölç

🏆 SONUÇLAR:
   • Seçilen grup sayısı: {best_k} (Silhouette yöntemiyle)
   • Simüle edilen kullanıcı sayısı: {user_behavior_df['user_id'].nunique()}
   • Toplam izleme kaydı: {len(user_behavior_df)}
   • Hibrit yöntem başarısı: {recommendation_stats['Hibrit']['başarı']/recommendation_stats['Hibrit']['toplam']*100:.1f}%

💾 KAYDEDILEN DOSYALAR:
   ✓ kmeans_pipeline.joblib - Kümeleme modeli
   ✓ recommendation_system.joblib - Öneri sistemi
   ✓ complete_project.joblib - Tüm veriler
"""

print(report)

# Final kayit
joblib.dump({
    "movies_with_clusters": movies_with_clusters,
    "user_behavior_df": user_behavior_df,
    "user_item_matrix": user_item_matrix,
    "recommendation_stats": recommendation_stats,
    "best_k": best_k,
    "feature_cols": feature_cols
}, "complete_project.joblib")

print("\n✅ PROJE TAMAMLANDI!")
print("💾 Tüm veriler kaydedildi: complete_project.joblib")
