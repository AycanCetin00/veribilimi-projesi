import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score
import joblib


def train_classifier(movies: pd.DataFrame, features, popularity_col="popularity", quantile=0.75):
    """Popüler filmi tahmin eden rastgele orman sınıflandırıcısı eğitir."""
    cols = list(dict.fromkeys(features + [popularity_col]))  # popülerlik iki kez eklenmesin

    df = movies.dropna(subset=cols).copy()
    df[cols] = df[cols].apply(pd.to_numeric, errors="coerce")
    df = df.dropna(subset=cols).copy()

    threshold = df[popularity_col].quantile(quantile)
    df["is_popular"] = (df[popularity_col] >= threshold).astype(int)

    X = df[features]
    y = df["is_popular"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train_scaled, y_train)

    y_pred = clf.predict(X_test_scaled)
    y_proba = clf.predict_proba(X_test_scaled)[:, 1]

    metrics = {
        "report": classification_report(y_test, y_pred, output_dict=False),
        "accuracy": accuracy_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba) if len(set(y_test)) > 1 else np.nan,
        "threshold": threshold,
        "test_size": len(y_test),
        "positives": int(y.sum()),
    }

    return clf, scaler, metrics


def predict_popularity_by_title(title: str, movies: pd.DataFrame, clf, scaler, features):
    """Başlık vererek filmin popüler olup olmayacağını döndürür."""
    if title not in movies["title"].values:
        return None

    row = movies.loc[movies["title"] == title, features]
    if row.isnull().any(axis=None):
        return None

    X_row = scaler.transform(row.values)
    pred = clf.predict(X_row)[0]
    proba = clf.predict_proba(X_row)[0, 1]
    return {"title": title, "is_popular": int(pred), "popularity_prob": float(proba)}


def save_pipeline(clf, scaler, features, threshold, path="classification_pipeline.joblib"):
    """Sınıflandırma modelini kaydet."""
    joblib.dump({"clf": clf, "scaler": scaler, "features": features, "threshold": threshold}, path)
