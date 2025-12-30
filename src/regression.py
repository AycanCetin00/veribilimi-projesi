import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib


def train_regressor(movies: pd.DataFrame, features, target="popularity"):
    """Popülerlik skorunu tahmin eden rastgele orman regresyonu eğitir."""
    cols = list(dict.fromkeys(features + [target]))  # hedef iki kez eklenmesin

    df = movies.dropna(subset=cols).copy()
    df[cols] = df[cols].apply(pd.to_numeric, errors="coerce")
    df = df.dropna(subset=cols).copy()

    X = df[features]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    reg = RandomForestRegressor(n_estimators=100, random_state=42)
    reg.fit(X_train_scaled, y_train)

    y_pred = reg.predict(X_test_scaled)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    metrics = {"rmse": rmse, "r2": r2, "test_size": len(y_test)}
    return reg, scaler, metrics


def predict_popularity_by_title(title: str, movies: pd.DataFrame, reg, scaler, features):
    """Başlık vererek tahmini popülerlik skorunu döndürür."""
    if title not in movies["title"].values:
        return None
    row = movies.loc[movies["title"] == title, features]
    if row.isnull().any(axis=None):
        return None
    X_row = scaler.transform(row.values)
    pred = float(reg.predict(X_row)[0])
    return {"title": title, "predicted_popularity": pred}


def save_pipeline(reg, scaler, features, path="regression_pipeline.joblib"):
    joblib.dump({"reg": reg, "scaler": scaler, "features": features}, path)
