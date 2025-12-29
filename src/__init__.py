"""Numarali modulleri okunabilir isimlerle yukler."""

from importlib.util import spec_from_file_location, module_from_spec
from pathlib import Path
import sys

_BASE = Path(__file__).parent


def _load(alias: str, filename: str):
    """Dosya adini alias ile modul olarak yukle."""
    spec = spec_from_file_location(f"{__name__}.{alias}", _BASE / filename)
    module = module_from_spec(spec)
    assert spec.loader
    spec.loader.exec_module(module)  # type: ignore[arg-type]
    sys.modules[f"{__name__}.{alias}"] = module
    setattr(sys.modules[__name__], alias, module)
    return module


# 01_data_io: ham veri setlerini okur ve örnekler.
data_io = _load("data_io", "01_data_io.py")
# 02_preprocess: film verisini temizler ve özellikler çıkarır.
preprocess = _load("preprocess", "02_preprocess.py")
# 03_visualize: temel grafikler ve tanılama görselleri üretir.
visualize = _load("visualize", "03_visualize.py")
# 04_clustering: k-ortalama kumeleme için veri hazırlayıp modeller.
clustering = _load("clustering", "04_clustering.py")
# 05_classification: popülerlik sınıflandırma modelini eğitir/kaydeder.
classification = _load("classification", "05_classification.py")
# 06_regression: popülerlik skorunu tahmin eden regresyonu kurar.
regression = _load("regression", "06_regression.py")
# 07_recommendation: içerik/işbirliği öneri yöntemlerini yönetir.
recommendation = _load("recommendation", "07_recommendation.py")
# 08_stats: genel ve küme bazlı istatistikleri raporlar.
stats = _load("stats", "08_stats.py")

__all__ = [
    "data_io",
    "preprocess",
    "visualize",
    "clustering",
    "classification",
    "regression",
    "recommendation",
    "stats",
]
