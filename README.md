# VeriBilimi

Bu repo, masaüstümdeki VeriBilimi projesini içerir. İçinde veri setleri, eğitim modelleri (`*.joblib`), ve analiz/plot kodları bulunur.

## İçerik
- `main.py` — Proje giriş dosyası
- `plots/` — Görselleştirmeler
- `VeriBilimiDataset/` — Veri setleri
- Model dosyaları: `*.joblib` (büyük dosyalar için Git LFS kullanılması önerilir)

## Kurulum
1. Python 3.10+ önerilir
2. Sanal ortam oluştur:
```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Kullanım
```bash
python main.py
```

## Notlar
- Büyük `*.joblib` dosyaları için Git LFS kullanın (`git lfs install` ve `git lfs track "*.joblib"`).
- Sorular: https://github.com/AycanCetin00
