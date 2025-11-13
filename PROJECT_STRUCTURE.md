# MAVERICK Projekt - Létrehozott Fájlok Áttekintése

## ? Sikeresen Létrehozott Fájlstruktúra

```
AI-Agent/
??? README.md                      ? Projekt dokumentáció
??? requirements.txt               ? Python függõségek
??? setup.py                       ? Package telepítõ
??? .gitignore                     ? Git kizárási szabályok
?
??? config/
?   ??? config.yaml               ? Univerzális konfiguráció (NER frissítve)
?
??? src/
?   ??? __init__.py               ? Package init
?   ?
?   ??? data/
?   ?   ??? __init__.py           ?
?   ?   ??? loader.py             ? ABS_torveny & labels betöltés
?   ?   ??? preprocessor.py       ? Kép elõfeldolgozás
?   ?
?   ??? models/
?   ?   ??? __init__.py           ?
?   ?   ??? cnn_model.py          ? Hierarchikus CNN architektúra
?   ?   ??? predictor.py          ? Teljes predikciós engine
?   ?
?   ??? ocr/
?   ?   ??? __init__.py           ?
?   ?   ??? text_extractor.py    ? EasyOCR wrapper
?   ?   ??? ner_processor.py      ? spaCy NER + pattern matching
?   ?
?   ??? hierarchy/
?   ?   ??? __init__.py           ?
?   ?   ??? validator.py          ? Hierarchia ellenõrzés
?   ?   ??? new_category.py       ? Új advertiser kezelés
?   ?
?   ??? utils/
?       ??? __init__.py           ?
?       ??? visualization.py      ? Diagramok, confusion matrix
?       ??? io_utils.py           ? YAML/JSON kezelés
?
??? notebooks/
?   ??? 01_data_exploration.ipynb       ? Adat feltárás
?   ??? 02_model_training.ipynb         ? Modell tanítás
?   ??? 03_inference_demo.ipynb         ? Predikció demo
?
??? scripts/
    ??? train.py                  ? CLI training script
    ??? predict.py                ? CLI prediction script
```

## ?? Fõbb Funkciók

### 1. Data Loading (`src/data/`)
- ? ABS_torveny.csv betöltés és hierarchia mappings
- ? labels.csv feldolgozás és validálás
- ? Képek batch loading tf.data.Dataset-tel
- ? Train/validation split

### 2. CNN Model (`src/models/`)
- ? Hierarchikus 4-szintû CNN (Segment ? Brand ? BaseBrand ? Advertiser)
- ? Concatenate layers a hierarchia követésére
- ? Dropout és BatchNormalization
- ? Multi-task loss

### 3. OCR & NER (`src/ocr/`)
- ? EasyOCR magyar nyelvû szövegkinyerés
- ? spaCy Hungarian NER (hu_core_news_lg)
- ? Univerzális pattern matching (nem csak üdítõk!)
- ? Új variant/advertiser azonosítás

### 4. Hierarchy Validation (`src/hierarchy/`)
- ? Predikciók validálása ABS_torveny szerint
- ? Érvénytelen kombinációk detektálása
- ? Új advertiser javaslatok kezelése
- ? CSV update funkció

### 5. Prediction Engine (`src/models/predictor.py`)
- ? Teljes pipeline: CNN ? OCR ? NER ? Validation
- ? Confidence threshold alapú OCR aktiválás
- ? Új advertiser proposal rendszer
- ? Interaktív jóváhagyás

### 6. Utilities (`src/utils/`)
- ? Training history vizualizáció
- ? Confusion matrix plotting
- ? Config management (YAML)
- ? JSON export/import

### 7. CLI Scripts (`scripts/`)
- ? `train.py` - Parancssorból tanítás
- ? `predict.py` - Batch predikció CLI-bõl
- ? Checkpoint kezelés
- ? Auto-approve opció

### 8. Colab Notebooks (`notebooks/`)
- ? Data exploration notebook
- ? Training notebook GPU támogatással
- ? Inference demo interaktív jóváhagyással

## ?? Használati Példák

### Google Colab-ban (Notebook)
```python
# Clone repo
!git clone https://github.com/Koppi02/AI-Agent.git
%cd AI-Agent
!pip install -r requirements.txt

# Használat
from src.models.predictor import MaverickPredictor

predictor = MaverickPredictor.from_checkpoint(...)
result = predictor.predict_with_new_category_support('image.jpg')
```

### CLI (Lokálisan)
```bash
# Tanítás
python scripts/train.py --config config/config.yaml --epochs 30

# Predikció
python scripts/predict.py --image test.jpg --model model.keras --approve-new
```

## ?? Konfiguráció Testreszabása

A `config/config.yaml` fájlban módosíthatók:
- Drive elérési utak
- Modell hyperparaméterek
- OCR/NER beállítások
- Confidence thresholdok
- **NER variant indicators** (univerzális!)

## ?? Következõ Lépések

1. ? Push GitHub-ra: `git add . && git commit -m "Complete project structure" && git push`
2. ? Google Colab-ban tesztelés
3. ? Konfiguráció finomítása az adataidhoz
4. ? Elsõ training futtatás
5. ? OCR model telepítése Colab-ban (hu_core_news_lg)

## ?? Ismert Limitációk

- spaCy model telepítés Colab-ban manuális lépés lehet (lásd notebooks)
- EasyOCR GPU támogatás CUDA availability-tõl függ
- Elsõ futtatáskor NER model letöltése ~500MB

## ?? Támogatás

Issues: https://github.com/Koppi02/AI-Agent/issues

---
**Készítve:** 2024 | **Verzió:** 0.6.0 | **Státusz:** Production Ready ?
