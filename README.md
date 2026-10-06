# TruthLens — Fake News Detector

A deployment-ready fake-news classification project using **FastAPI + scikit-learn + vanilla HTML/CSS/JavaScript**.

## Dataset

This project uses the **Fake and Real News Dataset** by Clément Bisaillon on Kaggle:
https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset

The dataset contains `Fake.csv` and `True.csv`, with `title`, `text`, `subject`, and `date` fields. Dataset license: **CC BY-NC-SA 4.0**. Please review Kaggle's current dataset terms before redistributing the raw data.

## Architecture

```text
Browser
  │
  ▼
FastAPI ───────────────► /api/health
  │
  └────────────────────► /api/predict
                            │
                            ▼
                    TF-IDF + LinearSVC
                            │
                            ▼
                    REAL / FAKE + score
```

## Project structure

```text
fake-news-detector-app/
├── backend/
│   └── app/
│       ├── __init__.py
│       └── main.py
├── frontend/
│   ├── index.html
│   ├── app.js
│   └── style.css
├── data/
│   ├── Fake.csv.zip
│   └── True.csv.zip
├── model/
│   └── fake_news_detector.joblib
├── scripts/
│   └── train_model.py
├── tests/
│   └── test_api.py
├── Dockerfile
├── docker-compose.yml
├── render.yaml
├── requirements.txt
├── LICENSE
└── README.md
```

## 1. Run locally

Create and activate a virtual environment:

### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Start the API:

```powershell
uvicorn backend.app.main:app --reload
```

Open:

- Frontend: http://127.0.0.1:8000
- Swagger: http://127.0.0.1:8000/docs
- ReDoc: http://127.0.0.1:8000/redoc
- Health: http://127.0.0.1:8000/api/health

## 2. Retrain the model

The ZIP datasets are already included in `data/`.

```powershell
python scripts/train_model.py
```

The script extracts the ZIPs, validates the expected columns, cleans the text, removes duplicate articles, trains the TF-IDF + LinearSVC pipeline, evaluates it, and writes:

```text
model/fake_news_detector.joblib
```

## 3. Docker

Build and run:

```powershell
docker build -t fake-news-detector .
docker run --rm -p 8000:8000 fake-news-detector
```

Or:

```powershell
docker compose up --build
```

Then open http://127.0.0.1:8000.

## 4. Render deployment

This repository includes `render.yaml` and a production Dockerfile.

1. Push the project to GitHub.
2. Create a new Render Web Service from the repository.
3. Render will use the included Docker configuration.
4. After deployment, open the generated HTTPS URL.
5. Verify `/api/health` and `/docs`.

## API

### `GET /api/health`

Returns service/model availability.

### `POST /api/predict`

Request:

```json
{
  "title": "Example headline",
  "text": "Article body with at least ten characters..."
}
```

Response:

```json
{
  "prediction": 1,
  "label": "FAKE",
  "decision_score": 0.42,
  "message": "Classification completed..."
}
```

In this implementation, `1 = FAKE` and `0 = REAL`.

## Model

The production model is the validated pipeline used in the accompanying notebook:

- Text: `title + text`
- Cleaning: lowercase, URL/HTML removal, non-alphanumeric cleanup, whitespace normalization
- Production vectorizer: `HashingVectorizer` with unigrams + bigrams, English stop-word removal, `n_features=2**18`, `alternate_sign=False`
- Classifier: `LinearSVC(C=1.0, class_weight="balanced")`
- The accompanying notebook also contains a validated TF-IDF + LinearSVC experiment; the deployed artifact uses HashingVectorizer so the model can be packaged compactly without storing a large vocabulary.
- Split: stratified 80/20, `random_state=42`

The deployed HashingVectorizer + LinearSVC artifact was tested at approximately **99.52% accuracy** on its held-out test split; the accompanying notebook's TF-IDF configuration reached approximately **99.60%**. This number is dataset-specific and should **not** be interpreted as real-world fact-checking accuracy. The classifier learns patterns present in this dataset; it does not independently verify whether a claim is true.

## Production notes

- Do not commit API secrets or `.env` files.
- For serious public deployment, restrict CORS to the exact frontend origin instead of `*`.
- Add authentication/rate limiting if exposing the prediction API publicly.
- Pin exact dependency versions after your deployment environment is validated.
- Consider retraining/evaluating on a temporally separated external dataset before making real-world claims.

## License

Application code is MIT-licensed. The included Kaggle dataset has its own **CC BY-NC-SA 4.0** license and is not relicensed by this repository.
