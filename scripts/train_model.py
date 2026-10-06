from pathlib import Path
import re
import zipfile
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "model"
MODEL_DIR.mkdir(exist_ok=True)

def extract_zip(zip_path: Path):
    target = DATA_DIR / zip_path.stem
    target.mkdir(exist_ok=True)
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(target)
    csvs = list(target.glob("*.csv"))
    if not csvs:
        raise FileNotFoundError(f"No CSV found in {zip_path}")
    return csvs[0]

def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()

def load_data():
    fake_csv = extract_zip(DATA_DIR / "Fake.csv.zip")
    true_csv = extract_zip(DATA_DIR / "True.csv.zip")
    required = {"title", "text", "subject", "date"}
    fake = pd.read_csv(fake_csv)
    true = pd.read_csv(true_csv)
    for name, frame in [("Fake.csv", fake), ("True.csv", true)]:
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"{name} missing columns: {sorted(missing)}")
    fake["category"] = 1
    true["category"] = 0
    df = pd.concat([fake, true], ignore_index=True)
    df["content"] = (df["title"].fillna("") + " " + df["text"].fillna("")).map(clean_text)
    df = df[df["content"].str.len() >= 20].drop_duplicates(subset=["content"]).reset_index(drop=True)
    return df

def main():
    df = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        df["content"], df["category"], test_size=0.20, random_state=42, stratify=df["category"]
    )
    pipeline = Pipeline([
        ("tfidf", HashingVectorizer(
            stop_words="english", ngram_range=(1, 2), n_features=2**18,
            alternate_sign=False, norm="l2"
        )),
        ("clf", LinearSVC(C=1.0, class_weight="balanced", random_state=42)),
    ])
    pipeline.fit(X_train, y_train)
    pred = pipeline.predict(X_test)
    print(f"Rows: {len(df):,}")
    print(f"Accuracy: {accuracy_score(y_test, pred):.4%}")
    print(classification_report(y_test, pred, target_names=["REAL", "FAKE"]))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, pred))
    output = MODEL_DIR / "fake_news_detector.joblib"
    joblib.dump(pipeline, output, compress=3)
    print(f"Saved model: {output}")

if __name__ == "__main__":
    main()
