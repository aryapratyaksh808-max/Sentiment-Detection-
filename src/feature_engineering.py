import joblib
import pandas as pd

from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)


def load_processed_data():
    train_df = pd.read_csv(DATA_DIR / "train_clean.csv")
    validate_df = pd.read_csv(DATA_DIR / "validate_clean.csv")
    test_df = pd.read_csv(DATA_DIR / "test_clean.csv")

    return train_df, validate_df, test_df


def build_tfidf_features(
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.95,
    max_features=50000,
    sublinear_tf=True,
):
    train_df, validate_df, test_df = load_processed_data()

    X_train = train_df["clean_text"]
    X_validate = validate_df["clean_text"]
    X_test = test_df["clean_text"]

    tfidf = TfidfVectorizer(
        ngram_range=ngram_range,
        min_df=min_df,
        max_df=max_df,
        max_features=max_features,
        sublinear_tf=sublinear_tf,
    )

    X_train_tfidf = tfidf.fit_transform(X_train)
    X_validate_tfidf = tfidf.transform(X_validate)
    X_test_tfidf = tfidf.transform(X_test)

    y_train = train_df["label"]
    y_validate = validate_df["label"]
    y_test = test_df["label"]

    return (
        tfidf,
        X_train_tfidf,
        X_validate_tfidf,
        X_test_tfidf,
        y_train,
        y_validate,
        y_test,
    )


def save_features(
    tfidf,
    X_train,
    X_validate,
    X_test,
    y_train,
    y_validate,
    y_test,
):
    joblib.dump(tfidf, MODELS_DIR / "tfidf_vectorizer.pkl")
    joblib.dump(X_train, MODELS_DIR / "X_train.pkl")
    joblib.dump(X_validate, MODELS_DIR / "X_validate.pkl")
    joblib.dump(X_test, MODELS_DIR / "X_test.pkl")
    joblib.dump(y_train, MODELS_DIR / "y_train.pkl")
    joblib.dump(y_validate, MODELS_DIR / "y_validate.pkl")
    joblib.dump(y_test, MODELS_DIR / "y_test.pkl")


def build_and_save_features(**kwargs):
    tfidf, X_train, X_validate, X_test, y_train, y_validate, y_test = build_tfidf_features(
        **kwargs
    )
    save_features(
        tfidf,
        X_train,
        X_validate,
        X_test,
        y_train,
        y_validate,
        y_test,
    )
    return (
        tfidf,
        X_train,
        X_validate,
        X_test,
        y_train,
        y_validate,
        y_test,
    )


def load_features():
    X_train = joblib.load(MODELS_DIR / "X_train.pkl")
    X_validate = joblib.load(MODELS_DIR / "X_validate.pkl")
    X_test = joblib.load(MODELS_DIR / "X_test.pkl")
    y_train = joblib.load(MODELS_DIR / "y_train.pkl")
    y_validate = joblib.load(MODELS_DIR / "y_validate.pkl")
    y_test = joblib.load(MODELS_DIR / "y_test.pkl")

    return X_train, X_validate, X_test, y_train, y_validate, y_test


def load_tfidf():
    return joblib.load(MODELS_DIR / "tfidf_vectorizer.pkl")
