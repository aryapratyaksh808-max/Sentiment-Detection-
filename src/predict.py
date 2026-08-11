import joblib

from pathlib import Path

from src.preprocessing import prepare_text

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"


def _load_model():
    model_path = MODELS_DIR / "logistic_regression_tfidf.pkl"
    return joblib.load(model_path)


def _load_vectorizer():
    vectorizer_path = MODELS_DIR / "tfidf_vectorizer.pkl"
    return joblib.load(vectorizer_path)


def predict_sentiment(text):
    """Predict sentiment for a single text sample."""
    model = _load_model()
    tfidf = _load_vectorizer()
    clean_text = prepare_text(text)
    features = tfidf.transform([clean_text])

    prediction = model.predict(features)[0]
    probabilities = model.predict_proba(features)[0]

    if prediction == 1:
        sentiment = "Positive"
        confidence = probabilities[1]
    else:
        sentiment = "Negative"
        confidence = probabilities[0]

    return sentiment, confidence
