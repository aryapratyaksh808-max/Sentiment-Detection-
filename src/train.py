import joblib

from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

from src.feature_engineering import build_and_save_features, load_features

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)


def train_model():
    """Build TF-IDF features, train the logistic regression model, and save it."""
    _, X_train, _, _, y_train, _, _ = build_and_save_features()

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    model_path = MODELS_DIR / "logistic_regression_tfidf.pkl"
    joblib.dump(model, model_path)

    return model_path


def load_model():
    """Load the trained model from disk."""
    model_path = MODELS_DIR / "logistic_regression_tfidf.pkl"
    return joblib.load(model_path)


def evaluate_model():
    """Evaluate the trained model on train, validate, and test splits."""
    model = load_model()
    X_train, X_validate, X_test, y_train, y_validate, y_test = load_features()

    metrics = {}

    for split_name, X_split, y_split in [
        ("train", X_train, y_train),
        ("validate", X_validate, y_validate),
        ("test", X_test, y_test),
    ]:
        y_pred = model.predict(X_split)
        metrics[split_name] = {
            "accuracy": accuracy_score(y_split, y_pred),
            "classification_report": classification_report(
                y_split,
                y_pred,
                output_dict=True,
                zero_division=0,
            ),
        }

    return metrics
