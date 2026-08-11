import joblib
import pandas as pd

from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer


BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "models"

MODELS_DIR.mkdir(parents=True, exist_ok=True)


# Load datasets
train_df = pd.read_csv(DATA_DIR / "train_clean.csv")
validate_df = pd.read_csv(DATA_DIR / "validate_clean.csv")
test_df = pd.read_csv(DATA_DIR / "test_clean.csv")


# Use clean_text if available
train_text = train_df["clean_text"]
validate_text = validate_df["clean_text"]
test_text = test_df["clean_text"]


# Create TF-IDF vectorizer
tfidf = TfidfVectorizer(
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.95,
    max_features=50000,
    sublinear_tf=True
)


# Fit only on training data
X_train = tfidf.fit_transform(train_text)

# Transform validation and test data
X_validate = tfidf.transform(validate_text)
X_test = tfidf.transform(test_text)


# Labels
y_train = train_df["label"]
y_validate = validate_df["label"]
y_test = test_df["label"]


# Save features
joblib.dump(X_train, MODELS_DIR / "X_train.pkl")
joblib.dump(X_validate, MODELS_DIR / "X_validate.pkl")
joblib.dump(X_test, MODELS_DIR / "X_test.pkl")

# Save labels
joblib.dump(y_train, MODELS_DIR / "y_train.pkl")
joblib.dump(y_validate, MODELS_DIR / "y_validate.pkl")
joblib.dump(y_test, MODELS_DIR / "y_test.pkl")

# Save TF-IDF vectorizer
joblib.dump(tfidf, MODELS_DIR / "tfidf_vectorizer.pkl")


print("Feature engineering completed.")
print("X_train:", X_train.shape)
print("X_validate:", X_validate.shape)
print("X_test:", X_test.shape)