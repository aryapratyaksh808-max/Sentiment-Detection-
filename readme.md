# Sentiment Analysis Project

This repository contains a sentiment analysis pipeline for movie reviews.

## Structure

- `app.py` - CLI entrypoint for training, evaluation, and prediction.
- `src/feature_engineering.py` - Builds TF-IDF features and saves them.
- `src/train.py` - Trains a logistic regression model and evaluates it.
- `src/predict.py` - Loads the saved model and predicts text sentiment.
- `src/preprocessing.py` - Cleans raw text before prediction.
- `data/processed` - Cleaned CSV files used for training and evaluation.
- `models` - Saved feature matrices, vectorizer, and trained model.

## Usage

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

Train the model:

```bash
python app.py train
```

Evaluate the model:

```bash
python app.py evaluate
```

Predict sentiment:

```bash
python app.py predict "This movie was fantastic and entertaining."
```
