import os
import sys

from flask import Flask, render_template, request, jsonify
import click

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.predict import predict_sentiment
from src.train import evaluate_model, train_model

app = Flask(__name__)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict_api():
    data = request.json or {}
    text = data.get("text", "")

    if not text.strip():
        return jsonify({"error": "Please provide text to analyze."}), 400

    sentiment, confidence = predict_sentiment(text)
    return jsonify({
        "sentiment": sentiment,
        "confidence": round(confidence * 100, 2),
        "text": text,
    })


@click.group()
def cli():
    """Sentiment analysis project CLI."""
    pass


@cli.command()
def train():
    """Build features and train the sentiment model."""
    train_model()


@cli.command()
def evaluate():
    """Evaluate the saved model on train, validate, and test splits."""
    metrics = evaluate_model()

    for split_name, values in metrics.items():
        click.echo(f"\n=== {split_name.capitalize()} metrics ===")
        click.echo(f"Accuracy: {values['accuracy']:.4f}")
        click.echo("Classification report:")
        for label, stats in values["classification_report"].items():
            if label in ["accuracy", "macro avg", "weighted avg"]:
                continue
            click.echo(
                f"  {label}: precision={stats['precision']:.4f}, recall={stats['recall']:.4f}, f1={stats['f1-score']:.4f}"
            )


@cli.command()
@click.argument("text", nargs=-1)
def predict(text):
    """Predict sentiment for a text string."""
    if not text:
        raise click.UsageError("Please provide text to predict. Example: python app.py predict 'I love this movie.'")

    text = " ".join(text)
    sentiment, confidence = predict_sentiment(text)
    click.echo(f"Sentiment: {sentiment}")
    click.echo(f"Confidence: {confidence * 100:.2f}%")


@cli.command()
def serve():
    """Run the Flask web frontend."""
    app.run(debug=True)


if __name__ == "__main__":
    cli()
