import os
import pickle
import yaml
import logging
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import lightgbm as lgb


# -----------------------------------
# Logging
# -----------------------------------
logger = logging.getLogger("model_building")
logger.setLevel(logging.DEBUG)

ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)

fh = logging.FileHandler("model_building_errors.log")
fh.setLevel(logging.ERROR)

fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
ch.setFormatter(fmt)
fh.setFormatter(fmt)

logger.addHandler(ch)
logger.addHandler(fh)


# -----------------------------------
# Helpers
# -----------------------------------
def get_root():
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))


def load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


# -----------------------------------
# TF-IDF Feature Engineering
# -----------------------------------
def apply_tfidf(train_df, max_features, ngram_range, vectorizer_path):
    try:
        logger.info("Applying TF-IDF...")

        vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=tuple(ngram_range),
        )

        X = train_df["clean_comment"].astype(str).values
        y = train_df["category"].values

        X_tfidf = vectorizer.fit_transform(X)

        # Save vectorizer
        with open(vectorizer_path, "wb") as f:
            pickle.dump(vectorizer, f)

        logger.info("TF-IDF complete. Shape: %s", X_tfidf.shape)
        return X_tfidf, y

    except Exception as e:
        logger.exception("Failed during TF-IDF transformation: %s", e)
        raise


# -----------------------------------
# Train LightGBM
# -----------------------------------
def train_model(X_train, y_train, params):
    try:
        logger.info("Training LightGBM model...")

        model = lgb.LGBMClassifier(
            objective="multiclass",
            num_class=3,
            metric="multi_logloss",
            class_weight="balanced",
            learning_rate=params["learning_rate"],
            max_depth=params["max_depth"],
            n_estimators=params["n_estimators"],
            reg_alpha=0.1,
            reg_lambda=0.1,
        )

        model.fit(X_train, y_train)
        logger.info("Model training completed.")
        return model

    except Exception as e:
        logger.exception("Model training failed: %s", e)
        raise


# -----------------------------------
# Save Model
# -----------------------------------
def save_model(model, path):
    try:
        with open(path, "wb") as f:
            pickle.dump(model, f)
        logger.info("Model saved to %s", path)
    except Exception as e:
        logger.exception("Failed to save model: %s", e)
        raise


# -----------------------------------
# Main
# -----------------------------------
def main():
    ROOT = get_root()
    params = load_yaml(os.path.join(ROOT, "params.yaml"))

    train_path = os.path.join(ROOT, params["preprocessing"]["processed_train_path"])

    models_dir = os.path.join(ROOT, "models")
    ensure_dir(models_dir)

    model_path = os.path.join(models_dir, "lgbm_model.pkl")
    vectorizer_path = os.path.join(models_dir, "tfidf_vectorizer.pkl")

    # Load processed train data
    logger.info("Loading processed training data...")
    train_df = pd.read_csv(train_path)

    # TF-IDF
    X_train, y_train = apply_tfidf(
        train_df,
        params["model_building"]["max_features"],
        params["model_building"]["ngram_range"],
        vectorizer_path,
    )

    # Train
    model = train_model(X_train, y_train, params["model_building"])

    # Save model
    save_model(model, model_path)

    logger.info("MODEL BUILDING COMPLETED SUCCESSFULLY")


if __name__ == "__main__":
    main()
