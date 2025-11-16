import os
import json
import yaml
import logging
import pickle
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


# -----------------------------------------------------
# Logging
# -----------------------------------------------------
logger = logging.getLogger("model_evaluation")
logger.setLevel(logging.DEBUG)

ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)

fh = logging.FileHandler("model_evaluation_errors.log")
fh.setLevel(logging.ERROR)

fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
ch.setFormatter(fmt)
fh.setFormatter(fmt)

logger.addHandler(ch)
logger.addHandler(fh)


# -----------------------------------------------------
# Helpers
# -----------------------------------------------------
def get_root():
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))


def load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


# -----------------------------------------------------
# Main
# -----------------------------------------------------
def main():
    ROOT = get_root()
    params = load_yaml(os.path.join(ROOT, "params.yaml"))

    # Prefer env var over params.yaml
    mlflow_uri = os.environ.get("MLFLOW_TRACKING_URI") or params.get("mlflow", {}).get("tracking_uri")
    if mlflow_uri:
        mlflow.set_tracking_uri(mlflow_uri)

    experiment_name = params.get("mlflow", {}).get("experiment_name", "default-experiment")
    mlflow.set_experiment(experiment_name)

    # Start MLflow run
    with mlflow.start_run():
        try:
            # -----------------------------
            # Load processed test data
            # -----------------------------
            test_path = os.path.join(ROOT, params["preprocessing"]["processed_test_path"])
            test_df = pd.read_csv(test_path)
            logger.info("Loaded test data: %s rows", len(test_df))

            # -----------------------------
            # Load TF-IDF vectorizer & model
            # -----------------------------
            models_dir = os.path.join(ROOT, "models")
            ensure_dir(models_dir)

            model_path = os.path.join(models_dir, "lgbm_model.pkl")
            vectorizer_path = os.path.join(models_dir, "tfidf_vectorizer.pkl")

            with open(model_path, "rb") as f:
                model = pickle.load(f)
            with open(vectorizer_path, "rb") as f:
                vectorizer = pickle.load(f)

            # -----------------------------
            # Transform test data
            # -----------------------------
            X_test_text = test_df["clean_comment"].astype(str).values
            y_test = test_df["category"].values

            X_test_tfidf = vectorizer.transform(X_test_text)

            # -----------------------------
            # Predict & evaluate
            # -----------------------------
            y_pred = model.predict(X_test_tfidf)

            report_dict = classification_report(y_test, y_pred, output_dict=True)
            cm = confusion_matrix(y_test, y_pred)

            # -----------------------------
            # Log MLflow metrics
            # -----------------------------
            for label, metrics in report_dict.items():
                if isinstance(metrics, dict):
                    mlflow.log_metric(f"{label}_precision", metrics.get("precision", 0.0))
                    mlflow.log_metric(f"{label}_recall", metrics.get("recall", 0.0))
                    mlflow.log_metric(f"{label}_f1", metrics.get("f1-score", 0.0))
                else:
                    mlflow.log_metric("accuracy", float(metrics))

            # Save classification report artifact
            report_path = os.path.join(models_dir, "classification_report.json")
            with open(report_path, "w") as f:
                json.dump(report_dict, f, indent=2)
            mlflow.log_artifact(report_path)

            # -----------------------------
            # Confusion matrix
            # -----------------------------
            cm_path = os.path.join(models_dir, "confusion_matrix_test.png")
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
            plt.title("Confusion Matrix - Test")
            plt.xlabel("Predicted")
            plt.ylabel("Actual")
            plt.tight_layout()
            plt.savefig(cm_path)
            mlflow.log_artifact(cm_path)
            plt.close()

            # -----------------------------
            # Signature + Input example
            # -----------------------------
            sample = X_test_tfidf[:3]
            input_example = pd.DataFrame(sample.toarray(), columns=vectorizer.get_feature_names_out())
            signature = infer_signature(input_example, model.predict(sample))

            mlflow.sklearn.log_model(
                model,
                artifact_path="model",
                signature=signature,
                input_example=input_example
            )

            mlflow.log_artifact(vectorizer_path, artifact_path="vectorizer")

            # -----------------------------
            # Save run info
            # -----------------------------
            run_info = {
                "run_id": mlflow.active_run().info.run_id,
                "model_artifact_path": "model"
            }

            run_info_path = os.path.join(models_dir, "run_info.json")
            with open(run_info_path, "w") as f:
                json.dump(run_info, f)
            mlflow.log_artifact(run_info_path)

            logger.info("Model evaluation completed successfully.")

        except Exception as e:
            logger.exception("Model evaluation failed: %s", e)
            raise


if __name__ == "__main__":
    main()
