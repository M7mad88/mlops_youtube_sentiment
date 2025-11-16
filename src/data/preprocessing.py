import os
import re
import logging
import yaml
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


# -----------------------------------------
# Logging
# -----------------------------------------
logger = logging.getLogger("data_preprocessing")
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

file_handler = logging.FileHandler("preprocessing_errors.log")
file_handler.setLevel(logging.ERROR)

formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)


# -----------------------------------------
# Load params
# -----------------------------------------
def load_params():
    ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
    with open(os.path.join(ROOT_DIR, "params.yaml"), "r") as f:
        return yaml.safe_load(f)


# -----------------------------------------
# Text preprocessing logic
# -----------------------------------------
def preprocess_comment(comment: str) -> str:
    try:
        comment = comment.lower().strip()
        comment = re.sub(r"\n", " ", comment)
        comment = re.sub(r"[^A-Za-z0-9\s!?.,]", "", comment)

        stop_words = set(stopwords.words("english")) - {
            "not", "no", "but", "however", "yet"
        }
        comment = " ".join(
            [word for word in comment.split() if word not in stop_words]
        )

        lemmatizer = WordNetLemmatizer()
        comment = " ".join([lemmatizer.lemmatize(word) for word in comment.split()])

        return comment

    except Exception as e:
        logger.error(f"Preprocessing failed: {e}")
        return comment


def normalize_text(df: pd.DataFrame) -> pd.DataFrame:
    df["clean_comment"] = df["clean_comment"].astype(str)
    df["clean_comment"] = df["clean_comment"].apply(preprocess_comment)
    return df


# -----------------------------------------
# Save processed data
# -----------------------------------------
def save_processed(train_df, test_df, train_path, test_path):
    os.makedirs(os.path.dirname(train_path), exist_ok=True)
    os.makedirs(os.path.dirname(test_path), exist_ok=True)

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    logger.info("Processed data saved successfully.")


# -----------------------------------------
# Main
# -----------------------------------------
def main():
    logger.info("Starting preprocessing stage...")

    ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
    params = load_params()

    ingestion_cfg = params["data_ingestion"]
    preprocess_cfg = params["preprocessing"]

    # Load raw train/test
    train_df = pd.read_csv(os.path.join(ROOT, ingestion_cfg["train_path"]))
    test_df = pd.read_csv(os.path.join(ROOT, ingestion_cfg["test_path"]))

    train_df = normalize_text(train_df)
    test_df = normalize_text(test_df)

    save_processed(
        train_df,
        test_df,
        os.path.join(ROOT, preprocess_cfg["processed_train_path"]),
        os.path.join(ROOT, preprocess_cfg["processed_test_path"]),
    )

    logger.info("Preprocessing completed.")


if __name__ == "__main__":
    main()
