import os
import yaml
import pandas as pd
from sklearn.model_selection import train_test_split
import logging

# ===============================
# Logging
# ===============================
logger = logging.getLogger("data_ingestion")
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
file_handler = logging.FileHandler("errors.log")
file_handler.setLevel(logging.ERROR)

formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)


# ===============================
# Utility: Load Params
# ===============================
def load_params(params_path: str):
    try:
        with open(params_path, "r") as f:
            params = yaml.safe_load(f)
        return params
    except Exception as e:
        logger.error(f"Failed to load params.yaml: {e}")
        raise


# ===============================
# Data Loading
# ===============================
def load_data(source_url: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(source_url)
        logger.info("Data loaded successfully.")
        return df
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        raise


# ===============================
# Preprocessing
# ===============================
def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    try:
        df.dropna(inplace=True)
        df.drop_duplicates(inplace=True)
        df = df[df["clean_comment"].str.strip() != ""]
        logger.info("Preprocessing completed.")
        return df
    except Exception as e:
        logger.error(f"Preprocessing failed: {e}")
        raise


# ===============================
# Save train/test data
# ===============================
def save_split(train_df, test_df, train_path, test_path):
    os.makedirs(os.path.dirname(train_path), exist_ok=True)
    os.makedirs(os.path.dirname(test_path), exist_ok=True)

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    logger.info("Train/Test saved successfully.")


# ===============================
# Main
# ===============================
def main():
    ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
    params = load_params(os.path.join(ROOT_DIR, "params.yaml"))

    ingestion_cfg = params["data_ingestion"]

    df = load_data(ingestion_cfg["source_url"])
    df = preprocess(df)

    train_df, test_df = train_test_split(
        df,
        test_size=ingestion_cfg["test_size"],
        random_state=42,
    )

    save_split(
        train_df,
        test_df,
        os.path.join(ROOT_DIR, ingestion_cfg["train_path"]),
        os.path.join(ROOT_DIR, ingestion_cfg["test_path"]),
    )


if __name__ == "__main__":
    main()
