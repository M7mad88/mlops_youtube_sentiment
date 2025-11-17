# mlops_youtube_sentiment
# 🎬 YouTube Sentiment Insights (MLOps Project)

A fully reproducible Machine Learning project built with **Python**, **DVC**, **MLflow**, and **GitHub** following industry-standard MLOps practices.

This repository demonstrates:
- Version-controlled code (Git/GitHub)
- Version-controlled datasets (DVC)
- Experiment tracking and model registry (MLflow)
- Reproducible pipelines
- Standardized project structure used in real MLOps teams

---

## 📁 Project Structure

project/

│
├── data/

│ ├── raw/ # Raw data (tracked with DVC, not Git)
│ └── processed/ # Cleaned data for modeling
│
├── notebooks/ # EDA and experiments
│
├── src/ # Source code

│ ├── data/

│ │ └── preprocess.py

│ ├── model/

│ │ └── train.py

│ └── init.py
│
├── models/ # Trained models (DVC/MLflow artifacts)
│
├── tests/ # Unit tests
│
├── params.yaml # Parameters for reproducibility

├── requirements.txt # Python dependencies

├── dvc.yaml # DVC pipeline definition

├── README.md

└── .gitignore


---
## Technologies Used

| Component           | Technology              |
| ------------------- | ----------------------- |
| Programming         | Python 3.11             |
| ML Pipeline         | DVC                     |
| Experiment Tracking | MLflow                  |
| Modeling            | LightGBM                |
| Feature Extraction  | TF-IDF / scikit-learn   |
| Environment         | Conda                   |
| Dataset             | YouTube/Reddit Comments |


## 🛠️ Setup Instructions

###  Create Conda Environment

```bash
conda create -n youtube python=3.11 -y
conda activate youtube

pip install -r requirements.txt

## Running the Pipeline
### Preprocessing Data
python src/data/preprocess.py

### Training the Model (with MLflow Tracking)
python src/model/train.py

### Running the full DVC pipeline
dvc repro

👨‍💻 Author

Mohamed Adel Hafez