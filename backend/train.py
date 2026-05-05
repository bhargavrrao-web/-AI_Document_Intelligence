import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import mlflow
import mlflow.sklearn
from sentence_transformers import SentenceTransformer
import os

# Configuration
DATA_PATH = "data/document_dataset.csv"
MODEL_PATH = "trained_model.pkl"
EXPERIMENT_NAME = "custom_document_classification"

def train_model():
    print("🚀 Starting Custom Model Training...")
    
    # 1. Load Dataset
    if not os.path.exists(DATA_PATH):
        print(f"❌ Error: Dataset not found at {DATA_PATH}")
        return

    df = pd.read_csv(DATA_PATH)
    print(f"📊 Loaded {len(df)} samples from dataset.")

    # 2. Extract Features using Sentence Transformer
    print("🧠 Extracting text embeddings...")
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    X = embedder.encode(df['text'].tolist())
    y = df['label']

    # 3. Split Data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 4. Setup MLflow
    mlflow.set_tracking_uri("sqlite:///mlruns.db")
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run():
        print("🛠️ Training Random Forest Classifier...")
        # 5. Train Model
        n_estimators = 100
        clf = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
        clf.fit(X_train, y_train)

        # 6. Evaluate
        predictions = clf.predict(X_test)
        acc = accuracy_score(y_test, predictions)
        print(f"✅ Training Complete. Accuracy: {acc:.4f}")

        # 7. Log to MLflow
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_metric("accuracy", acc)
        mlflow.sklearn.log_model(clf, "model")

        # 8. Save Model Locally
        joblib.dump(clf, MODEL_PATH)
        print(f"💾 Model saved to {MODEL_PATH}")

if __name__ == "__main__":
    train_model()
