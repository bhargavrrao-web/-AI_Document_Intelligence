import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import mlflow
import mlflow.pytorch
import os

# 1. Define the Neural Network Architecture (Deep Learning Concept)
class DocumentClassifierNet(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(DocumentClassifierNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_classes)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return self.softmax(x)

def train_dl_model():
    print("Starting Deep Learning (Neural Network) Training...")
    
    # 2. Data Preparation
    base_dir = os.path.dirname(__file__)
    data_path = os.path.join(base_dir, "data", "document_dataset.csv")
    df = pd.read_csv(data_path)
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    X = embedder.encode(df['text'].tolist())
    
    le = LabelEncoder()
    y = le.fit_transform(df['label'])
    num_classes = len(le.classes_)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Convert to PyTorch Tensors
    X_train_t = torch.FloatTensor(X_train)
    y_train_t = torch.LongTensor(y_train)
    X_test_t = torch.FloatTensor(X_test)
    y_test_t = torch.LongTensor(y_test)

    # 3. Initialize Model, Loss, and Optimizer
    model = DocumentClassifierNet(input_dim=384, num_classes=num_classes)
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 4. Setup MLflow
    mlflow.set_tracking_uri("sqlite:///mlruns.db")
    mlflow.set_experiment("deep_learning_training")

    with mlflow.start_run():
        print("Training the Neural Network...")
        # 5. Training Loop
        epochs = 50
        for epoch in range(epochs):
            optimizer.zero_grad()
            outputs = model(X_train_t)
            loss = criterion(outputs, y_train_t)
            loss.backward()
            optimizer.step()
            
            if (epoch+1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

        # 6. Evaluation
        model.eval()
        with torch.no_grad():
            test_outputs = model(X_test_t)
            _, predicted = torch.max(test_outputs, 1)
            accuracy = (predicted == y_test_t).sum().item() / y_test_t.size(0)
            print(f"DL Training Complete. Test Accuracy: {accuracy:.4f}")

        # 7. Log to MLflow
        mlflow.log_param("epochs", epochs)
        mlflow.log_param("optimizer", "Adam")
        mlflow.log_metric("accuracy", accuracy)
        mlflow.pytorch.log_model(model, "dl_model")
        
        # Save mappings for prediction
        torch.save({
            'model_state_dict': model.state_dict(),
            'classes': le.classes_,
            'input_dim': 384,
            'num_classes': num_classes
        }, "trained_model_dl.pth")
        print("Deep Learning model saved to trained_model_dl.pth")

if __name__ == "__main__":
    train_dl_model()
