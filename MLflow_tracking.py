import pandas as pd
import mlflow
import mlflow.sklearn
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

# STEP 1: Load dataset
data = pd.read_csv("dataset/students_clean.csv")

# STEP 2: Separate features and target
X = data[["study_hours", "attendance", "previous_marks"]]
y = data["result"]

# STEP 3: Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# STEP 4: Set MLflow experiment
mlflow.set_experiment("Student_Performance_Prediction")

# STEP 5: Compare different Random Forest models
tree_values = [50, 100, 200]

for trees in tree_values:

    model = RandomForestClassifier(
        n_estimators=trees,
        random_state=42
    )

    with mlflow.start_run():

        # Train model
        model.fit(X_train, y_train)

        # Make predictions
        y_pred = model.predict(X_test)

        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)

        precision = precision_score(
            y_test, y_pred,
            pos_label="Pass",
            zero_division=0
        )

        recall = recall_score(
            y_test, y_pred,
            pos_label="Pass",
            zero_division=0
        )

        f1 = f1_score(
            y_test, y_pred,
            pos_label="Pass",
            zero_division=0
        )

        # Log parameters
        mlflow.log_param("model_name", "Random Forest")
        mlflow.log_param("n_estimators", trees)
        mlflow.log_param("random_state", 42)
        mlflow.log_param("test_size", 0.2)

        # Log metrics
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)
        mlflow.sklearn.log_model(
            model,
            name="student_performance_model",
            skops_trusted_types=["sklearn.tree._tree.Tree"]
        )
        if trees == 100:
            os.makedirs("models", exist_ok=True)
            joblib.dump(model, "models/student_models.pkl")

        print("Trees:", trees)
        print("Accuracy:", accuracy)
        print("Precision:", precision)
        print("Recall:", recall)
        print("F1 Score:", f1)
        print("--------------------------")

print("All MLflow experiments completed!")
from fastapi import FastAPI

app = FastAPI()

@app.post("/predict")
def predict_endpoint():
    return {"student_result": "fail"}