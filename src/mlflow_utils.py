import pandas as pd
import time
import mlflow
import mlflow.sklearn

from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score


def load_data():
    wine = load_wine()
    df = pd.DataFrame(wine.data, columns=wine.feature_names)
    df["target"] = wine.target
    return df


def split_data(df):
    X = df.drop("target", axis=1)
    y = df["target"]

    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.2, random_state=42, stratify=y_train_full
    )

    print("Tamaño de X_train:", X_train.shape)
    print("Tamaño de X_val:", X_val.shape)
    print("Tamaño de X_test:", X_test.shape)

    return X_train, X_val, X_test, y_train, y_val, y_test


def train_model(X_train, y_train, learning_rate=0.1, max_depth=3, n_estimators=100):
    clf = GradientBoostingClassifier(
        learning_rate=learning_rate,
        max_depth=max_depth,
        n_estimators=n_estimators,
        random_state=42
    )

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", clf)
    ])

    pipeline.fit(X_train, y_train)
    return pipeline


def evaluate_model(model, X_eval, y_eval, dataset_name=""):
    """
    Evalúa el modelo y muestra el accuracy de train/test/val, según el dataset_name.
    """
    y_pred = model.predict(X_eval)
    accuracy = accuracy_score(y_eval, y_pred)
    print(f"Accuracy en {dataset_name}: {accuracy:.3f}")
    return accuracy

def train_and_log_models(nombre_experimento, X_train, X_val, y_train, y_val,
                         n_estimators_list, max_depth_list, learning_rate_list):
    mlflow.set_experiment(nombre_experimento)
    time.sleep(1)

    for n in n_estimators_list:
        for d in max_depth_list:
            for lr in learning_rate_list:
                with mlflow.start_run():
                    model = train_model(X_train, y_train,
                                        n_estimators=n,
                                        max_depth=d,
                                        learning_rate=lr)

                    acc = evaluate_model(model, X_val, y_val)

                    mlflow.log_param("n_estimators", n)
                    mlflow.log_param("max_depth", d)
                    mlflow.log_param("learning_rate", lr)
                    mlflow.log_metric("val_accuracy", acc)
                    mlflow.sklearn.log_model(model, "gradient_boosting_model")

    print("Gradient Boosting: entrenamiento y registro completado.")

    
