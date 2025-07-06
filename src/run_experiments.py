from mlflow_utils import load_data, split_data, train_and_log_models

def main():
    print("Ejecutando entrenamiento con Gradient Boosting + MLflow")

    nombre_job = "Wine_GradientBoosting_MLflow"
    n_estimators_list = [50, 100]
    max_depth_list = [3, 5]
    learning_rate_list = [0.05, 0.1]

    df = load_data()
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df)

    train_and_log_models(
        nombre_experimento=nombre_job,
        X_train=X_train,
        X_val=X_val,
        y_train=y_train,
        y_val=y_val,
        n_estimators_list=n_estimators_list,
        max_depth_list=max_depth_list,
        learning_rate_list=learning_rate_list
    )

if __name__ == "__main__":
    main()
