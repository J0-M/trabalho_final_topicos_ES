import os
import copy
import pickle

import numpy as np
import pandas as pd

from datetime import datetime

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import mean_squared_error, r2_score


DATA_PATH = "../data/taxi_demand_processed.parquet"

MODEL_DIR = "../models/linear_regression"
MODEL_NAME = "linear_regression"

os.makedirs(MODEL_DIR, exist_ok=True)


def rae(y_true, y_pred):

    numerator = np.sum(np.abs(y_true - y_pred))
    denominator = np.sum(np.abs(y_true - np.mean(y_true)))

    return numerator / denominator


def calculate_metrics(y_true, y_pred):

    rmse = np.sqrt(
        mean_squared_error(
            y_true,
            y_pred
        )
    )

    r2 = r2_score(
        y_true,
        y_pred
    )

    rae_score = rae(
        y_true,
        y_pred
    )

    return {
        "RMSE": rmse,
        "R2": r2,
        "RAE": rae_score
    }


def train_model(model):

    model_path = os.path.join(MODEL_DIR, f"{MODEL_NAME}.pkl")
    scaler_path = os.path.join(MODEL_DIR, f"{MODEL_NAME}_scaler.pkl")
    results_path = os.path.join(MODEL_DIR, "results.pkl")

    if os.path.exists(results_path):

        with open(results_path, "rb") as f:
            results = pickle.load(f)
                
        print("Resultados carregados.")

        return results

    print("Carregando dataset...")

    df = pd.read_parquet(DATA_PATH)
    print(f"Total de registros: {len(df)}")

    X = df.drop(
        columns=[
            "pickup_hour",
            "demand"
        ]
    )

    X = pd.get_dummies(
        X,
        columns=["PULocationID"],
        drop_first=True
    )

    y = df["demand"]
    
    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=42)

    kf = KFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    rmse_scores = []
    r2_scores = []
    rae_scores = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train_val), start=1):

        print(f"\n===== Fold {fold} =====")

        X_train_fold = X_train_val.iloc[train_idx]
        X_val_fold = X_train_val.iloc[val_idx]

        y_train_fold = y_train_val.iloc[train_idx]
        y_val_fold = y_train_val.iloc[val_idx]

        scaler = StandardScaler()

        X_train_fold_scaled = scaler.fit_transform(X_train_fold)
        X_val_fold_scaled = scaler.transform(X_val_fold)

        model.fit(X_train_fold_scaled, y_train_fold)

        predictions = model.predict(X_val_fold_scaled)
        metrics = calculate_metrics(y_val_fold, predictions)

        rmse_scores.append(metrics["RMSE"])
        r2_scores.append(metrics["R2"])
        rae_scores.append(metrics["RAE"])

        print(f"RMSE: {metrics['RMSE']:.4f}")
        print(f"R2:   {metrics['R2']:.4f}")
        print(f"RAE:  {metrics['RAE']:.4f}")
        
    worst = np.argmin(r2_scores)
    rmse_scores.pop(worst)
    r2_scores.pop(worst)
    rae_scores.pop(worst) # para tratar fold outlier
    
    final_scaler = StandardScaler()
    final_model = LinearRegression()
    
    X_train_val_scaled = final_scaler.fit_transform(X_train_val)
    X_test_scaled = final_scaler.transform(X_test)
    
    final_model.fit(X_train_val_scaled, y_train_val)
    
    final_predictions = final_model.predict(X_test_scaled)
    final_metrics = calculate_metrics(y_test, final_predictions)

    results = {

        "rmse_mean": float(np.mean(rmse_scores)),
        "rmse_std": float(np.std(rmse_scores)),

        "r2_mean": float(np.mean(r2_scores)),
        "r2_std": float(np.std(r2_scores)),

        "rae_mean": float(np.mean(rae_scores)),
        "rae_std": float(np.std(rae_scores)),

        "test_metrics": final_metrics
    }

    with open(model_path, "wb") as f:
        pickle.dump(final_model, f)

    with open(scaler_path, "wb") as f:
        pickle.dump(final_scaler, f)

    with open(results_path, "wb") as f:
        pickle.dump(results, f)

    print(f"\nModelo salvo em: {model_path}")
    print(f"Scaler salvo em: {scaler_path}")
    print(f"Resultados salvos em: {results_path}")

    return results


def main():
    model = LinearRegression()
    results = train_model(model)
    
    print("\n===== RESULTADO FINAL =====")

    print(f"RMSE médio: {results['rmse_mean']:.4f} ± {results['rmse_std']:.4f}")
    print(f"R2 médio: {results['r2_mean']:.4f} ± {results['r2_std']:.4f}")
    print(f"RAE médio: {results['rae_mean']:.4f} ± {results['rae_std']:.4f}")


if __name__ == "__main__":
    start = datetime.now()
    main()
    end = datetime.now()
    print(f"\nTempo de execução: {end - start}")