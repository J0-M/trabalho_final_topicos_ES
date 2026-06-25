import os
import copy
import pickle

import numpy as np
import pandas as pd

from datetime import datetime

from lightgbm import LGBMRegressor

from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score

DATA_PATH = "../data/taxi_demand_processed.parquet"
MODEL_DIR = "../models/lightgbm"
MODEL_NAME = "lightgbm"

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

    r2 = r2_score(y_true, y_pred)
    rae_score = rae(y_true, y_pred)

    return {
        "RMSE": rmse,
        "R2": r2,
        "RAE": rae_score
    }

def train_model(model):
    
    model_path = os.path.join(MODEL_DIR, f"{MODEL_NAME}.pkl")
    results_path = os.path.join(MODEL_DIR, "results.pkl")

    if os.path.exists(model_path):
        
        with open(model_path, "rb") as f:
            model = pickle.load(f)

        print("Modelo carregado com sucesso.")

        if os.path.exists(results_path):

            with open(results_path, "rb") as f:
                results = pickle.load(f)

            print("\nResultados armazenados:")

            for key, value in results.items():
                print(f"{key}: {value}")

        return model

    print("Carregando dataset...")
    df = pd.read_parquet(DATA_PATH)
    print(f"Total de registros: {len(df)}")

    X = df.drop(
        columns=[
            "pickup_hour",
            "demand"
        ]
    )

    y = df["demand"]


    kf = KFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    rmse_scores = []
    r2_scores = []
    rae_scores = []

    best_model = None
    best_r2 = -np.inf

    for fold, (train_idx, test_idx) in enumerate(kf.split(X), start=1):

        print(f"\n===== Fold {fold} =====")

        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]

        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]

        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        metrics = calculate_metrics(y_test, predictions)

        rmse_scores.append(metrics["RMSE"])
        r2_scores.append(metrics["R2"])
        rae_scores.append(metrics["RAE"])

        print(f"RMSE: {metrics['RMSE']:.4f}")
        print(f"R2:   {metrics['R2']:.4f}")
        print(f"RAE:  {metrics['RAE']:.4f}")

        if metrics["R2"] > best_r2:
            best_r2 = metrics["R2"]
            best_model = copy.deepcopy(model)


    print("\n===== RESULTADO FINAL =====")

    print(f"RMSE médio: {np.mean(rmse_scores):.4f} ± {np.std(rmse_scores):.4f}")
    print(f"R2 médio: {np.mean(r2_scores):.4f} ± {np.std(r2_scores):.4f}")
    print(f"RAE médio: {np.mean(rae_scores):.4f} ± {np.std(rae_scores):.4f}")

    with open(model_path, "wb") as f:
        pickle.dump(best_model, f)

    results = {

        "RMSE_mean": float(np.mean(rmse_scores)),
        "RMSE_std": float(np.std(rmse_scores)),

        "R2_mean": float(np.mean(r2_scores)),
        "R2_std": float(np.std(r2_scores)),

        "RAE_mean": float(np.mean(rae_scores)),
        "RAE_std": float(np.std(rae_scores)),

        "best_R2": float(best_r2),

        "n_estimators": model.n_estimators,
        "learning_rate": model.learning_rate,
        "max_depth": model.max_depth,
        "num_leaves": model.num_leaves,
        "subsample": model.subsample,
        "colsample_bytree": model.colsample_bytree
    }

    with open(results_path, "wb") as f:
        pickle.dump(results, f)

    print(f"\nMelhor R²: {best_r2:.4f}")
    print(f"Modelo salvo em: {model_path}")
    print(f"Resultados salvos em: {results_path}")

    return results
    
def main():

    model = LGBMRegressor(
        objective="regression",
        n_estimators=100,
        learning_rate=0.05,
        max_depth=8,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        verbosity=-1
    )
    
    results = train_model(model=model)

    return results


if __name__ == "__main__":
    startTime = datetime.now()
    main()
    endTime = datetime.now()
    print("\nTempo de execução = ", endTime - startTime)