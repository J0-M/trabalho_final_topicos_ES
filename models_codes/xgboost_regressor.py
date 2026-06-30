import os
import copy
import pickle

import numpy as np
import pandas as pd

from datetime import datetime
from itertools import product

from xgboost import XGBRegressor # árvores de decisão não precisam de normalização, logo sem StandardScaler

from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score

DATA_PATH = "../data/taxi_demand_processed.parquet"
MODEL_DIR = "../models/xgboost"
MODEL_NAME = "xgboost"

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

def train_model(combinations):
    
    model_path = os.path.join(MODEL_DIR, f"{MODEL_NAME}.pkl")
    results_path = os.path.join(MODEL_DIR, "results.pkl")

    if os.path.exists(results_path):
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

    y = df["demand"]


    kf = KFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    best_model = None
    best_params = None
    best_metrics = None
    best_r2 = -np.inf
    
    for n_estimators, max_depth, learning_rate in combinations:
        print("\n==============================")
        print(
            f"Testando: "
            f"n_estimators={n_estimators}, "
            f"max_depth={max_depth}, "
            f"lr={learning_rate}"
        )
        
        model = XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="reg:squarederror",
            random_state=42,
            n_jobs=-1
        )

        fold_r2 = []
        fold_rmse = []
        fold_rae = []
        
        for fold, (train_idx, test_idx) in enumerate(kf.split(X), start=1):

            print(f"\n===== Fold {fold} =====")

            X_train = X.iloc[train_idx]
            X_test = X.iloc[test_idx]

            y_train = y.iloc[train_idx]
            y_test = y.iloc[test_idx]

            model.fit(X_train, y_train)
            predictions = model.predict(X_test)
            metrics = calculate_metrics(y_test, predictions)

            fold_rmse.append(metrics["RMSE"])
            fold_r2.append(metrics["R2"])
            fold_rae.append(metrics["RAE"])

            print(f"RMSE: {metrics['RMSE']:.4f}")
            print(f"R2:   {metrics['R2']:.4f}")
            print(f"RAE:  {metrics['RAE']:.4f}")
        
        mean_r2 = np.mean(fold_r2)
        print("R2 médio:", mean_r2)

        if mean_r2 > best_r2:
            best_r2 = mean_r2
            best_params = {
                "n_estimators": n_estimators,
                "max_depth": max_depth,
                "learning_rate": learning_rate
            }
            best_metrics = {
                "rmse_mean": np.mean(fold_rmse),
                "rmse_std": np.std(fold_rmse),

                "r2_mean": np.mean(fold_r2),
                "r2_std": np.std(fold_r2),

                "rae_mean": np.mean(fold_rae),
                "rae_std": np.std(fold_rae)
            }
            best_model = copy.deepcopy(model)

    with open(model_path, "wb") as f:
        pickle.dump(best_model, f)

    results = {
        "rmse_mean": best_metrics["rmse_mean"],
        "rmse_std": best_metrics["rmse_std"],

        "r2_mean": best_metrics["r2_mean"],
        "r2_std": best_metrics["r2_std"],

        "rae_mean": best_metrics["rae_mean"],
        "rae_std": best_metrics["rae_std"],

        "best_params": best_params
    }

    with open(results_path, "wb") as f:
        pickle.dump(results, f)

    print(f"\nMelhor R2: {best_r2:.4f}")
    print(f"Modelo salvo em: {model_path}")
    print(f"Resultados salvos em: {results_path}")

    return results
    
def main():
    
    param_grid = {
        "n_estimators": [50, 100],
        "max_depth": [6, 8],
        "learning_rate": [0.05, 0.1]
    }
    
    combinations = list(product(
        param_grid["n_estimators"],
        param_grid["max_depth"],
        param_grid["learning_rate"]
    ))
    
    results = train_model(combinations)
    
    print("\n===== RESULTADO FINAL =====")

    print(f"Melhores Parâmetros: {results['best_params']}")
    
    print(f"RMSE médio: {results['rmse_mean']:.4f} ± {results['rmse_std']:.4f}")
    print(f"R2 médio: {results['r2_mean']:.4f} ± {results['r2_std']:.4f}")
    print(f"RAE médio: {results['rae_mean']:.4f} ± {results['rae_std']:.4f}")

    return results


if __name__ == "__main__":
    startTime = datetime.now()
    main()
    endTime = datetime.now()
    print("\nTempo de execução = ", endTime - startTime)