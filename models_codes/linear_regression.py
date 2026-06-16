import os
import pickle

import numpy as np
import pandas as pd

from datetime import datetime

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score

DATA_PATH = "../data/taxi_demand_processed.parquet"
MODEL_DIR = "../models/linear_regression"
MODEL_NAME = "linear_regression"
SCALER_NAME = "linear_regression_scaler"

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
    print("Carregando dataset...")
    
    df = pd.read_parquet(DATA_PATH)
    
    # print(df["demand"].describe())
    
    print(f"Total de registros: {len(df)}")
    
    ################

    X = df.drop(columns=["demand", "pickup_hour"])
    X = pd.get_dummies(
        X,
        columns=["PULocationID"],
        drop_first=True
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
    best_scaler = None
    best_r2 = -np.inf
    
    for fold, (train_idx, test_idx) in enumerate(kf.split(X), start=1):

        print(f"\n===== Fold {fold} =====")

        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]

        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        model.fit(X_train, y_train)

        predictions = model.predict(X_test)
        metrics = calculate_metrics(y_test, predictions)

        rmse_scores.append(metrics["RMSE"])
        r2_scores.append(metrics["R2"])
        rae_scores.append(metrics["RAE"])

        print(f"RMSE: {metrics['RMSE']:.4f}")
        print(f"R2: {metrics['R2']:.4f}")
        print(f"RAE: {metrics['RAE']:.4f}")

        if metrics["R2"] > best_r2:
            best_r2 = metrics["R2"]
            best_model = model
            best_scaler = scaler
            
    print("\n===== RESULTADO FINAL =====")

    print(f"    RMSE médio: {np.mean(rmse_scores):.4f} ± {np.std(rmse_scores):.4f}")
    print(f"    R2 médio: {np.mean(r2_scores):.4f} ± {np.std(r2_scores):.4f}")
    print(f"    RAE médio: {np.mean(rae_scores):.4f} ± {np.std(rae_scores):.4f}")
        
    model_path = os.path.join(MODEL_DIR, f"{MODEL_NAME}.pkl")
    scaler_path = os.path.join(MODEL_DIR, f"{MODEL_NAME}_scaler.pkl")

    with open(model_path, "wb") as f:
        pickle.dump(best_model, f)

    with open(scaler_path, "wb") as f:
        pickle.dump(best_scaler, f)

    print(f"Modelo salvo em: {model_path}")
    print(f"Scaler salvo em: {scaler_path}")

    return {
        "rmse": rmse_scores,
        "r2": r2_scores,
        "rae": rae_scores
    }
    
def main():

    model = LinearRegression()
    results = train_model(model=model)

    return results


if __name__ == "__main__":
    startTime = datetime.now()
    main()
    endTime = datetime.now()
    print("\nTempo de execução = ", endTime - startTime)