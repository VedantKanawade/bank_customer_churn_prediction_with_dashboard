import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier

from data_loader import load_data
from evaluate import evaluate_model


DATA_PATH = "../data/processed/engineered_data.csv"


def train_logistic():

    df = load_data(DATA_PATH)
    
    leakage_cols = [
    "Complain",
    "Risk_Score"]

    df = df.drop(columns=leakage_cols)


    X = df.drop(["Exited" , "Geography_Spain" ], axis=1)
    y = df["Exited"]

    # Train-test split with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # Pipeline: scaling + logistic regression
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            solver="liblinear"
        ))
    ])

    print("\nTraining Logistic Regression...")
    pipeline.fit(X_train, y_train)

    roc_auc = evaluate_model(pipeline, X_test, y_test)

    # Save model
    joblib.dump(pipeline, "../models/logistic_model.pkl")

    print("\nLogistic Regression model saved!")

    return roc_auc




## Random Forest Model 
def train_random_forest():

    df = load_data(DATA_PATH)

    leakage_cols = [
        "Complain",
        "Risk_Score"
    ]

    df = df.drop(columns=leakage_cols)

    X = df.drop("Exited", axis=1)
    y = df["Exited"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    rf = RandomForestClassifier(
        random_state=42,
        class_weight="balanced",
        n_jobs=-1
    )

    param_grid = {
        "n_estimators": [200, 400],
        "max_depth": [8, 12, None],
        "min_samples_split": [2, 5],
        "min_samples_leaf": [1, 3]
    }

    print("\nRunning GridSearch for Random Forest...")

    grid = GridSearchCV(
        rf,
        param_grid,
        cv=3,
        scoring="roc_auc",
        verbose=2,
        n_jobs=-1
    )

    grid.fit(X_train, y_train)

    best_rf = grid.best_estimator_

    print("\nBest Parameters:", grid.best_params_)

    roc_auc = evaluate_model(best_rf, X_test, y_test)

    joblib.dump(best_rf, "../models/random_forest_model.pkl")

    print("\nRandom Forest model saved!")

    return roc_auc

# XGBoost Model 

def train_xgboost():

    df = load_data(DATA_PATH)

    # Drop any post-event/leaky features
    leakage_cols = [
        "Complain",
        "Risk_Score"
    ]
    df = df.drop(columns=leakage_cols)

    X = df.drop("Exited", axis=1)
    y = df["Exited"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

    xgb_model = XGBClassifier(
        n_estimators=700,   # increase trees
        max_depth=8,        # deeper trees
        learning_rate=0.03, # slower boosting
        subsample=0.9,
        colsample_bytree=0.9,
        scale_pos_weight=scale_pos_weight,
        gamma=1,            # adds regularization
        reg_alpha=0.1,      # L1 regularization
        reg_lambda=1,       # L2 regularization
        random_state=42
    )
    print("\nTraining XGBoost...")

    # Just fit without eval_metric or early stopping
    xgb_model.fit(X_train, y_train)

    roc_auc = evaluate_model(xgb_model, X_test, y_test)

    joblib.dump(xgb_model, "../models/xgboost_model.pkl")

    print("\nXGBoost model saved!")

    return roc_auc



if __name__ == "__main__":

    log_auc = train_logistic()
    rf_auc = train_random_forest()
    xgb_auc = train_xgboost()

    print("\nFINAL MODEL COMPARISON")
    print("----------------------")
    print("Logistic ROC-AUC:", log_auc)
    print("Random Forest ROC-AUC:", rf_auc)
    print("XGBoost ROC-AUC:", xgb_auc)