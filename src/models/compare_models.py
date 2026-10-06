from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler,
)

from xgboost import XGBClassifier


DATA_PATH = Path(
    "data/processed/brfss_diabetes_2024.csv"
)

REPORT_DIR = Path(
    "reports/tables"
)


NUMERIC_FEATURES = [
    "age",
    "bmi",
]


CATEGORICAL_FEATURES = [
    "sex",
    "physical_activity",
    "smoking_status",
    "alcohol_use",
    "education",
    "income",
    "race_group",
]


TARGET = "diabetes"


def load_data():
    df = pd.read_csv(DATA_PATH)

    features = (
        NUMERIC_FEATURES
        + CATEGORICAL_FEATURES
    )

    X = df[features].copy()
    y = df[TARGET].copy()

    print("=" * 80)
    print("DIABETES MODEL COMPARISON")
    print("=" * 80)

    print(
        f"Observations: {len(df):,}"
    )

    print(
        f"Positive class: "
        f"{y.mean() * 100:.2f}%"
    )

    return X, y


def split_data(X, y):
    X_train, X_temp, y_train, y_temp = (
        train_test_split(
            X,
            y,
            test_size=0.30,
            random_state=42,
            stratify=y,
        )
    )

    X_val, X_test, y_val, y_test = (
        train_test_split(
            X_temp,
            y_temp,
            test_size=0.50,
            random_state=42,
            stratify=y_temp,
        )
    )

    print()
    print("=" * 80)
    print("DATA SPLIT")
    print("=" * 80)

    print(
        f"Train:      {len(X_train):,}"
    )

    print(
        f"Validation: {len(X_val):,}"
    )

    print(
        f"Test:       {len(X_test):,}"
    )

    return (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
    )


def build_preprocessor(
    scale_numeric=False,
):
    numeric_steps = [
        (
            "imputer",
            SimpleImputer(
                strategy="median"
            ),
        )
    ]

    if scale_numeric:
        numeric_steps.append(
            (
                "scaler",
                StandardScaler(),
            )
        )

    numeric_pipeline = Pipeline(
        steps=numeric_steps
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="most_frequent"
                ),
            ),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    drop="first",
                ),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_pipeline,
                NUMERIC_FEATURES,
            ),
            (
                "categorical",
                categorical_pipeline,
                CATEGORICAL_FEATURES,
            ),
        ]
    )


def build_models():
    logistic = Pipeline(
        steps=[
            (
                "preprocessor",
                build_preprocessor(
                    scale_numeric=True
                ),
            ),
            (
                "model",
                LogisticRegression(
                    max_iter=1000,
                    random_state=42,
                ),
            ),
        ]
    )

    random_forest = Pipeline(
        steps=[
            (
                "preprocessor",
                build_preprocessor(
                    scale_numeric=False
                ),
            ),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=200,
                    max_depth=15,
                    min_samples_leaf=10,
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    xgboost = Pipeline(
        steps=[
            (
                "preprocessor",
                build_preprocessor(
                    scale_numeric=False
                ),
            ),
            (
                "model",
                XGBClassifier(
                    n_estimators=300,
                    max_depth=5,
                    learning_rate=0.05,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    objective="binary:logistic",
                    eval_metric="logloss",
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    return {
        "Logistic Regression": logistic,
        "Random Forest": random_forest,
        "XGBoost": xgboost,
    }


def find_best_threshold(
    y_true,
    probabilities,
):
    thresholds = np.arange(
        0.05,
        0.51,
        0.01,
    )

    best_threshold = 0.50
    best_f1 = -1.0

    for threshold in thresholds:
        predictions = (
            probabilities >= threshold
        ).astype(int)

        score = f1_score(
            y_true,
            predictions,
        )

        if score > best_f1:
            best_f1 = score
            best_threshold = threshold

    return (
        float(best_threshold),
        float(best_f1),
    )


def evaluate_model(
    name,
    model,
    X_train,
    y_train,
    X_val,
    y_val,
    X_test,
    y_test,
):
    print()
    print("=" * 80)
    print(name.upper())
    print("=" * 80)

    start = perf_counter()

    print("Training...")

    model.fit(
        X_train,
        y_train,
    )

    training_seconds = (
        perf_counter() - start
    )

    print(
        f"Training time: "
        f"{training_seconds:.2f} seconds"
    )

    val_probabilities = (
        model.predict_proba(
            X_val
        )[:, 1]
    )

    (
        threshold,
        validation_f1,
    ) = find_best_threshold(
        y_val,
        val_probabilities,
    )

    print(
        f"Best validation threshold: "
        f"{threshold:.2f}"
    )

    print(
        f"Validation F1: "
        f"{validation_f1:.4f}"
    )

    test_probabilities = (
        model.predict_proba(
            X_test
        )[:, 1]
    )

    predictions = (
        test_probabilities
        >= threshold
    ).astype(int)

    roc_auc = roc_auc_score(
        y_test,
        test_probabilities,
    )

    pr_auc = average_precision_score(
        y_test,
        test_probabilities,
    )

    precision = precision_score(
        y_test,
        predictions,
    )

    recall = recall_score(
        y_test,
        predictions,
    )

    f1 = f1_score(
        y_test,
        predictions,
    )

    matrix = confusion_matrix(
        y_test,
        predictions,
    )

    tn, fp, fn, tp = matrix.ravel()

    specificity = (
        tn / (tn + fp)
    )

    print()
    print("Test metrics:")

    print(
        f"ROC-AUC:     {roc_auc:.4f}"
    )

    print(
        f"PR-AUC:      {pr_auc:.4f}"
    )

    print(
        f"Precision:   {precision:.4f}"
    )

    print(
        f"Recall:      {recall:.4f}"
    )

    print(
        f"Specificity: {specificity:.4f}"
    )

    print(
        f"F1-score:    {f1:.4f}"
    )

    print()
    print("Confusion matrix:")
    print(matrix)

    result = {
        "model": name,
        "threshold": threshold,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "training_seconds": training_seconds,
    }

    return result


def main():
    X, y = load_data()

    (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
    ) = split_data(
        X,
        y,
    )

    models = build_models()

    results = []

    for name, model in models.items():
        result = evaluate_model(
            name=name,
            model=model,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            X_test=X_test,
            y_test=y_test,
        )

        results.append(result)

    comparison = pd.DataFrame(
        results
    )

    comparison = comparison.sort_values(
        "roc_auc",
        ascending=False,
    )

    print()
    print("=" * 100)
    print("FINAL MODEL COMPARISON")
    print("=" * 100)

    columns = [
        "model",
        "threshold",
        "roc_auc",
        "pr_auc",
        "precision",
        "recall",
        "specificity",
        "f1",
        "training_seconds",
    ]

    print(
        comparison[
            columns
        ].to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        REPORT_DIR
        / "model_comparison_2024.csv"
    )

    comparison.to_csv(
        output_path,
        index=False,
    )

    print()
    print(
        f"Results saved: "
        f"{output_path}"
    )


if __name__ == "__main__":
    main()
