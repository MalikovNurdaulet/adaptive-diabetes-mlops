from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
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


DATA_PATH = Path(
    "data/processed/brfss_diabetes_2024.csv"
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

    return X, y


def split_data(X, y):

    # 70% train
    # 30% temporary
    X_train, X_temp, y_train, y_temp = (
        train_test_split(
            X,
            y,
            test_size=0.30,
            random_state=42,
            stratify=y,
        )
    )

    # Temporary 30% -> 15% validation + 15% test
    X_val, X_test, y_val, y_test = (
        train_test_split(
            X_temp,
            y_temp,
            test_size=0.50,
            random_state=42,
            stratify=y_temp,
        )
    )

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


def build_pipeline():

    numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                ),
            ),
            (
                "scaler",
                StandardScaler(),
            ),
        ]
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

    preprocessor = ColumnTransformer(
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

    model = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor,
            ),
            (
                "model",
                LogisticRegression(
                    max_iter=1000,
                ),
            ),
        ]
    )

    return model


def evaluate_thresholds(
    y_true,
    probabilities,
):

    results = []

    thresholds = np.arange(
        0.05,
        0.51,
        0.01,
    )

    for threshold in thresholds:

        predictions = (
            probabilities >= threshold
        ).astype(int)

        precision = precision_score(
            y_true,
            predictions,
            zero_division=0,
        )

        recall = recall_score(
            y_true,
            predictions,
        )

        f1 = f1_score(
            y_true,
            predictions,
        )

        results.append(
            {
                "threshold": threshold,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        )

    return pd.DataFrame(results)


def choose_threshold(
    results,
):

    best_row = results.loc[
        results["f1"].idxmax()
    ]

    return float(
        best_row["threshold"]
    )


def final_evaluation(
    y_test,
    probabilities,
    threshold,
):

    predictions = (
        probabilities >= threshold
    ).astype(int)

    roc_auc = roc_auc_score(
        y_test,
        probabilities,
    )

    pr_auc = average_precision_score(
        y_test,
        probabilities,
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
    print("=" * 80)
    print("FINAL TEST RESULTS")
    print("=" * 80)

    print(
        f"Threshold:   {threshold:.2f}"
    )

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


def main():

    X, y = load_data()

    (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
    ) = split_data(X, y)

    model = build_pipeline()

    print()
    print("Training Logistic Regression...")

    model.fit(
        X_train,
        y_train,
    )

    print("Training completed.")

    val_probabilities = (
        model.predict_proba(
            X_val
        )[:, 1]
    )

    results = evaluate_thresholds(
        y_val,
        val_probabilities,
    )

    print()
    print("=" * 80)
    print("VALIDATION THRESHOLDS")
    print("=" * 80)

    print(
        results.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    best_threshold = choose_threshold(
        results
    )

    print()
    print(
        f"Best F1 threshold: "
        f"{best_threshold:.2f}"
    )

    test_probabilities = (
        model.predict_proba(
            X_test
        )[:, 1]
    )

    final_evaluation(
        y_test=y_test,
        probabilities=test_probabilities,
        threshold=best_threshold,
    )


if __name__ == "__main__":
    main()
