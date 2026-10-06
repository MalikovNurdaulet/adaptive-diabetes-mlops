from pathlib import Path

import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    average_precision_score,
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

    print("=" * 80)
    print("LOGISTIC REGRESSION - ML BASELINE")
    print("=" * 80)

    print(f"Observations: {len(df):,}")
    print(f"Features: {len(features)}")
    return X, y

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

    pipeline = Pipeline(
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

    return pipeline


def evaluate(
    model,
    X_test,
    y_test,
):

    probabilities = model.predict_proba(
        X_test
    )[:, 1]

    predictions = model.predict(
        X_test
    )

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

    print()
    print("=" * 80)
    print("TEST METRICS")
    print("=" * 80)

    print(
        f"ROC-AUC:   {roc_auc:.4f}"
    )

    print(
        f"PR-AUC:    {pr_auc:.4f}"
    )

    print(
        f"Precision: {precision:.4f}"
    )

    print(
        f"Recall:    {recall:.4f}"
    )

    print(
        f"F1-score:  {f1:.4f}"
    )

    print()
    print("Confusion matrix:")

    print(
        confusion_matrix(
            y_test,
            predictions,
        )
    )

    print()
    print(
        classification_report(
            y_test,
            predictions,
            digits=4,
        )
    )


def main():

    X, y = load_data()

    (
        X_train,
        X_test,
        y_train,
        y_test,
    ) = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    print()
    print(
        f"Train observations: "
        f"{len(X_train):,}"
    )

    print(
        f"Test observations: "
        f"{len(X_test):,}"
    )

    print(
        f"Train diabetes rate: "
        f"{y_train.mean() * 100:.2f}%"
    )

    print(
        f"Test diabetes rate: "
        f"{y_test.mean() * 100:.2f}%"
    )

    model = build_pipeline()

    print()
    print("Training model...")

    model.fit(
        X_train,
        y_train,
    )

    print("Training completed.")

    evaluate(
        model=model,
        X_test=X_test,
        y_test=y_test,
    )


if __name__ == "__main__":
    main()
