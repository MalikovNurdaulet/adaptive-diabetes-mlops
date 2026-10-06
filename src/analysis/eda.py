from pathlib import Path

import pandas as pd


DATA_PATH = Path(
    "data/processed/brfss_diabetes_2024.csv"
)


def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)

    print("=" * 70)
    print("BRFSS 2024 - DIABETES EDA")
    print("=" * 70)

    print(f"Rows: {len(df):,}")
    print(f"Columns: {len(df.columns)}")

    return df


def target_summary(df: pd.DataFrame) -> None:
    print()
    print("=" * 70)
    print("1. DIABETES PREVALENCE")
    print("=" * 70)

    counts = (
        df["diabetes"]
        .value_counts()
        .sort_index()
    )

    proportions = (
        df["diabetes"]
        .value_counts(normalize=True)
        .sort_index()
        * 100
    )

    result = pd.DataFrame(
        {
            "count": counts,
            "percent": proportions.round(2),
        }
    )

    print(result)


def numerical_summary(df: pd.DataFrame) -> None:
    print()
    print("=" * 70)
    print("2. NUMERICAL VARIABLES")
    print("=" * 70)

    variables = [
        "age",
        "bmi",
    ]

    summary = (
        df.groupby("diabetes")[variables]
        .agg(
            [
                "count",
                "mean",
                "median",
                "std",
            ]
        )
        .round(2)
    )

    print(summary)


def missing_summary(df: pd.DataFrame) -> None:
    print()
    print("=" * 70)
    print("3. MISSING VALUES")
    print("=" * 70)

    missing = pd.DataFrame(
        {
            "missing_n": df.isna().sum(),
            "missing_pct": (
                df.isna().mean() * 100
            ).round(2),
        }
    )

    missing = missing[
        missing["missing_n"] > 0
    ].sort_values(
        "missing_pct",
        ascending=False,
    )

    print(missing)


def categorical_summary(
    df: pd.DataFrame,
) -> None:

    print()
    print("=" * 70)
    print("4. DIABETES BY CATEGORICAL FACTORS")
    print("=" * 70)

    variables = [
        "sex",
        "physical_activity",
        "smoking_status",
        "alcohol_use",
        "education",
        "income",
        "race_group",
        "general_health",
    ]

    for variable in variables:

        print()
        print("-" * 70)
        print(variable.upper())
        print("-" * 70)

        table = (
            df.groupby(
                variable,
                dropna=False,
            )["diabetes"]
            .agg(
                n="count",
                diabetes_cases="sum",
                diabetes_rate="mean",
            )
        )

        table["diabetes_rate"] = (
            table["diabetes_rate"]
            * 100
        ).round(2)

        print(table)


def weighted_prevalence(
    df: pd.DataFrame,
) -> None:

    print()
    print("=" * 70)
    print("5. WEIGHTED DIABETES PREVALENCE")
    print("=" * 70)

    valid = df[
        df["survey_weight"].notna()
    ].copy()

    weighted_cases = (
        valid["diabetes"]
        * valid["survey_weight"]
    ).sum()

    total_weight = (
        valid["survey_weight"]
    ).sum()

    prevalence = (
        weighted_cases
        / total_weight
        * 100
    )

    print(
        f"Unweighted prevalence: "
        f"{df['diabetes'].mean() * 100:.2f}%"
    )

    print(
        f"Weighted prevalence: "
        f"{prevalence:.2f}%"
    )


def main() -> None:
    df = load_data()

    target_summary(df)
    numerical_summary(df)
    missing_summary(df)
    categorical_summary(df)
    weighted_prevalence(df)


if __name__ == "__main__":
    main()
