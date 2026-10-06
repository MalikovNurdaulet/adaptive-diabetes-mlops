from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


DATA_PATH = Path(
    "data/processed/brfss_diabetes_2024.csv"
)


def cohens_d(x: pd.Series, y: pd.Series) -> float:
    x = x.dropna()
    y = y.dropna()

    nx = len(x)
    ny = len(y)

    pooled_variance = (
        ((nx - 1) * x.var(ddof=1))
        + ((ny - 1) * y.var(ddof=1))
    ) / (nx + ny - 2)

    return (
        x.mean() - y.mean()
    ) / np.sqrt(pooled_variance)


def cramers_v(table: pd.DataFrame) -> float:
    chi2, _, _, _ = stats.chi2_contingency(
        table
    )

    n = table.to_numpy().sum()

    rows, cols = table.shape

    return np.sqrt(
        (chi2 / n)
        / min(rows - 1, cols - 1)
    )


def numerical_tests(
    df: pd.DataFrame,
) -> None:

    print()
    print("=" * 80)
    print("1. NUMERICAL VARIABLES")
    print("=" * 80)

    variables = [
        "age",
        "bmi",
    ]

    for variable in variables:

        group_no = df.loc[
            df["diabetes"] == 0,
            variable,
        ].dropna()

        group_yes = df.loc[
            df["diabetes"] == 1,
            variable,
        ].dropna()

        t_stat, t_p = stats.ttest_ind(
            group_yes,
            group_no,
            equal_var=False,
        )

        mw_stat, mw_p = stats.mannwhitneyu(
            group_yes,
            group_no,
            alternative="two-sided",
        )

        d = cohens_d(
            group_yes,
            group_no,
        )

        print()
        print(f"Variable: {variable}")

        print(
            f"Mean without diabetes: "
            f"{group_no.mean():.2f}"
        )

        print(
            f"Mean with diabetes: "
            f"{group_yes.mean():.2f}"
        )

        print(
            f"Welch t-test: "
            f"t={t_stat:.3f}, "
            f"p={t_p:.6g}"
        )

        print(
            f"Mann-Whitney: "
            f"U={mw_stat:.0f}, "
            f"p={mw_p:.6g}"
        )

        print(
            f"Cohen's d: "
            f"{d:.3f}"
        )


def categorical_tests(
    df: pd.DataFrame,
) -> None:

    print()
    print("=" * 80)
    print("2. CATEGORICAL VARIABLES")
    print("=" * 80)

    variables = [
        "sex",
        "physical_activity",
        "smoking_status",
        "alcohol_use",
        "education",
        "income",
        "race_group",
    ]

    for variable in variables:

        subset = df[
            [variable, "diabetes"]
        ].dropna()

        table = pd.crosstab(
            subset[variable],
            subset["diabetes"],
        )

        chi2, p, dof, expected = (
            stats.chi2_contingency(
                table
            )
        )

        v = cramers_v(table)

        print()
        print(f"Variable: {variable}")

        print(
            f"Chi-square: "
            f"{chi2:.3f}"
        )

        print(
            f"df: {dof}"
        )

        print(
            f"p-value: {p:.6g}"
        )

        print(
            f"Cramer's V: {v:.3f}"
        )


def main() -> None:

    df = pd.read_csv(
        DATA_PATH
    )

    print("=" * 80)
    print("BRFSS 2024 - STATISTICAL TESTS")
    print("=" * 80)

    print(
        f"Observations: "
        f"{len(df):,}"
    )

    numerical_tests(df)
    categorical_tests(df)


if __name__ == "__main__":
    main()
