from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from patsy.contrasts import Treatment


DATA_PATH = Path(
    "data/processed/brfss_diabetes_2024.csv"
)

OUTPUT_DIR = Path(
    "reports/tables"
)


MODEL_VARIABLES = [
    "diabetes",
    "age",
    "bmi",
    "sex",
    "physical_activity",
    "smoking_status",
    "alcohol_use",
    "education",
    "income",
    "race_group",
]


def load_data() -> pd.DataFrame:

    df = pd.read_csv(DATA_PATH)

    print("=" * 80)
    print("BRFSS 2024 - MULTIVARIABLE LOGISTIC REGRESSION")
    print("=" * 80)

    print(f"Initial observations: {len(df):,}")

    return df


def prepare_data(
    df: pd.DataFrame,
) -> pd.DataFrame:

    model_df = (
        df[MODEL_VARIABLES]
        .dropna()
        .copy()
    )

    # Make numerical effects easier to interpret.
    #
    # age10:
    # one model unit = 10 years
    #
    # bmi5:
    # one model unit = 5 BMI points

    model_df["age10"] = (
        model_df["age"] / 10
    )

    model_df["bmi5"] = (
        model_df["bmi"] / 5
    )

    print()
    print(
        f"Complete-case observations: "
        f"{len(model_df):,}"
    )

    retained_pct = (
        len(model_df)
        / len(df)
        * 100
    )

    print(
        f"Retained observations: "
        f"{retained_pct:.2f}%"
    )

    print()
    print("Target distribution:")

    print(
        (
            model_df["diabetes"]
            .value_counts(normalize=True)
            .sort_index()
            * 100
        ).round(2)
    )

    return model_df


def fit_model(
    df: pd.DataFrame,
):

    print()
    print("=" * 80)
    print("MODEL FITTING")
    print("=" * 80)

    formula = """
    diabetes
    ~ age10
    + bmi5
    + C(sex, Treatment(reference=2))
    + C(physical_activity, Treatment(reference=1))
    + C(smoking_status, Treatment(reference=4))
    + C(alcohol_use, Treatment(reference=2))
    + C(education, Treatment(reference=4))
    + C(income, Treatment(reference=7))
    + C(race_group, Treatment(reference=1))
    """

    print("Fitting logistic regression...")

    model = smf.logit(
        formula=formula,
        data=df,
    )

    result = model.fit(
        maxiter=200,
        disp=False,
    )

    print("Model fitted successfully.")

    return result


def create_odds_ratio_table(
    result,
) -> pd.DataFrame:

    params = result.params
    conf = result.conf_int()

    table = pd.DataFrame(
        {
            "beta": params,
            "odds_ratio": np.exp(params),
            "ci_lower": np.exp(conf[0]),
            "ci_upper": np.exp(conf[1]),
            "p_value": result.pvalues,
        }
    )

    table = table.round(
        {
            "beta": 4,
            "odds_ratio": 4,
            "ci_lower": 4,
            "ci_upper": 4,
        }
    )

    return table


def print_results(
    result,
    table: pd.DataFrame,
) -> None:

    print()
    print("=" * 80)
    print("MODEL SUMMARY")
    print("=" * 80)

    print(
        f"N observations: "
        f"{int(result.nobs):,}"
    )

    print(
        f"Log-Likelihood: "
        f"{result.llf:.2f}"
    )

    print(
        f"McFadden pseudo R²: "
        f"{result.prsquared:.4f}"
    )

    print(
        f"AIC: "
        f"{result.aic:.2f}"
    )

    print()
    print("=" * 80)
    print("ODDS RATIOS")
    print("=" * 80)

    pd.set_option(
        "display.max_rows",
        100,
    )

    pd.set_option(
        "display.max_colwidth",
        100,
    )

    print(table)


def save_results(
    table: pd.DataFrame,
) -> None:

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        OUTPUT_DIR
        / "logistic_regression_2024.csv"
    )

    table.to_csv(
        output_path,
        index=True,
    )

    print()
    print(
        f"Results saved: "
        f"{output_path}"
    )


def main() -> None:

    df = load_data()

    model_df = prepare_data(df)

    result = fit_model(model_df)

    table = create_odds_ratio_table(
        result
    )

    print_results(
        result=result,
        table=table,
    )

    save_results(table)


if __name__ == "__main__":
    main()
