import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import pyreadstat


RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")


FEATURE_COLUMNS = [
    "SEQNO",
    "_STATE",
    "_LLCPWT",
    "_STSTR",
    "_PSU",
    
    "DIABETE4",
    "_AGE80",
    "_SEX",
    "_BMI5",
    "_TOTINDA",
    "_SMOKER3",
    "DRNKANY6",
    "_EDUCAG",
    "_INCOMG1",
    "_RACEGR3",
    "GENHLTH",
]


def load_brfss(year: int) -> pd.DataFrame:
    xpt_path = (
        RAW_DIR
        / str(year)
        / f"LLCP{year}.XPT"
    )

    if not xpt_path.exists():
        raise FileNotFoundError(
            f"BRFSS XPT file not found: {xpt_path}"
        )

    print("=" * 70)
    print("Adaptive Diabetes MLOps - Dataset Builder")
    print("=" * 70)
    print(f"Year: {year}")
    print(f"Source: {xpt_path}")
    print()

    print("Reading selected BRFSS variables...")

    df, _ = pyreadstat.read_xport(
        str(xpt_path),
        usecols=FEATURE_COLUMNS,
        encoding="LATIN1",
    )

    print(f"Raw selected rows: {len(df):,}")
    print(f"Selected columns: {len(df.columns)}")

    return df


def create_target(df: pd.DataFrame) -> pd.DataFrame:
    print()
    print("Creating binary diabetes target...")

    print("Original DIABETE4 distribution:")
    print(
        df["DIABETE4"]
        .value_counts(dropna=False)
        .sort_index()
    )

    # BRFSS DIABETE4:
    #
    # 1 = Yes
    # 2 = Yes, but female told only during pregnancy
    # 3 = No
    # 4 = No, pre-diabetes or borderline diabetes
    # 7 = Don't know / Not sure
    # 9 = Refused
    #
    # For a clean binary comparison:
    # 1 -> diabetes
    # 3 -> no diabetes
    #
    # Gestational diabetes, prediabetes and unknown/refused
    # responses are excluded.

    df = df[
        df["DIABETE4"].isin([1, 3])
    ].copy()

    df["diabetes"] = (
        df["DIABETE4"] == 1
    ).astype(int)

    df = df.drop(
        columns=["DIABETE4"]
    )

    print()
    print("Binary target distribution:")
    print(
        df["diabetes"]
        .value_counts()
        .sort_index()
    )

    print()
    print("Target proportions:")
    print(
        (
            df["diabetes"]
            .value_counts(normalize=True)
            .sort_index()
            * 100
        ).round(2)
    )

    return df


def clean_features(df: pd.DataFrame, year: int,) -> pd.DataFrame:
    print()
    print("Cleaning variables...")
    df = df.rename(
         columns={
            "SEQNO": "respondent_id",
            "_STATE": "state",
            "_LLCPWT": "survey_weight",
            "_STSTR": "survey_stratum",
            "_PSU": "survey_psu",
            "_AGE80": "age",
            "_SEX": "sex",
            "_BMI5": "bmi",
            "_TOTINDA": "physical_activity",
            "_SMOKER3": "smoking_status",
            "DRNKANY6": "alcohol_use",
            "_EDUCAG": "education",
            "_INCOMG1": "income",
            "_RACEGR3": "race_group",
            "GENHLTH": "general_health",
        }
    )
    # Source dataset year.
    # Needed because BRFSS sequence numbers are unique
    # only within a state and year.
    df["source_year"] = year

    # Create a globally unique record identifier.
    state_id = (
        df["state"]
        .astype("Int64")
        .astype(str)
        .str.zfill(2)
    )

    respondent_id = (
        df["respondent_id"]
        .astype(str)
        .str.strip()
    )

    df["record_id"] = (
        df["source_year"].astype(str)
        + "-"
        + state_id
        + "-"
        + respondent_id
    )
    # BRFSS stores BMI multiplied by 100.
    # Example: 2745 means BMI = 27.45.
    df["bmi"] = df["bmi"] / 100

    # Replace explicit missing / unknown codes.

    # Sex:
    # 1 = Male
    # 2 = Female
    df.loc[
        ~df["sex"].isin([1, 2]),
        "sex"
    ] = np.nan

    # Physical activity:
    # 1 = Had physical activity
    # 2 = No physical activity
    # 9 = Don't know/refused/missing
    df.loc[
        ~df["physical_activity"].isin([1, 2]),
        "physical_activity"
    ] = np.nan

    # Smoking calculated status:
    # 1 = Current smoker every day
    # 2 = Current smoker some days
    # 3 = Former smoker
    # 4 = Never smoked
    # 9 = Don't know/refused/missing
    df.loc[
        ~df["smoking_status"].isin([1, 2, 3, 4]),
        "smoking_status"
    ] = np.nan

    # Alcohol:
    # 1 = Yes
    # 2 = No
    # 7/9 = unknown/refused
    df.loc[
        ~df["alcohol_use"].isin([1, 2]),
        "alcohol_use"
    ] = np.nan

    # Education:
    # valid calculated groups = 1..4
    df.loc[
        ~df["education"].isin([1, 2, 3, 4]),
        "education"
    ] = np.nan

    # Income:
    # valid calculated groups are 1..7
    df.loc[
        ~df["income"].isin(
            [1, 2, 3, 4, 5, 6, 7]
        ),
        "income"
    ] = np.nan

    # Race/ethnicity calculated groups:
    # valid groups = 1..5
    df.loc[
        ~df["race_group"].isin(
            [1, 2, 3, 4, 5]
        ),
        "race_group"
    ] = np.nan

    # General health:
    # 1 Excellent
    # 2 Very good
    # 3 Good
    # 4 Fair
    # 5 Poor
    # 7/9 unknown/refused
    df.loc[
        ~df["general_health"].isin(
            [1, 2, 3, 4, 5]
        ),
        "general_health"
    ] = np.nan

    # Basic physiological plausibility checks.
    df.loc[
        ~df["age"].between(18, 80),
        "age"
    ] = np.nan

    df.loc[
        ~df["bmi"].between(12, 100),
        "bmi"
    ] = np.nan

    return df


def show_quality_report(
    df: pd.DataFrame,
) -> None:
    print()
    print("=" * 70)
    print("DATA QUALITY REPORT")
    print("=" * 70)

    report = pd.DataFrame(
        {
            "dtype": df.dtypes.astype(str),
            "missing_n": df.isna().sum(),
            "missing_pct": (
                df.isna().mean() * 100
            ).round(2),
            "unique": df.nunique(),
        }
    )

    print(report)

    print()
    print(f"Final rows: {len(df):,}")
    duplicated_record_ids = (
    df["record_id"]
    .duplicated()
    .sum()
    )

    print(
        f"Duplicated global record IDs: "
        f"{duplicated_record_ids:,}"
    )

    print(
        f"Unique global record IDs: "
        f"{df['record_id'].nunique():,}"
    )

    if duplicated_record_ids > 0:
        raise ValueError(
            "Duplicate global record IDs detected."
        )


def save_dataset(
    df: pd.DataFrame,
    year: int,
) -> Path:
    PROCESSED_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        PROCESSED_DIR
        / f"brfss_diabetes_{year}.csv"
    )

    df.to_csv(
        output_path,
        index=False,
    )

    print()
    print(f"Dataset saved: {output_path}")

    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build analytical diabetes dataset "
            "from BRFSS."
        )
    )

    parser.add_argument(
        "--year",
        type=int,
        required=True,
    )

    args = parser.parse_args()

    df = load_brfss(args.year)

    df = create_target(df)

    df = clean_features(
        df=df,
        year=args.year,
    )

    show_quality_report(df)

    save_dataset(
        df=df,
        year=args.year,
    )


if __name__ == "__main__":
    main()
