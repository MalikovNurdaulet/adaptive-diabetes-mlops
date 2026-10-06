import argparse
import shutil
import zipfile
from pathlib import Path

import pyreadstat


RAW_DIR = Path("data/raw")


def extract_xpt(year: int) -> Path:
    zip_path = RAW_DIR / f"LLCP{year}XPT.zip"

    if not zip_path.exists():
        raise FileNotFoundError(
            f"Dataset ZIP not found: {zip_path}"
        )

    extract_dir = RAW_DIR / str(year)
    extract_dir.mkdir(parents=True, exist_ok=True)

    output_path = extract_dir / f"LLCP{year}.XPT"

    if output_path.exists():
        print(f"XPT already extracted: {output_path}")
        return output_path

    print(f"Opening archive: {zip_path}")

    with zipfile.ZipFile(zip_path, "r") as archive:

        files = [
            info
            for info in archive.infolist()
            if not info.is_dir()
        ]

        if not files:
            raise RuntimeError(
                "ZIP archive is empty."
            )

        print()
        print("Files inside ZIP:")

        for info in files:
            print(
                f"  {info.filename} "
                f"({info.file_size / 1024 / 1024:.2f} MB)"
            )

        # The BRFSS transport archive contains the dataset.
        # We select the largest file instead of relying on
        # the internal filename or extension.
        dataset_file = max(
            files,
            key=lambda info: info.file_size,
        )

        print()
        print(
            f"Selected archive member: "
            f"{dataset_file.filename}"
        )

        print(
            f"Extracting as: {output_path}"
        )

        with archive.open(dataset_file) as source:
            with output_path.open("wb") as destination:
                shutil.copyfileobj(
                    source,
                    destination,
                )

    print("Extraction completed.")

    return output_path


def inspect_dataset(year: int) -> None:
    xpt_path = extract_xpt(year)

    print()
    print("=" * 70)
    print(f"BRFSS {year} DATASET INSPECTION")
    print("=" * 70)

    print(f"File: {xpt_path}")
    print(
        f"Size: "
        f"{xpt_path.stat().st_size / 1024 / 1024:.2f} MB"
    )

    print()
    print("Reading metadata...")

    _, metadata = pyreadstat.read_xport(
        str(xpt_path),
        metadataonly=True,
        encoding="LATIN1",
    )

    columns = metadata.column_names

    print()
    print(f"Number of variables: {len(columns)}")

    print()
    print("First 30 variables:")
    print("-" * 70)

    for column in columns[:30]:
        label = (
            metadata.column_names_to_labels.get(
                column,
                "",
            )
            or ""
        )

        print(
            f"{column:15} | {label}"
        )

    print()
    print("Potential diabetes-related variables:")
    print("-" * 70)

    keywords = [
        "DIAB",
        "BMI",
        "AGE",
        "SEX",
        "SMOK",
        "EXER",
        "INCOME",
        "BPHIGH",
        "GENHLTH",
        "PHYSHLTH",
        "MENTHLTH",
        "WEIGHT",
        "HEIGHT",
        "CHOL",
        "CVD",
    ]

    found = []

    for column in columns:
        column_upper = column.upper()

        if any(
            keyword in column_upper
            for keyword in keywords
        ):
            label = (
                metadata.column_names_to_labels.get(
                    column,
                    "",
                )
                or ""
            )

            found.append(
                (column, label)
            )

    for column, label in found:
        print(
            f"{column:15} | {label}"
        )

    print()
    print(
        f"Found {len(found)} "
        f"potentially relevant variables."
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Extract and inspect a BRFSS "
            "SAS Transport dataset."
        )
    )

    parser.add_argument(
        "--year",
        type=int,
        required=True,
        help="BRFSS dataset year.",
    )

    args = parser.parse_args()

    inspect_dataset(
        year=args.year,
    )


if __name__ == "__main__":
    main()
