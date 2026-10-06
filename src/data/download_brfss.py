import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import requests


RAW_DIR = Path("data/raw")
MANIFEST_PATH = RAW_DIR / "manifest.json"

CDC_URL_TEMPLATE = (
    "https://www.cdc.gov/brfss/annual_data/"
    "{year}/files/LLCP{year}XPT.zip"
)


def calculate_sha256(file_path: Path) -> str:
    sha256 = hashlib.sha256()

    with file_path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            sha256.update(chunk)

    return sha256.hexdigest()


def load_manifest() -> dict:
    if not MANIFEST_PATH.exists():
        return {"datasets": []}

    with MANIFEST_PATH.open("r", encoding="utf-8") as file:
        return json.load(file)


def save_manifest(manifest: dict) -> None:
    with MANIFEST_PATH.open("w", encoding="utf-8") as file:
        json.dump(
            manifest,
            file,
            indent=4,
            ensure_ascii=False,
        )


def dataset_exists(year: int, manifest: dict) -> bool:
    for dataset in manifest["datasets"]:
        if dataset["year"] == year:
            file_path = RAW_DIR / dataset["file"]

            if file_path.exists():
                return True

    return False


def download_brfss(year: int, force: bool = False) -> Path:
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    manifest = load_manifest()

    file_name = f"LLCP{year}XPT.zip"
    output_path = RAW_DIR / file_name
    temp_path = RAW_DIR / f"{file_name}.part"

    if dataset_exists(year, manifest) and not force:
        print(f"BRFSS {year} already exists.")
        print(f"File: {output_path}")
        return output_path

    url = CDC_URL_TEMPLATE.format(year=year)

    print("=" * 60)
    print("Adaptive Diabetes MLOps - BRFSS ingestion")
    print("=" * 60)
    print(f"Dataset year: {year}")
    print(f"Source: {url}")
    print(f"Destination: {output_path}")
    print()
    print("Downloading...")

    headers = {
        "User-Agent": (
            "Mozilla/5.0 Adaptive-Diabetes-MLOps/1.0"
        )
    }

    try:
        with requests.get(
            url,
            headers=headers,
            stream=True,
            timeout=60,
        ) as response:

            response.raise_for_status()

            total_size = int(
                response.headers.get("content-length", 0)
            )

            downloaded = 0

            with temp_path.open("wb") as file:
                for chunk in response.iter_content(
                    chunk_size=1024 * 1024
                ):
                    if not chunk:
                        continue

                    file.write(chunk)
                    downloaded += len(chunk)

                    if total_size > 0:
                        percent = downloaded / total_size * 100

                        print(
                            f"\rDownloaded: "
                            f"{downloaded / 1024 / 1024:.1f} MB "
                            f"({percent:.1f}%)",
                            end="",
                        )

        print()

        temp_path.replace(output_path)

    except requests.RequestException as error:
        if temp_path.exists():
            temp_path.unlink()

        raise RuntimeError(
            f"Failed to download BRFSS {year}: {error}"
        ) from error

    file_size = output_path.stat().st_size
    sha256 = calculate_sha256(output_path)

    print("Download completed.")
    print(f"Size: {file_size / 1024 / 1024:.2f} MB")
    print(f"SHA256: {sha256}")

    manifest["datasets"] = [
        dataset
        for dataset in manifest["datasets"]
        if dataset["year"] != year
    ]

    manifest["datasets"].append(
        {
            "year": year,
            "file": file_name,
            "source_url": url,
            "size_bytes": file_size,
            "sha256": sha256,
            "downloaded_at": datetime.now(
                timezone.utc
            ).isoformat(),
        }
    )

    manifest["datasets"] = sorted(
        manifest["datasets"],
        key=lambda item: item["year"],
    )

    save_manifest(manifest)

    print()
    print("Manifest updated:")
    print(MANIFEST_PATH)

    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download CDC BRFSS annual dataset."
    )

    parser.add_argument(
        "--year",
        type=int,
        required=True,
        help="BRFSS year, for example 2024.",
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Download the dataset again even if it exists.",
    )

    args = parser.parse_args()

    download_brfss(
        year=args.year,
        force=args.force,
    )


if __name__ == "__main__":
    main()
