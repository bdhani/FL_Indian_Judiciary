from pathlib import Path

import boto3
from botocore import UNSIGNED
from botocore.config import Config


# ============================================================
# Configuration
# ============================================================

BUCKET = "indian-high-court-judgments"

YEAR = "2024"

OUTPUT_DIR = Path(
    "F:/FL_Indian_Judiciary/data/raw"
)


# We begin with five courts.
# Each court will become one FL client.

COURTS = {
    "27_1": "bombay",
    "7_26": "delhi",
    "29_3": "karnataka",
    "32_4": "kerala",
    "23_23": "madhya_pradesh",
}


# ============================================================
# Main
# ============================================================

def main():

    s3 = boto3.client(
        "s3",
        region_name="ap-south-1",
        config=Config(
            signature_version=UNSIGNED
        ),
    )

    for court_code, court_name in COURTS.items():

        prefix = (
            f"metadata/parquet/"
            f"year={YEAR}/"
            f"court={court_code}/"
        )

        output_dir = (
            OUTPUT_DIR / court_name
        )

        output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        print()
        print("=" * 60)
        print(f"COURT: {court_name}")
        print("=" * 60)

        print(
            f"S3 prefix: {prefix}"
        )

        paginator = s3.get_paginator(
            "list_objects_v2"
        )

        downloaded = 0

        for page in paginator.paginate(
            Bucket=BUCKET,
            Prefix=prefix,
        ):

            for obj in page.get(
                "Contents",
                [],
            ):

                key = obj["Key"]

                if not key.endswith(
                    ".parquet"
                ):
                    continue

                filename = Path(
                    key
                ).name

                output_path = (
                    output_dir / filename
                )

                print(
                    f"Downloading {filename}"
                )

                s3.download_file(
                    BUCKET,
                    key,
                    str(output_path),
                )

                downloaded += 1

        print(
            f"Downloaded {downloaded} files"
        )


if __name__ == "__main__":
    main()