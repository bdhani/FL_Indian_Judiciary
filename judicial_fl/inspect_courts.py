from pathlib import Path

import pandas as pd


RAW_DIR = Path(
    "F:/FL_Indian_Judiciary/data/raw"
)


def main():

    for court_dir in sorted(RAW_DIR.iterdir()):

        if not court_dir.is_dir():
            continue

        print()
        print("=" * 60)
        print(f"COURT: {court_dir.name}")
        print("=" * 60)

        files = list(
            court_dir.glob("*.parquet")
        )

        for file in files:

            print()
            print(f"FILE: {file.name}")

            df = pd.read_parquet(file)

            print(
                f"Shape: {df.shape}"
            )

            print("Columns:")

            for column in df.columns:
                print(
                    f"  {column}"
                )


if __name__ == "__main__":
    main()