from pathlib import Path

import pandas as pd


DATA_DIR = Path(
    "F:/FL_Indian_Judiciary/data/raw"
)


def main():

    files = list(
        DATA_DIR.rglob("*.parquet")
    )

    print(
        f"Found {len(files)} parquet files."
    )

    if not files:
        return

    file = files[0]

    print()
    print("Inspecting:")
    print(file)

    df = pd.read_parquet(file)

    print()
    print("Shape:")
    print(df.shape)

    print()
    print("Columns:")

    for column in df.columns:
        print(
            f"  {column}"
        )

    print()
    print("First 5 rows:")

    print(
        df.head()
    )


if __name__ == "__main__":
    main()