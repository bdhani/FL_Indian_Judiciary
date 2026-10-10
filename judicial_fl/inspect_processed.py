from pathlib import Path

import pandas as pd


PROCESSED_DIR = Path(
    "F:/FL_Indian_Judiciary/data/processed"
)


def main():

    files = sorted(
        PROCESSED_DIR.glob("*.parquet")
    )

    for file in files:

        print()
        print("=" * 70)
        print(f"COURT: {file.stem}")
        print("=" * 70)

        df = pd.read_parquet(file)

        print()
        print("Columns:")
        print(df.columns.tolist())

        print()
        print("Shape:")
        print(df.shape)

        print()
        print("Data types:")
        print(df.dtypes)

        print()
        print("Missing values:")
        print(df.isna().sum())

        print()
        print("Unique values:")

        for column in [
            "client",
            "cnr",
            "judge",
            "disposal_nature",
            "court",
            "disposal_days",
        ]:

            print(
                f"{column}: "
                f"{df[column].nunique(dropna=True)}"
            )

        print()
        print("-" * 70)
        print("JUDGE COUNTS")
        print("-" * 70)

        print(
            df["judge"]
            .value_counts(dropna=False)
            .head(20)
        )

        print()
        print("-" * 70)
        print("DISPOSAL NATURE COUNTS")
        print("-" * 70)

        print(
            df["disposal_nature"]
            .value_counts(dropna=False)
        )

        print()
        print("-" * 70)
        print("DISPOSAL DAYS")
        print("-" * 70)

        print(
            df["disposal_days"].describe()
        )

        print()
        print("Disposal days percentiles:")

        print(
            df["disposal_days"].quantile(
                [
                    0.01,
                    0.05,
                    0.25,
                    0.50,
                    0.75,
                    0.95,
                    0.99,
                ]
            )
        )


if __name__ == "__main__":
    main()