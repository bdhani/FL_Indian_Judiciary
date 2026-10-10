from pathlib import Path

import pandas as pd


RAW_DIR = Path(
    "F:/FL_Indian_Judiciary/data/raw"
)


# Columns that exist in the common 12-column files
COMMON_COLUMNS = [
    "title",
    "description",
    "judge",
    "date_of_registration",
    "decision_date",
    "disposal_nature",
]


# Additional columns available in the richer mobile files
ADDITIONAL_COLUMNS = [
    "case_type",
    "case_no",
    "bench_name",
    "judicial_section",
    "petitioner",
    "respondent",
    "pet_advocate",
    "res_advocate",
    "fir_no",
    "fir_year",
    "police_station_code",
    "lower_court",
    "lower_court_decision_date",
    "disposal_code",
    "purpose_name",
    "est_code",
    "order_number",
    "order_type",
    "is_final",
]


def print_examples(df, column, n=5):

    if column not in df.columns:
        return

    print()
    print(f"--- {column} ---")

    values = (
        df[column]
        .dropna()
        .astype(str)
        .drop_duplicates()
        .head(n)
    )

    if len(values) == 0:
        print("No non-null values.")
        return

    for i, value in enumerate(values, 1):

        # Prevent extremely long output
        value = value.replace("\n", " ")

        if len(value) > 300:
            value = value[:300] + "..."

        print(f"{i}. {value}")


def inspect_column(df, column):

    if column not in df.columns:
        return

    series = df[column]

    print()
    print("=" * 70)
    print(f"COLUMN: {column}")
    print("=" * 70)

    print(f"Dtype: {series.dtype}")
    print(f"Rows: {len(series):,}")
    print(f"Missing: {series.isna().sum():,}")
    print(
        f"Missing %: "
        f"{series.isna().mean() * 100:.2f}%"
    )

    print(
        f"Unique values: "
        f"{series.nunique(dropna=True):,}"
    )

    # Text statistics
    if (
        pd.api.types.is_string_dtype(series)
        or series.dtype == "object"
    ):

        text = (
            series
            .dropna()
            .astype(str)
            .str.replace(r"\s+", " ", regex=True)
        )

        if len(text) > 0:

            lengths = text.str.len()

            print(
                f"Average text length: "
                f"{lengths.mean():.2f}"
            )

            print(
                f"Median text length: "
                f"{lengths.median():.2f}"
            )

            print(
                f"Maximum text length: "
                f"{lengths.max():,}"
            )

    # Show a few values
    print_examples(df, column)


def inspect_file(file):

    print()
    print("#" * 80)
    print(f"FILE: {file}")
    print("#" * 80)

    df = pd.read_parquet(file)

    print()
    print("Shape:")
    print(df.shape)

    print()
    print("Columns:")
    for column in df.columns:
        print(f"  {column}")

    print()
    print("Data types:")
    print(df.dtypes)

    # --------------------------------------------------
    # Common fields
    # --------------------------------------------------

    print()
    print("=" * 80)
    print("COMMON FIELDS")
    print("=" * 80)

    for column in COMMON_COLUMNS:
        inspect_column(df, column)

    # --------------------------------------------------
    # Additional fields
    # --------------------------------------------------

    available_additional = [
        column
        for column in ADDITIONAL_COLUMNS
        if column in df.columns
    ]

    if available_additional:

        print()
        print("=" * 80)
        print("ADDITIONAL FIELDS")
        print("=" * 80)

        for column in available_additional:
            inspect_column(df, column)

    # --------------------------------------------------
    # Dates
    # --------------------------------------------------

    for column in [
        "date_of_registration",
        "decision_date",
        "lower_court_decision_date",
    ]:

        if column not in df.columns:
            continue

        print()
        print("=" * 70)
        print(f"DATE RANGE: {column}")
        print("=" * 70)

        dates = pd.to_datetime(
            df[column],
            errors="coerce",
            dayfirst=True,
        )

        print(f"Valid dates: {dates.notna().sum():,}")
        print(f"Invalid/missing: {dates.isna().sum():,}")

        if dates.notna().any():

            print(
                f"Earliest: "
                f"{dates.min()}"
            )

            print(
                f"Latest: "
                f"{dates.max()}"
            )


def main():

    files = sorted(
        RAW_DIR.glob("**/*.parquet")
    )

    print(
        f"Found {len(files)} parquet files."
    )

    for file in files:

        inspect_file(file)

    print()
    print("=" * 80)
    print("INSPECTION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()