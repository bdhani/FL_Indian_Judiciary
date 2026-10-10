from pathlib import Path

import pandas as pd


# ============================================================
# Configuration
# ============================================================

RAW_DIR = Path(
    "F:/FL_Indian_Judiciary/data/raw"
)

PROCESSED_DIR = Path(
    "F:/FL_Indian_Judiciary/data/processed"
)


# ============================================================
# Data loading
# ============================================================

def load_court_data(court_dir):

    files = list(
        court_dir.glob("*.parquet")
    )

    dataframes = []

    for file in files:

        print(
            f"Reading {file.name}"
        )

        df = pd.read_parquet(file)

        dataframes.append(df)

    if not dataframes:
        return pd.DataFrame()

    df = pd.concat(
        dataframes,
        ignore_index=True,
    )

    return df


# ============================================================
# Data preparation
# ============================================================

def prepare_data(df, court_name):

    print()
    print("Original shape:")
    print(df.shape)

    # --------------------------------------------------------
    # Remove duplicate cases
    # --------------------------------------------------------

    if "cnr" in df.columns:

        before = len(df)

        # Keep non-null CNR values unique.
        # Rows without a CNR are retained.
        has_cnr = df["cnr"].notna()

        with_cnr = df[has_cnr].drop_duplicates(
            subset=["cnr"]
        )

        without_cnr = df[~has_cnr]

        df = pd.concat(
            [
                with_cnr,
                without_cnr,
            ],
            ignore_index=True,
        )

        print(
            f"Removed {before - len(df)} duplicate rows."
        )

    # --------------------------------------------------------
    # Convert dates
    # --------------------------------------------------------

    df["date_of_registration"] = pd.to_datetime(
        df["date_of_registration"],
        errors="coerce",
        dayfirst=True,
    )

    df["decision_date"] = pd.to_datetime(
        df["decision_date"],
        errors="coerce",
        dayfirst=True,
    )

    # --------------------------------------------------------
    # Remove cases without both dates
    # --------------------------------------------------------

    df = df.dropna(
        subset=[
            "date_of_registration",
            "decision_date",
        ]
    ).copy()

    # --------------------------------------------------------
    # Calculate disposal duration
    # --------------------------------------------------------

    df["disposal_days"] = (
        df["decision_date"]
        - df["date_of_registration"]
    ).dt.days

    # --------------------------------------------------------
    # Remove impossible durations
    # --------------------------------------------------------

    df = df[
        df["disposal_days"] >= 0
    ].copy()

    # --------------------------------------------------------
    # Add explicit client/court name
    # --------------------------------------------------------

    df["client"] = court_name

    # --------------------------------------------------------
    # Keep only fields available across all courts
    # --------------------------------------------------------

    columns = [
        "client",
        "cnr",
        "judge",
        "disposal_nature",
        "court",
        "disposal_days",
    ]

    df = df[columns].copy()

    return df


# ============================================================
# Main
# ============================================================

def main():

    PROCESSED_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    court_dirs = [
        path
        for path in RAW_DIR.iterdir()
        if path.is_dir()
    ]

    print(
        f"Found {len(court_dirs)} courts."
    )

    for court_dir in sorted(court_dirs):

        court_name = court_dir.name

        print()
        print("=" * 60)
        print(
            f"PROCESSING: {court_name}"
        )
        print("=" * 60)

        df = load_court_data(
            court_dir
        )

        if df.empty:
            print(
                "No data found."
            )
            continue

        df = prepare_data(
            df,
            court_name,
        )

        output_file = (
            PROCESSED_DIR
            / f"{court_name}.parquet"
        )

        df.to_parquet(
            output_file,
            index=False,
        )

        print()
        print(
            f"Final shape: {df.shape}"
        )

        print(
            f"Saved: {output_file}"
        )


if __name__ == "__main__":
    main()