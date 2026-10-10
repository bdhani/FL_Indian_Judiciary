from pathlib import Path

import json
import pandas as pd
from sklearn.model_selection import train_test_split


PROCESSED_DIR = Path(
    "F:/FL_Indian_Judiciary/data/processed"
)

FEDERATED_DIR = Path(
    "F:/FL_Indian_Judiciary/data/federated"
)

RANDOM_STATE = 42


def clean_dataframe(df):

    # Keep only the columns relevant to modelling.
    df = df[
        [
            "client",
            "judge",
            "disposal_nature",
            "disposal_days",
        ]
    ].copy()

    # Missing disposal nature is treated as its own category.
    df["disposal_nature"] = (
        df["disposal_nature"]
        .fillna("__MISSING__")
        .astype(str)
    )

    # Judge is categorical.
    df["judge"] = (
        df["judge"]
        .fillna("__MISSING__")
        .astype(str)
    )

    # Remove impossible negative values.
    df = df[
        df["disposal_days"] >= 0
    ].copy()

    return df


def split_court(df):

    train_df, temp_df = train_test_split(
        df,
        test_size=0.30,
        random_state=RANDOM_STATE,
    )

    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        random_state=RANDOM_STATE,
    )

    return train_df, val_df, test_df


def main():

    FEDERATED_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    all_train = []

    files = sorted(
        PROCESSED_DIR.glob("*.parquet")
    )

    for file in files:

        court = file.stem

        print()
        print("=" * 70)
        print(f"PROCESSING: {court}")
        print("=" * 70)

        df = pd.read_parquet(file)

        print(
            f"Original shape: {df.shape}"
        )

        df = clean_dataframe(df)

        print(
            f"Cleaned shape: {df.shape}"
        )

        train_df, val_df, test_df = split_court(df)

        court_dir = FEDERATED_DIR / court

        court_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        train_df.to_parquet(
            court_dir / "train.parquet",
            index=False
        )

        val_df.to_parquet(
            court_dir / "val.parquet",
            index=False
        )

        test_df.to_parquet(
            court_dir / "test.parquet",
            index=False
        )

        print(
            f"Train: {len(train_df)}"
        )

        print(
            f"Validation: {len(val_df)}"
        )

        print(
            f"Test: {len(test_df)}"
        )

        all_train.append(train_df)

    # --------------------------------------------------
    # Build a shared categorical vocabulary.
    # --------------------------------------------------

    all_train_df = pd.concat(
        all_train,
        ignore_index=True
    )

    judge_values = sorted(
        all_train_df["judge"]
        .astype(str)
        .unique()
        .tolist()
    )

    disposal_values = sorted(
        all_train_df["disposal_nature"]
        .astype(str)
        .unique()
        .tolist()
    )

    metadata = {
        "judges": judge_values,
        "disposal_nature": disposal_values,
    }

    with open(
        FEDERATED_DIR / "vocabulary.json",
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            metadata,
            f,
            indent=2,
        )

    print()
    print("=" * 70)
    print("VOCABULARY")
    print("=" * 70)

    print(
        f"Judges: {len(judge_values)}"
    )

    print(
        f"Disposal nature values: "
        f"{len(disposal_values)}"
    )

    print()
    print(
        f"Saved to: {FEDERATED_DIR}"
    )


if __name__ == "__main__":
    main()