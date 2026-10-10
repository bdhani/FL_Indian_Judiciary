from pathlib import Path
import json

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


FEDERATED_DIR = Path(
    "F:/FL_Indian_Judiciary/data/federated"
)

RANDOM_SEED = 42

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


class JudicialMLP(nn.Module):

    def __init__(self, input_size):

        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),

            nn.Linear(64, 32),
            nn.ReLU(),

            nn.Linear(32, 1),
        )

    def forward(self, x):

        return self.network(x)


def load_vocabulary():

    with open(
        FEDERATED_DIR / "vocabulary.json",
        "r",
        encoding="utf-8",
    ) as f:

        return json.load(f)


def encode_dataframe(df, vocabulary):

    judge_map = {
        value: i
        for i, value in enumerate(
            vocabulary["judges"]
        )
    }

    disposal_map = {
        value: i
        for i, value in enumerate(
            vocabulary["disposal_nature"]
        )
    }

    judge_unknown = len(judge_map)
    disposal_unknown = len(disposal_map)

    judge = (
        df["judge"]
        .astype(str)
        .map(judge_map)
        .fillna(judge_unknown)
        .to_numpy()
    )

    disposal = (
        df["disposal_nature"]
        .astype(str)
        .map(disposal_map)
        .fillna(disposal_unknown)
        .to_numpy()
    )

    # One-hot encoding.
    num_judges = len(judge_map) + 1
    num_disposal = len(disposal_map) + 1

    X_judge = np.eye(
        num_judges,
        dtype=np.float32
    )[judge.astype(int)]

    X_disposal = np.eye(
        num_disposal,
        dtype=np.float32
    )[disposal.astype(int)]

    X = np.concatenate(
        [
            X_judge,
            X_disposal,
        ],
        axis=1,
    )

    y = np.log1p(
        df["disposal_days"]
        .to_numpy(dtype=np.float32)
    )

    return X, y


def evaluate(model, X, y):

    model.eval()

    with torch.no_grad():

        predictions = (
            model(
                torch.tensor(
                    X,
                    dtype=torch.float32,
                )
            )
            .squeeze()
            .numpy()
        )

    actual_days = np.expm1(y)
    predicted_days = np.maximum(
        0,
        np.expm1(predictions)
    )

    mae = mean_absolute_error(
        actual_days,
        predicted_days,
    )

    rmse = np.sqrt(
        mean_squared_error(
            actual_days,
            predicted_days,
        )
    )

    r2 = r2_score(
        actual_days,
        predicted_days,
    )

    return mae, rmse, r2


def main():

    vocabulary = load_vocabulary()

    train_parts = []
    test_parts = []

    courts = [
        "bombay",
        "delhi",
        "karnataka",
        "kerala",
        "madhya_pradesh",
    ]

    for court in courts:

        court_dir = (
            FEDERATED_DIR / court
        )

        train_parts.append(
            pd.read_parquet(
                court_dir / "train.parquet"
            )
        )

        test_parts.append(
            pd.read_parquet(
                court_dir / "test.parquet"
            )
        )

    train_df = pd.concat(
        train_parts,
        ignore_index=True,
    )

    test_df = pd.concat(
        test_parts,
        ignore_index=True,
    )

    X_train, y_train = encode_dataframe(
        train_df,
        vocabulary,
    )

    X_test, y_test = encode_dataframe(
        test_df,
        vocabulary,
    )

    print()
    print("Training shape:", X_train.shape)
    print("Test shape:", X_test.shape)

    model = JudicialMLP(
        input_size=X_train.shape[1]
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.001,
    )

    loss_function = nn.MSELoss()

    X_tensor = torch.tensor(
        X_train,
        dtype=torch.float32,
    )

    y_tensor = torch.tensor(
        y_train,
        dtype=torch.float32,
    ).reshape(-1, 1)

    for epoch in range(100):

        model.train()

        optimizer.zero_grad()

        predictions = model(
            X_tensor
        )

        loss = loss_function(
            predictions,
            y_tensor,
        )

        loss.backward()

        optimizer.step()

        if (epoch + 1) % 10 == 0:

            print(
                f"Epoch {epoch + 1:03d} "
                f"Loss: {loss.item():.4f}"
            )

    mae, rmse, r2 = evaluate(
        model,
        X_test,
        y_test,
    )

    print()
    print("=" * 60)
    print("CENTRALIZED BASELINE")
    print("=" * 60)

    print(
        f"MAE : {mae:.2f} days"
    )

    print(
        f"RMSE: {rmse:.2f} days"
    )

    print(
        f"R²  : {r2:.4f}"
    )


if __name__ == "__main__":
    main()