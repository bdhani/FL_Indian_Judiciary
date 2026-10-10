from pathlib import Path
import json

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import flwr as fl


FEDERATED_DIR = Path(
    "F:/FL_Indian_Judiciary/data/federated"
)

COURTS = [
    "bombay",
    "delhi",
    "karnataka",
    "kerala",
    "madhya_pradesh",
]

NUM_ROUNDS = 20
LOCAL_EPOCHS = 3

DEVICE = torch.device("cpu")


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
        .astype(int)
        .to_numpy()
    )

    disposal = (
        df["disposal_nature"]
        .astype(str)
        .map(disposal_map)
        .fillna(disposal_unknown)
        .astype(int)
        .to_numpy()
    )

    num_judges = len(judge_map) + 1
    num_disposal = len(disposal_map) + 1

    X_judge = np.eye(
        num_judges,
        dtype=np.float32,
    )[judge]

    X_disposal = np.eye(
        num_disposal,
        dtype=np.float32,
    )[disposal]

    X = np.concatenate(
        [
            X_judge,
            X_disposal,
        ],
        axis=1,
    )

    y = np.log1p(
        df["disposal_days"]
        .to_numpy(
            dtype=np.float32
        )
    )

    return X, y


def get_parameters(model):

    return [
        value.detach()
        .cpu()
        .numpy()
        for _, value in model.state_dict().items()
    ]


def set_parameters(model, parameters):

    state_dict = model.state_dict()

    new_state_dict = {}

    for (key, _), value in zip(
        state_dict.items(),
        parameters,
    ):

        new_state_dict[key] = torch.tensor(
            value
        )

    model.load_state_dict(
        new_state_dict
    )


class CourtClient(fl.client.NumPyClient):

    def __init__(self, court):

        self.court = court

        self.vocabulary = (
            load_vocabulary()
        )

        court_dir = (
            FEDERATED_DIR / court
        )

        train_df = pd.read_parquet(
            court_dir / "train.parquet"
        )

        val_df = pd.read_parquet(
            court_dir / "val.parquet"
        )

        self.X_train, self.y_train = (
            encode_dataframe(
                train_df,
                self.vocabulary,
            )
        )

        self.X_val, self.y_val = (
            encode_dataframe(
                val_df,
                self.vocabulary,
            )
        )

        self.model = JudicialMLP(
            input_size=self.X_train.shape[1]
        ).to(DEVICE)

    def get_parameters(
        self,
        config,
    ):

        return get_parameters(
            self.model
        )

    def fit(
        self,
        parameters,
        config,
    ):

        set_parameters(
            self.model,
            parameters,
        )

        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=0.001,
        )

        loss_function = nn.MSELoss()

        X = torch.tensor(
            self.X_train,
            dtype=torch.float32,
        )

        y = torch.tensor(
            self.y_train,
            dtype=torch.float32,
        ).reshape(-1, 1)

        self.model.train()

        for _ in range(
            LOCAL_EPOCHS
        ):

            optimizer.zero_grad()

            prediction = self.model(X)

            loss = loss_function(
                prediction,
                y,
            )

            loss.backward()

            optimizer.step()

        print(
            f"{self.court}: "
            f"{len(self.X_train)} training samples"
        )

        return (
            get_parameters(self.model),
            len(self.X_train),
            {
                "train_loss":
                    float(loss.item())
            },
        )

    def evaluate(
        self,
        parameters,
        config,
    ):

        set_parameters(
            self.model,
            parameters,
        )

        self.model.eval()

        X = torch.tensor(
            self.X_val,
            dtype=torch.float32,
        )

        y = torch.tensor(
            self.y_val,
            dtype=torch.float32,
        ).reshape(-1, 1)

        with torch.no_grad():

            prediction = self.model(X)

            loss = nn.MSELoss()(
                prediction,
                y,
            )

        return (
            float(loss.item()),
            len(self.X_val),
            {},
        )


def client_fn(context):

    # Flower Simulation passes the client ID.
    client_id = int(
        context.node_config[
            "partition-id"
        ]
    )

    court = COURTS[client_id]

    return CourtClient(
        court
    ).to_client()


def weighted_average(metrics):

    total = sum(
        num_examples
        for num_examples, _ in metrics
    )

    weighted_loss = sum(
        num_examples * loss
        for num_examples, loss in metrics
    )

    return {
        "loss":
            weighted_loss / total
    }


def main():

    vocabulary = load_vocabulary()

    input_size = (
        len(vocabulary["judges"]) + 1
        +
        len(vocabulary["disposal_nature"]) + 1
    )

    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=5,
        min_evaluate_clients=5,
        min_available_clients=5,
    )

    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=5,
        config=fl.server.ServerConfig(
            num_rounds=NUM_ROUNDS
        ),
        strategy=strategy,
        client_resources={
            "num_cpus": 1
        },
    )


if __name__ == "__main__":
    main()