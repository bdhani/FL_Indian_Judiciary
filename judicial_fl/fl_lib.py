"""Data loading, model, metrics and a plain-PyTorch FL simulator.

A hand-written FL loop (instead of Flower/Ray) is used on purpose: it is ~60
lines, runs fast on Windows, and the aggregation weights are one function you
can swap for your performance-aware / trust-aware strategies later."""
import copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import spearmanr
from sklearn.metrics import f1_score

from .common import *

# Feature sets -------------------------------------------------------------
BASE = ([f"grp_{g}" for g in GROUPS] + [f"hash_{i}" for i in range(N_HASH)] +
        ["n_judges", "is_division", "judge_missing", "judge_load", "title_len",
         "pet_state", "res_state", "pet_company", "res_company",
         "pet_multi", "res_multi"])
DATE = ["log_max_days", "filed_in_window"]
FEATURE_SETS = {"A_no_dates": BASE, "B_with_dates": BASE + DATE}


def load_clients(feature_set="B_with_dates"):
    cols = FEATURE_SETS[feature_set]
    data = {}
    for c in COURTS:
        d = {}
        for part in ("train", "val", "test"):
            df = pd.read_parquet(PROCESSED_DIR / c / f"{part}.parquet")
            d[part] = (df[cols].to_numpy(np.float32),
                       (np.log1p(df["days"].to_numpy()) / LOG_SCALE).astype(np.float32),
                       df["days"].to_numpy())
        data[c] = d
    return data


def make_model(n_in):
    return nn.Sequential(nn.Linear(n_in, 128), nn.ReLU(), nn.Dropout(0.1),
                         nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1))


def train_epochs(model, X, y, epochs, lr=1e-3, bs=256, prox_mu=0.0, global_params=None, seed=0):
    g = torch.Generator().manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    Xt, yt = torch.from_numpy(X), torch.from_numpy(y).view(-1, 1)
    model.train()
    for _ in range(epochs):
        perm = torch.randperm(len(Xt), generator=g)
        for i in range(0, len(Xt), bs):
            idx = perm[i:i + bs]
            opt.zero_grad()
            loss = nn.functional.huber_loss(model(Xt[idx]), yt[idx], delta=0.1)
            if prox_mu > 0:                                   # FedProx term
                loss = loss + 0.5 * prox_mu * sum(
                    ((p - q) ** 2).sum() for p, q in zip(model.parameters(), global_params))
            loss.backward()
            opt.step()


@torch.no_grad()
def predict_days(model, X):
    model.eval()
    z = model(torch.from_numpy(X)).squeeze(1).numpy()
    return np.expm1(np.clip(z, 0, 1.2) * LOG_SCALE)


def metrics(days_true, days_pred):
    lt, lp = np.log1p(days_true), np.log1p(days_pred)
    bt, bp = duration_bucket(days_true), duration_bucket(days_pred)
    return {
        "MAE_log": float(np.mean(np.abs(lt - lp))),                 # ~ relative error
        "MedAE_days": float(np.median(np.abs(days_true - days_pred))),
        "R2_log": float(1 - np.sum((lt - lp) ** 2) / np.sum((lt - lt.mean()) ** 2)),
        "Spearman": float(spearmanr(days_true, days_pred)[0]),
        "Acc4": float((bt == bp).mean()),
        "MacroF1_4": float(f1_score(bt, bp, average="macro")),
    }


def eval_clients(models, data, part="test"):
    """models: one model or {court: model}. Returns DataFrame of per-client metrics."""
    rows = {}
    for c in COURTS:
        m = models[c] if isinstance(models, dict) else models
        X, _, days = data[c][part]
        rows[c] = metrics(days, predict_days(m, X))
    df = pd.DataFrame(rows).T
    df.loc["MEAN"] = df.mean()
    return df


# ---------------------------------------------------------------- baselines
def run_court_median(data):
    """Dumb baseline every model must beat: predict the court's training median."""
    rows = {}
    for c in COURTS:
        med = np.median(data[c]["train"][2])
        _, _, days = data[c]["test"]
        rows[c] = metrics(days, np.full(len(days), med))
    df = pd.DataFrame(rows).T
    df.loc["MEAN"] = df.mean()
    return df


def run_central(data, epochs=15, seed=0):
    X = np.concatenate([data[c]["train"][0] for c in COURTS])
    y = np.concatenate([data[c]["train"][1] for c in COURTS])
    torch.manual_seed(seed)
    m = make_model(X.shape[1])
    train_epochs(m, X, y, epochs, seed=seed)
    return m


def run_local(data, epochs=15, seed=0):
    models = {}
    for c in COURTS:
        torch.manual_seed(seed)
        m = make_model(data[c]["train"][0].shape[1])
        train_epochs(m, data[c]["train"][0], data[c]["train"][1], epochs, seed=seed)
        models[c] = m
    return models


# --------------------------------------------------------- federated learning
def size_weights(ns, **_):
    return np.array(ns, float) / np.sum(ns)


def uniform_weights(ns, **_):
    return np.ones(len(ns)) / len(ns)


def run_fedavg(data, rounds=30, local_epochs=2, weight_fn=size_weights,
               prox_mu=0.0, seed=0, log_every=5, verbose=True, attack=None):
    """Generic FL loop.  weight_fn(ns, val_losses, updates, global_state) -> weights.
    `attack`: optional {court: callable(update_state_dict)->state_dict} to simulate
    a malicious client later in the project."""
    n_in = data[COURTS[0]]["train"][0].shape[1]
    torch.manual_seed(seed)
    gmodel = make_model(n_in)
    hist = []
    for r in range(1, rounds + 1):
        gstate = copy.deepcopy(gmodel.state_dict())
        gparams = [p.detach().clone() for p in gmodel.parameters()]
        states, ns, vlosses = [], [], []
        for c in COURTS:
            lm = make_model(n_in)
            lm.load_state_dict(gstate)
            X, y, _ = data[c]["train"]
            train_epochs(lm, X, y, local_epochs, prox_mu=prox_mu,
                         global_params=gparams, seed=seed * 1000 + r)
            st = lm.state_dict()
            if attack and c in attack:
                st = attack[c](st)
            states.append(st)
            ns.append(len(X))
            Xv, yv, dv = data[c]["val"]
            vlosses.append(metrics(dv, predict_days(lm, Xv))["MAE_log"])
        w = weight_fn(ns=ns, val_losses=vlosses, updates=states, global_state=gstate)
        new = {k: sum(float(wi) * st[k].float() for wi, st in zip(w, states)) for k in gstate}
        gmodel.load_state_dict(new)
        if verbose and (r % log_every == 0 or r == 1):
            v = eval_clients(gmodel, data, "val")
            print(f"  round {r:3d}  val MAE_log mean={v.loc['MEAN','MAE_log']:.3f} "
                  f"worst={v.drop('MEAN')['MAE_log'].max():.3f}  weights={np.round(w, 2)}")
        hist.append(w)
    return gmodel, hist
