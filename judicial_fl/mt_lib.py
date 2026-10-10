"""Multi-task data + model + metrics + FL loop (Tasks 1 and 2 together).

Inputs come from exports/judgments_labelled.csv and exports/task2_multihot.csv
(created by export_csv.py).  Text -> hashed bag of 1-2 word n-grams, so every
court uses the same feature space WITHOUT sharing a vocabulary.
"""
import copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import spearmanr
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.metrics import f1_score

from .common import *

EXPORT_DIR = ROOT / "exports"
OUTCOMES = ["ALLOWED", "DISMISSED", "DISPOSED", "WITHDRAWN"]      # UNCLEAR rows are ignored
N_TEXT = 2 ** 13
HID = 256
WINDOW_START = pd.Timestamp("2023-12-31")


# ------------------------------------------------------------------ data
def _struct(df, use_dates):
    f = pd.DataFrame(index=df.index)
    for g in GROUPS:
        f[f"grp_{g}"] = (df["case_type_group"] == g).astype(float)
    hb = df["case_type_raw"].fillna("UNK").map(hash_bucket)
    for i in range(N_HASH):
        f[f"hash_{i}"] = (hb == i).astype(float)
    nj = df["n_judges"].fillna(1)
    f["n_judges"] = nj / 5.0
    f["is_division"] = (nj > 1).astype(float)
    f["judge_missing"] = df["judge_raw"].isna().astype(float)
    pet, res = df["petitioner"].fillna(""), df["respondent"].fillna("")
    f["pet_state"] = pet.str.contains(STATE_RE).astype(float)
    f["res_state"] = res.str.contains(STATE_RE).astype(float)
    f["pet_company"] = pet.str.contains(COMPANY_RE).astype(float)
    f["res_company"] = res.str.contains(COMPANY_RE).astype(float)
    f["pet_multi"] = pet.str.contains(MULTI_RE).astype(float)
    f["res_multi"] = res.str.contains(MULTI_RE).astype(float)
    if use_dates:
        reg = pd.to_datetime(df["date_of_registration"])
        f["log_max_days"] = np.log1p((WINDOW_END - reg).dt.days.clip(lower=0)) / LOG_SCALE
        f["filed_in_window"] = (reg >= WINDOW_END - pd.Timedelta(days=365)).astype(float)
    return f.to_numpy(np.float32)


def load_multitask(use_dates=True):
    df = pd.read_csv(EXPORT_DIR / "judgments_labelled.csv", low_memory=False)
    mh = pd.read_csv(EXPORT_DIR / "task2_multihot.csv")
    label_cols = [c for c in mh.columns if c != "cnr"]
    df = df[~df["garbled"].astype(bool)].merge(mh, on="cnr", how="left").reset_index(drop=True)
    hv = HashingVectorizer(n_features=N_TEXT, ngram_range=(1, 2), alternate_sign=False,
                           binary=True, norm="l2", token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z]+\b")
    Xt = hv.transform(df["input_text"].fillna("")).tocsr()
    Xs = _struct(df, use_dates)
    data = {}
    for c in COURTS:
        data[c] = {}
        for part in ("train", "val", "test"):
            idx = np.where((df["court"] == c) & (df["split"] == part))[0]
            sub = df.iloc[idx]
            days = sub["duration_days"].to_numpy(float)
            yo = sub["outcome_from_text"].map({o: i for i, o in enumerate(OUTCOMES)}).fillna(-1).astype(int).to_numpy()
            ys = sub[label_cols].fillna(0).to_numpy(np.float32)
            ms = sub["has_task2_label"].astype(bool).to_numpy()
            data[c][part] = dict(Xt=Xt[idx], Xs=Xs[idx], days=days,
                                 y_dur=(np.log1p(days) / LOG_SCALE).astype(np.float32),
                                 y_out=yo, y_st=ys, m_st=ms)
    return data, label_cols, Xs.shape[1]


# ------------------------------------------------------------------ model
class MultiTaskNet(nn.Module):
    def __init__(self, n_struct, n_labels):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(N_TEXT + n_struct, HID), nn.ReLU(), nn.Dropout(0.3))
        self.dur = nn.Linear(HID, 1)
        self.out = nn.Linear(HID, len(OUTCOMES))
        self.st = nn.Linear(HID, n_labels)

    def forward(self, x):
        h = self.enc(x)
        return self.dur(h).squeeze(1), self.out(h), self.st(h)


def batch(d, idx):
    x = np.concatenate([d["Xt"][idx].toarray().astype(np.float32), d["Xs"][idx]], axis=1)
    return torch.from_numpy(x)


def task_losses(model, d, idx, w=(1.0, 1.0, 1.0)):
    z_d, z_o, z_s = model(batch(d, idx))
    zero = torch.zeros(())
    l_d = nn.functional.huber_loss(z_d, torch.from_numpy(d["y_dur"][idx]), delta=0.1) * 10
    mo = d["y_out"][idx] >= 0
    l_o = (nn.functional.cross_entropy(z_o[torch.from_numpy(mo)], torch.from_numpy(d["y_out"][idx][mo]))
           if mo.any() else zero)
    ms = d["m_st"][idx]
    l_s = (nn.functional.binary_cross_entropy_with_logits(z_s[torch.from_numpy(ms)], torch.from_numpy(d["y_st"][idx][ms])) * 20
           if ms.any() else zero)
    return w[0] * l_d + w[1] * l_o + w[2] * l_s


def train_epochs(model, d, epochs, lr=1e-3, bs=64, prox_mu=0.0, gparams=None, seed=0, w=(1, 1, 1)):
    g = np.random.RandomState(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    n = len(d["days"])
    model.train()
    for _ in range(epochs):
        perm = g.permutation(n)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            opt.zero_grad()
            loss = task_losses(model, d, idx, w)
            if prox_mu > 0:
                loss = loss + 0.5 * prox_mu * sum(((p - q) ** 2).sum() for p, q in zip(model.parameters(), gparams))
            loss.backward()
            opt.step()


@torch.no_grad()
def predict(model, d):
    model.eval()
    outs = [[], [], []]
    for i in range(0, len(d["days"]), 512):
        idx = np.arange(i, min(i + 512, len(d["days"])))
        for k, z in enumerate(model(batch(d, idx))):
            outs[k].append(z)
    return [torch.cat(o) for o in outs]


@torch.no_grad()
def val_loss(model, d):
    model.eval()
    idx = np.arange(len(d["days"]))
    tot = 0.0
    for i in range(0, len(idx), 512):
        tot += float(task_losses(model, d, idx[i:i + 512])) * len(idx[i:i + 512])
    return tot / len(idx)


# ------------------------------------------------------------------ metrics
def precision_at_k(y_true, score, k=3):
    top = np.argsort(-score, axis=1)[:, :k]
    hits = np.take_along_axis(y_true, top, axis=1)
    return float(hits.mean())


def micro_f1(y_true, pred):
    tp = (y_true * pred).sum(); fp = ((1 - y_true) * pred).sum(); fn = (y_true * (1 - pred)).sum()
    return float(2 * tp / max(2 * tp + fp + fn, 1))


def best_threshold(model, dval):
    z = torch.sigmoid(predict(model, dval)[2]).numpy()
    m = dval["m_st"]
    if m.sum() < 5:
        return 0.3
    scores = {t: micro_f1(dval["y_st"][m], (z[m] >= t).astype(float)) for t in (0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5)}
    return max(scores, key=scores.get)


def evaluate(model, d, thr):
    z_d, z_o, z_s = predict(model, d)
    days_pred = np.expm1(np.clip(z_d.numpy(), 0, 1.2) * LOG_SCALE)
    lt, lp = np.log1p(d["days"]), np.log1p(days_pred)
    bt, bp = duration_bucket(d["days"]), duration_bucket(days_pred)
    r = {"dur_MAE_log": float(np.mean(np.abs(lt - lp))),
         "dur_Spearman": float(spearmanr(d["days"], days_pred)[0]),
         "dur_F1_4cls": float(f1_score(bt, bp, average="macro"))}
    mo = d["y_out"] >= 0
    po = z_o.argmax(1).numpy()
    r["out_acc"] = float((po[mo] == d["y_out"][mo]).mean())
    r["out_F1"] = float(f1_score(d["y_out"][mo], po[mo], average="macro"))
    ms = d["m_st"]
    zs = torch.sigmoid(z_s).numpy()[ms]
    r["stat_microF1"] = micro_f1(d["y_st"][ms], (zs >= thr).astype(float))
    r["stat_P@3"] = precision_at_k(d["y_st"][ms], zs, 3)
    return r


def baseline_prior(data):
    """Court median / majority outcome / most frequent 3 labels in the court's train set."""
    rows = {}
    for c in COURTS:
        tr, te = data[c]["train"], data[c]["test"]
        med = np.median(tr["days"])
        lt, lp = np.log1p(te["days"]), np.log1p(np.full(len(te["days"]), med))
        mo_tr, mo_te = tr["y_out"] >= 0, te["y_out"] >= 0
        maj = np.bincount(tr["y_out"][mo_tr]).argmax()
        freq = tr["y_st"][tr["m_st"]].mean(0)
        ms = te["m_st"]
        top3 = np.argsort(-freq)[:3]
        pred = np.zeros_like(te["y_st"][ms]); pred[:, top3] = 1
        rows[c] = {"dur_MAE_log": float(np.mean(np.abs(lt - lp))), "dur_Spearman": np.nan,
                   "dur_F1_4cls": float(f1_score(duration_bucket(te["days"]), duration_bucket(np.full(len(lt), med)), average="macro")),
                   "out_acc": float((te["y_out"][mo_te] == maj).mean()),
                   "out_F1": float(f1_score(te["y_out"][mo_te], np.full(mo_te.sum(), maj), average="macro")),
                   "stat_microF1": micro_f1(te["y_st"][ms], pred),
                   "stat_P@3": float(te["y_st"][ms][:, top3].mean())}
    return pd.DataFrame(rows).T


# ------------------------------------------------------------------ training regimes
def new_model(n_struct, n_labels, seed):
    torch.manual_seed(seed)
    return MultiTaskNet(n_struct, n_labels)


def fit_with_early_stop(model, d_train, d_val_list, max_epochs, seed, w=(1, 1, 1)):
    best, best_state, bad = 1e9, None, 0
    for ep in range(max_epochs):
        train_epochs(model, d_train, 1, seed=seed * 100 + ep, w=w)
        v = float(np.mean([val_loss(model, dv) for dv in d_val_list]))
        if v < best - 1e-4:
            best, best_state, bad = v, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= 2:
                break
    model.load_state_dict(best_state)
    return model


def concat_parts(parts):
    return dict(Xt=__import__("scipy.sparse", fromlist=["vstack"]).vstack([p["Xt"] for p in parts]).tocsr(),
                Xs=np.concatenate([p["Xs"] for p in parts]), days=np.concatenate([p["days"] for p in parts]),
                y_dur=np.concatenate([p["y_dur"] for p in parts]), y_out=np.concatenate([p["y_out"] for p in parts]),
                y_st=np.concatenate([p["y_st"] for p in parts]), m_st=np.concatenate([p["m_st"] for p in parts]))


def size_weights(ns, **_):
    return np.array(ns, float) / np.sum(ns)


def run_fl(data, n_struct, n_labels, rounds=20, local_epochs=2, weight_fn=size_weights,
           prox_mu=0.0, seed=0, verbose=True):
    gmodel = new_model(n_struct, n_labels, seed)
    best, best_state = 1e9, None
    for r in range(1, rounds + 1):
        gstate = copy.deepcopy(gmodel.state_dict())
        gparams = [p.detach().clone() for p in gmodel.parameters()]
        states, ns, vl = [], [], []
        for c in COURTS:
            lm = new_model(n_struct, n_labels, seed)
            lm.load_state_dict(gstate)
            train_epochs(lm, data[c]["train"], local_epochs, prox_mu=prox_mu, gparams=gparams, seed=seed * 1000 + r)
            states.append(lm.state_dict()); ns.append(len(data[c]["train"]["days"]))
            vl.append(val_loss(lm, data[c]["val"]))
        w = weight_fn(ns=ns, val_losses=vl, updates=states, global_state=gstate)
        gmodel.load_state_dict({k: sum(float(wi) * st[k].float() for wi, st in zip(w, states)) for k in gstate})
        v = float(np.mean([val_loss(gmodel, data[c]["val"]) for c in COURTS]))
        if v < best:
            best, best_state = v, copy.deepcopy(gmodel.state_dict())
        if verbose and (r == 1 or r % 5 == 0):
            print(f"    round {r:2d}  mean val loss {v:.3f}  weights {np.round(w, 2)}", flush=True)
    gmodel.load_state_dict(best_state)
    return gmodel