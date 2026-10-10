"""STEP 5 - baselines for Tasks 1 + 2 together (duration, outcome, statutes).

Run (needs exports/ from export_csv.py):
    python -m judicial_fl.run_multitask 1      # 1 seed, quick check (~10 min)
    python -m judicial_fl.run_multitask 3      # 3 seeds for the real numbers
Output: results/multitask_summary.csv, results/multitask_per_client.csv
"""
import sys
import numpy as np
import pandas as pd

from .mt_lib import *

METRICS = ["dur_MAE_log", "dur_Spearman", "dur_F1_4cls", "out_acc", "out_F1", "stat_microF1", "stat_P@3"]
WORST = {"dur_MAE_log": "max", "out_F1": "min", "stat_microF1": "min"}   # what 'worst court' means


def eval_global(model, data):
    rows = {}
    for c in COURTS:
        thr = best_threshold(model, data[c]["val"])
        rows[c] = evaluate(model, data[c]["test"], thr)
    return pd.DataFrame(rows).T


def eval_local(models, data):
    rows = {}
    for c in COURTS:
        thr = best_threshold(models[c], data[c]["val"])
        rows[c] = evaluate(models[c], data[c]["test"], thr)
    return pd.DataFrame(rows).T


def main(n_seeds=1):
    RESULTS_DIR.mkdir(exist_ok=True)
    data, labels, n_struct = load_multitask(use_dates=True)
    nl = len(labels)
    print(f"features: text {N_TEXT} + structured {n_struct} | Task-2 labels: {nl}")
    per_client = []

    def keep(name, seed, df):
        x = df.copy(); x["method"], x["seed"], x["court"] = name, seed, x.index
        per_client.append(x.reset_index(drop=True))

    for seed in range(n_seeds):
        print(f"\n===== seed {seed} =====")
        keep("prior (median/majority/top3)", seed, baseline_prior(data))

        print("local-only ...", flush=True)
        loc = {}
        for c in COURTS:
            m = new_model(n_struct, nl, seed)
            loc[c] = fit_with_early_stop(m, data[c]["train"], [data[c]["val"]], 15, seed)
        keep("local-only", seed, eval_local(loc, data))

        print("centralized (pooled data; upper bound, NOT federated) ...", flush=True)
        pooled = concat_parts([data[c]["train"] for c in COURTS])
        m = fit_with_early_stop(new_model(n_struct, nl, seed), pooled, [data[c]["val"] for c in COURTS], 15, seed)
        keep("centralized", seed, eval_global(m, data))

        print("FedAvg ...", flush=True)
        g = run_fl(data, n_struct, nl, seed=seed)
        keep("FedAvg", seed, eval_global(g, data))

        print("FedProx (mu=0.01) ...", flush=True)
        gp = run_fl(data, n_struct, nl, seed=seed, prox_mu=0.01)
        keep("FedProx", seed, eval_global(gp, data))

        print("FedAvg + local fine-tune ...", flush=True)
        ft = {}
        for c in COURTS:
            mm = new_model(n_struct, nl, seed)
            mm.load_state_dict(g.state_dict())
            train_epochs(mm, data[c]["train"], 2, lr=5e-4, seed=seed)
            ft[c] = mm
        keep("FedAvg + local fine-tune", seed, eval_local(ft, data))

    allr = pd.concat(per_client, ignore_index=True)
    allr.to_csv(RESULTS_DIR / "multitask_per_client.csv", index=False)
    order = list(dict.fromkeys(allr["method"]))
    mean_over_courts = allr.groupby(["method", "seed"])[METRICS].mean().groupby("method").agg(["mean", "std"])
    summ = pd.DataFrame({f"{m}": [f"{mean_over_courts.loc[k, (m, 'mean')]:.3f}" + (f" ±{mean_over_courts.loc[k, (m, 'std')]:.3f}" if n_seeds > 1 else "")
                                  for k in order] for m in METRICS}, index=order)
    for m, how in WORST.items():
        w = allr.groupby(["method", "seed", "court"])[m].mean().unstack("court")
        w = (w.max(axis=1) if how == "max" else w.min(axis=1)).groupby("method").mean()
        summ[f"worst_court_{m}"] = [f"{w[k]:.3f}" for k in order]
    summ.to_csv(RESULTS_DIR / "multitask_summary.csv")
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
    print("\nSUMMARY (test set; mean over 5 courts; ±std over seeds)\n", summ.to_string())
    last = allr[allr.method == "FedAvg"].groupby("court")[METRICS].mean().round(3)
    print("\nFedAvg per court:\n", last.to_string())
    print("\nDONE. Send me results\\multitask_summary.csv (and multitask_per_client.csv).")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 1)