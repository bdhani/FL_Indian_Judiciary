"""Step 2: court-median vs local vs centralized vs FedAvg (vs FedProx).

Run:  python -m judicial_fl.run_experiments
Outputs CSV tables into  results/
"""
import sys
import pandas as pd
from .fl_lib import *

pd.set_option("display.width", 200)


def main(feature_set="B_with_dates", rounds=30):
    RESULTS_DIR.mkdir(exist_ok=True)
    data = load_clients(feature_set)
    print(f"\n##### feature set: {feature_set}  ({len(FEATURE_SETS[feature_set])} features)")

    out = {}
    out["court_median"] = run_court_median(data)
    out["local_only"] = eval_clients(run_local(data), data)
    out["centralized"] = eval_clients(run_central(data), data)
    print("FedAvg (size-weighted):")
    out["fedavg"] = eval_clients(run_fedavg(data, rounds=rounds)[0], data)
    print("FedProx mu=0.01:")
    out["fedprox"] = eval_clients(run_fedavg(data, rounds=rounds, prox_mu=0.01)[0], data)

    cols = ["MAE_log", "Spearman", "Acc4", "MacroF1_4"]
    summary = pd.DataFrame({k: v.loc["MEAN", cols] for k, v in out.items()}).T
    worst = pd.Series({k: v.drop("MEAN")["MAE_log"].max()
                       for k, v in out.items()}, name="worst_client_MAE_log")
    summary = summary.join(worst)
    print("\nSUMMARY (test set, mean over 5 courts)\n", summary.round(3))
    summary.to_csv(RESULTS_DIR / f"summary_{feature_set}.csv")
    for k, v in out.items():
        v.to_csv(RESULTS_DIR / f"per_client_{feature_set}_{k}.csv")
    print("\nPer-client MAE_log, FedAvg:\n", out["fedavg"][cols].round(3))


if __name__ == "__main__":
    fs = sys.argv[1] if len(sys.argv) > 1 else "B_with_dates"
    main(fs)
