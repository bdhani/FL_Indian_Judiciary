"""Step 1: raw parquet -> one clean, FEATURE-CONSISTENT table per court.

Run:  python -m judicial_fl.prepare_v2
"""
import pandas as pd
import numpy as np

from judicial_fl.common import *


def load_court(court: str) -> pd.DataFrame:
    parts = []
    for f in sorted((RAW_DIR / court).glob("*.parquet")):
        d = pd.read_parquet(f)
        d["src_file"] = f.name
        if "is_final" not in d.columns:
            d["is_final"] = True
        parts.append(d[["cnr", "title", "judge", "date_of_registration",
                        "decision_date", "disposal_nature", "is_final", "src_file"]])
    return pd.concat(parts, ignore_index=True)


def build_court(court: str) -> pd.DataFrame:
    d = load_court(court)
    n0 = len(d)

    # 1. one row per case: keep the final order (latest decision) for each CNR
    d["decision_date"] = pd.to_datetime(d["decision_date"])
    d = (d.sort_values(["is_final", "decision_date"])
           .drop_duplicates("cnr", keep="last"))
    # Some cases in the 'mobile' files only have INTERIM orders (no final order).
    # Their decision_date is not the disposal date, so the duration would be wrong.
    n_interim = int((~d["is_final"].astype(bool)).sum())
    d = d[d["is_final"].astype(bool)]
    print(f"{court:15s} dropped {n_interim} cases with no final order")

    # 2. dates (mixed formats handled) and target
    d["reg"] = parse_dates(d["date_of_registration"])
    d["days"] = (d["decision_date"] - d["reg"]).dt.days
    d = d[d["reg"].notna() & (d["days"] >= 0) & (d["reg"].dt.year >= 1980)].copy()

    # 3. title -> case type / parties (same format in all 5 courts)
    m = d["title"].str.extract(TITLE_RE)
    d["case_type_raw"] = m["type"].fillna("UNK")
    parties = m["rest"].fillna("").str.split(SPLIT_RE, n=1, regex=True)
    d["pet"] = parties.str[0].fillna("")
    d["res"] = parties.str[1].fillna("")

    # 4. judge info (only what is comparable across courts)
    j = d["judge"].fillna("").astype(str)
    j = j.where(~j.str.contains("NOT ALLOCATED", case=False), "")
    d["judge_missing"] = (j.str.strip() == "").astype(float)
    d["n_judges"] = j.str.split(",").map(lambda x: len([p for p in x if p.strip()])).clip(0, 5)
    d.loc[d["judge_missing"] == 1, "n_judges"] = 1          # unknown -> assume single
    cnt = j.map(j.value_counts())                             # cases per judge (local)
    d["judge_load"] = np.where(d["judge_missing"] == 1, 0.0,
                               np.log1p(cnt) / np.log1p(cnt.max()))

    # 5. feature table -------------------------------------------------------
    f = pd.DataFrame(index=d.index)
    f["cnr"] = d["cnr"]
    f["client"] = court
    f["decision_date"] = d["decision_date"]
    f["days"] = d["days"].astype(float)

    group = d["case_type_raw"].map(case_group)
    for g in GROUPS:
        f[f"grp_{g}"] = (group == g).astype(float)
    hb = d["case_type_raw"].map(hash_bucket)
    for i in range(N_HASH):
        f[f"hash_{i}"] = (hb == i).astype(float)

    f["n_judges"] = d["n_judges"] / 5.0
    f["is_division"] = (d["n_judges"] > 1).astype(float)
    f["judge_missing"] = d["judge_missing"]
    f["judge_load"] = d["judge_load"]
    f["title_len"] = np.log1p(d["title"].str.len()) / 6.0
    f["pet_state"] = d["pet"].str.contains(STATE_RE).astype(float)
    f["res_state"] = d["res"].str.contains(STATE_RE).astype(float)
    f["pet_company"] = d["pet"].str.contains(COMPANY_RE).astype(float)
    f["res_company"] = d["res"].str.contains(COMPANY_RE).astype(float)
    f["pet_multi"] = d["pet"].str.contains(MULTI_RE).astype(float)
    f["res_multi"] = d["res"].str.contains(MULTI_RE).astype(float)

    # date features (set B): how long the case COULD have been pending
    max_days = (WINDOW_END - d["reg"]).dt.days.clip(lower=0)
    f["log_max_days"] = np.log1p(max_days) / LOG_SCALE
    f["filed_in_window"] = (d["reg"] >= WINDOW_END - pd.Timedelta(days=365)).astype(float)

    print(f"{court:15s} raw={n0:6d}  clean={len(f):6d}  "
          f"OTHER-group={float((group=='OTHER').mean()):.1%}  "
          f"median days={f['days'].median():.0f}")
    return f.reset_index(drop=True)


def temporal_split(f: pd.DataFrame):
    """Oldest 70% of decision dates -> train, next 15% -> val, newest 15% -> test."""
    q70, q85 = f["decision_date"].quantile([0.70, 0.85])
    tr = f[f["decision_date"] <= q70]
    va = f[(f["decision_date"] > q70) & (f["decision_date"] <= q85)]
    te = f[f["decision_date"] > q85]
    return tr, va, te


def main():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    for court in COURTS:
        f = build_court(court)
        tr, va, te = temporal_split(f)
        out = PROCESSED_DIR / court
        out.mkdir(exist_ok=True)
        tr.to_parquet(out / "train.parquet", index=False)
        va.to_parquet(out / "val.parquet", index=False)
        te.to_parquet(out / "test.parquet", index=False)
        print(f"{'':15s} train={len(tr)} val={len(va)} test={len(te)}")


if __name__ == "__main__":
    main()