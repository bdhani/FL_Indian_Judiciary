"""STEP 4 - build the Task-1 (outcome) and Task-2 (acts / sections / articles)
labels from the judgment text, plus a leak-free model input text.

Run:   python -m judicial_fl.build_labels
Takes about 1.5 minutes. Output:
   data/labels/<court>.parquet      one row per judgment
   data/labels/vocab.json           fixed label vocabulary for Task 2
   results/label_summary.txt        <- send me this file
"""
import collections
import json
import sys

import pandas as pd

from judicial_fl.common import ROOT, COURTS, RESULTS_DIR
from judicial_fl.legal_labels import (extract_outcome, extract_statutes, make_input_text, is_garbled)

TEXT_DIR = ROOT / "data" / "texts"
LABEL_DIR = ROOT / "data" / "labels"
MIN_DOCS = {"acts": 30, "secs": 40, "arts": 15}      # keep labels seen in >= this many judgments

lines = []


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    lines.append(s)


def main():
    LABEL_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(exist_ok=True)
    frames = []
    for court in COURTS:
        d = pd.read_parquet(TEXT_DIR / f"{court}.parquet")
        print(f"processing {court} ({len(d)}) ...", flush=True)
        out = d["text"].map(extract_outcome)
        d["outcome"] = out.str[0]
        d["outcome_evidence"] = out.str[1]
        d["garbled"] = d["text"].map(is_garbled)
        st = d["text"].map(extract_statutes)
        d["acts"], d["secs"], d["arts"] = st.str[0], st.str[1], st.str[2]
        d["input_text"] = d["text"].map(make_input_text)
        frames.append(d.drop(columns=["text"]))
    df = pd.concat(frames, ignore_index=True)

    # ---- fixed Task-2 vocabulary (pooled over courts, rare labels dropped)
    vocab = {}
    for col in ("acts", "secs", "arts"):
        cnt = collections.Counter(x for l in df.loc[~df["garbled"], col] for x in l)
        vocab[col] = sorted([k for k, v in cnt.items() if v >= MIN_DOCS[col]],
                            key=lambda k: -cnt[k])
    with open(LABEL_DIR / "vocab.json", "w") as f:
        json.dump(vocab, f, indent=1)

    for col in ("acts", "secs", "arts"):                     # lists -> 'A|B|C' for parquet
        keep = set(vocab[col])
        df[col + "_kept"] = df[col].map(lambda l: "|".join(x for x in l if x in keep))
        df[col] = df[col].map("|".join)
    df["has_task2_label"] = (df["acts_kept"] != "") | (df["secs_kept"] != "") | (df["arts_kept"] != "")
    for court in COURTS:
        df[df["client"] == court].to_parquet(LABEL_DIR / f"{court}.parquet", index=False)

    # ---- summary
    say("=" * 70, "\nTASK 1: OUTCOME LABELS (share per court)\n" + "=" * 70)
    say(pd.crosstab(df["client"], df["outcome"], normalize="index").round(3).to_string())
    say("\ngarbled/unreadable PDFs per court:", df.groupby("client")["garbled"].mean().round(3).to_dict())

    say("\n" + "=" * 70, "\nTASK 2: STATUTE LABELS\n" + "=" * 70)
    say("vocabulary sizes:", {k: len(v) for k, v in vocab.items()})
    say("\nshare of judgments with >=1 usable Task-2 label, per court:")
    say(df.groupby("client")["has_task2_label"].mean().round(3).to_string())
    say("\ntotal judgments with a Task-2 label (non-garbled):",
        int((df["has_task2_label"] & ~df["garbled"]).sum()), "of", len(df))
    say("\nTOP 15 acts:", vocab["acts"][:15])
    say("TOP 20 sections:", vocab["secs"][:20])
    say("TOP articles:", vocab["arts"][:8])

    say("\nact share per court (how different the courts are = non-IID):")
    rows = {}
    for court in COURTS:
        g = df[(df["client"] == court) & ~df["garbled"]]
        rows[court] = {a: round(g["acts_kept"].str.contains(rf"(?:^|\|){a}(?:\||$)").mean(), 3)
                       for a in vocab["acts"][:10]}
    say(pd.DataFrame(rows).to_string())

    say("\ntrain/val/test sizes per court:")
    say(pd.crosstab(df["client"], df["split"]).to_string())
    say("\nrandom outcome evidence (for manual checking):")
    for _, r in df[~df["garbled"]].sample(12, random_state=0).iterrows():
        say(f"  [{r['client']:14s}] {r['outcome']:9s} <- ...{r['outcome_evidence'][-110:]!r}")

    (RESULTS_DIR / "label_summary.txt").write_text("\n".join(lines), encoding="utf-8")
    print("\nDONE. Send me results\\label_summary.txt")


if __name__ == "__main__":
    main()