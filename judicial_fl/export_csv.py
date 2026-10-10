"""STEP 4b - export analysis-ready CSV files (open in Excel / use in any other model).

Run AFTER build_labels:   python -m judicial_fl.export_csv

Creates folder  exports/ :
  1. judgments_labelled.csv   one row per judgment in the text sample (about 12,500)
                              metadata + dates + outcome + acts/sections/articles + model input text
  2. task2_multihot.csv       cnr + one 0/1 column per Task-2 label (y_CRPC_439 ...)
  3. metadata_all_cases.csv   ALL cleaned cases (about 190k) - no text, no labels
                              (features available from the metadata alone)

Columns whose name starts with  audit_  are for FAIRNESS CHECKING ONLY.
Do not feed them to a model as inputs.
"""
import json
import re

import numpy as np
import pandas as pd

from .common import *
from .prepare_v2 import load_court
from .legal_labels import ACTS

LABEL_DIR = ROOT / "data" / "labels"
EXPORT_DIR = ROOT / "exports"

FEMALE = re.compile(r"\b(SMT|SMT\.|MRS|MRS\.|MS|MS\.|KU|KU\.|KUMARI|MISS|W/O|D/O|WIDOW|BEGUM|DEVI|BAI)\b", re.I)
MALE = re.compile(r"\b(SHRI|SRI|MR|MR\.|S/O|MASTER|SHRI\.|SRI\.)\b", re.I)


def party_type(s: str) -> str:
    if STATE_RE.search(s):
        return "STATE_OR_PUBLIC_BODY"
    if COMPANY_RE.search(s):
        return "COMPANY_OR_ORG"
    return "INDIVIDUAL"


def gender_marker(s: str, ptype: str) -> str:
    if ptype != "INDIVIDUAL":
        return "NA"
    f, m = bool(FEMALE.search(s)), bool(MALE.search(s))
    if f and not m:
        return "FEMALE_MARKER"
    if m and not f:
        return "MALE_MARKER"
    return "UNKNOWN"


def court_table(court: str) -> pd.DataFrame:
    d = load_court(court)
    d["decision_date"] = pd.to_datetime(d["decision_date"])
    d = d.sort_values(["is_final", "decision_date"]).drop_duplicates("cnr", keep="last")
    d = d[d["is_final"].astype(bool)].copy()
    d["date_of_registration"] = parse_dates(d["date_of_registration"])
    d["duration_days"] = (d["decision_date"] - d["date_of_registration"]).dt.days
    d = d[d["date_of_registration"].notna() & (d["duration_days"] >= 0)
          & (d["date_of_registration"].dt.year >= 1980)].copy()

    m = d["title"].str.extract(TITLE_RE)
    d["case_type_raw"] = m["type"].fillna("UNK")
    d["case_type_group"] = d["case_type_raw"].map(case_group)
    d["case_number"] = m["no"]
    d["case_year"] = m["yr"]
    parties = m["rest"].fillna("").str.split(SPLIT_RE, n=1, regex=True)
    d["petitioner"] = parties.str[0].fillna("")
    d["respondent"] = parties.str[1].fillna("")

    j = d["judge"].fillna("").astype(str)
    j = j.where(~j.str.contains("NOT ALLOCATED", case=False), "")
    d["judge_raw"] = j
    d["n_judges"] = j.str.split(",").map(lambda x: len([p for p in x if p.strip()])).replace(0, np.nan)

    d["audit_petitioner_type"] = d["petitioner"].map(party_type)
    d["audit_respondent_type"] = d["respondent"].map(party_type)
    d["audit_petitioner_gender_marker"] = [gender_marker(s, t) for s, t in zip(d["petitioner"], d["audit_petitioner_type"])]
    d["audit_respondent_gender_marker"] = [gender_marker(s, t) for s, t in zip(d["respondent"], d["audit_respondent_type"])]

    d["duration_bucket"] = pd.Series(duration_bucket(d["duration_days"]), index=d.index).map(
        dict(enumerate(BUCKET_NAMES)))
    d["court"] = court
    d["disposal_nature_metadata"] = d["disposal_nature"]
    cols = ["cnr", "court", "case_type_raw", "case_type_group", "case_number", "case_year",
            "date_of_registration", "decision_date", "duration_days", "duration_bucket",
            "disposal_nature_metadata", "n_judges", "judge_raw", "petitioner", "respondent",
            "audit_petitioner_type", "audit_respondent_type",
            "audit_petitioner_gender_marker", "audit_respondent_gender_marker"]
    out = d[cols].copy()
    for c in ("date_of_registration", "decision_date"):
        out[c] = out[c].dt.strftime("%Y-%m-%d")
    return out


def main():
    EXPORT_DIR.mkdir(exist_ok=True)
    tables = [court_table(c) for c in COURTS]
    meta = pd.concat(tables, ignore_index=True)
    meta.to_csv(EXPORT_DIR / "metadata_all_cases.csv", index=False, encoding="utf-8-sig")
    print(f"metadata_all_cases.csv : {len(meta)} rows")

    lab = pd.concat([pd.read_parquet(LABEL_DIR / f"{c}.parquet") for c in COURTS], ignore_index=True)
    lab = lab.rename(columns={"outcome": "outcome_from_text", "acts_kept": "acts",
                              "secs_kept": "sections", "arts_kept": "articles",
                              "secs": "sections_all", "acts": "acts_all", "arts": "articles_all"})
    # 'acts'/'secs' were renamed in two steps above; make sure we keep the filtered ones
    lab = lab.drop(columns=[c for c in ["client", "days", "decision_date"] if c in lab.columns])
    full = meta.merge(lab, on="cnr", how="inner")
    keep = ["cnr", "court", "split", "case_type_raw", "case_type_group", "case_number", "case_year",
            "date_of_registration", "decision_date", "duration_days", "duration_bucket",
            "disposal_nature_metadata", "outcome_from_text", "outcome_evidence",
            "acts", "sections", "articles", "has_task2_label", "garbled",
            "n_judges", "judge_raw", "petitioner", "respondent",
            "audit_petitioner_type", "audit_respondent_type",
            "audit_petitioner_gender_marker", "audit_respondent_gender_marker",
            "n_pages", "n_chars", "input_text"]
    full = full[keep]
    full.to_csv(EXPORT_DIR / "judgments_labelled.csv", index=False, encoding="utf-8-sig")
    print(f"judgments_labelled.csv : {len(full)} rows, {len(keep)} columns")

    vocab = json.load(open(LABEL_DIR / "vocab.json"))
    mh = pd.DataFrame({"cnr": full["cnr"]})
    for col, src in (("acts", "acts"), ("secs", "sections"), ("arts", "articles")):
        sets = full[src].fillna("").map(lambda s: set(s.split("|")) - {""})
        for lab_name in vocab[col]:
            mh[f"y_{lab_name}"] = sets.map(lambda s: int(lab_name in s))
    mh.to_csv(EXPORT_DIR / "task2_multihot.csv", index=False)
    print(f"task2_multihot.csv     : {mh.shape[1]-1} label columns")
    print("\nDONE. Files are in", EXPORT_DIR)


if __name__ == "__main__":
    main()