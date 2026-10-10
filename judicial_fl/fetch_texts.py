"""STEP 3 - download a random sample of judgments per court and extract text.

Why a sample: the full 2024 PDFs are ~20 GB. 2,500 per court (12,500 total) is
plenty to build and test Tasks 1-2, and the download resumes if interrupted.

SMOKE TEST FIRST (2-3 min):   python -m judicial_fl.fetch_texts 100
FULL SAMPLE (about 20-40 min): python -m judicial_fl.fetch_texts 2500

Needs:  pip install pymupdf   (pypdf also works, just slower)
Output: data/texts/<court>.parquet   (cnr, split, days, n_pages, n_chars, text)
        The sample is chosen from the SAME cases as your processed_v2 tables,
        so the temporal train/val/test split is preserved.
"""
import io
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import boto3
import pandas as pd
from botocore import UNSIGNED
from botocore.config import Config

from .common import RAW_DIR, PROCESSED_DIR, COURTS, ROOT

BUCKET = "indian-high-court-judgments"
CODES = {"bombay": "27_1", "delhi": "7_26", "karnataka": "29_3",
         "kerala": "32_4", "madhya_pradesh": "23_23"}
TEXT_DIR = ROOT / "data" / "texts"
MAX_CHARS = 80_000
THREADS = 16

s3 = boto3.client("s3", region_name="ap-south-1",
                  config=Config(signature_version=UNSIGNED, max_pool_connections=THREADS + 4,
                                retries={"max_attempts": 3}))


def select_sample(court, n, seed=42):
    """Pick n cases that exist in processed_v2, with their PDF link + split."""
    parts = []
    for f in sorted((RAW_DIR / court).glob("*.parquet")):
        d = pd.read_parquet(f)
        if "is_final" not in d.columns:
            d["is_final"] = True
        parts.append(d[["cnr", "pdf_link", "decision_date", "is_final"]])
    raw = pd.concat(parts, ignore_index=True)
    raw["decision_date"] = pd.to_datetime(raw["decision_date"])
    # same rule as prepare_v2: keep the final order of each case
    raw = raw.sort_values(["is_final", "decision_date"]).drop_duplicates("cnr", keep="last")
    # only links like 'court/cnrorders/<bench>/orders/<file>.pdf' map to the S3 folders
    # (the 'mobile' rows use another naming scheme, so those cases are skipped here)
    raw = raw[raw["pdf_link"].fillna("").str.startswith("court/")]

    proc = []
    for split in ("train", "val", "test"):
        p = pd.read_parquet(PROCESSED_DIR / court / f"{split}.parquet", columns=["cnr", "days"])
        p["split"] = split
        proc.append(p)
    proc = pd.concat(proc)
    m = proc.merge(raw[["cnr", "pdf_link", "decision_date"]], on="cnr", how="inner")
    m = m.sample(n=min(n, len(m)), random_state=seed).reset_index(drop=True)
    # Re-do the temporal split INSIDE the sample (per court): the downloadable PDFs of
    # some courts (e.g. MP) cover only part of 2024, so the old split would be lopsided.
    q70, q85 = m["decision_date"].quantile([0.70, 0.85])
    m["split"] = "train"
    m.loc[m["decision_date"] > q70, "split"] = "val"
    m.loc[m["decision_date"] > q85, "split"] = "test"
    # pdf_link looks like court/cnrorders/<bench>/orders/<file>.pdf
    m["bench"] = m["pdf_link"].str.split("/").str[2]
    m["fname"] = m["pdf_link"].str.split("/").str[-1]
    m["key"] = ("data/pdf/year=2024/court=" + CODES[court] + "/bench=" + m["bench"] + "/" + m["fname"])
    return m


def download(key):
    try:
        return key, s3.get_object(Bucket=BUCKET, Key=key)["Body"].read()
    except Exception as e:
        return key, None


def extract(pdf_bytes):
    """Returns (text, n_pages). Runs in the main thread (PyMuPDF is not thread-safe)."""
    try:
        import fitz
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        txt = "\n".join(p.get_text() for p in doc)
        return txt[:MAX_CHARS], len(doc)
    except ImportError:
        from pypdf import PdfReader
        rd = PdfReader(io.BytesIO(pdf_bytes))
        txt = "\n".join((p.extract_text() or "") for p in rd.pages)
        return txt[:MAX_CHARS], len(rd.pages)
    except Exception:
        return "", 0


def run_court(court, n):
    out = TEXT_DIR / f"{court}.parquet"
    if out.exists():
        print(f"{court}: already done ({out.name}); delete it to redo")
        return
    sample = select_sample(court, n)
    print(f"\n{court}: fetching {len(sample)} PDFs ...", flush=True)
    t0 = time.time()
    results = {}
    with ThreadPoolExecutor(THREADS) as ex:
        futs = [ex.submit(download, k) for k in sample["key"]]
        for i, f in enumerate(as_completed(futs), 1):
            key, data = f.result()
            if data is not None:
                results[key] = extract(data)
            if i % 250 == 0:
                print(f"   {i}/{len(sample)}  ({time.time()-t0:.0f}s)", flush=True)

    sample["text"] = sample["key"].map(lambda k: results.get(k, ("", 0))[0])
    sample["n_pages"] = sample["key"].map(lambda k: results.get(k, ("", 0))[1])
    sample["n_chars"] = sample["text"].str.len()
    failed = (~sample["key"].isin(results)).sum()
    sample["client"] = court
    TEXT_DIR.mkdir(parents=True, exist_ok=True)
    sample[["cnr", "client", "split", "days", "decision_date", "key", "n_pages", "n_chars", "text"]].to_parquet(out, index=False)

    print(f"{court}: saved {len(sample)} rows | download failures={failed} | "
          f"empty/scanned (<300 chars)={(sample['n_chars'] < 300).mean():.1%} | "
          f"median chars={sample['n_chars'].median():.0f} | median pages={sample['n_pages'].median():.0f}")


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    for court in COURTS:
        run_court(court, n)
    print("\nDONE. Send me the lines starting with the court names above.")


if __name__ == "__main__":
    main()