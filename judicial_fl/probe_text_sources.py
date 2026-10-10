"""STEP 1 - find out where the judgment TEXT lives and how big it is.
Only reads/lists. Downloads nothing large.

Run:  python -m judicial_fl.probe_text_sources > probe_output.txt
Then send me probe_output.txt.
"""
import boto3
import pandas as pd
from botocore import UNSIGNED
from botocore.config import Config

from .common import RAW_DIR, COURTS

BUCKET = "indian-high-court-judgments"
CODES = {"bombay": "27_1", "delhi": "7_26", "karnataka": "29_3",
         "kerala": "32_4", "madhya_pradesh": "23_23"}
s3 = boto3.client("s3", region_name="ap-south-1",
                  config=Config(signature_version=UNSIGNED))


def list_level(prefix, limit=15):
    r = s3.list_objects_v2(Bucket=BUCKET, Prefix=prefix, Delimiter="/", MaxKeys=limit)
    for p in r.get("CommonPrefixes", [])[:limit]:
        print("   [dir] ", p["Prefix"])
    for o in r.get("Contents", [])[:limit]:
        print(f"   [file] {o['Key']}  ({o['Size']/1e6:.1f} MB)")


print("=" * 70, "\nA. BUCKET LAYOUT\n" + "=" * 70)
for top in ["", "data/", "data/pdf/", "data/tar/", "metadata/"]:
    print(f"\nprefix '{top}':")
    try:
        list_level(top)
    except Exception as e:
        print("   error:", e)

print("\n" + "=" * 70, "\nB. PER-COURT: pdf_link examples + folder contents\n" + "=" * 70)
for court in COURTS:
    f = sorted((RAW_DIR / court).glob("*.parquet"))[-1]
    d = pd.read_parquet(f, columns=["cnr", "pdf_link", "pdf_exists", "title"])
    print(f"\n--- {court} ({f.name}, {len(d)} rows)")
    print("pdf_exists:", d["pdf_exists"].value_counts(dropna=False).to_dict())
    print("pdf_link null %:", round(d["pdf_link"].isna().mean() * 100, 2))
    for v in d["pdf_link"].dropna().head(3):
        print("   pdf_link ->", v)
    for pre in [f"data/pdf/year=2024/court={CODES[court]}/",
                f"data/tar/year=2024/court={CODES[court]}/"]:
        print(" listing", pre)
        try:
            list_level(pre, 8)
        except Exception as e:
            print("   error:", e)
print("\nDONE")