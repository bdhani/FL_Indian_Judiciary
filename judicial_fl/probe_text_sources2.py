"""STEP 2 - (a) look inside the PDF folders, (b) check the 'case_details'
metadata (may already contain acts/sections), (c) download 3 sample PDFs per
court and test whether text can be extracted.

Run:   pip install pypdf
       python -m judicial_fl.probe_text_sources2 > probe2_output.txt
Then send me probe2_output.txt (and tell me if anything crashed).
Downloads: a few small PDFs + one small parquet per court. No big files.
"""
import io
import traceback

import boto3
import pandas as pd
from botocore import UNSIGNED
from botocore.config import Config

from .common import RAW_DIR, COURTS, ROOT

BUCKET = "indian-high-court-judgments"
CODES = {"bombay": "27_1", "delhi": "7_26", "karnataka": "29_3",
         "kerala": "32_4", "madhya_pradesh": "23_23"}
s3 = boto3.client("s3", region_name="ap-south-1",
                  config=Config(signature_version=UNSIGNED))
SAMPLE_DIR = ROOT / "data" / "sample_pdfs"


def ls(prefix, n=6, delim=False):
    kw = dict(Bucket=BUCKET, Prefix=prefix, MaxKeys=n)
    if delim:
        kw["Delimiter"] = "/"
    r = s3.list_objects_v2(**kw)
    for p in r.get("CommonPrefixes", []):
        print("   [dir] ", p["Prefix"])
    for o in r.get("Contents", []):
        print(f"   [file] {o['Key']}  ({o['Size']/1e6:.2f} MB)")
    return r


def extract_text(path, max_pages=3):
    try:
        import fitz                                   # PyMuPDF if installed
        doc = fitz.open(path)
        return "\n".join(doc[i].get_text() for i in range(min(max_pages, len(doc))))
    except ImportError:
        from pypdf import PdfReader
        rd = PdfReader(str(path))
        return "\n".join((rd.pages[i].extract_text() or "") for i in range(min(max_pages, len(rd.pages))))


print("=" * 70, "\nA. INSIDE THE PDF / TAR FOLDERS (first files of the main bench)\n" + "=" * 70)
main_bench = {}
for court in COURTS:
    f = sorted((RAW_DIR / court).glob("metadata.parquet"))[0]
    d = pd.read_parquet(f, columns=["pdf_link"])
    links = d["pdf_link"].dropna()
    main_bench[court] = links.iloc[0].split("/")[2]    # e.g. 'testcase', 'dhcdb'
    print(f"\n--- {court}: bench in pdf_link = {main_bench[court]}")
    for kind in ["pdf", "tar"]:
        print(f" {kind}:")
        ls(f"data/{kind}/year=2024/court={CODES[court]}/bench={main_bench[court]}/", 4)

print("\n" + "=" * 70, "\nB. metadata/parquet_case_details (extra columns?)\n" + "=" * 70)
for court in COURTS:
    pre = f"metadata/parquet_case_details/year=2024/court={CODES[court]}/"
    print(f"\n--- {court}: {pre}")
    try:
        r = s3.list_objects_v2(Bucket=BUCKET, Prefix=pre, MaxKeys=50)
        objs = [o for o in r.get("Contents", []) if o["Key"].endswith(".parquet")]
        for o in objs[:5]:
            print(f"   [file] {o['Key']} ({o['Size']/1e6:.2f} MB)")
        if not objs:
            print("   (no parquet files found)")
            continue
        o = sorted(objs, key=lambda x: x["Size"])[-1]
        if o["Size"] > 300e6:
            print("   too big, skipped")
            continue
        body = s3.get_object(Bucket=BUCKET, Key=o["Key"])["Body"].read()
        df = pd.read_parquet(io.BytesIO(body))
        print(f"   read {o['Key'].split('/')[-1]}: shape {df.shape}")
        for c in df.columns:
            nn = df[c].notna().mean() * 100
            ex = df[c].dropna().astype(str).head(1).tolist()
            print(f"     {c:32s} non-null {nn:5.1f}%  e.g. {str(ex[0])[:90] if ex else ''}")
    except Exception:
        traceback.print_exc()

print("\n" + "=" * 70, "\nC. SAMPLE PDFs: can we download + extract text?\n" + "=" * 70)
for court in COURTS:
    d = pd.read_parquet(sorted((RAW_DIR / court).glob("metadata.parquet"))[0], columns=["pdf_link"])
    bench = main_bench[court]
    out = SAMPLE_DIR / court
    out.mkdir(parents=True, exist_ok=True)
    print(f"\n--- {court}")
    got = 0
    for link in d["pdf_link"].dropna().sample(8, random_state=1):
        base = link.split("/")[-1]
        for key in [f"data/pdf/year=2024/court={CODES[court]}/bench={bench}/{base}",
                    f"data/pdf/year=2024/court={CODES[court]}/bench={bench}/orders/{base}"]:
            try:
                s3.download_file(BUCKET, key, str(out / base))
            except Exception:
                continue
            txt = extract_text(out / base)
            print(f"  OK  {key}\n      chars={len(txt)}  start: {txt[:160]!r}")
            got += 1
            break
        else:
            print(f"  MISSING (tried 2 key patterns) for {base}")
        if got == 3:
            break
print("\nDONE")