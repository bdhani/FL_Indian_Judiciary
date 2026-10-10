"""Shared settings + helpers. Paths are relative to the project folder, so the
same code runs on F:\\FL_Indian_Judiciary (Windows) or anywhere else."""
from pathlib import Path
import re
import zlib

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed_v2"
RESULTS_DIR = ROOT / "results"

COURTS = ["bombay", "delhi", "karnataka", "kerala", "madhya_pradesh"]

# Window end of the data you downloaded (year=2024). Used for the
# "maximum possible duration" feature. Change it if you add other years.
WINDOW_END = pd.Timestamp("2024-12-31")

DURATION_BINS = [30, 180, 730]          # days -> 4 classes
BUCKET_NAMES = ["<=30d", "31-180d", "181-730d", ">730d"]
LOG_SCALE = 10.0                         # fixed constant: y = log1p(days)/10

# ------------------------------------------------------------------ dates
def parse_dates(s: pd.Series) -> pd.Series:
    """Parse a column that mixes 2024-04-01 (mobile files) and 01-04-2024
    (normal files). pandas guesses ONE format from the first row and turns
    everything else into NaT - that is what dropped most Bombay / MP rows."""
    s = s.astype("string")
    iso = pd.to_datetime(s.where(s.str.match(r"^\d{4}-\d{2}-\d{2}$")),
                         format="%Y-%m-%d", errors="coerce")
    dmy = pd.to_datetime(s.where(s.str.match(r"^\d{2}-\d{2}-\d{4}$")),
                         format="%d-%m-%Y", errors="coerce")
    return iso.fillna(dmy)

# ------------------------------------------------------------------ title
# Every court writes the title as:  TYPE/NUMBER/YEAR of PETITIONER Vs RESPONDENT
TITLE_RE = re.compile(
    r"^\s*(?P<type>.+?)\s*/\s*(?P<no>\d+)\s*/\s*(?P<yr>\d{4})\s+of\s+(?P<rest>.*)$",
    re.S)
SPLIT_RE = re.compile(r"\s+vs\.?\s+", re.I)

STATE_RE = re.compile(
    r"\b(?:STATE|UNION OF INDIA|GOVT|GOVERNMENT|COMMISSIONER|COLLECTOR|SECRETARY|"
    r"DIRECTOR|POLICE|MUNICIPAL|CORPORATION|BOARD|AUTHORITY|DEPARTMENT|"
    r"TAHSILDAR|REGISTRAR|SUPERINTENDENT)\b", re.I)
COMPANY_RE = re.compile(
    r"\b(?:M/S|LTD|LIMITED|PVT|PRIVATE|BANK|INSURANCE|COMPANY|CO\.|FINANCE|"
    r"INDUSTRIES|ENTERPRISES|TRUST|SOCIETY|ASSOCIATION)\b", re.I)
MULTI_RE = re.compile(r"\b(?:ORS|OTHERS|ANR|ANOTHER|OTHER)\b", re.I)


def norm_type(t: str) -> str:
    return re.sub(r"[^A-Z]", "", str(t).upper())


def case_group(t: str) -> str:
    """Map each court's own case-type code to a shared family."""
    n = norm_type(t)
    if not n:
        return "OTHER"
    if "BAIL" in n or n in {"BA", "ABA", "MCRC", "CRLBA", "ANTBAIL"}:
        return "BAIL"
    if n.startswith("WP") or n in {"PIL", "WRIT"}:
        return "WRIT"
    if n.startswith("CONT") or n in {"CONC", "CCC", "CONCASE", "COMA"}:
        return "CONTEMPT"
    if n.startswith("ARB") or n in {"AA", "AP", "OMP", "ARBP"}:
        return "ARBITRATION"
    if n.startswith("CRL") or n in {"CRA", "CRR", "APL", "APPA", "CRIA"}:
        return "CRIMINAL_OTHER"
    if n in {"RP", "CRP", "REVIEW", "REV", "CMM", "CM", "CMAPPL"} or n.startswith("REV"):
        return "REVIEW_REVISION"
    if n.startswith("ITA") or n.startswith("TAX") or n in {"ST", "WT", "VAT", "CEA", "ITR"}:
        return "TAX"
    if n.startswith("MAC") or n.endswith("MACA"):
        return "MOTOR_ACCIDENT"
    if n in {"FA", "RFA", "MFA", "SA", "RSA", "MA", "FAO", "CMA", "WA", "LPA", "AO",
             "CAF", "CAO", "CAW", "APPEAL", "OP", "OS", "CS", "CM"}:
        return "CIVIL_APPEAL_OR_SUIT"
    return "OTHER"


GROUPS = ["WRIT", "BAIL", "CRIMINAL_OTHER", "CIVIL_APPEAL_OR_SUIT", "CONTEMPT",
          "ARBITRATION", "REVIEW_REVISION", "TAX", "MOTOR_ACCIDENT", "OTHER"]
N_HASH = 32


def hash_bucket(s: str, n: int = N_HASH) -> int:
    return zlib.crc32(norm_type(s).encode()) % n


# ------------------------------------------------------------------ labels
def duration_bucket(days) -> np.ndarray:
    return np.digitize(np.asarray(days), DURATION_BINS, right=True)
