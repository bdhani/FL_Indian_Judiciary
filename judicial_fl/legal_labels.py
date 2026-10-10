"""Rule-based label extraction from judgment text (no manual annotation).

 * extract_outcome(text)  -> ALLOWED / DISMISSED / WITHDRAWN / DISPOSED / UNCLEAR
 * extract_statutes(text) -> (acts, sections, articles)
 * make_input_text(text)  -> text safe to feed a model: statute references masked,
                             operative ending removed (so labels are not leaked).
"""
import re

# ------------------------------------------------------------------ cleaning
BOILER = [
    r"This is a digitally signed order.*?(?:\d{2}:\d{2}:\d{2}|$)",
    r"Signed by:.*?Signature Not Verified",
    r"Signature Not Verified",
    r"Digitally signed by.*?(?=\s[A-Z]{2,}|\Z)",
    r"Page \d+ of \d+",
    r"-\s*\d+\s*-",
    r"Sd/-",
    r"List No\.:\s*\d+\s*Sl No\.:\s*\d+",
    r"NC:\s*\d{4}:KHC[^ ]*",
    r"\b\d{4}:KHC:\d+(?:-DB)?",
    r"\*{3,}",
]
APPENDIX = re.compile(
    r"APPENDIX|PETITIONER'?S?\s+(?:EXHIBITS|ANNEXURES?)|RESPONDENT'?S?\s+(?:EXHIBITS|ANNEXURES?)|"
    r"(?:EXHIBIT|Exhibit)\s*-?P-?1\s+(?:A\s+)?(?:TRUE|True)|ANNEXURE\s*[-]?\s*A\s+(?:TRUE|True)")   # case-sensitive on purpose


def clean(text: str) -> str:
    t = re.sub(r"\s+", " ", text or "")
    for b in BOILER:
        t = re.sub(b, " ", t, flags=re.I | re.S)
    return re.sub(r"\s+", " ", t).strip()


def body_without_appendix(t: str) -> str:
    """Cut the exhibit list that Kerala (and others) put after the order.
    Only heading-style markers in the last 60% of the text count, so the word
    'Annexure-A' inside the prayer / body does not truncate the order."""
    for m in APPENDIX.finditer(t):
        if m.start() > len(t) * 0.40:
            return t[:m.start()]
    return t


def is_garbled(text: str) -> bool:
    """Some PDFs have broken font encoding -> unreadable text."""
    w = re.findall(r"[A-Za-z]+", (text or "")[:3000])
    if len(w) < 30:
        return True
    common = sum(x.lower() in {"the", "of", "and", "to", "in", "is", "for", "that", "court", "petitioner", "order"} for x in w)
    return common / len(w) < 0.04


# ------------------------------------------------------------------ outcome
OUT = {
    "WITHDRAWN": r"withdrawn|withdraw(?:s|ing)? the (?:petition|appeal|application|case)|permitted to withdraw|memo of withdrawal",
    "DISMISSED": r"dismiss(?:ed|es|al)|rejected|not entertained|no merit|devoid of merit|without merit|declined|not inclined to interfere",
    "ALLOWED": r"(?:partly |partially |is |are |stands? |be |hereby |accordingly )allowed|allowed in part|is granted|are granted|bail is granted|be released on bail|quashed|set aside|stands? allowed|allowed and disposed|petition succeeds|appeal succeeds|rule is made absolute|rule absolute|admitted",
    "DISPOSED": r"dispos(?:ed|es|al)[ -]?(?:of|off)?|infructuous|closed|stands? closed|nothing survives|no further order|no orders? (?:are )?required|accordingly\.?$",
}
OUT_RE = {k: re.compile(v, re.I) for k, v in OUT.items()}
NEG_ALLOWED = re.compile(r"not\s+(?:be\s+)?allowed|disallowed|cannot be allowed|no ground to allow", re.I)


def operative_window(text: str, n=4000) -> str:
    t = body_without_appendix(clean(text))
    return t[-n:]


def extract_outcome(text: str):
    """Looks at the LAST operative statements and returns (label, evidence)."""
    if is_garbled(text):
        return "UNCLEAR", "GARBLED_TEXT"
    w = operative_window(text)
    best = None                                # (end_pos, label, snippet)
    for label, rx in OUT_RE.items():
        for m in rx.finditer(w):
            if label == "ALLOWED" and NEG_ALLOWED.search(w[max(0, m.start() - 15): m.end()]):
                continue
            if label == "ALLOWED" and m.group(0).lower() == "admitted":
                continue
            cand = (m.end(), label, w[max(0, m.start() - 60): m.end() + 40])
            if best is None or cand[0] > best[0]:
                best = cand
    if best is None:
        return "UNCLEAR", ""
    label, snippet = best[1], best[2]
    # 'dismissed as withdrawn', 'disposed of as withdrawn' -> WITHDRAWN
    tail = w[max(0, best[0] - 80): best[0] + 60]
    if label in ("DISMISSED", "DISPOSED") and re.search(r"withdraw", tail, re.I):
        label = "WITHDRAWN"
    if label == "DISMISSED" and re.search(r"infructuous", tail, re.I):
        label = "DISPOSED"
    # a late 'disposed of' after an earlier allowed/dismissed: 'allowed and disposed of'
    if label == "DISPOSED":
        for l2 in ("ALLOWED", "DISMISSED", "WITHDRAWN"):
            m2 = list(OUT_RE[l2].finditer(w[-400:]))
            ok = [m for m in m2 if not NEG_ALLOWED.search(w[-400:][max(0, m.start() - 15): m.end()])]
            if ok and best[0] - (len(w) - 400 + ok[-1].end()) < 80:
                label = l2
                break
    return label, snippet


# ------------------------------------------------------------------ statutes
ACTS = {  # code -> regex that names the act (case-insensitive)
    "IPC": r"I\.?\s?P\.?\s?C\b\.?|Indian\s+Penal\s+Code",
    "BNS": r"B\.?\s?N\.?\s?S\b\.?(?!S)|Bharatiya\s+Nyaya\s+Sanhita",
    "CRPC": r"Cr\.?\s?P\.?\s?C\b\.?|Code\s+of\s+Criminal\s+Procedure",
    "BNSS": r"B\.?\s?N\.?\s?S\.?\s?S\b\.?|Bharatiya\s+Nagarik\s+Suraksha\s+Sanhita",
    "CPC": r"(?<!Cr\.)(?<!Cr)\bC\.?\s?P\.?\s?C\b\.?|Code\s+of\s+Civil\s+Procedure",
    "NI_ACT": r"N\.?\s?I\.?\s?Act|Negotiable\s+Instruments\s+Act",
    "MV_ACT": r"M\.?\s?V\.?\s?Act|Motor\s+Vehicles?\s+Act",
    "ARB_ACT": r"Arbitration\s+(?:and|&)\s+Conciliation\s+Act|A\s?&\s?C\s+Act",
    "NDPS": r"N\.?D\.?P\.?S\.?\s*Act|Narcotic\s+Drugs\s+and\s+Psychotropic",
    "POCSO": r"POCSO|Protection\s+of\s+Children\s+from\s+Sexual\s+Offences",
    "IT_ACT": r"Income[\s-]*Tax\s+Act|I\.?T\.?\s+Act",
    "COMPANIES": r"Companies\s+Act",
    "SARFAESI": r"SARFAESI|Securitisation\s+and\s+Reconstruction",
    "IBC": r"Insolvency\s+and\s+Bankruptcy\s+Code|I\.?B\.?C\b",
    "DV_ACT": r"Domestic\s+Violence",
    "HMA": r"Hindu\s+Marriage\s+Act",
    "LIMITATION": r"Limitation\s+Act",
    "EVIDENCE": r"Evidence\s+Act|Bharatiya\s+Sakshya",
    "SC_ST": r"Scheduled\s+Castes?\s+and\s+(?:the\s+)?Scheduled\s+Tribes?|SC\s?/\s?ST\s*\(?P\.?A\.?\)?\s*Act",
    "LAND_ACQ": r"Land\s+Acquisition|Right\s+to\s+Fair\s+Compensation",
    "CONTEMPT": r"Contempt\s+of\s+Courts?\s+Act",
    "RTI": r"Right\s+to\s+Information\s+Act",
    "ELECTRICITY": r"Electricity\s+Act",
    "INDUSTRIAL": r"Industrial\s+Disputes\s+Act",
    "PMLA": r"PMLA|Prevention\s+of\s+Money[\s-]*Laundering",
    "UAPA": r"UAPA|Unlawful\s+Activities",
    "CUSTOMS": r"Customs\s+Act",
    "GST": r"GST\s+Act|Goods\s+and\s+Services\s+Tax",
}
ACT_RX = {k: re.compile(v, re.I) for k, v in ACTS.items()}
ANY_ACT = "(?:" + "|".join(f"(?P<a_{k}>{v})" for k, v in ACTS.items()) + ")"
NUM = r"\d+[A-Za-z]?(?:\(\d+\))?(?:\([a-z]\))?"
NUMLIST = rf"{NUM}(?:\s*(?:,|and|&|/|to|or)\s*{NUM})*"
SEC_RX = re.compile(
    rf"(?:Sections?|Secs?\.?|S\.|u/s\.?|ss\.)\s*(?P<nums>{NUMLIST})"
    rf"(?P<gap>[^.;]{{0,45}}?)\b(?:of\s+(?:the\s+)?|under\s+(?:the\s+)?|r/w\s+|read\s+with\s+)?{ANY_ACT}",
    re.I)
ART_RX = re.compile(rf"Articles?\s*(?P<nums>{NUMLIST})", re.I)
CONST_RX = re.compile(r"Constitution", re.I)


def _base(num: str) -> str:
    return re.match(r"\d+[A-Za-z]?", num).group(0).upper()


def extract_statutes(text: str):
    t = clean(text)
    acts, secs, arts = set(), set(), set()
    for k, rx in ACT_RX.items():                       # act mentions (even without a section)
        if rx.search(t):
            acts.add(k)
    for m in SEC_RX.finditer(t):
        act = next(k for k in ACTS if m.group(f"a_{k}"))
        for n in re.findall(NUM, m.group("nums")):
            secs.add(f"{act}_{_base(n)}")
            acts.add(act)
    if CONST_RX.search(t):
        acts.add("CONSTITUTION")
        for m in ART_RX.finditer(t):
            for n in re.findall(NUM, m.group("nums")):
                arts.add(f"ART_{_base(n)}")
    return sorted(acts), sorted(secs), sorted(arts)


# ------------------------------------------------------------------ model input
MASK_RX = [SEC_RX, ART_RX]


def make_input_text(text: str, max_chars=3000) -> str:
    """Model input: no statute references, no operative ending."""
    t = body_without_appendix(clean(text))
    t = t[: max(0, len(t) - 1200)] if len(t) > 2500 else t[: int(len(t) * 0.7)]   # drop the ending
    t = SEC_RX.sub(" [STATUTE] ", t)
    t = ART_RX.sub(" [STATUTE] ", t)
    for rx in ACT_RX.values():
        t = rx.sub(" [ACT] ", t)
    return re.sub(r"\s+", " ", t).strip()[:max_chars]