"""
query_utils.py
==============
Medical query cleaning and expansion utilities.

  clean_query          — normalises abbreviations, strips boilerplate
  expand_medical_query — appends domain synonyms and intent keywords

Usage
-----
  from query_utils import clean_query, expand_medical_query
  q = expand_medical_query(clean_query("45 y/o M c/o SOB & CP. Dx?"))
"""

import re
import html

# ── Abbreviation expansion map ────────────────────────────────────
ABBREVIATION_MAP = {
    r"\by/o\b": "year old", r"\byr\b": "year", r"\byrs\b": "years",
    r"\bM\b": "male", r"\bF\b": "female",
    r"\bpt\b": "patient", r"\bpts\b": "patients",
    r"\bhx\b": "history", r"\bh/o\b": "history of",
    r"\bpmh\b": "past medical history",
    r"\bc/o\b": "complains of", r"\bcc\b": "chief complaint",
    r"\bsob\b": "shortness of breath", r"\bcp\b": "chest pain",
    r"\bn/v\b": "nausea and vomiting",
    r"\bha\b": "headache", r"\babd\b": "abdominal",
    r"\bams\b": "altered mental status",
    r"\bhr\b": "heart rate", r"\brr\b": "respiratory rate",
    r"\bbp\b": "blood pressure", r"\btemp\b": "temperature",
    r"\bdm\b": "diabetes mellitus", r"\bdm2\b": "diabetes mellitus type 2",
    r"\bhtn\b": "hypertension", r"\bhf\b": "heart failure",
    r"\bchf\b": "congestive heart failure", r"\bcad\b": "coronary artery disease",
    r"\bmi\b": "myocardial infarction", r"\bstemi\b": "ST-elevation myocardial infarction",
    r"\bacs\b": "acute coronary syndrome",
    r"\bpe\b": "pulmonary embolism", r"\bdvt\b": "deep vein thrombosis",
    r"\bcva\b": "cerebrovascular accident stroke", r"\btia\b": "transient ischemic attack",
    r"\bcopd\b": "chronic obstructive pulmonary disease",
    r"\buti\b": "urinary tract infection", r"\bckd\b": "chronic kidney disease",
    r"\baki\b": "acute kidney injury", r"\bgerd\b": "gastroesophageal reflux disease",
    r"\bcbc\b": "complete blood count", r"\bcmp\b": "comprehensive metabolic panel",
    r"\becg\b": "electrocardiogram", r"\bekg\b": "electrocardiogram",
    r"\bcxr\b": "chest x-ray", r"\bct\b": "computed tomography",
    r"\bmri\b": "magnetic resonance imaging",
    r"\bwbc\b": "white blood cell", r"\brbc\b": "red blood cell",
    r"\bhgb\b": "hemoglobin", r"\binr\b": "international normalised ratio",
    r"\bbun\b": "blood urea nitrogen", r"\bgfr\b": "glomerular filtration rate",
    r"\btsh\b": "thyroid stimulating hormone",
    r"\bnsaid\b": "non-steroidal anti-inflammatory drug",
    r"\bace\b": "angiotensin converting enzyme",
    r"\bssri\b": "selective serotonin reuptake inhibitor",
    r"\bppi\b": "proton pump inhibitor",
    r"\biv\b": "intravenous", r"\bim\b": "intramuscular",
    r"\bpo\b": "oral by mouth", r"\bprn\b": "as needed",
    r"\bdx\b": "diagnosis", r"\bddx\b": "differential diagnosis",
    r"\btx\b": "treatment", r"\bmgmt\b": "management",
    r"\b&\b": "and",
    r"↑": "increased", r"↓": "decreased",
}

BOILERPLATE_PATTERNS = [
    r"comes?\s+to\s+the\s+(physician|doctor|emergency\s+department|clinic|hospital)",
    r"is\s+brought\s+to\s+the\s+(emergency\s+department|hospital|physician)",
    r"presents?\s+to\s+the\s+(emergency\s+department|clinic|physician|hospital|office)",
    r"which\s+of\s+the\s+following\s+(is|are|best|most|would)",
    r"what\s+is\s+the\s+(most\s+)?(likely|appropriate|best|next)",
]

# ── Medical synonym dictionary ────────────────────────────────────
MEDICAL_SYNONYMS = {
    "chest pain":      "chest pain angina myocardial infarction ACS pleuritis",
    "shortness of breath": "dyspnea shortness of breath SOB respiratory distress hypoxia",
    "fever":           "fever pyrexia febrile hyperthermia infection sepsis",
    "headache":        "headache cephalgia migraine tension intracranial pressure",
    "diabetes":        "diabetes mellitus type 1 type 2 insulin glucose HbA1c hyperglycemia",
    "hypertension":    "hypertension high blood pressure antihypertensive cardiovascular risk",
    "heart failure":   "heart failure HF CHF cardiomyopathy ejection fraction BNP",
    "myocardial infarction": "myocardial infarction MI STEMI NSTEMI troponin ACS coronary",
    "stroke":          "stroke CVA ischemic hemorrhagic TIA thrombolysis tPA",
    "pneumonia":       "pneumonia lung infection respiratory consolidation antibiotic",
    "copd":            "COPD chronic obstructive pulmonary emphysema bronchitis FEV1",
    "sepsis":          "sepsis SIRS bacteremia systemic infection organ failure",
    "anemia":          "anemia hemoglobin hematocrit iron deficiency B12 folate hemolysis",
    "renal failure":   "renal failure acute kidney injury AKI CKD creatinine BUN GFR",
    "cancer":          "cancer malignancy tumor neoplasm oncology metastasis carcinoma",
    "tuberculosis":    "tuberculosis TB mycobacterium AFB Ghon complex granuloma",
    "antibiotic":      "antibiotic antimicrobial bactericidal bacteriostatic resistance",
    "sensitivity":     "sensitivity true positive rate diagnostic test recall screening",
    "specificity":     "specificity true negative rate false positive rule out",
}

# ── Intent detection ──────────────────────────────────────────────
INTENT_PATTERNS = {
    "diagnosis": [r"most likely diagnosis", r"what is the diagnosis",
                  r"most likely.*caus", r"most likely.*condition"],
    "treatment": [r"most appropriate.*treatment", r"best.*management",
                  r"next.*step.*management", r"first.?line", r"drug of choice"],
    "mechanism": [r"mechanism of action", r"pathophysiology",
                  r"which.*pathway", r"which.*receptor"],
    "anatomy":   [r"which.*nerve", r"which.*artery", r"innervat", r"blood supply"],
    "prognosis": [r"most likely.*complication", r"risk.*factor", r"prognosis"],
    "ethics":    [r"most appropriate.*action", r"ethical",
                  r"should.*physician", r"informed consent"],
    "statistics":[r"sensitivity", r"specificity", r"odds ratio",
                  r"confidence interval", r"number needed"],
}

INTENT_KEYWORDS = {
    "diagnosis":  "diagnosis differential diagnosis clinical presentation signs symptoms workup",
    "treatment":  "treatment management therapy first-line pharmacotherapy guidelines",
    "mechanism":  "mechanism pathophysiology biochemistry molecular pathway receptor enzyme",
    "anatomy":    "anatomy structure nerve artery innervation blood supply location function",
    "prognosis":  "prognosis complication risk factor outcome long-term prevention",
    "ethics":     "medical ethics duty disclosure informed consent autonomy beneficence",
    "statistics": "biostatistics sensitivity specificity predictive value study design",
}


# ── Public API ────────────────────────────────────────────────────

def clean_query(query: str, max_words: int = 150) -> str:
    """Normalise a raw medical query for retrieval."""
    if not query or not isinstance(query, str):
        return ""

    text = html.unescape(query.strip())
    text = re.sub(r"[\r\n\t]+", " ", text)

    CASE_SENSITIVE = {r"\bPTT\b", r"\bPT\b"}
    for pattern, expansion in sorted(ABBREVIATION_MAP.items(),
                                     key=lambda x: len(x[0]), reverse=True):
        flags = 0 if pattern in CASE_SENSITIVE else re.IGNORECASE
        text  = re.sub(pattern, f" {expansion} ", text, flags=flags)

    for pattern in BOILERPLATE_PATTERNS:
        text = re.sub(pattern, " ", text, flags=re.IGNORECASE)

    text   = re.sub(r'[\"\'`|#_{}\\]', " ", text)
    text   = re.sub(r"(?<!\d)[,;:!?]+(?!\d)", " ", text)
    text   = re.sub(r"\s*-{2,}\s*", " ", text)
    text   = re.sub(r"(?<!\w)-|-(?!\w)", " ", text)
    text   = re.sub(r"\s+", " ", text).strip()
    tokens = [t for t in text.split() if not (len(t) == 1 and not t.isalpha())]
    text   = " ".join(tokens[:max_words])
    return text.strip()


def expand_medical_query(query: str) -> str:
    """Append domain synonyms and intent keywords to the query."""
    if not query or not query.strip():
        return query

    q_lower = query.lower().strip()
    parts   = []
    matched = set()

    for term, synonyms in MEDICAL_SYNONYMS.items():
        if term in q_lower and term not in matched:
            parts.append(synonyms)
            matched.add(term)

    detected_intent = None
    for intent, patterns in INTENT_PATTERNS.items():
        if any(re.search(p, q_lower) for p in patterns):
            detected_intent = intent
            break

    if detected_intent is None:
        if re.search(r"\d+.year.old|\bpatient\b|\bpresents?\b", q_lower):
            detected_intent = "diagnosis"

    if detected_intent:
        parts.append(INTENT_KEYWORDS[detected_intent])

    parts.append("clinical medicine USMLE pathophysiology")
    return (query.strip() + " " + " ".join(parts)) if parts else query.strip()
