"""
Consumer Complaint Classifier  -  Gradio app
============================================
Routes a customer complaint to one of 5 categories with a fine-tuned DistilBERT model.

Features
  * Single complaint: category, confidence, suggested team, auto-route / human-review decision
  * Explainability: the words that pushed the model towards its prediction
  * Batch mode: classify a CSV file and download the results
  * Session history and a feedback form (human-in-the-loop corrections saved to CSV)

Run
  python app.py                      # local: http://127.0.0.1:7860
  GRADIO_SHARE=1 python app.py       # also creates a public gradio.live link

Model folder `distilbert_complaints` is searched in: $MODEL_DIR, ./distilbert_complaints,
./models/distilbert_complaints, /kaggle/input/**, /kaggle/working/**.
"""
from __future__ import annotations

import csv
import glob
import html
import logging
import os
import re
import tempfile
import threading
import time
from datetime import datetime
from html import unescape
from urllib.parse import quote

import gradio as gr
import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("complaint-app")

# =========================================================================== #
# 1. Configuration
# =========================================================================== #
MAX_LEN = 256                  # max tokens (same as training)
MAX_CHARS = 6000               # input guard
MIN_WORDS = 3                  # minimum meaningful words after cleaning
DEFAULT_THRESHOLD = 0.60       # confidence below this -> human review
MAX_BATCH_ROWS = 2000
BATCH_SIZE = 32
EXPLAIN_MAX_WORDS = 60         # words tested by the explainability step
HISTORY_LIMIT = 25
FEEDBACK_PATH = os.environ.get("FEEDBACK_PATH", "feedback_log.csv")
_feedback_lock = threading.Lock()

# Illustrative routing table (replace with your organisation's teams)
CATEGORY_INFO = {
    "credit_card":         {"title": "Credit card",       "team": "Card Services",              "color": "#7c3aed", "icon": "&#128179;"},
    "credit_reporting":    {"title": "Credit reporting",  "team": "Credit Report Disputes",     "color": "#0284c7", "icon": "&#128202;"},
    "debt_collection":     {"title": "Debt collection",   "team": "Collections Compliance",     "color": "#ea580c", "icon": "&#128222;"},
    "mortgages_and_loans": {"title": "Mortgages & loans", "team": "Lending & Mortgage Support", "color": "#059669", "icon": "&#127968;"},
    "retail_banking":      {"title": "Retail banking",    "team": "Retail Banking Support",     "color": "#db2777", "icon": "&#127974;"},
}

# Test-set results (18,625 unseen complaints) from the project notebooks
RESULTS = [
    # name, type, accuracy, macro-F1, train time, params
    ("TF-IDF + Logistic Regression", "Baseline",      0.8628, 0.8507, "78 s",     "-"),
    ("DistilBERT (fine-tuned)",      "Transformer",   0.8479, 0.8382, "38.7 min", "67.0 M"),
    ("GRU",                          "RNN (scratch)", 0.8453, 0.8342, "0.9 min",  "2.6 M"),
    ("LSTM",                         "RNN (scratch)", 0.8306, 0.8172, "1.2 min",  "2.6 M"),
    ("SimpleRNN",                    "RNN (scratch)", 0.8154, 0.7962, "2.8 min",  "2.6 M"),
]
DEPLOYED_F1 = 0.8382
DEPLOYED_ACC = 0.8479


def info(label: str) -> dict:
    return CATEGORY_INFO.get(label, {"title": label, "team": "-", "color": "#64748b", "icon": "&#128196;"})


# =========================================================================== #
# 2. Text cleaning (same pipeline the training data went through)
#    lowercase -> remove punctuation/digits -> remove stop words -> lemmatize
# =========================================================================== #
STOP = set("""i me my myself we our ours ourselves you your yours yourself yourselves he him his himself she her hers
herself it its itself they them their theirs themselves what which who whom this that these those am is are was were be
been being have has had having do does did doing a an the and but if or because as until while of at by for with about
against between into through during before after above below to from up down in out on off over under again further then
once here there when where why how all any both each few more most other some such no nor not only own same so than too
very s t can will just don should now d ll m o re ve y ain aren couldn didn doesn hadn hasn haven isn ma mightn mustn
needn shan shouldn wasn weren won wouldn""".split())

try:
    from nltk.stem import WordNetLemmatizer

    _lem = WordNetLemmatizer()
    _lem.lemmatize("tests")                       # raises LookupError if wordnet is missing
    lemmatize = _lem.lemmatize
except Exception:                                 # lemmatization is optional
    lemmatize = lambda w: w                       # noqa: E731


def clean_text(text) -> str:
    text = re.sub(r"\bx{2,}\b", " ", str(text).lower())      # masked data such as XXXX
    text = re.sub(r"[^a-z\s]", " ", text)                    # punctuation, digits, symbols
    return " ".join(lemmatize(w) for w in text.split() if w not in STOP and len(w) > 1)


# =========================================================================== #
# 3. Model
# =========================================================================== #
def find_model_dir() -> str:
    here = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
    candidates = [os.environ.get("MODEL_DIR"), "distilbert_complaints", "models/distilbert_complaints",
                  os.path.join(here, "distilbert_complaints")]
    candidates += glob.glob("/kaggle/input/**/distilbert_complaints", recursive=True)
    candidates += glob.glob("/kaggle/working/**/distilbert_complaints", recursive=True)
    for p in candidates:
        if p and os.path.isfile(os.path.join(p, "config.json")):
            return p
    raise FileNotFoundError(
        "Model folder 'distilbert_complaints' not found. Put it next to app.py, add the Step 5 output as a "
        "Kaggle Input, or set the MODEL_DIR environment variable."
    )


MODEL_DIR = find_model_dir()
device = "cuda" if torch.cuda.is_available() else "cpu"
_t0 = time.perf_counter()
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR).to(device).eval()
id2label = {int(k): v for k, v in model.config.id2label.items()}
LABELS = [id2label[i] for i in range(len(id2label))]
try:
    N_PARAMS = sum(p.numel() for p in model.parameters())
except Exception:
    N_PARAMS = 0
log.info("Model loaded from %s on %s in %.1fs", MODEL_DIR, device, time.perf_counter() - _t0)


@torch.inference_mode()
def predict_proba(cleaned_texts: list[str], batch_size: int = BATCH_SIZE) -> np.ndarray:
    """Cleaned strings -> (n, n_classes) softmax probabilities."""
    outputs = []
    for i in range(0, len(cleaned_texts), batch_size):
        chunk = cleaned_texts[i:i + batch_size]
        enc = tokenizer(chunk, truncation=True, max_length=MAX_LEN, padding=True, return_tensors="pt").to(device)
        outputs.append(torch.softmax(model(**enc).logits.float(), dim=-1).cpu().numpy())
    return np.vstack(outputs) if outputs else np.zeros((0, len(LABELS)))


def predict(text: str) -> dict | None:
    """One complaint -> {label, confidence, scores, cleaned}; None if the text is too short."""
    cleaned = clean_text((text or "")[:MAX_CHARS])
    if len(cleaned.split()) < MIN_WORDS:
        return None
    probs = predict_proba([cleaned])[0]
    scores = {LABELS[i]: float(probs[i]) for i in range(len(LABELS))}
    top = max(scores, key=scores.get)
    return {"label": top, "confidence": scores[top], "scores": scores, "cleaned": cleaned}


def explain(cleaned: str, label: str, top_k: int = 8) -> list[tuple[str, float]]:
    """Occlusion explanation: how much does the predicted-class probability drop when a word is removed?"""
    tokens = cleaned.split()[:EXPLAIN_MAX_WORDS]
    unique = list(dict.fromkeys(tokens))
    if len(unique) < 2:
        return []
    base = predict_proba([cleaned])[0][LABELS.index(label)]
    variants = [" ".join(w for w in cleaned.split() if w != u) for u in unique]
    probs = predict_proba(variants)[:, LABELS.index(label)]
    drops = sorted(zip(unique, (base - probs).tolist()), key=lambda x: x[1], reverse=True)
    return [(w, d) for w, d in drops[:top_k] if d > 0.005]


# =========================================================================== #
# 4. Visual assets (inline SVG, no external files needed)
# =========================================================================== #
HERO_ART = (
    '''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 300 190" width="100%" height="100%"><!-- bank building --><g fill="rgba(255,255,255,.18)" stroke="#fff" stroke-width="2.2" stroke-linejoin="round"><polygon points="14,70 84,26 154,70"/><rect x="22" y="70" width="124" height="9" rx="2"/><rect x="32" y="84" width="16" height="56" rx="2"/><rect x="60" y="84" width="16" height="56" rx="2"/><rect x="88" y="84" width="16" height="56" rx="2"/><rect x="116" y="84" width="16" height="56" rx="2"/><rect x="16" y="144" width="136" height="10" rx="2"/><rect x="8" y="158" width="152" height="9" rx="2"/></g><circle cx="84" cy="52" r="8" fill="none" stroke="#fff" stroke-width="2.2"/><!-- credit card --><g transform="rotate(-10 205 120)"><rect x="150" y="82" width="124" height="78" rx="11" fill="#ffffff"/><rect x="150" y="98" width="124" height="14" fill="#0f766e"/><rect x="162" y="122" width="26" height="19" rx="4" fill="#fbbf24"/><rect x="162" y="148" width="58" height="5" rx="2.5" fill="#cbd5e1"/><rect x="228" y="148" width="32" height="5" rx="2.5" fill="#cbd5e1"/><circle cx="250" cy="134" r="8" fill="#f97316" opacity=".9"/><circle cx="260" cy="134" r="8" fill="#ef4444" opacity=".75"/></g><!-- chat bubble with check --><g><path d="M178 14h74a12 12 0 0 1 12 12v30a12 12 0 0 1-12 12h-40l-14 14v-14h-20a12 12 0 0 1-12-12V26a12 12 0 0 1 12-12z" fill="#ffffff"/><circle cx="198" cy="41" r="10" fill="#16a34a"/><path d="M192.5 41.5l4 4 7-8" fill="none" stroke="#fff" stroke-width="2.8" stroke-linecap="round" stroke-linejoin="round"/><rect x="215" y="32" width="38" height="5" rx="2.5" fill="#cbd5e1"/><rect x="215" y="44" width="26" height="5" rx="2.5" fill="#cbd5e1"/></g><!-- coins --><g><circle cx="282" cy="172" r="13" fill="#fbbf24" stroke="#f59e0b" stroke-width="2"/><circle cx="282" cy="172" r="8" fill="none" stroke="#f59e0b" stroke-width="1.6"/><circle cx="258" cy="178" r="10" fill="#fcd34d" stroke="#f59e0b" stroke-width="2"/></g></svg>'''
)


def _pattern_tile(stroke: str, opacity: float) -> str:
    """Faint finance-themed icons (card, coin, bank, envelope, shield, chat, document) as a repeating tile."""
    svg = (
        '<svg xmlns="http://www.w3.org/2000/svg" width="240" height="240" viewBox="0 0 240 240">'
        f'<g fill="none" stroke="{stroke}" stroke-opacity="{opacity}" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round">'
        '<g transform="rotate(-12 45 40)"><rect x="22" y="24" width="46" height="30" rx="5"/><path d="M22 34h46M28 46h12"/></g>'
        '<g><circle cx="160" cy="38" r="16"/><circle cx="160" cy="38" r="10"/>'
        '<path d="M160 31v14M164 34.5c-1-2-7-2-7 1s7 2 7 5-6 3-8 1"/></g>'
        '<g transform="translate(24 128)"><path d="M0 20L24 4l24 16zM6 24v26M18 24v26M30 24v26M42 24v26M0 54h48"/></g>'
        '<g transform="rotate(10 160 150)"><rect x="136" y="132" width="46" height="32" rx="4"/><path d="M136 136l23 17 23-17"/></g>'
        '<g transform="translate(92 84)"><path d="M20 0l20 7v14c0 14-9 23-20 28C9 44 0 35 0 21V7z"/><path d="M12 24l6 6 12-13"/></g>'
        '<g transform="translate(180 186) rotate(-8)"><path d="M6 0h34a6 6 0 0 1 6 6v18a6 6 0 0 1-6 6H24l-8 8v-8H6a6 6 0 0 1-6-6V6a6 6 0 0 1 6-6z"/></g>'
        '<g transform="translate(70 186) rotate(6)"><rect x="0" y="0" width="30" height="40" rx="4"/><path d="M7 11h16M7 19h16M7 27h10"/></g>'
        '</g></svg>'
    )
    return "data:image/svg+xml;utf8," + quote(svg)


PATTERN_LIGHT = _pattern_tile("#0e7490", 0.16)
PATTERN_DARK = _pattern_tile("#5eead4", 0.14)


# =========================================================================== #
# 5. HTML building blocks
# =========================================================================== #


def _badge(text: str, bg: str) -> str:
    return (f'<span style="background:{bg};color:#fff;padding:3px 11px;border-radius:999px;'
            f'font-size:12px;font-weight:600;white-space:nowrap">{text}</span>')


def category_chips_html() -> str:
    """The 5 categories as coloured pills (hover shows the team they are routed to)."""
    return "".join(
        f'<span title="Routed to: {html.escape(v["team"])}" style="display:inline-flex;align-items:center;gap:7px;'
        f'background:#fff;color:#0f172a;padding:3px 13px 3px 4px;border-radius:999px;font-size:12.5px;font-weight:700;'
        f'box-shadow:0 2px 8px rgba(0,0,0,.18)">'
        f'<span style="width:24px;height:24px;border-radius:50%;background:{v["color"]};display:inline-flex;'
        f'align-items:center;justify-content:center;font-size:13px">{v["icon"]}</span>{html.escape(v["title"])}</span>'
        for v in CATEGORY_INFO.values())


def header_html() -> str:
    chips = [("5", "categories"), (f"{DEPLOYED_F1:.3f}", "macro-F1 (test)"), (f"{DEPLOYED_ACC:.1%}", "accuracy (test)"),
             (f"{N_PARAMS/1e6:.0f} M" if N_PARAMS else "DistilBERT", "parameters"), (device.upper(), "running on")]
    stats = "".join(
        f'<div style="background:rgba(255,255,255,.14);border:1px solid rgba(255,255,255,.22);border-radius:14px;'
        f'padding:9px 16px;min-width:100px;backdrop-filter:blur(6px)">'
        f'<div style="font-size:19px;font-weight:800">{v}</div>'
        f'<div style="font-size:11px;opacity:.85;letter-spacing:.03em">{k}</div></div>' for v, k in chips)
    steps = "".join(
        f'<div class="cc-card" style="flex:1;min-width:170px;display:flex;align-items:center;gap:12px;padding:12px 16px;'
        f'border:1px solid rgba(128,128,128,.28);border-radius:16px;background:rgba(128,128,128,.07)">'
        f'<div style="width:34px;height:34px;border-radius:50%;background:linear-gradient(135deg,#0f766e,#1d4ed8);color:#fff;'
        f'display:flex;align-items:center;justify-content:center;font-weight:800;flex-shrink:0">{n}</div>'
        f'<div><div style="font-weight:700;font-size:14px">{t}</div><div style="font-size:12px;opacity:.7">{d}</div></div></div>'
        for n, t, d in [(1, "Paste a complaint", "or upload a CSV file"), (2, "AI reads it", "DistilBERT scores 5 categories"),
                        (3, "Routed to a team", "or flagged for a human")])
    return f"""
<div style="position:relative;overflow:hidden;padding:30px 32px;border-radius:22px;color:#fff;margin-bottom:14px;
            background:linear-gradient(120deg,#0f766e 0%,#0e7490 50%,#1d4ed8 100%);box-shadow:0 12px 34px rgba(14,116,144,.30)">
  <div style="position:absolute;right:-60px;top:-80px;width:240px;height:240px;border-radius:50%;background:rgba(255,255,255,.07)"></div>
  <div style="position:absolute;left:38%;bottom:-110px;width:200px;height:200px;border-radius:50%;background:rgba(255,255,255,.05)"></div>
  <div style="position:relative;display:flex;align-items:center;justify-content:space-between;gap:24px">
    <div style="min-width:0;flex:1">
      <div style="font-size:12px;font-weight:700;letter-spacing:.14em;opacity:.8">AI-POWERED COMPLAINT ROUTING</div>
      <div style="font-size:31px;font-weight:800;letter-spacing:-.02em;line-height:1.15;margin-top:4px">Consumer Complaint Classifier</div>
      <div style="opacity:.93;margin:8px 0 16px;font-size:15px;max-width:580px">
        Reads a customer complaint and routes it to the right team. Low-confidence cases are flagged for a human agent.
      </div>
      <div style="display:flex;gap:9px;flex-wrap:wrap;margin-bottom:18px">{category_chips_html()}</div>
      <div style="display:flex;gap:10px;flex-wrap:wrap">{stats}</div>
    </div>
    <div class="cc-hero-art" style="width:300px;height:190px;flex-shrink:0">{HERO_ART}</div>
  </div>
</div>
<div style="display:flex;gap:12px;flex-wrap:wrap;margin-bottom:10px">{steps}</div>"""


def footer_html() -> str:
    return ('<div style="text-align:center;font-size:12px;opacity:.55;margin:16px 0 6px">'
            'DistilBERT fine-tuned on CFPB consumer complaints &middot; built with Hugging Face Transformers and Gradio</div>')


def _panel(title: str, body: str) -> str:
    return (f'<div class="cc-card" style="border:1px solid rgba(128,128,128,.3);border-radius:16px;padding:16px 20px">'
            f'<div style="font-size:11.5px;letter-spacing:.1em;text-transform:uppercase;opacity:.65;margin-bottom:6px">{title}</div>'
            f'{body}</div>')


PLACEHOLDER_CARD = """
<div class="cc-card" style="border:2px dashed rgba(128,128,128,.4);border-radius:18px;padding:36px 20px;text-align:center">
  <svg width="86" height="70" viewBox="0 0 86 70" fill="none" stroke="#0e7490" stroke-width="2.6" stroke-linecap="round" stroke-linejoin="round">
    <rect x="6" y="14" width="56" height="40" rx="6" fill="rgba(14,116,144,.10)"/><path d="M6 22l28 19 28-19"/>
    <circle cx="62" cy="44" r="12" fill="rgba(20,184,166,.18)" stroke="#0f766e"/><path d="M71 53l9 9"/>
  </svg>
  <div style="font-weight:700;font-size:16px;margin-top:10px">No complaint analysed yet</div>
  <div style="font-size:13px;margin-top:6px;opacity:.75">Write a complaint or pick an example, then press <b>Classify</b>.</div>
</div>"""


def warning_card(message: str) -> str:
    return (f'<div style="border:1px solid #f59e0b;background:rgba(245,158,11,.12);border-radius:16px;padding:16px">'
            f'<div style="font-weight:700;color:#d97706">&#9888; {html.escape(message)}</div></div>')


def _gauge(conf: float, color: str) -> str:
    circ = 2 * 3.14159265 * 40
    return (f'<svg width="104" height="104" viewBox="0 0 104 104" style="flex-shrink:0">'
            f'<circle cx="52" cy="52" r="40" fill="none" stroke="rgba(128,128,128,.22)" stroke-width="10"/>'
            f'<circle cx="52" cy="52" r="40" fill="none" stroke="{color}" stroke-width="10" stroke-linecap="round" '
            f'stroke-dasharray="{conf*circ:.1f} {circ:.1f}" transform="rotate(-90 52 52)"/>'
            f'<text x="52" y="55" text-anchor="middle" font-size="23" font-weight="800" fill="currentColor">{conf:.0%}</text>'
            f'<text x="52" y="71" text-anchor="middle" font-size="9" opacity=".65" fill="currentColor">confidence</text></svg>')


def result_card(label: str, conf: float, threshold: float, latency_ms: float, note: str = "") -> str:
    meta = info(label)
    color = meta["color"]
    if conf >= threshold:
        status = _badge("&#10003; Auto-route", "#16a34a")
        route = f'Route to <b>{html.escape(meta["team"])}</b>'
    else:
        status = _badge("&#9888; Human review", "#d97706")
        route = f'Below {threshold:.0%}: send to a <b>human agent</b> (guess: {html.escape(meta["team"])})'
    note_html = f'<div style="margin-top:6px;font-size:12px;color:#d97706">{html.escape(note)}</div>' if note else ""
    return f"""
<div class="cc-card" style="border:1px solid rgba(128,128,128,.3);border-top:5px solid {color};border-radius:18px;
            padding:20px 22px;background:linear-gradient(135deg,{color}18,transparent 65%)">
  <div style="display:flex;justify-content:space-between;align-items:center;gap:12px">
    <div style="min-width:0">
      <div style="font-size:11px;letter-spacing:.1em;text-transform:uppercase;opacity:.65;margin-bottom:7px">Predicted category</div>
      <div style="display:flex;align-items:center;gap:10px">
        <div style="width:46px;height:46px;border-radius:14px;background:{color}22;display:flex;align-items:center;
                    justify-content:center;font-size:24px;flex-shrink:0">{meta["icon"]}</div>
        <div style="font-size:28px;font-weight:800;color:{color};line-height:1.1">{html.escape(meta["title"])}</div>
      </div>
      <div style="margin-top:12px">{status}</div>
    </div>
    {_gauge(conf, color)}
  </div>
  <div style="margin-top:14px;padding-top:12px;border-top:1px solid rgba(128,128,128,.22);display:flex;justify-content:space-between;
              gap:10px;font-size:14px"><span>{route}</span>
    <span style="opacity:.5;font-size:11px;white-space:nowrap">{latency_ms:.0f} ms &middot; {device.upper()}</span></div>{note_html}
</div>"""


def probs_html(scores: dict | None = None) -> str:
    """Probability bars for all categories. With scores=None it renders an empty (greyed) placeholder."""
    placeholder = scores is None
    scores = scores or {k: 0.0 for k in CATEGORY_INFO}
    best = max(scores.values())
    rows = ""
    for k, v in sorted(scores.items(), key=lambda kv: kv[1], reverse=True):
        meta = info(k)
        strong = (not placeholder) and v == best
        value = "&ndash;" if placeholder else f"{v:.1%}"
        rows += (f'<div style="display:flex;align-items:center;gap:10px;margin:11px 0;font-size:13.5px;'
                 f'{"font-weight:700;" if strong else "opacity:.85;"}">'
                 f'<div style="width:20px;text-align:center">{meta["icon"]}</div>'
                 f'<div style="width:128px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap">{html.escape(meta["title"])}</div>'
                 f'<div style="flex:1;height:10px;background:rgba(128,128,128,.2);border-radius:999px;overflow:hidden">'
                 f'<div style="width:{0 if placeholder else max(v*100, 0.8):.1f}%;height:100%;background:{meta["color"]};'
                 f'border-radius:999px"></div></div><div style="width:50px;text-align:right">{value}</div></div>')
    return _panel("Probability per category", rows)


PLACEHOLDER_PROBS = probs_html(None)


def explanation_html(items: list[tuple[str, float]], label: str) -> str:
    title = "Key words behind the prediction"
    if not items:
        return _panel(title, '<div style="font-size:12.5px;opacity:.7;padding:4px 0">No single word clearly drives this '
                             'prediction: the decision comes from the overall context.</div>')
    color = info(label)["color"]
    top = max(d for _, d in items)
    rows = "".join(
        f'<div style="display:flex;align-items:center;gap:10px;margin:8px 0;font-size:13px">'
        f'<div style="width:104px;font-weight:600;overflow:hidden;text-overflow:ellipsis;white-space:nowrap">{html.escape(w)}</div>'
        f'<div style="flex:1;height:8px;background:rgba(128,128,128,.22);border-radius:999px;overflow:hidden">'
        f'<div style="width:{max(d/top*100, 6):.0f}%;height:100%;background:{color}"></div></div>'
        f'<div style="width:44px;text-align:right;opacity:.75">-{d:.0%}</div></div>' for w, d in items)
    return _panel(title, f'<div style="font-size:11.5px;opacity:.65;margin-bottom:2px">Confidence drop when the word is removed</div>{rows}')


def explanation_message(msg: str) -> str:
    return _panel("Key words behind the prediction", f'<div style="font-size:12.5px;opacity:.7;padding:4px 0">{msg}</div>')


EXPL_EMPTY = explanation_message("The words that matter most will appear here after you press <b>Classify</b>.")
EXPL_OFF = explanation_message("Key-word analysis is turned off. Tick <b>Show key words</b> to enable it.")


# =========================================================================== #
# 6. Callbacks - single complaint
# =========================================================================== #
HISTORY_COLUMNS = ["Time", "Complaint", "Category", "Confidence", "Status"]


def classify_one(text, threshold, explain_on, history):
    history = history or []
    t0 = time.perf_counter()
    raw = (text or "").strip()
    note = f"Text longer than {MAX_CHARS:,} characters was truncated." if len(raw) > MAX_CHARS else ""
    result = predict(raw)
    if result is None:
        warn = warning_card("Please enter a longer complaint (at least a few meaningful words).")
        return warn, PLACEHOLDER_PROBS, EXPL_EMPTY, clean_text(raw[:MAX_CHARS]), None, history, _history_df(history)

    label, conf = result["label"], result["confidence"]
    ms = (time.perf_counter() - t0) * 1000
    card = result_card(label, conf, threshold, ms, note)
    expl = explanation_html(explain(result["cleaned"], label, top_k=8), label) if explain_on else EXPL_OFF
    status = "Auto-route" if conf >= threshold else "Human review"
    history = ([{"Time": datetime.now().strftime("%H:%M:%S"), "Complaint": raw[:90] + ("..." if len(raw) > 90 else ""),
                 "Category": info(label)["title"], "Confidence": f"{conf:.1%}", "Status": status}] + history)[:HISTORY_LIMIT]
    last = {"text": raw, "label": label, "confidence": conf}
    return card, probs_html(result["scores"]), expl, result["cleaned"], last, history, _history_df(history)


def _history_df(history) -> pd.DataFrame:
    return pd.DataFrame(history, columns=HISTORY_COLUMNS) if history else pd.DataFrame(columns=HISTORY_COLUMNS)


def clear_single():
    return "", PLACEHOLDER_CARD, PLACEHOLDER_PROBS, EXPL_EMPTY, "", None, "", ""


def clear_history():
    return [], _history_df([])


def save_feedback(last, correct_label, comment):
    if not last:
        return "Classify a complaint first, then send feedback."
    if not correct_label:
        return "Please choose the correct category."
    row = {"timestamp": datetime.now().isoformat(timespec="seconds"), "complaint": last["text"],
           "predicted": last["label"], "confidence": round(last["confidence"], 4),
           "correct_label": correct_label, "is_correct": correct_label == last["label"], "comment": comment or ""}
    try:
        with _feedback_lock:
            new_file = not os.path.exists(FEEDBACK_PATH)
            with open(FEEDBACK_PATH, "a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(row))
                if new_file:
                    writer.writeheader()
                writer.writerow(row)
    except OSError as e:
        log.error("Could not write feedback: %s", e)
        return f"Could not save feedback: {e}"
    return "Thank you! Feedback saved." + ("" if row["is_correct"] else " The correction will help improve the model.")


# =========================================================================== #
# 7. Callbacks - batch CSV
# =========================================================================== #
TEXT_COLUMNS = ["narrative", "complaint", "complaints", "text", "consumer_complaint_narrative", "description", "message"]


def classify_file(file, threshold):
    if file is None:
        return warning_card("Please upload a CSV file first."), None, None
    path = file if isinstance(file, str) else getattr(file, "name", str(file))
    try:
        df = pd.read_csv(path)
    except Exception as e:  # noqa: BLE001
        return warning_card(f"Could not read the CSV file: {e}"), None, None

    lower = {str(c).lower(): c for c in df.columns}
    col = next((lower[c] for c in TEXT_COLUMNS if c in lower), None)
    if col is None:
        obj_cols = [c for c in df.columns if df[c].dtype == object]
        if not obj_cols:
            return warning_card("No text column found. Add a column named 'narrative'."), None, None
        col = obj_cols[0]

    note = ""
    if len(df) > MAX_BATCH_ROWS:
        df = df.head(MAX_BATCH_ROWS).copy()
        note = f" (only the first {MAX_BATCH_ROWS:,} rows were processed)"

    cleaned = [clean_text(str(x)[:MAX_CHARS]) if isinstance(x, str) else "" for x in df[col]]
    valid = [i for i, c in enumerate(cleaned) if len(c.split()) >= MIN_WORDS]
    pred, conf, status = [""] * len(df), [np.nan] * len(df), ["Too short"] * len(df)
    if valid:
        probs = predict_proba([cleaned[i] for i in valid])
        for k, i in enumerate(valid):
            j = int(probs[k].argmax())
            pred[i], conf[i] = LABELS[j], float(probs[k][j])
            status[i] = "Auto-route" if conf[i] >= threshold else "Human review"

    out = df.copy()
    out["predicted_category"] = pred
    out["confidence"] = np.round(conf, 4)
    out["status"] = status
    out["suggested_team"] = [info(p)["team"] if p else "" for p in pred]

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".csv", prefix="predictions_")
    tmp.close()
    out.to_csv(tmp.name, index=False)

    n = len(out)
    counts = out["status"].value_counts()
    n_auto, n_human, n_short = (int(counts.get(k, 0)) for k in ("Auto-route", "Human review", "Too short"))
    dist = out.loc[out["predicted_category"] != "", "predicted_category"].value_counts()
    top_n = max(dist.max(), 1) if len(dist) else 1
    bars = "".join(
        f'<div style="display:flex;align-items:center;gap:10px;margin:5px 0;font-size:13px">'
        f'<div style="width:150px">{html.escape(info(k)["title"])}</div>'
        f'<div style="flex:1;height:10px;background:rgba(128,128,128,.22);border-radius:999px;overflow:hidden">'
        f'<div style="width:{v/top_n*100:.0f}%;height:100%;background:{info(k)["color"]}"></div></div>'
        f'<div style="width:64px;text-align:right"><b>{v:,}</b></div></div>' for k, v in dist.items())
    summary = f"""
<div style="border:1px solid rgba(128,128,128,.35);border-radius:14px;padding:18px">
  <div style="font-weight:700;margin-bottom:10px">Processed {n:,} complaints from column <code>{html.escape(str(col))}</code>{note}</div>
  <div style="display:flex;gap:8px;flex-wrap:wrap;margin-bottom:12px">
    {_badge(f"Auto-route: {n_auto:,} ({n_auto/n:.0%})", "#16a34a")}
    {_badge(f"Human review: {n_human:,} ({n_human/n:.0%})", "#d97706")}
    {_badge(f"Too short: {n_short:,}", "#64748b")}
  </div>{bars}
</div>"""
    preview = out.head(200).copy()
    preview[col] = preview[col].astype(str).str.slice(0, 110)
    return summary, preview, tmp.name


# =========================================================================== #
# 8. Static content
# =========================================================================== #
EXAMPLE_ITEMS = [
    ("credit_card", "I was charged an annual fee on my credit card even though the offer said no fee for the first year. I called the bank twice and nobody fixed it."),
    ("credit_reporting", "There is an account on my credit report that I never opened. I disputed it with the bureau but they refuse to remove the inaccurate information."),
    ("debt_collection", "A collection agency keeps calling me about a debt that is not mine. They never sent me a validation letter and they threaten legal action."),
    ("mortgages_and_loans", "My mortgage servicer lost my payment and now they say my loan is late. I also asked about refinancing and nobody answered."),
    ("retail_banking", "The bank froze my checking account without notice and I cannot access my money. A debit card charge was also posted twice."),
]
EXAMPLES = [[text] for _, text in EXAMPLE_ITEMS]


def make_sample_csv() -> str:
    path = os.path.join(tempfile.gettempdir(), "sample_complaints.csv")
    pd.DataFrame({"narrative": [e[0] for e in EXAMPLES]}).to_csv(path, index=False)
    return path


def about_html() -> str:
    rows = ""
    for name, kind, acc, f1, t, params in RESULTS:
        deployed = name.startswith("DistilBERT")
        style = "font-weight:700;background:rgba(13,148,136,.14);" if deployed else ""
        mark = " &#9733; deployed" if deployed else ""
        cells = (f"<td style='padding:7px 10px'>{name}{mark}</td><td style='padding:7px 10px'>{kind}</td>"
                 f"<td style='padding:7px 10px;text-align:right'>{acc:.4f}</td>"
                 f"<td style='padding:7px 10px;text-align:right'>{f1:.4f}</td>"
                 f"<td style='padding:7px 10px;text-align:right'>{t}</td>"
                 f"<td style='padding:7px 10px;text-align:right'>{params}</td>")
        rows += f"<tr style='{style}'>{cells}</tr>"
    cats = "".join(f"<li><b>{v['title']}</b> &rarr; {v['team']}</li>" for v in CATEGORY_INFO.values())
    return f"""
<div style="line-height:1.6;font-size:14.5px">
  <h3>How it works</h3>
  <ol>
    <li>The complaint is cleaned like the training data (lowercase, no punctuation or digits, no stop words).</li>
    <li>DistilBERT reads up to {MAX_LEN} tokens and outputs a probability for each of the 5 categories.</li>
    <li>If the top probability reaches the threshold the complaint is auto-routed, otherwise it goes to a human agent.</li>
    <li>The <i>key words</i> panel removes each word in turn and shows how much the confidence drops (occlusion).</li>
  </ol>
  <h3>Categories and suggested routing</h3><ul>{cats}</ul>
  <h3>Test-set results (18,625 unseen complaints)</h3>
  <div style="overflow-x:auto"><table style="border-collapse:collapse;width:100%;font-size:14px">
    <thead><tr style="border-bottom:2px solid rgba(128,128,128,.4);text-align:left">
      <th style="padding:7px 10px">Model</th><th style="padding:7px 10px">Type</th>
      <th style="padding:7px 10px;text-align:right">Accuracy</th><th style="padding:7px 10px;text-align:right">Macro-F1</th>
      <th style="padding:7px 10px;text-align:right">Train time</th><th style="padding:7px 10px;text-align:right">Params</th>
    </tr></thead><tbody>{rows}</tbody></table></div>
  <h3>Limitations</h3>
  <ul>
    <li>The model always picks one of the 5 categories, even for text unrelated to banking. Low confidence is the warning sign.</li>
    <li>Confidence is the raw model probability and is <b>not calibrated</b>. The threshold is a practical setting, not a guarantee.</li>
    <li>The hardest pair is <i>credit reporting</i> vs <i>debt collection</i> / <i>credit card</i>: complaints often mention several topics.</li>
    <li>A TF-IDF baseline scored slightly higher than every neural model on this pre-cleaned dataset, so the transformer's cost is a trade-off.</li>
    <li>The routing teams shown here are illustrative. Feedback is stored locally in <code>{html.escape(FEEDBACK_PATH)}</code>.</li>
  </ul>
</div>"""


CSS = """
body {
  background-color: #f1f8fa !important;
  background-image:
    radial-gradient(1100px 520px at 88% -8%, rgba(14,165,233,.16), transparent 62%),
    radial-gradient(900px 520px at -6% 18%, rgba(20,184,166,.14), transparent 60%),
    url("__PL__") !important;
  background-attachment: fixed !important;
}
body.dark, .dark body {
  background-color: #0b1320 !important;
  background-image:
    radial-gradient(1100px 520px at 88% -8%, rgba(14,165,233,.16), transparent 62%),
    radial-gradient(900px 520px at -6% 18%, rgba(20,184,166,.12), transparent 60%),
    url("__PD__") !important;
}
gradio-app, .gradio-app, .gradio-container {background: transparent !important;}
.gradio-container {max-width: 1120px !important; margin: auto !important;}
.tabitem {background: transparent !important; border: none !important;}
footer {display: none !important;}
button[role="tab"] {font-weight: 600 !important; font-size: 15px !important;}
.cc-card {transition: box-shadow .2s ease, transform .2s ease;}
.cc-card:hover {box-shadow: 0 8px 22px rgba(0, 0, 0, .12); transform: translateY(-1px);}
.cc-hero-art {animation: cc-float 6s ease-in-out infinite;}
@keyframes cc-float {0%, 100% {transform: translateY(0);} 50% {transform: translateY(-6px);}}
.cc-ex-label {font-size: 12px; opacity: .7; white-space: nowrap; padding-top: 7px;}
.cc-ex {font-size: 12.5px !important; font-weight: 600 !important; border-left-width: 5px !important; border-style: solid !important;
        min-width: 0 !important;}
__EXCOLORS__
@media (max-width: 760px) {.cc-hero-art {display: none;}}
@media (prefers-reduced-motion: reduce) {.cc-hero-art {animation: none;}}
"""
_ex_colors = "\n".join(f".cc-ex-{k} {{border-color: {v['color']} !important;}}" for k, v in CATEGORY_INFO.items())
CSS = (CSS.replace("__PL__", PATTERN_LIGHT).replace("__PD__", PATTERN_DARK).replace("__EXCOLORS__", _ex_colors))
try:
    THEME = gr.themes.Soft(
        primary_hue="teal", secondary_hue="cyan", neutral_hue="slate",
        radius_size=gr.themes.sizes.radius_lg,
        font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui", "sans-serif"],
    ).set(
        button_primary_background_fill="linear-gradient(90deg, #0f766e, #0e7490)",
        button_primary_background_fill_hover="linear-gradient(90deg, #115e59, #155e75)",
        button_primary_text_color="white",
    )
except Exception:                                  # older Gradio: fall back to the plain theme
    THEME = gr.themes.Soft(primary_hue="teal")
GRADIO_MAJOR = int(gr.__version__.split(".")[0])


# =========================================================================== #
# 9. Interface
# =========================================================================== #
def build_demo() -> gr.Blocks:
    title = "Consumer Complaint Classifier"
    demo = gr.Blocks(title=title) if GRADIO_MAJOR >= 6 else gr.Blocks(title=title, theme=THEME, css=CSS)

    with demo:
        gr.HTML(header_html())
        last_state = gr.State(None)
        history_state = gr.State([])

        with gr.Tabs():
            # ------------------------------------------------ single complaint
            with gr.Tab("Classify a complaint"):
                with gr.Row(equal_height=True):
                    gr.HTML('<div class="cc-ex-label">Try an example:</div>', min_width=110, scale=0)
                    ex_buttons = [
                        gr.Button(f'{unescape(info(k)["icon"])} {info(k)["title"]}', size="sm", min_width=120,
                                  elem_classes=["cc-ex", f"cc-ex-{k}"])
                        for k, _ in EXAMPLE_ITEMS]

                with gr.Row(equal_height=False):
                    # ---- left: input
                    with gr.Column(scale=5, min_width=320):
                        inp = gr.Textbox(label="Complaint narrative", lines=10, max_lines=20,
                                         placeholder="Paste or type the customer complaint here...")
                        thr = gr.Slider(0.30, 0.95, value=DEFAULT_THRESHOLD, step=0.05, label="Human-review threshold",
                                        info="Below this confidence the complaint goes to a human agent.")
                        explain_on = gr.Checkbox(value=True, label="Show key words behind the prediction")
                        with gr.Row():
                            btn = gr.Button("Classify", variant="primary", scale=3)
                            clr = gr.Button("Clear", scale=1)
                        with gr.Accordion("Cleaned text (what the model sees)", open=False):
                            cleaned_box = gr.Textbox(lines=4, interactive=False, show_label=False)
                        with gr.Accordion("Was this prediction correct? Send feedback", open=False):
                            fb_label = gr.Dropdown(choices=LABELS, label="Correct category")
                            fb_comment = gr.Textbox(label="Comment (optional)", lines=2)
                            fb_btn = gr.Button("Submit feedback")
                            fb_msg = gr.Markdown()
                    # ---- right: results
                    with gr.Column(scale=5, min_width=320):
                        card = gr.HTML(PLACEHOLDER_CARD)
                        lab = gr.HTML(PLACEHOLDER_PROBS)
                        expl_box = gr.HTML(EXPL_EMPTY)

                with gr.Accordion("Session history", open=False):
                    hist_table = gr.Dataframe(value=_history_df([]), interactive=False, wrap=True, label=None)
                    hist_clear = gr.Button("Clear history", size="sm")

                outs = [card, lab, expl_box, cleaned_box, last_state, history_state, hist_table]
                run_inputs = [inp, thr, explain_on, history_state]
                btn.click(classify_one, run_inputs, outs)
                for b_, (_, text) in zip(ex_buttons, EXAMPLE_ITEMS):
                    b_.click(lambda t=text: t, None, inp, queue=False).then(classify_one, run_inputs, outs)
                clr.click(clear_single, None, [inp, card, lab, expl_box, cleaned_box, last_state, fb_msg, fb_comment])
                fb_btn.click(save_feedback, [last_state, fb_label, fb_comment], fb_msg)
                hist_clear.click(clear_history, None, [history_state, hist_table])

            # ------------------------------------------------ batch
            with gr.Tab("Batch (CSV)"):
                with gr.Row(equal_height=False):
                    with gr.Column(scale=4, min_width=290):
                        gr.Markdown(f"Upload a CSV with a column named **narrative** (or any text column). "
                                    f"Up to {MAX_BATCH_ROWS:,} rows are processed.")
                        up = gr.File(label="CSV file", file_types=[".csv"], type="filepath")
                        thr_b = gr.Slider(0.30, 0.95, value=DEFAULT_THRESHOLD, step=0.05, label="Human-review threshold")
                        with gr.Row():
                            run_b = gr.Button("Classify file", variant="primary", scale=3)
                            gr.DownloadButton("Sample CSV", value=make_sample_csv(), scale=2)
                    with gr.Column(scale=6, min_width=320):
                        summary = gr.HTML()
                        out_file = gr.File(label="Download predictions (CSV)")
                table = gr.Dataframe(label="Preview (first 200 rows)", interactive=False, wrap=True)
                run_b.click(classify_file, [up, thr_b], [summary, table, out_file])

            # ------------------------------------------------ about
            with gr.Tab("About the model"):
                gr.HTML(about_html())
        gr.HTML(footer_html())
    return demo


demo = build_demo()
demo.queue(max_size=16)


def launch(share: bool = False, **kwargs):
    """Launch the app. Gradio 6 takes theme/css here instead of in gr.Blocks()."""
    if GRADIO_MAJOR >= 6:
        kwargs.update(theme=THEME, css=CSS)
    return demo.launch(share=share, **kwargs)


if __name__ == "__main__":
    SHARE = os.environ.get("GRADIO_SHARE", "1" if os.path.exists("/kaggle") else "0") == "1"
    launch(share=SHARE)