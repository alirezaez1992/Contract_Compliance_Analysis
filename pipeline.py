#%%writefile pipeline.py   #This is for running on Colab
import io
import json
import re
import difflib
from dataclasses import dataclass
from typing import List, Literal

import numpy as np
import faiss
import fitz
import pdfplumber
import torch
from json_repair import repair_json
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig


# ---------------------------
# 1) Configuration
# ---------------------------
MODEL = "mistralai/Mistral-7B-Instruct-v0.2"

COMPLIANCE_QUESTIONS = [
    "1 Password Management. The contract must require a documented password standard covering password length/strength, prohibition of default and known-compromised passwords, secure storage (no plaintext; salted hashing if stored), brute-force protections (lockout/rate limiting), prohibition on password sharing, vaulting of privileged credentials/recovery codes, and time-based rotation for break-glass credentials. Based on the contract language and exhibits, what is the compliance state for Password Management?",
    "2 IT Asset Management. The contract must require an in-scope asset inventory (including cloud accounts/subscriptions, workloads, databases, security tooling), define minimum inventory fields, require at least quarterly reconciliation/review, and require secure configuration baselines with drift remediation and prohibition of insecure defaults. Based on the contract language and exhibits, what is the compliance state for IT Asset Management?",
    "3 Security Training & Background Checks. The contract must require security awareness training on hire and at least annually, and background screening for personnel with access to Company Data to the extent permitted by law, including maintaining a screening policy and attestation/evidence. Based on the contract language and exhibits, what is the compliance state for Security Training and Background Checks?",
    "4 Data in Transit Encryption. The contract must require encryption of Company Data in transit using TLS 1.2+ (preferably TLS 1.3 where feasible) for Company-to-Service traffic, administrative access pathways, and applicable Service-to-Subprocessor transfers, with certificate management and avoidance of insecure cipher suites. Based on the contract language and exhibits, what is the compliance state for Data in Transit Encryption?",
    "5 Network Authentication & Authorization Protocols. The contract must specify the authentication mechanisms (e.g., SAML SSO for users, OAuth/token-based for APIs), require MFA for privileged/production access, require secure admin pathways (bastion/secure gateway) with session logging, and require RBAC authorization. Based on the contract language and exhibits, what is the compliance state for Network Authentication and Authorization Protocols?",
]

# ---------------------------
# 2) Schema
# ---------------------------
class ComplianceItem(BaseModel):
    compliance_question: str
    compliance_state: Literal["Fully Compliant", "Partially Compliant", "Non-Compliant"]
    confidence: float = Field(ge=0, le=1)
    relevant_quotes: List[str]
    rationale: str

class ComplianceReport(BaseModel):
    items: List[ComplianceItem] = Field(min_items=5, max_items=5)

# ---------------------------
# 3) PDF parsing
# ---------------------------
@dataclass
class PageContent:
    page: int
    text: str
    tables: List[str]

def extract_pdf_content(pdf_bytes: bytes) -> List[PageContent]:
    pages: List[PageContent] = []

    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    text_pages = {}
    for i in range(len(doc)):
        t = doc[i].get_text("text") or ""
        t = "\n".join(l.strip() for l in t.splitlines() if l.strip())
        text_pages[i + 1] = t

    table_pages = {}
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for i, p in enumerate(pdf.pages, start=1):
            tables = []
            for tbl in p.extract_tables() or []:
                rows = ["\t".join("" if c is None else str(c) for c in row) for row in tbl]
                table_txt = "\n".join(rows).strip()
                if table_txt:
                    tables.append(table_txt)
            table_pages[i] = tables

    for p in range(1, len(doc) + 1):
        pages.append(PageContent(
            page=p,
            text=text_pages.get(p, ""),
            tables=table_pages.get(p, [])
        ))
    return pages

def chunk_text(text, size=1500, overlap=200):
    text = re.sub(r"\n{3,}", "\n\n", text)
    chunks = []
    i = 0
    while i < len(text):
        j = min(len(text), i + size)
        chunk = text[i:j].strip()
        if chunk:
            chunks.append(chunk)
        i = max(j - overlap, i + 1)
    return chunks

def build_corpus(pages: List[PageContent]):
    corpus = []
    for p in pages:
        for c in chunk_text(p.text):
            corpus.append({"page": p.page, "text": c})
        for t in p.tables:
            for c in chunk_text(t):
                corpus.append({"page": p.page, "text": c})
    return corpus

# ---------------------------
# 4) Retriever
# ---------------------------
_embedder = None
def get_embedder():
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return _embedder

def build_index(embedder, corpus):
    texts = [c["text"] for c in corpus]
    emb = embedder.encode(texts, batch_size=64, show_progress_bar=False)
    emb = np.asarray(emb, dtype="float32")
    faiss.normalize_L2(emb)

    index = faiss.IndexFlatIP(emb.shape[1])
    index.add(emb)
    return index

def retrieve(embedder, index, corpus, query, k=12):
    q = embedder.encode([query]).astype("float32")
    faiss.normalize_L2(q)
    scores, ids = index.search(q, k)
    return [{**corpus[int(i)], "score": float(s)} for s, i in zip(scores[0], ids[0])]

# ---------------------------
# 5) Local LLM
# ---------------------------
_tokenizer = None
_model = None
def get_llm():
    global _tokenizer, _model
    if _tokenizer is None or _model is None:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
        _tokenizer = AutoTokenizer.from_pretrained(MODEL, use_fast=True)
        _model = AutoModelForCausalLM.from_pretrained(
            MODEL,
            device_map="auto",
            quantization_config=bnb_config,
            torch_dtype=torch.float16,
        )
    return _tokenizer, _model

def generate_text(tokenizer, model, prompt, max_new_tokens=450):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            repetition_penalty=1.05,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )
    full = tokenizer.decode(out_ids[0], skip_special_tokens=True)
    return full[len(prompt):].strip()

# ---------------------------
# 6) JSON + quote grounding
# ---------------------------
def normalize(s: str) -> str:
    s = s.strip()
    s = re.sub(r"\s+", " ", s)
    return s

def extract_first_json(text: str) -> str:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("No JSON object found.")
    return text[start:end+1]

def find_best_verbatim_snippet(quote: str, hits, max_len=260):
    q = normalize(re.sub(r"\(PAGE\s+\d+\)\s*$", "", quote))
    blocks = [(h["page"], normalize(h["text"])) for h in hits]

    for page, tb in blocks:
        if q and q in tb:
            i = tb.find(q)
            start = max(0, i - 90)
            end = min(len(tb), i + len(q) + 120)
            snippet = tb[start:end].strip()[:max_len]
            return f"{snippet} (PAGE {page})"

    candidates = []
    for page, tb in blocks:
        sents = re.split(r"(?<=[\.\;\:])\s+", tb)
        for s in sents:
            s2 = s.strip()
            if 60 <= len(s2) <= 520:
                candidates.append((page, s2))

    best, best_score = None, 0.0
    for page, s in candidates[:5000]:
        score = difflib.SequenceMatcher(None, q, s).ratio()
        if score > best_score:
            best_score, best = score, (page, s)

    if best is None:
        page = hits[0]["page"]
        snippet = normalize(hits[0]["text"])[:max_len]
        return f"{snippet} (PAGE {page})"

    page, s = best
    return f"{s[:max_len].strip()} (PAGE {page})"

def ground_quotes(quotes, hits):
    grounded = [find_best_verbatim_snippet(q, hits) for q in quotes]
    out, seen = [], set()
    for q in grounded:
        k = q.lower()
        if k not in seen:
            seen.add(k)
            out.append(q)
    return out[:4]

def build_prompt(question: str, hits, short=False):
    if short:
        context = "\n\n".join([f"(PAGE {h['page']}) {h['text'][:900]}" for h in hits])
    else:
        context = "\n\n".join([f"(PAGE {h['page']}) {h['text']}" for h in hits])

    return f"""
Return a SINGLE JSON object only.

Compliance requirement:
{question}

Evidence (verbatim):
{context}

JSON schema (keys must match exactly):
{{
  "compliance_question": string,
  "compliance_state": "Fully Compliant" | "Partially Compliant" | "Non-Compliant",
  "confidence": number between 0 and 1,
  "relevant_quotes": array of 1 to 4 strings,
  "rationale": string
}}

Rules:
- Output MUST start with {{ and end with }} (JSON only).
- Use only the Evidence; do not invent.
- Include 1–4 relevant quotes supported by Evidence and include page tags when possible.

Now output the JSON object:
""".strip()

def run_compliance_one(embedder, index, corpus, tokenizer, model, question: str, k=12):
    hits = retrieve(embedder, index, corpus, question, k=k)
    last_err = None
    for short in [False, True]:
        prompt = build_prompt(question, hits, short=short)
        raw = generate_text(tokenizer, model, prompt)
        try:
            obj = repair_json(extract_first_json(raw))
            data = json.loads(obj)
            item = ComplianceItem(**data)
            item.relevant_quotes = ground_quotes(item.relevant_quotes, hits)
            return item
        except Exception as e:
            last_err = e
    return ComplianceItem(
        compliance_question=question,
        compliance_state="Partially Compliant",
        confidence=0.2,
        relevant_quotes=[f"Parsing/validation failed after retries: {str(last_err)[:160]} (PAGE N/A)"],
        rationale="Model output was not valid JSON; returned conservative default."
    )

# ---------------------------
# 7) Main entrypoint for UI
# ---------------------------
def analyze_pdf_bytes(pdf_bytes: bytes) -> dict:
    pages = extract_pdf_content(pdf_bytes)
    corpus = build_corpus(pages)

    embedder = get_embedder()
    index = build_index(embedder, corpus)

    tokenizer, model = get_llm()

    items = [run_compliance_one(embedder, index, corpus, tokenizer, model, q, k=12) for q in COMPLIANCE_QUESTIONS]
    report = ComplianceReport(items=items)
    return report.model_dump()
