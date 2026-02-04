#%%writefile pipeline.py  For running on Colab
import io
import json
import re
import difflib
import hashlib
from dataclasses import dataclass
from typing import List, Literal, Optional, Dict, Any

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
# 2) Schemas
# ---------------------------
class ComplianceItem(BaseModel):
    compliance_question: str
    compliance_state: Literal["Fully Compliant", "Partially Compliant", "Non-Compliant"]
    confidence: float = Field(ge=0, le=1)
    relevant_quotes: List[str]
    rationale: str

class ComplianceReport(BaseModel):
    items: List[ComplianceItem] = Field(min_items=5, max_items=5)

# New: Chat schema
class ChatAnswer(BaseModel):
    question: str
    answer: str
    confidence: float = Field(ge=0, le=1)
    relevant_quotes: List[str]


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
    text_pages: Dict[int, str] = {}
    for i in range(len(doc)):
        t = doc[i].get_text("text") or ""
        t = "\n".join(l.strip() for l in t.splitlines() if l.strip())
        text_pages[i + 1] = t

    table_pages: Dict[int, List[str]] = {}
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

def chunk_text(text: str, size: int = 1500, overlap: int = 200) -> List[str]:
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

def build_corpus(pages: List[PageContent]) -> List[Dict[str, Any]]:
    corpus: List[Dict[str, Any]] = []
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

def build_index(embedder, corpus: List[Dict[str, Any]]):
    texts = [c["text"] for c in corpus]
    emb = embedder.encode(texts, batch_size=64, show_progress_bar=False)
    emb = np.asarray(emb, dtype="float32")
    faiss.normalize_L2(emb)

    index = faiss.IndexFlatIP(emb.shape[1])
    index.add(emb)
    return index

def retrieve(embedder, index, corpus: List[Dict[str, Any]], query: str, k: int = 12):
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

def generate_text(tokenizer, model, prompt: str, max_new_tokens: int = 450) -> str:
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

def find_best_verbatim_snippet(quote: str, hits: List[Dict[str, Any]], max_len: int = 260) -> str:
    q = normalize(re.sub(r"\(PAGE\s+\d+\)\s*$", "", quote))
    blocks = [(h["page"], normalize(h["text"])) for h in hits]

    # Exact containment first
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
        page = hits[0]["page"] if hits else -1
        snippet = normalize(hits[0]["text"])[:max_len] if hits else ""
        return f"{snippet} (PAGE {page if page != -1 else 'N/A'})"

    page, s = best
    return f"{s[:max_len].strip()} (PAGE {page})"

def ground_quotes(quotes: List[str], hits: List[Dict[str, Any]]) -> List[str]:
    grounded = [find_best_verbatim_snippet(q, hits) for q in quotes]
    out, seen = [], set()
    for q in grounded:
        k = q.lower()
        if k not in seen:
            seen.add(k)
            out.append(q)
    return out[:4]


# ---------------------------
# 7) Prompts
# ---------------------------
def build_compliance_prompt(question: str, hits: List[Dict[str, Any]], short: bool = False) -> str:
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

def build_chat_prompt(user_question: str, hits: List[Dict[str, Any]], chat_history: Optional[List[Dict[str, str]]] = None, short: bool = False) -> str:
    if short:
        context = "\n\n".join([f"(PAGE {h['page']}) {h['text'][:900]}" for h in hits])
    else:
        context = "\n\n".join([f"(PAGE {h['page']}) {h['text']}" for h in hits])

    history_txt = ""
    if chat_history:
        trimmed = chat_history[-6:]  # last 6 turns
        history_txt = "\n".join([f"{m.get('role','user').upper()}: {m.get('content','')}" for m in trimmed])

    return f"""
Return a SINGLE JSON object only.

You are answering a user's question about a PDF.
Use ONLY the Evidence. If the answer is not supported by Evidence, say so clearly.

Chat history (may help disambiguate; do NOT treat as evidence):
{history_txt if history_txt else "(none)"}

User question:
{user_question}

Evidence (verbatim):
{context}

JSON schema (keys must match exactly):
{{
  "question": string,
  "answer": string,
  "confidence": number between 0 and 1,
  "relevant_quotes": array of 1 to 4 strings
}}

Rules:
- Output MUST start with {{ and end with }} (JSON only).
- Do not invent facts not present in Evidence.
- If Evidence is insufficient, answer: "Not enough information in the provided document excerpts." and set confidence <= 0.3.
- relevant_quotes MUST be supported by Evidence; include page tags when possible.

Now output the JSON object:
""".strip()


# ---------------------------
# 8) Runners
# ---------------------------
def run_compliance_one(embedder, index, corpus, tokenizer, model, question: str, k: int = 12) -> ComplianceItem:
    hits = retrieve(embedder, index, corpus, question, k=k)
    last_err = None
    for short in [False, True]:
        prompt = build_compliance_prompt(question, hits, short=short)
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

def run_chat_one(embedder, index, corpus, tokenizer, model, user_question: str, chat_history: Optional[List[Dict[str, str]]] = None, k: int = 12) -> Dict[str, Any]:
    hits = retrieve(embedder, index, corpus, user_question, k=k)
    last_err = None

    for short in [False, True]:
        prompt = build_chat_prompt(user_question, hits, chat_history=chat_history, short=short)
        raw = generate_text(tokenizer, model, prompt)
        try:
            obj = repair_json(extract_first_json(raw))
            data = json.loads(obj)
            ans = ChatAnswer(**data)
            ans.relevant_quotes = ground_quotes(ans.relevant_quotes, hits)
            return ans.model_dump()
        except Exception as e:
            last_err = e

    return ChatAnswer(
        question=user_question,
        answer="Not enough information in the provided document excerpts.",
        confidence=0.2,
        relevant_quotes=[f"Parsing/validation failed after retries: {str(last_err)[:160]} (PAGE N/A)"],
    ).model_dump()


# ---------------------------
# 9) Optional tiny cache (keeps chat fast across multiple questions per same PDF)
# ---------------------------
# Cache is per-PDF-bytes SHA1; stores (corpus, index). Embedder/LLM are already global-cached.
_PDF_CACHE: Dict[str, Dict[str, Any]] = {}

def _pdf_key(pdf_bytes: bytes) -> str:
    return hashlib.sha1(pdf_bytes).hexdigest()

def _get_or_build_retrieval(pdf_bytes: bytes):
    key = _pdf_key(pdf_bytes)
    cached = _PDF_CACHE.get(key)
    if cached is not None:
        return cached["corpus"], cached["index"]

    pages = extract_pdf_content(pdf_bytes)
    corpus = build_corpus(pages)
    embedder = get_embedder()
    index = build_index(embedder, corpus)

    _PDF_CACHE[key] = {"corpus": corpus, "index": index}
    return corpus, index


# ---------------------------
# 10) Main entrypoints
# ---------------------------
def analyze_pdf_bytes(pdf_bytes: bytes, user_question: Optional[str] = None, chat_history: Optional[List[Dict[str, str]]] = None) -> dict:
    """
    Backward compatible:
      - If user_question is None -> returns ComplianceReport (your original behavior)
      - If user_question is provided -> returns ChatAnswer dict
    """
    embedder = get_embedder()
    corpus, index = _get_or_build_retrieval(pdf_bytes)
    tokenizer, model = get_llm()

    if user_question is not None:
        return run_chat_one(embedder, index, corpus, tokenizer, model, user_question, chat_history=chat_history, k=12)

    items = [run_compliance_one(embedder, index, corpus, tokenizer, model, q, k=12) for q in COMPLIANCE_QUESTIONS]
    report = ComplianceReport(items=items)
    return report.model_dump()

def chat_with_pdf_bytes(pdf_bytes: bytes, user_question: str, chat_history: Optional[List[Dict[str, str]]] = None) -> dict:
    """
    Dedicated chat function (cleaner for Streamlit chat UIs).
    """
    return analyze_pdf_bytes(pdf_bytes, user_question=user_question, chat_history=chat_history)
