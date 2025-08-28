# bot.py
# Version 5.4 - v2.3.0 with per-run API key rotation, targeted sanitization, and form-fill cache.

import os
import time
import json
from playwright.sync_api import sync_playwright, Page, TimeoutError
import gspread
from google.oauth2.service_account import Credentials
import google.generativeai as genai
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
import pickle 
from dotenv import load_dotenv
import itertools
from typing import Any, Dict, Optional, List
import re
import logging
from logging.handlers import RotatingFileHandler
from html import unescape
# PDF helper will be defined inline below

# --- API KEY ROTATION SETUP ---
load_dotenv() # Load variables from .env file

API_KEYS = [
    os.getenv("GEMINI_API_KEY_1"),
    os.getenv("GEMINI_API_KEY_2"),
    os.getenv("GEMINI_API_KEY_3"),
]
# Filter out any keys that might be missing
VALID_API_KEYS = [key for key in API_KEYS if key]

if not VALID_API_KEYS:
    raise ValueError("FATAL ERROR: No Gemini API keys found in .env file. Please check your .env configuration.")

# API key cycle is initialized after logging setup (uses logs_dir). We'll rotate per run.
api_key_cycler = None

def get_next_api_key():
    """Returns the next API key from the rotation."""
    global api_key_cycler
    if api_key_cycler is None:
        api_key_cycler = itertools.cycle(VALID_API_KEYS)
    return next(api_key_cycler)

# --- CONFIGURATION ---


# --- IMPORTANT: FILL IN YOUR DETAILS HERE ---
# Pull sensitive values from environment (.env)
TRIZETTO_USERNAME = os.getenv("TRIZETTO_USERNAME", "")
TRIZETTO_PASSWORD = os.getenv("TRIZETTO_PASSWORD", "")

# The email option the bot should click to receive the OTP
OTP_EMAIL_ADDRESS_TEXT = os.getenv("OTP_EMAIL_ADDRESS_TEXT", "")

# Google Services Configuration
SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive"
]
SPREADSHEET_ID = os.getenv("SPREADSHEET_ID", "")
SHEET_NAME = os.getenv("SHEET_NAME", "")
DRIVE_FOLDER_ID = os.getenv("DRIVE_FOLDER_ID", "")
# PDF_DRIVE_FOLDER_ID may be empty if unused
PDF_DRIVE_FOLDER_ID = os.getenv("PDF_DRIVE_FOLDER_ID", "")
CHECK_INTERVAL_SECONDS = int(os.getenv("CHECK_INTERVAL_SECONDS", "60"))
RUN_ONCE = str(os.getenv("RUN_ONCE", "False")).strip().lower() in ("1", "true", "yes", "on")
HEADLESS = str(os.getenv("HEADLESS", "True")).strip().lower() in ("1", "true", "yes", "on")
OPERATION_TIMEOUT_SECONDS = int(os.getenv("OPERATION_TIMEOUT_SECONDS", "300"))
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
REPORT_HTML_MAX_CHARS = int(os.getenv("REPORT_HTML_MAX_CHARS", "100000"))
KEEP_BROWSER_OPEN = str(os.getenv("KEEP_BROWSER_OPEN", "False")).strip().lower() in ("1", "true", "yes", "on")

# --- File/Path Configuration ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CLIENT_SECRETS_FILE = os.path.join(SCRIPT_DIR, 'client_secret.json') # Use OAuth secrets
TOKEN_FILE = os.path.join(SCRIPT_DIR, 'token.json') # Stores user's access token
STATE_FILE = os.path.join(SCRIPT_DIR, "login_state.json")
SCREENSHOT_DIR = os.path.join(SCRIPT_DIR, "Screenshots")
PAYER_CACHE_FILE = os.getenv("PAYER_CACHE_FILE", os.path.join(SCRIPT_DIR, "payer_cache.json"))
CHROME_PROFILE_DIR = os.getenv("CHROME_PROFILE_DIR", os.path.join(SCRIPT_DIR, "chrome-profile"))
COOKIES_FILE = os.getenv("COOKIES_FILE", os.path.join(SCRIPT_DIR, "Trizetto_cookies.pkl"))
FORM_FILL_CACHE_FILE = os.getenv("FORM_FILL_CACHE_FILE", os.path.join(SCRIPT_DIR, "form_fill_cache.json"))

# --- LOGGING SETUP ---
logs_dir = os.path.join(SCRIPT_DIR, "logs")
os.makedirs(logs_dir, exist_ok=True)
logger = logging.getLogger("eligibility_bot")
if not logger.handlers:
    logger.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))
    ch = logging.StreamHandler()
    ch.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))
    fh = RotatingFileHandler(os.path.join(logs_dir, "eligibility-bot.log"), maxBytes=1_000_000, backupCount=3)
    fh.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    ch.setFormatter(fmt)
    fh.setFormatter(fmt)
    logger.addHandler(ch)
    logger.addHandler(fh)

    # --- PDF SAVE HELPER (inlined) ---
    def save_current_page_pdf(page: Page, base_dir: str, patient_name: str, subdir: str = "logs/PDFs") -> Optional[str]:
        """Save the current page as a PDF to base_dir/subdir/<patient_name>.pdf using Chromium CDP.

        Returns absolute file path on success; None on failure.
        """
        try:
            pdf_dir = os.path.join(base_dir, subdir)
            os.makedirs(pdf_dir, exist_ok=True)
            try:
                page.emulate_media(media="print")
            except Exception:
                pass
            client = page.context.new_cdp_session(page)
            result = client.send('Page.printToPDF', {
                'printBackground': True,
                'preferCSSPageSize': True,
                'scale': 0.95,
                'landscape': False,
                'transferMode': 'ReturnAsBase64'
            })
            data_b64 = result.get('data')
            if not data_b64:
                return None
            def _sanitize(name: str) -> str:
                return "".join(c for c in name if c.isalnum() or c in ("_", "-", "."))[:128]
            safe = _sanitize(patient_name or f"report_{time.strftime('%Y%m%d-%H%M%S')}")
            pdf_path = os.path.join(pdf_dir, f"{safe}.pdf")
            import base64 as _b64
            with open(pdf_path, 'wb') as f:
                f.write(_b64.b64decode(data_b64))
            return pdf_path
        except Exception:
            return None

# Initialize API key cycle with per-run rotation and persist last start index under logs/
def _init_api_key_cycle():
    global api_key_cycler
    try:
        n = len(VALID_API_KEYS)
        if n == 0:
            raise ValueError("No valid API keys available for rotation.")
        idx_path = os.path.join(logs_dir, "key_index.json")
        last_idx = -1
        try:
            if os.path.exists(idx_path):
                with open(idx_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    last_idx = int(data.get("last_used_index", -1))
        except Exception:
            last_idx = -1
        start_idx = (last_idx + 1) % n
        ordered = VALID_API_KEYS[start_idx:] + VALID_API_KEYS[:start_idx]
        api_key_cycler = itertools.cycle(ordered)
        # Persist new start for next run
        try:
            with open(idx_path, "w", encoding="utf-8") as f:
                json.dump({"last_used_index": start_idx}, f)
        except Exception:
            pass
        print(f"-> API key rotation: starting with key index {start_idx} (***{(ordered[0] or '')[-4:]})")
    except Exception as e:
        print(f"-!- Failed to initialize API key rotation: {e}. Using default order.")
        api_key_cycler = itertools.cycle(VALID_API_KEYS)

_init_api_key_cycle()

# --- AI AND AUTOMATION LOGIC ---

# Token usage accounting (per-row)
USAGE_TOKENS = {"prompt": 0, "candidates": 0, "total": 0}

def _accumulate_usage(resp: Any):
    try:
        um = getattr(resp, "usage_metadata", None)
        if not um:
            return
        # Support both attribute and dict-like access
        p = getattr(um, "prompt_token_count", None) or um.get("prompt_token_count", 0)
        c = getattr(um, "candidates_token_count", None) or um.get("candidates_token_count", 0)
        t = getattr(um, "total_token_count", None) or um.get("total_token_count", 0)
        USAGE_TOKENS["prompt"] += int(p or 0)
        USAGE_TOKENS["candidates"] += int(c or 0)
        USAGE_TOKENS["total"] += int(t or 0)
    except Exception:
        pass

SYSTEM_PROMPTS = {
    "form_filling": """
    You are a meticulous web automation assistant. Your task is to analyze the provided HTML of a web form and a JSON object of patient data. Create a JSON array of steps to fill ALL necessary fields based on the patient data provided.

    **CRITICAL INSTRUCTIONS:**
    1.  **Identify All Required Fields:** You MUST find and create steps for the input fields associated with the following labels: Date of Service (and Date of Service End, if present), Subscriber ID (or Member ID), Subscriber First Name (or First Name), Subscriber Last Name (or Last Name), and Subscriber Date of Birth (or DOB).
    2.  **Map Data:** Use the provided patient data to map values to the fields you identified. If you find both "Date of Service" and "Date of Service End", use the `dos` value for both.
    3.  **Generate Robust CSS Selectors:** Create a precise CSS selector for each input field. Prefer using element IDs.
    4.  **Ignore Dropdowns:** Your plan must ONLY include actions for text `<input>` elements. Do NOT generate steps for `<select>` elements like "Search By".
    5.  **Return a JSON Array ONLY:** The output must be a valid JSON array of objects. Each object must have a "selector" key and a "value" key.
    """,
    "payer_selection": """
    You are an expert web automation assistant. Analyze the provided HTML of a payer list and a target payer name. Generate a two-step JSON plan to select the correct payer.
    1.  Find the Best Match: From the HTML, determine which category the target payer belongs to. Then, find the best and most logical match for the target name within that category's list. For example, if the target is "UMR", match it to "UMR-Wausau". If the target is "BCBS North Carolina", find the "Blue Cross Blue Shield" category and then the "BCBS North Carolina" link.
    2.  Return ONLY a JSON object with two keys: "category_text" (the exact text of the category link) and "payer_text" (the exact text of the final payer link).
    """,
    "report_generation": """
    You are an expert data structuring engine for U.S. healthcare insurance reports. Transform the provided HTML into a single JSON object with only these fields:

    OUTPUT RULES (STRICT):
    - Respond with a single valid JSON object only. No markdown, no code fences, no commentary.
    - Use double quotes for all keys and string values. No trailing commas. No comments.
    - If a value is missing, use the string "Not Found". Do not invent data.

    REQUIRED FIELDS:
    - status: A specific coverage status like "Active Coverage" or "Inactive Coverage" (do not shorten to just "Active").
    - policy_begin: Policy start date string (or "Not Found").
    - policy_end: Policy end date string (or "Not Found").
    - summaries: Object with keys copay, deductible, coinsurance, out_of_pocket. Each value is a concise, newline-separated string summarizing relevant benefits.
    """
}

# --- PAYER PLAN CACHE ---
_payer_cache: Dict[str, Dict[str, str]] = {}
_form_fill_cache: Dict[str, list] = {}

def _normalize_payer_key(name: str) -> str:
    return (name or "").strip().lower()

def _load_payer_cache():
    global _payer_cache
    try:
        if os.path.exists(PAYER_CACHE_FILE):
            with open(PAYER_CACHE_FILE, "r", encoding="utf-8") as f:
                _payer_cache = json.load(f)
                if not isinstance(_payer_cache, dict):
                    _payer_cache = {}
    except Exception as e:
        logger.warning(f"Failed to load payer cache: {e}")
        _payer_cache = {}

def _save_payer_cache():
    try:
        with open(PAYER_CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(_payer_cache, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.warning(f"Failed to save payer cache: {e}")

def _load_form_fill_cache():
    global _form_fill_cache
    try:
        if os.path.exists(FORM_FILL_CACHE_FILE):
            with open(FORM_FILL_CACHE_FILE, "r", encoding="utf-8") as f:
                _form_fill_cache = json.load(f)
                if not isinstance(_form_fill_cache, dict):
                    _form_fill_cache = {}
    except Exception as e:
        logger.warning(f"Failed to load form-fill cache: {e}")
        _form_fill_cache = {}

def _save_form_fill_cache():
    try:
        with open(FORM_FILL_CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(_form_fill_cache, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.warning(f"Failed to save form-fill cache: {e}")

def _check_timeout(deadline_ts: Optional[float] = None):
    if deadline_ts and time.time() > deadline_ts:
        raise TimeoutError("Operation timed out for this row.")

# --- INPUT SANITIZATION FOR AI ---
def _sanitize_html_for_ai(html: str, max_chars: int = REPORT_HTML_MAX_CHARS) -> str:
    """Targeted sanitization to retain only essential eligibility content.

    Keep:
    - <h1 id="trnEligibilityStatus">...</h1>
    - <div id="BasicProfile"> (prefer only Plan Begin/End rows)
    - Benefit sections + tables for: Co-Payment, Co-Insurance, Deductible, Out of Pocket (including common label variants)

    Otherwise remove scripts/styles/comments, collapse whitespace, and truncate.
    Fallback to base cleanup if targeted extraction fails.
    """
    try:
        raw = html
        # Strip scripts/styles/comments
        raw = re.sub(r"<script[\s\S]*?</script>", " ", raw, flags=re.IGNORECASE)
        raw = re.sub(r"<style[\s\S]*?</style>", " ", raw, flags=re.IGNORECASE)
        raw = re.sub(r"<!--([\s\S]*?)-->", " ", raw)

        pieces: List[str] = []
        seen_table_ids: set[str] = set()
        # 1) Eligibility Status
        m = re.search(r"<h1[^>]*id=\"trnEligibilityStatus\"[^>]*>\s*([\s\S]*?)\s*</h1>", raw, re.IGNORECASE)
        if m:
            pieces.append(f"<h1 id=\"trnEligibilityStatus\">{m.group(1)}</h1>")

        # 2) Basic Profile (Plan Begin/End)
        m = re.search(r"<div[^>]*id=\"BasicProfile\"[^>]*>[\s\S]*?</div>", raw, re.IGNORECASE)
        if m:
            bp = m.group(0)
            keep_rows: List[str] = []
            for lab in ["Plan Begin Date", "Plan End Date"]:
                mm = re.search(rf"<dt>\s*{re.escape(lab)}\s*:</dt>\s*<dd>([^<]+)</dd>", bp, re.IGNORECASE)
                if mm:
                    keep_rows.append(f"<dt>{lab}:</dt><dd>{mm.group(1)}</dd>")
            if keep_rows:
                pieces.append("<div id=\"BasicProfile\"><dl>" + "".join(keep_rows) + "</dl></div>")
            else:
                pieces.append(bp)

        # Track where intro (status + basic profile) ends so we can insert high-priority OOP context right after
        intro_end_index = len(pieces)

        # 3) Benefits sections of interest (broadened matching, fewer structure assumptions)
        # Core labels
        core_labels = [
            "Co-Payment",
            "Co-Insurance",
            "Deductible",
        ]
        # OOP variants
        oop_variants = [
            "Out of Pocket",
            "Out-of-Pocket",
            "Out of Pocket (Stop Loss)",
            "Out-of-Pocket (Stop Loss)",
            "Maximum Out of Pocket",
            "Out-of-Pocket Maximum",
            "Out of Pocket Maximum",
            "Out-of-Pocket Limit",
            "Out of Pocket Limit",
            "Out-of-Pocket Expenses",
            "Out of Pocket Expenses",
            "Out-of-Pocket Accumulators",
            "Out of Pocket Accumulators",
            "OOP Max",
            "OOP Maximum",
            "MOOP",
        ]

        found_oop = False
        oop_context_blocks: List[str] = []

        def _append_label_and_table(label: str, html_block: str):
            # Try to extract table id to avoid duplicates
            nonlocal seen_table_ids
            tid = None
            try:
                m_id = re.search(r'id\s*=\s*"(BenefitsTable\d+)"', html_block, re.IGNORECASE)
                if m_id:
                    tid = m_id.group(1)
            except Exception:
                tid = None
            if tid and tid in seen_table_ids:
                return
            if tid:
                seen_table_ids.add(tid)
            pieces.append(f"<div class=\"benefit-section-label\">{label}</div>" + html_block)

        # Accept headings/anchors/spans/divs containing label followed by a BenefitsTable div
        def _find_and_append_by_labels(labels: List[str], mark_oop: bool = False):
            nonlocal found_oop
            for lab in labels:
                pat = re.compile(
                    rf"((<a[^>]*>\s*{re.escape(lab)}\s*</a>)|(<h[1-6][^>]*>\s*{re.escape(lab)}\s*</h[1-6]>)|(<span[^>]*>\s*{re.escape(lab)}\s*</span>)|(<div[^>]*>\s*{re.escape(lab)}\s*</div>))[\s\S]*?(<div[^>]*id=\"BenefitsTable\d+\"[\s\S]*?</div>)",
                    re.IGNORECASE,
                )
                mm = pat.search(raw)
                if mm:
                    _append_label_and_table(lab, mm.group(6))
                    if mark_oop:
                        found_oop = True

        _find_and_append_by_labels(core_labels, mark_oop=False)
        _find_and_append_by_labels(oop_variants, mark_oop=True)

        # If OOP not found by structural labels, try direct keyword search and capture local context window(s)
        if not found_oop:
            # Combine variants with a few common phrases
            oop_keywords = oop_variants + [
                "Out of Pocket Responsibility",
                "Out-of-Pocket Responsibility",
                "Out of Pocket Amount",
                "Maximum Out of Pocket",
                "Max Out of Pocket",
                "OOP",
            ]
            # Build regex that tolerates minor punctuation/spacing differences
            alt = "|".join([re.escape(k) for k in oop_keywords])
            pat = re.compile(alt, re.IGNORECASE)
            max_blocks = 2  # avoid oversized output
            for mi, mm in enumerate(pat.finditer(raw)):
                if mi >= max_blocks:
                    break
                start = max(0, mm.start() - 2000)
                end = min(len(raw), mm.end() + 4000)
                snippet = raw[start:end]
                # Try to expand to nearest surrounding <div>...</div> to avoid broken HTML chunks
                try:
                    # Find nearest preceding <div and subsequent </div>
                    div_start = snippet.rfind("<div", 0, mm.start() - start)
                    div_end_rel = snippet.find("</div>", mm.end() - start)
                    if div_start != -1 and div_end_rel != -1:
                        snippet = snippet[div_start:div_end_rel + len("</div>")]
                except Exception:
                    pass
                oop_context_blocks.append(
                    f"<div class=\"benefit-section-label\">Out-of-Pocket (Context)</div>" + snippet
                )
            if oop_context_blocks:
                found_oop = True
                # Insert OOP context blocks immediately after intro so they survive truncation
                pieces[intro_end_index:intro_end_index] = oop_context_blocks

        # Safe fallback: if we failed to detect any OOP block, include all benefit tables for AI to summarize
        if not found_oop:
            for mtab in re.finditer(r"(<div[^>]*id=\"BenefitsTable\d+\"[\s\S]*?</div>)", raw, re.IGNORECASE):
                block = mtab.group(1)
                m_id = re.search(r'id\s*=\s*"(BenefitsTable\d+)"', block, re.IGNORECASE)
                tid = m_id.group(1) if m_id else None
                if tid and tid in seen_table_ids:
                    continue
                if tid:
                    seen_table_ids.add(tid)
                pieces.append(block)

        content = "\n".join(pieces) if pieces else raw
        content = re.sub(r"\s+", " ", content).strip()
        if len(content) > max_chars:
            content = content[:max_chars]
        return content
    except Exception:
        # Fall back to base cleanup
        pass
    try:
        fallback = re.sub(r"<script[\s\S]*?</script>", " ", html, flags=re.IGNORECASE)
        fallback = re.sub(r"<style[\s\S]*?</style>", " ", fallback, flags=re.IGNORECASE)
        fallback = re.sub(r"<!--([\s\S]*?)-->", " ", fallback)
        fallback = re.sub(r"\s+", " ", fallback).strip()
        if len(fallback) > max_chars:
            fallback = fallback[:max_chars]
        return fallback
    except Exception:
        return html

def _extract_json_object(text: str) -> Optional[str]:
    """Extract the first top-level JSON object from text using brace matching.

    Handles quotes and escapes to avoid counting braces inside strings.
    Returns None if no complete object is found.
    """
    stack = 0
    in_str = False
    esc = False
    start = -1
    for i, ch in enumerate(text):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        else:
            if ch == '"':
                in_str = True
            elif ch == '{':
                if stack == 0:
                    start = i
                stack += 1
            elif ch == '}':
                if stack > 0:
                    stack -= 1
                    if stack == 0 and start != -1:
                        return text[start:i+1]
    return None

def _fallback_parse_report(html: str) -> Dict[str, Any]:
    """Lightweight DOM regex-based fallback to extract key fields when AI fails.

    Returns a dict matching a subset of the JSON schema with safe defaults.
    """
    try:
        text = unescape(html)
        # Status: look for stsactive/inactive or text near title
        status = "Not Found"
        m = re.search(r"id=\"trnEligibilityStatus\"[^>]*>([^<]+)", text, re.IGNORECASE)
        if m:
            status = m.group(1).strip()
        else:
            m = re.search(r"Active Coverage|Inactive Coverage", text, re.IGNORECASE)
            if m:
                status = m.group(0)

        # Plan dates
        plan_begin = "Not Found"
        plan_end = "Not Found"
        mb = re.search(r"Plan\s*Begin\s*Date:\s*</dt>\s*<dd>([^<]+)</dd>", text, re.IGNORECASE)
        if mb:
            plan_begin = mb.group(1).strip()
        me = re.search(r"Plan\s*End\s*Date:\s*</dt>\s*<dd>([^<]+)</dd>", text, re.IGNORECASE)
        if me:
            plan_end = me.group(1).strip()

        # Summaries: naive snippets from tables
        def _summary_for(keyword: str, is_regex: bool = False) -> str:
            # Grab a small window around keyword occurrences. If is_regex=True, treat keyword as a regex alternation.
            safe_kw = keyword if is_regex else re.escape(keyword)
            pat = re.compile(rf"<a[^>]*>\s*({safe_kw})\s*</a>[\s\S]*?<table[\s\S]*?</table>", re.IGNORECASE)
            m = pat.search(text)
            if not m:
                return "Not Found"
            snippet = re.sub(r"<[^>]+>", " ", m.group(0))
            snippet = re.sub(r"\s+", " ", snippet).strip()
            return snippet[:400]

        summaries = {
            "copay": _summary_for("Co-Payment"),
            "deductible": _summary_for("Deductible"),
            "coinsurance": _summary_for("Co-Insurance"),
            "out_of_pocket": _summary_for("Out of Pocket|Out-of-Pocket", is_regex=True),
        }

        return {
            "status": status,
            "policy_begin": plan_begin,
            "policy_end": plan_end,
            "summaries": summaries,
        }
    except Exception as e:
        logger.warning(f"Fallback parse failed: {e}")
        return {
            "status": "Not Found",
            "policy_begin": "Not Found",
            "policy_end": "Not Found",
            "summaries": {"copay": "Not Found", "deductible": "Not Found", "coinsurance": "Not Found", "out_of_pocket": "Not Found"},
        }

# --- LOGIN HELPERS ---
def _is_logged_in(page: Page) -> bool:
    try:
        page.goto("https://mytools.gatewayedi.com/default.aspx", timeout=60000)
        page.locator("#NavCtrl_navHome").wait_for(timeout=15000)
        return True
    except Exception:
        return False

def _save_cookies(context, path: str):
    try:
        cookies = context.cookies()
        with open(path, "wb") as f:
            pickle.dump(cookies, f)
        print(f"-> Cookies saved to '{path}'.")
    except Exception as e:
        print(f"-!- Failed to save cookies: {e}")

def _load_cookies_if_exists(context, path: str) -> bool:
    try:
        if os.path.exists(path):
            with open(path, "rb") as f:
                cookies = pickle.load(f)
            if isinstance(cookies, list) and cookies:
                context.add_cookies(cookies)
                print(f"-> Loaded {len(cookies)} cookies from '{path}'.")
                return True
    except Exception as e:
        print(f"-!- Failed to load cookies from '{path}': {e}")
    return False

# --- METRICS ---
PROCESSED_COUNT = 0
OK_COUNT = 0
FAIL_COUNT = 0
RETRIED_COUNT = 0

def _safe_response_text(response) -> str:
    """Best-effort extraction of text from a Gemini response, even if response.text is unavailable."""
    # Fast path
    try:
        t = getattr(response, "text", None)
        if t:
            return t
    except Exception:
        pass
    # Fallback: concatenate candidate parts' text
    try:
        candidates = getattr(response, "candidates", None) or []
        for cand in candidates:
            content = getattr(cand, "content", None)
            parts = getattr(content, "parts", None) if content else None
            if parts:
                texts = []
                for p in parts:
                    txt = getattr(p, "text", None)
                    if txt:
                        texts.append(txt)
                if texts:
                    return "\n".join(texts)
    except Exception:
        pass
    return ""

def make_ai_call_with_retry(chat_session, prompt_text, max_retries=3):
    """
    Sends a message to the AI chat session with an automatic retry mechanism
    for 500-level server errors.
    """
    global RETRIED_COUNT
    for attempt in range(max_retries):
        try:
            response = chat_session.send_message(prompt_text)
            return response
        except Exception as e:
            # Check if the error is a 500-level server error
            if "500" in str(e) or "internal error" in str(e).lower():
                print(f"   -!- AI server error (500), attempt {attempt + 1} of {max_retries}. Retrying in 5 seconds...")
                RETRIED_COUNT += 1
                time.sleep(5)
                continue # Go to the next attempt
            else:
                # If it's a different error, raise it immediately
                raise e
    # If all retries fail, raise the last exception
    raise Exception(f"AI call failed after {max_retries} retries.")

def _ai_text_with_retries(model_names, system_prompt: str, user_prompt: str, attempts: int = 4, base_backoff: float = 1.5) -> str:
    """Generate text from Gemini with retries, key rotation, and model fallback.

    - Tries models in order from model_names for each attempt; last attempt will always use the last model.
    - Rotates API keys on every attempt via get_next_api_key().
    - Exponential backoff between attempts on transient errors (5xx, timeouts, empty response).
    """
    last_err = None
    for i in range(attempts):
        model_name = model_names[min(i, len(model_names) - 1)]
        try:
            api_key = get_next_api_key()
            alias = (api_key or "")[-4:]
            print(f"   - AI text attempt {i+1}/{attempts}: model={model_name}, key=***{alias}")
            genai.configure(api_key=api_key)  # type: ignore[attr-defined]
            model = genai.GenerativeModel(model_name)  # type: ignore[attr-defined]
            chat = model.start_chat()
            if system_prompt:
                sys_resp = chat.send_message(system_prompt)
                _accumulate_usage(sys_resp)
            response = chat.send_message(user_prompt)
            _accumulate_usage(response)
            text = _safe_response_text(response).strip()
            if not text:
                raise RuntimeError("Empty AI response text")
            return text
        except Exception as e:
            last_err = e
            # Heuristic: transient if 5xx, internal error, timeout/deadline, or empty
            msg = str(e).lower()
            transient = any(s in msg for s in ["500", "internal", "timeout", "deadline", "temporarily", "unavailable"]) or isinstance(e, TimeoutError)
            if i < attempts - 1 and transient:
                # Exponential backoff with small jitter via base_backoff multiplier
                sleep_s = (base_backoff ** i) * 2
                # Respect server-advised retry delay if provided
                try:
                    m = re.search(r"retry_delay\s*\{\s*seconds:\s*(\d+)", str(e))
                    if m:
                        sleep_s = max(sleep_s, int(m.group(1)))
                except Exception:
                    pass
                print(f"   -!- AI attempt {i+1} failed on {model_name}: {e}. Retrying in {int(sleep_s)}s with rotated key/model...")
                try:
                    time.sleep(sleep_s)
                except Exception:
                    pass
                continue
            break
    raise RuntimeError(f"AI text generation failed after {attempts} attempts: {last_err}")

def _ai_json_with_schema(
    model_names,
    system_prompt: str,
    user_prompt: str,
    schema: Dict[str, Any],
    attempts: int = 3,
    base_backoff: float = 1.6,
) -> Dict[str, Any]:
    """Generate strict JSON from Gemini using response_schema.

    Tries models with API key rotation and exponential backoff. Returns parsed JSON dict.
    If the SDK/model doesn't support structured output or errors occur, raises RuntimeError.
    """
    last_err = None
    generation_config = {
        "response_mime_type": "application/json",
        "response_schema": schema,
    }
    for i in range(attempts):
        model_name = model_names[min(i, len(model_names) - 1)]
        try:
            api_key = get_next_api_key()
            alias = (api_key or "")[-4:]
            print(f"   - AI structured attempt {i+1}/{attempts}: model={model_name}, key=***{alias}")
            genai.configure(api_key=api_key)  # type: ignore[attr-defined]
            model = genai.GenerativeModel(model_name, system_instruction=system_prompt)  # type: ignore[attr-defined]
            response = model.generate_content(user_prompt, generation_config=generation_config)  # type: ignore[arg-type]
            _accumulate_usage(response)
            text = _safe_response_text(response).strip()
            if not text:
                raise RuntimeError("Empty AI response text")
            data = json.loads(text)
            if not isinstance(data, dict):
                raise RuntimeError("Structured output was not a JSON object")
            return data
        except Exception as e:
            last_err = e
            msg = str(e).lower()
            transient = any(s in msg for s in ["500", "internal", "timeout", "deadline", "temporarily", "unavailable", "quota"]) or isinstance(e, TimeoutError)
            if i < attempts - 1 and transient:
                sleep_s = (base_backoff ** i) * 2
                try:
                    m = re.search(r"retry_delay\s*\{\s*seconds:\s*(\d+)", str(e))
                    if m:
                        sleep_s = max(sleep_s, int(m.group(1)))
                except Exception:
                    pass
                print(f"   -!- AI (structured) attempt {i+1} failed on {model_name}: {e}. Retrying in {int(sleep_s)}s...")
                try:
                    time.sleep(sleep_s)
                except Exception:
                    pass
                continue
            break
    raise RuntimeError(f"AI structured JSON failed after {attempts} attempts: {last_err}")

def upload_file_to_drive(drive_service, folder_id, file_path, mimetype):
    """Uploads a file to a specific Google Drive folder and returns its shareable link."""
    file_name = os.path.basename(file_path)
    try:
        print(f"   - Uploading '{file_name}' to Google Drive...")
        file_metadata = {'name': file_name, 'parents': [folder_id]}
        media = MediaFileUpload(file_path, mimetype=mimetype, resumable=True)
        file = drive_service.files().create(
            body=file_metadata,
            media_body=media,
            fields='id, webViewLink'
        ).execute()
        file_id = file.get('id')
        drive_service.permissions().create(fileId=file_id, body={'type': 'anyone', 'role': 'reader'}).execute()
        print(f"   - Upload successful for {file_name}.")
        return file.get('webViewLink')
    except Exception as e:
        print(f"   -!- Google Drive upload failed for {file_name}: {e}")
        return "Drive Upload Failed"

def generate_form_fill_plan(page_html: str, patient_data: dict) -> list:
    """Uses a Gemini chat session to generate a form-filling plan efficiently."""
    print("   - Asking AI to generate a form-filling plan (Model: gemini-2.5-flash)...")
    try:
        # Try payer-specific cached plan first
        payer_key = _normalize_payer_key(patient_data.get("payer_name", ""))
        cached = _form_fill_cache.get(payer_key)
        if cached:
            # Personalize cached placeholder plan with current patient's values
            print("   - Using cached form-fill plan for this payer.")
            try:
                allowed_keys = {"dos", "member_id", "first_name", "last_name", "dob"}

                def _norm_placeholder_token(s: str) -> Optional[str]:
                    if not isinstance(s, str):
                        return None
                    raw = s.strip()
                    low = raw.lower()
                    # Strip common wrapper syntaxes like {{key}}, ${key}, <<key>>
                    low = re.sub(r"^\{\{\s*|\}\}$", "", low)
                    low = re.sub(r"^\$\{\s*|\}$", "", low)
                    low = re.sub(r"^<<\s*|\s*>>$", "", low)
                    low = low.strip()
                    # Map synonyms to canonical keys
                    synonyms = {
                        "dos": "dos",
                        "date_of_service": "dos",
                        "service_date": "dos",
                        "start_date": "dos",
                        "visit_date": "dos",
                        "dob": "dob",
                        "date_of_birth": "dob",
                        "birth_date": "dob",
                        "member_id": "member_id",
                        "subscriber_id": "member_id",
                        "policy_number": "member_id",
                        "id": "member_id",
                        "first_name": "first_name",
                        "subscriber_first_name": "first_name",
                        "last_name": "last_name",
                        "subscriber_last_name": "last_name",
                    }
                    return synonyms.get(low)
                personalized = []
                for step in cached:
                    sel = step.get("selector", "")
                    val = step.get("value", "")
                    key = _norm_placeholder_token(val)
                    if key in allowed_keys:
                        personalized.append({"selector": sel, "value": str(patient_data.get(key, ""))})
                    else:
                        personalized.append(step)
                return personalized
            except Exception:
                return cached
        system_prompt = SYSTEM_PROMPTS['form_filling']
        data_prompt = f"**Patient Data:**\n```json\n{json.dumps(patient_data, indent=2)}\n```\n\n**Form HTML:**\n```html\n{page_html}\n```"
        raw = _ai_text_with_retries(['gemini-2.5-flash','gemini-2.5-flash','gemini-2.5-pro'], system_prompt, data_prompt, attempts=3)
        cleaned_text = raw.strip().replace("```json", "").replace("```", "").strip()
        plan = json.loads(cleaned_text)
        # Save a placeholder-based version to cache so values are not hard-coded
        if payer_key:
            try:
                reverse_map = {}
                # Build reverse map from value string -> key
                def _normalized_date_variants(s: str) -> List[str]:
                    s = s.strip()
                    outs = {s}
                    # Normalize common separators and leading zeros for mm/dd/yyyy
                    m = re.match(r"^(\d{1,4})[\-/\.](\d{1,2})[\-/\.](\d{1,4})$", s)
                    if m:
                        a, b, c = m.group(1), m.group(2), m.group(3)
                        # Try to detect year position (YYYY first or last)
                        if len(a) == 4:
                            y, mm, dd = a, b.zfill(2), c.zfill(2)
                        elif len(c) == 4:
                            y, mm, dd = c, a.zfill(2), b.zfill(2)
                        else:
                            y, mm, dd = c, a.zfill(2), b.zfill(2)
                        outs.add(f"{mm}/{dd}/{y}")
                        outs.add(f"{int(mm)}/{int(dd)}/{y}")
                        outs.add(f"{y}-{mm}-{dd}")
                    return list(outs)

                for k in ["dos", "member_id", "first_name", "last_name", "dob"]:
                    v = str(patient_data.get(k, "")).strip()
                    if not v:
                        continue
                    # Map raw
                    reverse_map[v] = k
                    # Map normalized variants for dates
                    if k in ("dos", "dob"):
                        for vv in _normalized_date_variants(v):
                            reverse_map[vv] = k
                cached_plan = []
                for step in plan:
                    sel = step.get("selector", "")
                    val = str(step.get("value", "")).strip()
                    key = reverse_map.get(val)
                    if not key:
                        # Infer by selector heuristics if value match failed (useful for DOS/DOB mismatched formatting)
                        low_sel = sel.lower()
                        if any(tok in low_sel for tok in ["dos", "service", "svc", "startdate", "servicedate"]):
                            key = "dos"
                        elif any(tok in low_sel for tok in ["dob", "birth", "birthdate", "dateofbirth"]):
                            key = "dob"
                        elif any(tok in low_sel for tok in ["member", "subscriber", "policy", "subid", "memberid"]):
                            key = "member_id"
                        elif any(tok in low_sel for tok in ["first", "fname", "firstname"]):
                            key = "first_name"
                        elif any(tok in low_sel for tok in ["last", "lname", "lastname", "surname"]):
                            key = "last_name"
                    if key:
                        cached_plan.append({"selector": sel, "value": key})
                    else:
                        cached_plan.append({"selector": sel, "value": val})
                _form_fill_cache[payer_key] = cached_plan
                _save_form_fill_cache()
            except Exception:
                # On any error, fall back to caching raw plan (better than nothing)
                _form_fill_cache[payer_key] = plan
                _save_form_fill_cache()
        print("   - AI form-fill plan generated successfully.")
        return plan
    except Exception as e:
        print(f"   -!- CRITICAL: AI failed to generate a valid form-fill plan: {e}")
        return []

def _try_fill_field(page: Page, label_candidates: list[str], value: str, timeout_ms: int = 2000, is_date: bool = False) -> bool:
    """Best-effort fill by trying common label and attribute strategies.

    - Tries get_by_label on each candidate
    - Falls back to placeholder/aria-label contains
    - Tries input by id/name contains normalized tokens
    Returns True if filled, else False.
    """
    def _date_variants(v: str) -> List[str]:
        v = (v or "").strip()
        outs = []
        # normalize digits
        m = re.match(r"^(\d{1,4})[\-/\. ](\d{1,2})[\-/\. ](\d{1,4})$", v)
        if m:
            a, b, c = m.group(1), m.group(2), m.group(3)
            if len(a) == 4:
                y, mm, dd = a, b.zfill(2), c.zfill(2)
            elif len(c) == 4:
                y, mm, dd = c, a.zfill(2), b.zfill(2)
            else:
                y, mm, dd = c, a.zfill(2), b.zfill(2)
            outs += [f"{mm}/{dd}/{y}", f"{int(mm)}/{int(dd)}/{y}", f"{y}-{mm}-{dd}", f"{mm}{dd}{y}"]
        # prefer original first
        return [v] + [x for x in outs if x != v]

    def _safe_fill(locator_expr) -> bool:
        try:
            loc = locator_expr
            loc.wait_for(state="visible", timeout=timeout_ms)
            loc.scroll_into_view_if_needed()
            page.wait_for_timeout(100)
            # Handle date specifics
            if is_date:
                # Try type=date format first if applicable
                try:
                    t = (loc.get_attribute("type") or "").lower()
                except Exception:
                    t = ""
                for candidate in _date_variants(value):
                    try:
                        cand = candidate
                        if t == "date":
                            # Expect yyyy-mm-dd
                            m = re.match(r"^(\d{1,2})/(\d{1,2})/(\d{4})$", candidate)
                            if m:
                                cand = f"{m.group(3)}-{m.group(1).zfill(2)}-{m.group(2).zfill(2)}"
                        loc.fill(cand, timeout=timeout_ms)
                        # Verify
                        try:
                            iv = loc.input_value(timeout=timeout_ms) or (loc.get_attribute("value") or "")
                        except Exception:
                            iv = loc.get_attribute("value") or ""
                        if iv and (cand in iv or re.sub(r"\D", "", cand) == re.sub(r"\D", "", iv)):
                            return True
                        # Try sequential typing for masked fields
                        loc.fill("")
                        loc.press_sequentially(cand, delay=50)
                        try:
                            iv = loc.input_value(timeout=timeout_ms) or (loc.get_attribute("value") or "")
                        except Exception:
                            iv = loc.get_attribute("value") or ""
                        if iv and (cand in iv or re.sub(r"\D", "", cand) == re.sub(r"\D", "", iv)):
                            return True
                    except Exception:
                        continue
                return False
            else:
                loc.fill(value, timeout=timeout_ms)
                # Verify basic fill took effect; if not, try sequential typing
                try:
                    iv = loc.input_value(timeout=timeout_ms) or (loc.get_attribute("value") or "")
                except Exception:
                    iv = loc.get_attribute("value") or ""
                if not iv or (len(value) > 0 and value not in iv and re.sub(r"\W", "", value) != re.sub(r"\W", "", iv)):
                    loc.fill("")
                    loc.press_sequentially(str(value), delay=50)
                return True
        except Exception:
            return False

    # Try accessible labels
    for lbl in label_candidates:
        try:
            loc = page.get_by_label(lbl, exact=False)
            if _safe_fill(loc.first):
                return True
        except Exception:
            pass

    # Try placeholder or aria-label contains
    for lbl in label_candidates:
        try:
            css = f"input[placeholder*='{lbl}'], input[aria-label*='{lbl}']"
            loc = page.locator(css)
            if _safe_fill(loc.first):
                return True
        except Exception:
            pass

    # Try id/name contains normalized tokens
    tokens = []
    for lbl in label_candidates:
        tokens += re.split(r"\W+", lbl.lower())
    tokens = [t for t in tokens if t]
    if tokens:
        try:
            css = "input"
            locs = page.locator(css)
            count = locs.count()
            for i in range(min(count, 50)):
                el = locs.nth(i)
                try:
                    el_id = (el.get_attribute("id") or "").lower()
                    el_name = (el.get_attribute("name") or "").lower()
                    if any(t in el_id or t in el_name for t in tokens):
                        if _safe_fill(el):
                            return True
                except Exception:
                    continue
        except Exception:
            pass
    return False

def get_all_report_data(html_content: str, patient_base_name: str) -> dict:
    """Uses Gemini to generate a single JSON object with all structured data from the report.

    Root-cause fixes:
    - Sanitizes HTML to reduce noise and length to prevent truncation.
    - Strengthens system prompt to strictly require JSON-only output.
    - Adds robust JSON extraction and diagnostics on repeated parse failure.
    """
    print("   - Asking AI to generate all report data (structured JSON with model fallback)...")

    sanitized_html = _sanitize_html_for_ai(html_content)
    # Save the exact sanitized HTML we send to AI for inspection, using firstname_lastname
    try:
        sanitized_dir = os.path.join(logs_dir, "Sanitized HTMLs")
        os.makedirs(sanitized_dir, exist_ok=True)
        sanitized_path = os.path.join(sanitized_dir, f"{patient_base_name}.html")
        with open(sanitized_path, "w", encoding="utf-8") as f:
            f.write(sanitized_html)
        print(f"   - Saved sanitized report HTML to: {sanitized_path}")
    except Exception:
        pass

    # First, try strict structured output using Gemini response_schema for a clean JSON object
    schema = {
        "type": "object",
        "properties": {
            "status": {"type": "string"},
            "policy_begin": {"type": "string"},
            "policy_end": {"type": "string"},
            "summaries": {
                "type": "object",
                "properties": {
                    "copay": {"type": "string"},
                    "deductible": {"type": "string"},
                    "coinsurance": {"type": "string"},
                    "out_of_pocket": {"type": "string"},
                },
                "required": ["copay", "deductible", "coinsurance", "out_of_pocket"],
            },
        },
        "required": ["status", "policy_begin", "policy_end", "summaries"]
    }
    try:
        user_prompt_struct = (
            "Return a single strictly valid JSON object as specified in the schema. "
            "Do not include any text, explanations, or code fences before or after the JSON.\n\n"
            f"HTML Content:\n{sanitized_html}"
        )
        models_struct = ["gemini-2.5-pro", "gemini-2.5-pro", "gemini-2.5-flash"]
        data_struct = _ai_json_with_schema(models_struct, SYSTEM_PROMPTS['report_generation'], user_prompt_struct, schema, attempts=3)
        # Save success raw JSON/TXT to AI Response with firstname_lastname
        try:
            ai_dir = os.path.join(logs_dir, "AI Response")
            os.makedirs(ai_dir, exist_ok=True)
            raw_ok_path = os.path.join(ai_dir, f"{patient_base_name}.json")
            with open(raw_ok_path, "w", encoding="utf-8") as f:
                json.dump(data_struct, f, ensure_ascii=False, indent=2)
            print(f"   - Saved AI report JSON to: {raw_ok_path}")
            raw_txt_path = os.path.join(ai_dir, f"{patient_base_name}.txt")
            try:
                with open(raw_txt_path, "w", encoding="utf-8") as tf:
                    tf.write(json.dumps(data_struct, ensure_ascii=False, indent=2))
            except Exception:
                pass
        except Exception:
            pass
        print("   - AI successfully generated consolidated report data (structured output).")
        return data_struct
    except Exception as struct_err:
        print(f"   - Structured output not available or failed ({struct_err}). Falling back to text JSON prompt...")

    def _prompt_with_retries() -> str:
        system_prompt = SYSTEM_PROMPTS['report_generation']
        user_prompt = (
            "Return a single strictly valid JSON object as specified in the system instructions. "
            "Do not include any text, explanations, or code fences before or after the JSON.\n\n"
            f"HTML Content:\n{sanitized_html}"
        )
        # Try pro, then flash as fallback on later attempts
        models = ['gemini-2.5-pro', 'gemini-2.5-pro', 'gemini-2.5-flash', 'gemini-2.5-flash']
        return _ai_text_with_retries(models, system_prompt, user_prompt, attempts=4, base_backoff=1.8)

    # Attempt 1 (text JSON prompt)
    try:
        raw_text = _prompt_with_retries()
        candidate = _extract_json_object(raw_text) or raw_text.replace("```json", "").replace("```", "").strip()
        try:
            data = json.loads(candidate)
            # Save success raw JSON/TXT to AI Response with firstname_lastname
            try:
                ai_dir = os.path.join(logs_dir, "AI Response")
                os.makedirs(ai_dir, exist_ok=True)
                raw_ok_path = os.path.join(ai_dir, f"{patient_base_name}.json")
                with open(raw_ok_path, "w", encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
                print(f"   - Saved AI report JSON to: {raw_ok_path}")
                raw_txt_path = os.path.join(ai_dir, f"{patient_base_name}.txt")
                with open(raw_txt_path, "w", encoding="utf-8") as tf:
                    tf.write(raw_text)
            except Exception:
                pass
            print("   - AI successfully generated consolidated report data.")
            return data
        except Exception as parse_err:
            print(f"   -!- JSON parse failed (attempt 1), retrying once: {parse_err}")
            time.sleep(2)
            raw_text = _prompt_with_retries()
            candidate = _extract_json_object(raw_text) or raw_text.replace("```json", "").replace("```", "").strip()
            data = json.loads(candidate)
            # Save success raw JSON/TXT to AI Response with firstname_lastname (retry)
            try:
                ai_dir = os.path.join(logs_dir, "AI Response")
                os.makedirs(ai_dir, exist_ok=True)
                raw_ok_path = os.path.join(ai_dir, f"{patient_base_name}.json")
                with open(raw_ok_path, "w", encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
                print(f"   - Saved AI report JSON to: {raw_ok_path}")
                raw_txt_path = os.path.join(ai_dir, f"{patient_base_name}.txt")
                with open(raw_txt_path, "w", encoding="utf-8") as tf:
                    tf.write(raw_text)
            except Exception:
                pass
            print("   - AI successfully generated consolidated report data (retry).")
            return data
    except Exception as e:
        # Diagnostics: save raw output and sanitized HTML for investigation
        try:
            ai_dir = os.path.join(logs_dir, "AI Response")
            os.makedirs(ai_dir, exist_ok=True)
            diag_txt_path = os.path.join(ai_dir, f"{patient_base_name}.fail.txt")
            with open(diag_txt_path, "w", encoding="utf-8") as f:
                f.write(locals().get('raw_text', ''))
            print(f"   -!- Diagnostics written to: {diag_txt_path}")
        except Exception as diag_err:
            print(f"   -!- Failed to write diagnostics: {diag_err}")
        print(f"   -!- AI failed to generate consolidated JSON after retry: {e}. Falling back to DOM parsing...")
        # Fallback best-effort parse from DOM
        try:
            fallback = _fallback_parse_report(html_content)
            return fallback
        except Exception:
            return {"Error": f"Failed to parse report: {e}"}

def select_payer_with_ai(page: Page, payer_name: str, deadline_ts: Optional[float] = None):
    """Uses a Gemini chat session to find and select the correct payer."""
    print("   - Selecting Payer...")
    try:
        _check_timeout(deadline_ts)
        page.goto("https://mytools.gatewayedi.com/ManagePatients/RealTimeEligibility/Index", wait_until="domcontentloaded")
        payer_list_container = page.locator("#InsurerAccordion")
        payer_list_container.wait_for(state="visible", timeout=30000)
        list_html = payer_list_container.inner_html()

        # Try cache first
        key = _normalize_payer_key(payer_name)
        plan = _payer_cache.get(key)
        if plan and "category_text" in plan and "payer_text" in plan:
            print("   - Using cached payer selection plan.")
        else:
            print("   - Asking AI to find the best payer match and create a plan...")
            _check_timeout(deadline_ts)
            # Try AI plan with retries and key rotation
            system_prompt = SYSTEM_PROMPTS['payer_selection']
            user_prompt = f"The user wants to select: **\"{payer_name}\"**.\n\n**Payer List HTML:**\n```html\n{list_html}\n```"
            try:
                raw = _ai_text_with_retries(['gemini-2.5-flash','gemini-2.5-flash','gemini-2.5-pro'], system_prompt, user_prompt, attempts=3)
                cleaned_text = raw.strip().replace("```json", "").replace("```", "").strip()
                plan = json.loads(cleaned_text)
            except Exception as ai_err:
                print(f"   -!- AI payer planning failed ({ai_err}). Falling back to substring search...")
                plan = None
            if isinstance(plan, dict) and "category_text" in plan and "payer_text" in plan:
                _payer_cache[key] = {"category_text": plan["category_text"], "payer_text": plan["payer_text"]}
                _save_payer_cache()

        if plan:
            print(f"   - AI Plan Received. Category: '{plan['category_text']}', Payer: '{plan['payer_text']}'")

        # Expand category and wait for nested list
        clicked = False
        if plan:
            # Planned path using category then payer
            _check_timeout(deadline_ts)
            category_element = page.get_by_text(plan['category_text'], exact=True).first
            category_element.scroll_into_view_if_needed()
            nested_list = page.locator(f"li[id='{plan['category_text']}'] ul.insurersDetail")
            for _ in range(2):
                category_element.click()
                page.wait_for_timeout(500)
                if nested_list.is_visible():
                    break
            try:
                nested_list.wait_for(state="visible", timeout=15000)
            except Exception:
                pass
            for attempt in range(4):
                try:
                    _check_timeout(deadline_ts)
                    payer_link = nested_list.get_by_text(plan['payer_text'], exact=True).first
                    payer_link.scroll_into_view_if_needed()
                    try:
                        cls = payer_link.get_attribute("class") or ""
                        if "disabled" in cls:
                            page.wait_for_timeout(300)
                    except Exception:
                        pass
                    page.wait_for_timeout(150 + attempt * 200)
                    payer_link.click(timeout=5000)
                    clicked = True
                    break
                except Exception:
                    continue
            if not clicked:
                try:
                    _check_timeout(deadline_ts)
                    fallback_link = nested_list.get_by_text(plan.get('payer_text',''), exact=False).first
                    fallback_link.scroll_into_view_if_needed()
                    page.wait_for_timeout(250)
                    fallback_link.click(timeout=6000)
                    clicked = True
                except Exception:
                    pass
        if not clicked:
            # Heuristic fallback: expand visible categories and click first link containing payer name
            try:
                _check_timeout(deadline_ts)
                headers = payer_list_container.locator("li > a").all()
                for h in headers:
                    try:
                        h.scroll_into_view_if_needed()
                        h.click()
                        page.wait_for_timeout(150)
                    except Exception:
                        continue
                guess_link = payer_list_container.locator("a", has_text=payer_name).first
                guess_link.scroll_into_view_if_needed()
                page.wait_for_timeout(200)
                guess_link.click(timeout=6000)
                clicked = True
                print("   - Fallback payer selection (substring) succeeded.")
            except Exception as fb_err:
                print(f"   -!- Fallback payer selection failed: {fb_err}")
                clicked = False

        page.wait_for_timeout(1200)
        if not clicked:
            raise RuntimeError("Unable to select payer via AI or fallback heuristics")
        print(f"   - AI Payer Selection for '{payer_name}' successful.")

    except Exception as e:
        print(f"   -!- CRITICAL: AI-driven payer selection failed: {e}")
        raise
        
def process_patient(page: Page, drive_service, patient_data: dict) -> dict:
    """Handles all parsing, PDF generation, and uploads for a patient using a single AI call."""
    pdf_link = "N/A"
    # Enforce per-row timeout
    deadline_ts = time.time() + OPERATION_TIMEOUT_SECONDS if OPERATION_TIMEOUT_SECONDS > 0 else None
    # Reset per-row AI token usage counters
    try:
        USAGE_TOKENS["prompt"] = 0
        USAGE_TOKENS["candidates"] = 0
        USAGE_TOKENS["total"] = 0
    except Exception:
        pass
    try:
        # (Form filling logic is unchanged)
        _check_timeout(deadline_ts)
        form_html = page.locator("body").inner_html()
        fill_plan = generate_form_fill_plan(form_html, patient_data)
        if not fill_plan:
            raise ValueError("AI did not return a valid form-filling plan.")
        print("   - Executing AI form-filling plan...")
        # Field mapping to support fallback matching by semantic name
        field_hints = {
            "dos": ["Date of Service", "Service Date", "DOS", "Start Date"],
            "member_id": ["Subscriber ID", "Member ID", "Subscriber Number", "Policy Number", "ID"],
            "first_name": ["First Name", "Subscriber First Name"],
            "last_name": ["Last Name", "Subscriber Last Name"],
            "dob": ["Date of Birth", "DOB", "Birth Date"],
        }
        # Attempt to execute plan; on failure per step, fallback to label-based fill
        for step in fill_plan:
            sel = step.get('selector', '')
            val = step.get('value', '')
            try:
                loc = page.locator(sel)
                loc.wait_for(state="visible", timeout=2000)
                loc.scroll_into_view_if_needed()
                page.wait_for_timeout(100)
                loc.fill(val, timeout=2000)
            except Exception:
                # Guess the field by matching value to patient_data, then use label-based fallback
                matched_key = None
                for k in ["dos", "member_id", "first_name", "last_name", "dob"]:
                    if str(patient_data.get(k, "")) == str(val):
                        matched_key = k
                        break
                if matched_key:
                    is_date = matched_key in ("dos", "dob")
                    if _try_fill_field(page, field_hints[matched_key], val, is_date=is_date):
                        continue
                # As a last resort, try all hint groups with the provided value
                all_labels = sum(field_hints.values(), [])
                # Heuristic: decide if looks like a date
                looks_like_date = bool(re.match(r"\d{1,4}[\-/\. ]\d{1,2}[\-/\. ]\d{1,4}", str(val)))
                if _try_fill_field(page, all_labels, val, is_date=looks_like_date):
                    continue
                raise
        print("   - Form filled according to AI plan.")
        _check_timeout(deadline_ts)
        # Submit: click label if visible; otherwise fallback to real input or JS click
        try:
            btn_label = page.locator("#btnUploadButton")
            btn_label.scroll_into_view_if_needed()
            btn_label.wait_for(state="visible", timeout=2000)
            btn_label.click()
        except Exception:
            try:
                btn_input = page.locator("#btnUpload")
                btn_input.scroll_into_view_if_needed()
                btn_input.click()
            except Exception:
                try:
                    page.evaluate("() => { const el = document.getElementById('btnUpload'); if(el) el.click(); }")
                except Exception:
                    # As a last resort, try clicking any visible green submit near form area
                    try:
                        page.locator("label.btn_green, .btn_green").first.click()
                    except Exception:
                        raise
        page.wait_for_selector("#eligibilityRequestResponse, #EligibilityValidationErrors", timeout=90000)
        error_div = page.locator("#EligibilityValidationErrors")
        if error_div.is_visible() and len(error_div.inner_text().strip()) > 0:
            raise ValueError(f"Form submission error on page: {error_div.inner_text().strip()}")
        
        print("   - Expanding full benefit report...")
        _check_timeout(deadline_ts)
        page.locator("#tab_benefitinformation").click()
        page.wait_for_timeout(500)
        page.locator("#ExpandInfo > li > a").click()
        page.wait_for_timeout(1000)
        # Ensure network is idle before proceeding to parse and capture
        try:
            page.wait_for_load_state("networkidle", timeout=15000)
        except Exception:
            pass
        print("   - Report expanded.")

        report_container = page.locator("#eligibilityRequestResponse")
        report_html = report_container.inner_html()
        
        # Build firstname_lastname base name for artifact naming and save raw HTML
        patient_base_name = f"{patient_data['first_name']}_{patient_data['last_name']}"
        patient_base_name = re.sub(r"[^a-zA-Z0-9._-]+", "_", patient_base_name.strip())
        try:
            raw_dir = os.path.join(logs_dir, "Raw HTMLs")
            os.makedirs(raw_dir, exist_ok=True)
            raw_path = os.path.join(raw_dir, f"{patient_base_name}.html")
            with open(raw_path, "w", encoding="utf-8") as f:
                f.write(report_html)
            print(f"   - Saved raw report HTML to: {raw_path}")
        except Exception:
            pass

        # --- NEW: SINGLE, CONSOLIDATED AI CALL ---
        _check_timeout(deadline_ts)
        all_data = get_all_report_data(report_html, patient_base_name)
        if "Error" in all_data:
            raise ValueError(all_data["Error"])

        patient_name_str = f"{patient_data['last_name']}_{patient_data['first_name']}"
        # --- Save and upload PDF instead of screenshot ---
        try:
            page.wait_for_load_state("networkidle", timeout=15000)
        except Exception:
            pass
        _check_timeout(deadline_ts)
        pdf_path = save_current_page_pdf(page, SCRIPT_DIR, patient_name_str)
        if pdf_path and os.path.exists(pdf_path):
            pdf_link = upload_file_to_drive(drive_service, DRIVE_FOLDER_ID, pdf_path, 'application/pdf')
        else:
            print("   -!- PDF was not created; skipping upload.")
        # Prepare results for Google Sheet update
        summaries = all_data.get("summaries", {})
        # Policy end fallback to end-of-year if missing
        policy_begin = all_data.get("policy_begin", "Not Found")
        policy_end = all_data.get("policy_end", "Not Found")
        if isinstance(policy_end, str) and policy_end.strip().lower() == "not found":
            # Extract year from begin date
            year_match = re.search(r"(20\d{2})", str(policy_begin))
            if year_match:
                policy_end = f"12/31/{year_match.group(1)}"
        final_results = {
                "status": all_data.get("status", "Not Found"),
                "policy_begin": policy_begin,
                "policy_end": policy_end,
                "copay": summaries.get("copay", "Not Found"),
                "deductible": summaries.get("deductible", "Not Found"),
                "coinsurance": summaries.get("coinsurance", "Not Found"),
                "out_of_pocket": summaries.get("out_of_pocket", "Not Found"),
                "pdf_link": pdf_link
        }
        
        print(f"-> Check complete for {patient_data['first_name']}. Status: {final_results.get('status')}")
        logger.info(f"Result: name={patient_data.get('last_name','')},{patient_data.get('first_name','')} status={final_results.get('status')} payer={patient_data.get('payer_name','')} member={patient_data.get('member_id','')}")
        # Print a compact per-row token usage summary
        try:
            print(f"   - AI token usage (this row): prompt={USAGE_TOKENS.get('prompt',0)}, candidates={USAGE_TOKENS.get('candidates',0)}, total={USAGE_TOKENS.get('total',0)}")
            logger.info(f"AI tokens (row): prompt={USAGE_TOKENS.get('prompt',0)} candidates={USAGE_TOKENS.get('candidates',0)} total={USAGE_TOKENS.get('total',0)}")
        except Exception:
            pass
        return final_results

    except Exception as e:
        print(f"   -!- An error occurred during patient processing: {e}")
        # On error we won't try to capture screenshot; PDF flow is best-effort only during success path
        # Print a compact per-row token usage summary even on error
        try:
            print(f"   - AI token usage (this row): prompt={USAGE_TOKENS.get('prompt',0)}, candidates={USAGE_TOKENS.get('candidates',0)}, total={USAGE_TOKENS.get('total',0)}")
            logger.info(f"AI tokens (row): prompt={USAGE_TOKENS.get('prompt',0)} candidates={USAGE_TOKENS.get('candidates',0)} total={USAGE_TOKENS.get('total',0)}")
        except Exception:
            pass
        return {
            "status": f"Error: {type(e).__name__}", "policy_begin": f"{e}", "policy_end": "",
            "copay": "N/A", "deductible": "N/A", "coinsurance": "N/A", "out_of_pocket": "N/A",
            "pdf_link": "Upload Failed"
        }


def get_oauth_credentials(scopes: list) -> Any:
    """Gets user credentials using the manual OAuth 2.0 console flow."""
    creds = None
    # The file token.json stores the user's access and refresh tokens.
    if os.path.exists(TOKEN_FILE):
        with open(TOKEN_FILE, 'rb') as token:
            creds = pickle.load(token)
    
    # If there are no (valid) credentials available, let the user log in.
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            print("-> Refreshing expired OAuth token...")
            creds.refresh(Request())
        else:
            print("-> No valid token found. Initiating manual authorization flow...")
            flow = InstalledAppFlow.from_client_secrets_file(CLIENT_SECRETS_FILE, scopes)
            
            # This is the key part from your other script:
            # Set the redirect URI to Out-of-Band (OOB) for command-line apps.
            flow.redirect_uri = "urn:ietf:wg:oauth:2.0:oob" 

            auth_url, _ = flow.authorization_url(prompt='consent')
            
            print("\n--- GOOGLE AUTHORIZATION REQUIRED ---")
            print("Please open the following URL in your browser:")
            print(f"\n{auth_url}\n")
            print("After authorizing, copy the verification code from your browser and paste it here:")
            
            verification_code = input("Enter the verification code: ")
            
            flow.fetch_token(code=verification_code)
            creds = flow.credentials
            
        # Save the credentials for the next run using pickle, just like your other script
        with open(TOKEN_FILE, 'wb') as token:
            pickle.dump(creds, token)
        print(f"-> Token saved to '{TOKEN_FILE}' for future use.")
            
    return creds

# --- MAIN BOT LOOP ---

def main():
    """Main function to run the bot loop."""
    print("--- Eligibility Bot (AI Full Suite v2.3.0) Starting Up ---")

    print("-> Authenticating with Google Services...")
    creds = get_oauth_credentials(SCOPES)
    sheets_client = gspread.authorize(creds)
    drive_service = build('drive', 'v3', credentials=creds)
    sheet = sheets_client.open_by_key(SPREADSHEET_ID).worksheet(SHEET_NAME)
    print("-> Google Sheets & Drive authentication successful.")
    # Load payer cache once at startup
    _load_payer_cache()
    # Load form-fill cache once at startup
    _load_form_fill_cache()

    with sync_playwright() as p:
        context = None
        page = None
        browser = None
        # 1) Primary: Persistent profile
        try:
            if CHROME_PROFILE_DIR:
                os.makedirs(CHROME_PROFILE_DIR, exist_ok=True)
                print(f"-> Launching persistent context with profile dir: {CHROME_PROFILE_DIR}")
                context = p.chromium.launch_persistent_context(
                    CHROME_PROFILE_DIR,
                    headless=HEADLESS,
                    slow_mo=50
                )
            else:
                browser = p.chromium.launch(headless=HEADLESS, slow_mo=50)
                context = browser.new_context()
            page = context.new_page()
        except Exception as e:
            raise RuntimeError(f"Failed to start browser context: {e}")

        # Check login via profile
        if _is_logged_in(page):
            print("-> Session valid via persistent profile.")
        else:
            print("-> Persistent profile not authenticated. Trying cookies fallback...")
            loaded = _load_cookies_if_exists(context, COOKIES_FILE)
            if loaded and _is_logged_in(page):
                print("-> Session valid via cookies.")
            else:
                # 3) Automated login with manual OTP, then save cookies
                print("-> Performing manual login to Trizetto (OTP required)...")
                page.goto("https://mytools.gatewayedi.com/LogOn")
                page.fill('input[name="UserName"]', TRIZETTO_USERNAME)
                page.fill('input[type="password"]', TRIZETTO_PASSWORD)
                page.click('input[type="submit"]')
                print("   - Handling OTP step...")
                page.locator(f'text={OTP_EMAIL_ADDRESS_TEXT}').click()
                otp_code = input(">>> Please check your email for the OTP and enter it here: ")
                page.locator("#AuthCode").press_sequentially(otp_code, delay=100)
                page.locator("#btnVerify").click()
                page.wait_for_url("**/default.aspx**", timeout=30000)
                if not _is_logged_in(page):
                    raise RuntimeError("Login appeared to succeed but home page check failed.")
                print("-> LOGIN SUCCESSFUL! Saving cookies...")
                _save_cookies(context, COOKIES_FILE)

        # Ensure page has been initialized correctly
        if page is None:
            raise RuntimeError("Playwright page failed to initialize.")

        global PROCESSED_COUNT, OK_COUNT, FAIL_COUNT, RETRIED_COUNT
        while True:
            try:
                print(f"\n--- Checking for new records... ({time.ctime()}) ---")
                all_rows = sheet.get_all_values()

                row_to_process, row_index_to_process = (None, -1)
                for i, row in enumerate(all_rows[1:], start=2):
                    if len(row) >= 6 and all(str(item).strip() for item in row[:6]) and (len(row) < 7 or not str(row[6]).strip()):
                        row_to_process, row_index_to_process = row, i
                        break

                if row_to_process:
                    current_payer = row_to_process[4].strip()
                    print(f"-> Found record in row {row_index_to_process} for Payer: '{current_payer}'")

                    sheet.update_cell(row_index_to_process, 7, "Processing...")

                    # Set a per-row deadline and pass to payer selection
                    deadline_ts = time.time() + OPERATION_TIMEOUT_SECONDS if OPERATION_TIMEOUT_SECONDS > 0 else None
                    select_payer_with_ai(page, current_payer, deadline_ts)

                    patient_data = {
                        "dos": row_to_process[0], "first_name": row_to_process[1],
                        "last_name": row_to_process[2], "dob": row_to_process[3],
                        "payer_name": current_payer, "member_id": row_to_process[5]
                    }

                    PROCESSED_COUNT += 1
                    results = process_patient(page, drive_service, patient_data)

                    print(f"-> Writing results back to row {row_index_to_process}...")
                    # Existing columns
                    sheet.update_cell(row_index_to_process, 7, results.get("status", "Error"))
                    sheet.update_cell(row_index_to_process, 8, results.get("policy_begin", ""))
                    sheet.update_cell(row_index_to_process, 9, results.get("policy_end", ""))
                    sheet.update_cell(row_index_to_process, 10, results.get("pdf_link", "Upload Failed"))
                    # NEW: Additional benefit detail columns
                    sheet.update_cell(row_index_to_process, 11, results.get("copay", "Not Found"))
                    sheet.update_cell(row_index_to_process, 12, results.get("deductible", "Not Found"))
                    sheet.update_cell(row_index_to_process, 13, results.get("coinsurance", "Not Found"))
                    sheet.update_cell(row_index_to_process, 14, results.get("out_of_pocket", "Not Found"))
                    status = results.get("status", "Error")
                    if status.lower().startswith("error"):
                        FAIL_COUNT += 1
                    else:
                        OK_COUNT += 1
                    print("-> Sheet updated with comprehensive details and links.")

                    # After finishing the row updates, save a PDF of the current report page for audit/debug and update link if needed
                    try:
                        patient_name_str = f"{patient_data['last_name']}_{patient_data['first_name']}"
                        pdf_path = save_current_page_pdf(page, SCRIPT_DIR, patient_name_str)
                        if pdf_path:
                            print(f"-> Saved eligibility report PDF: {pdf_path}")
                            link = upload_file_to_drive(drive_service, DRIVE_FOLDER_ID, pdf_path, 'application/pdf')
                            try:
                                sheet.update_cell(row_index_to_process, 10, link)
                            except Exception:
                                pass
                        else:
                            print("-!- Failed to save eligibility report PDF for this row.")
                    except Exception:
                        print("-!- Exception occurred while saving eligibility report PDF.")

                else:
                    if RUN_ONCE:
                        print("-> No new records found. RUN_ONCE is enabled; exiting.")
                        break
                    print(f"-> No new records found. Waiting {CHECK_INTERVAL_SECONDS} seconds...")
                    time.sleep(CHECK_INTERVAL_SECONDS)

            except gspread.exceptions.APIError as e:
                print(f"-!- GOOGLE SHEETS API ERROR: {e}. Waiting 5 minutes...")
                time.sleep(300)
            except Exception as e:
                print(f"-!- AN UNEXPECTED ERROR IN THE MAIN LOOP: {e}")
                print("-!- Resetting state. Will re-navigate on next record. Waiting 60 seconds...")
                time.sleep(60)

        # End-of-run summary
        logger.info(f"Summary: processed={PROCESSED_COUNT} ok={OK_COUNT} fail={FAIL_COUNT} retried={RETRIED_COUNT}")
        # Optionally keep the browser open for review
        if KEEP_BROWSER_OPEN:
            try:
                input("\nPress Enter to close the browser and exit...")
            except Exception:
                pass

if __name__ == "__main__":
    main()