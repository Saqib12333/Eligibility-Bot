# Eligibility-Bot

Automates real-time insurance eligibility checks in TriZetto, extracts structured benefit data with Gemini, and logs results to Google Sheets with PDFs uploaded to Google Drive.

## Features
- Playwright browser automation (headless or UI)
- AI-assisted form filling, payer selection, and report parsing (Gemini)
- Hardened AI parsing: sanitized HTML input, strict JSON prompt, robust JSON extractor, and diagnostics on failure
- Persistent login: Chromium profile directory + cookies fallback; OTP handled on-demand
- Google Sheets updates and Drive uploads (PDFs)
- PDF export via Chromium CDP and upload to Drive
- Saves raw and sanitized HTML, and AI JSON/TXT artifacts per patient
- Logging with rotating log file and end-of-run metrics (processed/ok/fail/retried)

## Prerequisites
- Windows with PowerShell
- Python 3.10+
- Google Cloud project with OAuth Client (Desktop app)
- Access to TriZetto portal
- Gemini API keys (1–3 keys supported with rotation)

## Quick start
1) Clone and create a virtual environment
2) Copy `.env.example` to `.env` and fill required values
3) Place your Google OAuth `client_secret.json` in the project root
4) Install dependencies and run

### Setup
- Hardened AI parsing: targeted HTML sanitization (keeps status, plan dates, and key benefits tables), strict JSON prompt with structured-output, robust JSON extractor, and diagnostics on failure
- Gemini API keys (1–3 keys supported with per-run start rotation)
- Optional: put existing TriZetto cookies into `Trizetto_cookies.pkl` to avoid OTP on first run.

	 - Input HTML is targeted-sanitized (status h1, BasicProfile plan dates, and benefits tables for Co-Payment, Co-Insurance, Deductible, Out-of-Pocket; noise removed; whitespace collapsed; truncated safely).
	 - Uses structured-output first; falls back to text JSON prompt if needed. A robust extractor parses JSON if the model adds extra text.

- Per-row AI token usage summary printed to console and logs (prompt/candidates/total).
- One-shot mode: set `RUN_ONCE=true` to process the next pending row and exit.
	- `GEMINI_API_KEY_1`, `GEMINI_API_KEY_2`, `GEMINI_API_KEY_3` (rotation with per-run start index)
## Configuration (env vars)
- Sheets/Drive: `SPREADSHEET_ID`, `SHEET_NAME`, `DRIVE_FOLDER_ID` (optional `PDF_DRIVE_FOLDER_ID`)
- TriZetto: `TRIZETTO_USERNAME`, `TRIZETTO_PASSWORD`, `OTP_EMAIL_ADDRESS_TEXT`
- AI: `GEMINI_API_KEY_1`, `GEMINI_API_KEY_2`, `GEMINI_API_KEY_3` (rotation between runs)
- Behavior: `HEADLESS`, `RUN_ONCE`, `OPERATION_TIMEOUT_SECONDS`, `LOG_LEVEL`
- Paths/cache: `CHROME_PROFILE_DIR`, `COOKIES_FILE`, `PAYER_CACHE_FILE`
- HTML size cap: `REPORT_HTML_MAX_CHARS` to keep token usage reasonable while preserving Deductible/Out-of-Pocket sections

Note: Per-run key rotation helps with 429s; the bot honors server-provided retry delays and may fall back to a lighter model when necessary.

## What’s new (v2.3.0)
- PDFs replace screenshots: export via Chromium CDP, upload to Drive, and write PDF link to the sheet
- OOP-aware sanitizer: preserves Out-of-Pocket variants like "Out of Pocket (Stop Loss)", escapes labels in regex, and injects OOP context near the top if not structurally found
- Artifacts per patient (First_Last):
	- AI outputs: `logs/AI Response/First_Last.json` and `.txt` (on failure: `First_Last.fail.txt`)
	- Sanitized HTML: `logs/Sanitized HTMLs/First_Last.html`
	- Raw HTML: `logs/Raw HTMLs/First_Last.html`
- Config: `REPORT_HTML_MAX_CHARS` to cap sanitized HTML length
- Minor reliability: strict JSON prompting + robust JSON extractor; clearer diagnostics

## How it works
1) Persistent session: launches Playwright with a persistent Chromium profile; falls back to cookies; otherwise performs login with manual OTP and saves cookies.
2) Finds next pending row in Google Sheet and selects the payer (AI-assisted, with a cached plan per payer).
3) AI generates a precise form-fill plan (CSS selectors + values); bot fills and submits.
4) Expands report UI, captures HTML, and asks AI for a single JSON object.
	 - Input HTML is sanitized (scripts/styles/comments removed, whitespace collapsed, truncated to safe length).
	 - Prompt enforces strict JSON only; a robust extractor parses the JSON object from the response.
	 - On repeated failure, diagnostics are written under `logs/AI Response/First_Last.fail.txt`; raw and sanitized HTMLs are saved under `logs/Raw HTMLs/` and `logs/Sanitized HTMLs/`.
5) Exports a PDF of the report and uploads to Google Drive; updates the sheet with status and summaries.

## Logging, diagnostics, artifacts
- Logs: `logs/eligibility-bot.log` (rotating file) and console output.
- AI outputs: `logs/AI Response/First_Last.json` and `.txt`; on failure: `logs/AI Response/First_Last.fail.txt`.
- HTMLs: `logs/Sanitized HTMLs/First_Last.html` and `logs/Raw HTMLs/First_Last.html`.
- PDFs: `logs/PDFs/Last_First.pdf` (also uploaded to Drive).
- End-of-run summary: processed/ok/fail/retried counters in logs.

## Troubleshooting
- Google auth errors: delete `token.json` and re-run to re-authorize.
- TriZetto session issues: the bot will try profile → cookies → login with OTP; ensure `OTP_EMAIL_ADDRESS_TEXT` matches your email option.
- AI parsing errors: check diagnostics in `logs/` and consider increasing `OPERATION_TIMEOUT_SECONDS` if pages are slow.

## Security notes
- Never commit `.env`, credentials, tokens, cookies, logs, or session files. `.gitignore` is configured to ignore these.
- Rotate API keys and credentials regularly.

## License
Proprietary. Internal use only unless otherwise stated.

## Roadmap

### Phase A (Reliability, Observability)
Status: in progress; major items implemented
1. Cache payer selection plans per payer name to reduce AI calls and variability. (Done)
2. Waits before screenshots and after expand; network-idle waits. (Done)
3. Per-row operation timeout with clear sheet errors. (Done)
4. Logging (console + rotating file), INFO default, DEBUG via env. (Done)
5. Compact result summaries and simple counters at run end. (Done)
6. Harden AI parsing with sanitized inputs, strict prompts, JSON extractor, and diagnostics. (Done)

### Phase B (Performance, Maintainability)
1. Minimize AI context (ongoing).
2. Reuse browser/context across rows; consider preloading pages.
3. Split into modules: `ai.py`, `sheets_drive.py`, `trizetto.py`, `main.py`.

### Phase C (Artifacts, Safety, UX)
1. Save raw report HTML and upload alongside screenshot.
2. Timestamp artifact filenames and include row index to avoid collisions.
3. Add print-to-PDF export and upload as a stable artifact.
4. Add “Processing…” watchdog for stale rows and graceful shutdown to flush logs.

---

## Installation
1) Ensure Python 3.10+ is installed
2) Create and activate a virtual environment
3) Install dependencies from `requirements.txt`

## Environment setup
- Copy `.env.example` to `.env` and fill all required values (see Configuration above).
- Place `client_secret.json` (OAuth Desktop Client) in the project root.

## How to run
- Default:
	- Set `HEADLESS=false` in `.env` if you want to see the browser.
	- Run the bot; it will poll for new rows and process them.
- One-shot:
	- Set `RUN_ONCE=true` to process only the next pending row.

## Google Sheet format
Inputs (A–F): DOS, First Name, Last Name, DOB, Payer Name, Member/Subscriber ID.
Outputs (G–N): Status, Policy Begin, Policy End, PDF Link, Copay, Deductible, Coinsurance, Out of Pocket.

## FAQs
- Q: I still get OTP prompts every run.
	- A: Ensure `CHROME_PROFILE_DIR` points to a persistent folder and `COOKIES_FILE` exists/updates after manual OTP. Subsequent runs should use those.
- Q: Why do I see “Failed to parse report”? 
	- A: Check `logs/AI Response/First_Last.fail.txt` and compare `logs/Raw HTMLs/First_Last.html` vs `logs/Sanitized HTMLs/First_Last.html`. The model may have produced non-JSON or truncated output; sanitized inputs and the extractor help, and diagnostics will show exact content.
- Q: Where are PDFs stored?
	- A: Locally under `logs/PDFs/` (named `Last_First.pdf`) and uploaded to the Google Drive folder set by `DRIVE_FOLDER_ID`.

## Support
Open an issue internally or provide failing diagnostics files to maintainers for prompt fixes.
