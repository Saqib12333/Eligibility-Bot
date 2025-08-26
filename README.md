# Eligibility-Bot

Automates real-time insurance eligibility checks in TriZetto, extracts structured benefit data with Gemini, and logs results to Google Sheets with screenshots uploaded to Google Drive.

## Features
- Playwright browser automation (headless or UI)
- AI-assisted form filling, payer selection, and report parsing (Gemini)
- Hardened AI parsing: sanitized HTML input, strict JSON prompt, robust JSON extractor, and diagnostics on failure
- Persistent login: Chromium profile directory + cookies fallback; OTP handled on-demand
- Google Sheets updates and Drive uploads
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
- Create `.env` from `.env.example` and fill all required values.
- Ensure `client_secret.json` (OAuth Desktop Client) is present.
- Optional: put existing TriZetto cookies into `Trizetto_cookies.pkl` to avoid OTP on first run.

### First run (manual OAuth)
On first run, you’ll be prompted to visit a URL and paste back a verification code. Token is stored in `token.json` (ignored by Git).

## Run
- HEADLESS mode: set `HEADLESS=false` in `.env` for visible runs.
- One-shot mode: set `RUN_ONCE=true` to process the next pending row and exit.
- The bot polls Google Sheets for new records and writes back results, including a screenshot link.

## Configuration (env vars)
- Google/Sheets/Drive
	- `SPREADSHEET_ID`, `SHEET_NAME`, `DRIVE_FOLDER_ID`, optional `PDF_DRIVE_FOLDER_ID`
	- OAuth files: `client_secret.json` in project root; token persisted to `token.json`
- TriZetto login
	- `TRIZETTO_USERNAME`, `TRIZETTO_PASSWORD`, `OTP_EMAIL_ADDRESS_TEXT`
	- Persistent profile dir: `CHROME_PROFILE_DIR` (default `./chrome-profile`)
	- Cookies file: `COOKIES_FILE` (default `./Trizetto_cookies.pkl`)
- AI (Gemini)
	- `GEMINI_API_KEY_1`, `GEMINI_API_KEY_2`, `GEMINI_API_KEY_3` (rotation)
- Behavior and reliability
	- `HEADLESS` (true/false), `RUN_ONCE` (true/false)
	- `OPERATION_TIMEOUT_SECONDS` (per-row timeout)
	- `LOG_LEVEL` (INFO by default)
	- `PAYER_CACHE_FILE` (default `./payer_cache.json`)

## How it works
1) Persistent session: launches Playwright with a persistent Chromium profile; falls back to cookies; otherwise performs login with manual OTP and saves cookies.
2) Finds next pending row in Google Sheet and selects the payer (AI-assisted, with a cached plan per payer).
3) AI generates a precise form-fill plan (CSS selectors + values); bot fills and submits.
4) Expands report UI, captures HTML, and asks AI for a single JSON object.
	 - Input HTML is sanitized (scripts/styles/comments removed, whitespace collapsed, truncated to safe length).
	 - Prompt enforces strict JSON only; a robust extractor parses the JSON object from the response.
	 - On repeated failure, diagnostics are written to `logs/ai_report_fail_*.json` and `logs/report_html_*.html`.
5) Takes an element screenshot (falls back to full page) and uploads to Google Drive; updates the sheet with status and summaries.

## Logging, diagnostics, artifacts
- Logs: `logs/eligibility-bot.log` (rotating file) and console output.
- Diagnostics (on AI parse failure): `logs/ai_report_fail_*.json`, `logs/report_html_*.html`.
- Screenshots: `Screenshots/` and uploaded to Drive.
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
Outputs (G–N): Status, Policy Begin, Policy End, Screenshot Link, Copay, Deductible, Coinsurance, Out of Pocket.

## FAQs
- Q: I still get OTP prompts every run.
	- A: Ensure `CHROME_PROFILE_DIR` points to a persistent folder and `COOKIES_FILE` exists/updates after manual OTP. Subsequent runs should use those.
- Q: Why do I see “Failed to parse report”? 
	- A: Check `logs/ai_report_fail_*.json` and `logs/report_html_*.html`. The model may have produced non-JSON or truncated output; sanitized inputs and extractor help, but diagnostics will show exact content.
- Q: Where are screenshots stored?
	- A: Locally under `Screenshots/` and uploaded to Google Drive folder set by `DRIVE_FOLDER_ID`.

## Support
Open an issue internally or provide failing diagnostics files to maintainers for prompt fixes.
