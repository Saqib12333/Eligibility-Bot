# Eligibility-Bot

Automates real-time insurance eligibility checks in TriZetto, extracts structured benefit data with Gemini, and logs results to Google Sheets with screenshots uploaded to Google Drive.

## Features
- Headless Playwright browser automation
- AI-assisted form filling, payer selection, and report parsing (Gemini)
- Google Sheets updates and Drive uploads
- Resilient retries and session persistence

## Prerequisites
- Windows with PowerShell
- Python 3.10+
- Google Cloud project with OAuth Client (Desktop app)
- Access to TriZetto portal
- Gemini API keys

## Quick start
1) Clone and create a virtual environment
2) Copy .env.example to .env and fill required values
3) Place your Google OAuth `client_secret.json` in the project root
4) Install dependencies and run

### Setup
- Create `.env` from `.env.example` and fill all required values.
- Ensure `client_secret.json` (OAuth Desktop Client) is present.

### First run (manual OAuth)
On first run, you’ll be prompted to visit a URL and paste back a verification code. Token is stored in `token.json` (ignored by Git).

## Run
- Default headless mode can be toggled via `HEADLESS=false` in `.env`.
- The bot polls Google Sheets for new records and writes back results.

## Troubleshooting
- If Google APIs fail with auth errors, delete `token.json` and re-run to re-authorize.
- If TriZetto session expires, the bot will re-login and update `login_state.json`.

## Security notes
- Never commit `.env`, credentials, tokens, or session files. `.gitignore` is configured accordingly.
- Rotate API keys and credentials regularly.

## License
Proprietary. Internal use only unless otherwise stated.

## Roadmap

### Phase A (Reliability, Observability)
1. Cache payer selection plans per payer name to reduce AI calls and variability.
2. Add waits before screenshots: `page.wait_for_load_state('networkidle')` and a short pause after expand.
3. Per-row operation timeout; fail fast and write a clear error to the sheet.
4. Switch to Python logging (console + rotating file), INFO by default, DEBUG via env.
5. Log compact result summaries and classify errors (AI, network, page, sheet).
6. Emit simple metrics counters (processed/ok/fail/retried) at the end of each run.

### Phase B (Performance, Maintainability)
1. Minimize AI context: pre-trim report HTML and payer list before prompting.
2. Reuse browser/context across rows (already in place); consider preloading pages after login.
3. Split codebase into modules: `ai.py`, `sheets_drive.py`, `trizetto.py`, `main.py`.

### Phase C (Artifacts, Safety, UX)
1. Save raw report HTML and upload alongside screenshot.
2. Timestamp artifact filenames and include row index to avoid collisions.
3. Add print-to-PDF export and upload as a stable artifact.
4. Add “Processing…” watchdog (e.g., mark stale > 30 mins as error with reason).
5. Graceful shutdown (handle Ctrl+C) to flush logs and release resources.
