---
applyTo: '**'
---
---
applyTo: '**'
---
# Copilot instructions for Eligibility-Bot

Context for GitHub Copilot agents and human developers.

## Project overview
- Single-file Python bot (`bot.py`) using Playwright for browser automation.
- AI via `google-generativeai` (Gemini): form filling, payer selection, report parsing.
- Google APIs (Sheets & Drive) via OAuth user flow.

## Constraints and conventions
- All sensitive/config values must come from environment variables; see `.env.example`.
- Do not hardcode secrets in source code or docs.
- Use HEADLESS env to switch between UI and headless runs.
- Use small, focused functions; add retries for flaky I/O.

## Key integration points
- OAuth credentials file path from `CLIENT_SECRETS_FILE`.
- Token path from `TOKEN_FILE`.
- Session state from `STATE_FILE`.
- Output artifacts go to `SCREENSHOT_DIR` and uploaded to Drive `DRIVE_FOLDER_ID`.

## Editing guidelines for Copilot
- When adding new config, wire it via `.env` and document it in `.env.example`.
- Prefer explicit waits and robust selectors in Playwright.
- Avoid reflowing the entire file on small changes; keep diffs minimal.
- Add type hints where obvious, but keep pragmatic for dynamic libs.
 - Keep AI outputs strictly JSON for report parsing; target-sanitize report HTML inputs (keep status h1, plan begin/end, and key benefit tables; remove noise) to reduce token pressure.
 - On parse failure, write diagnostics (raw AI text + sanitized HTML) under `logs/` with timestamps.

## Testing tips
- Set `HEADLESS=false` during debugging.
- Use a test worksheet to avoid production data.
- Mock or stub AI calls if adding unit tests; keep API key rotation logic intact.

## Security and secrets
- `.gitignore` excludes credentials, tokens, `.env`, and generated artifacts.
- If secrets were committed, remove with git history rewriting before pushing.
- Rotate API keys and credentials regularly.
 - Also ignore `chrome-profile/`, `Trizetto_cookies.pkl`, payer cache, and logs.

## Roadmap for changes

### Phase A (Reliability, Observability)
- Cache payer selection per payer to avoid repeated AI calls and variability.
- Add waits before screenshots `page.wait_for_load_state('networkidle')` and a brief pause post-expand.
- Implement per-row operation timeout and fail fast with clear sheet errors.
- Switch to logging (console + rotating file), INFO default, DEBUG via env.
- Log compact result summaries and classify errors; expose simple counters at run end.
 - Harden AI parsing: sanitized HTML, strict JSON prompts, robust JSON extraction, diagnostics on failure.

### Phase B (Performance, Maintainability)
- Pre-trim AI prompt contexts (report HTML, payer list) before prompting.
- Keep reusing context; consider preloading eligibility page after login.
- Split into modules: `ai.py`, `sheets_drive.py`, `trizetto.py`, `main.py`.

### Phase C (Artifacts, Safety, UX)
- Save raw report HTML and upload with screenshots.
- Timestamp artifacts and include row indices.
- Add print-to-PDF export upload.
- Add a “Processing…” watchdog and graceful shutdown.

---

## Architecture and project structure

Minimal single-file implementation with clear responsibilities inside `bot.py`:
- Environment and config: .env values are loaded early; never hardcode secrets.
- Google OAuth: `get_oauth_credentials()` writes/reads `token.json`.
- Browser/session: Playwright persistent context using `CHROME_PROFILE_DIR`; cookies fallback via `COOKIES_FILE`.
- AI helpers:
	- Key rotation across `GEMINI_API_KEY_1..3`.
	- `_safe_response_text()` for robust text extraction.
	- `_sanitize_html_for_ai()` to reduce HTML noise/size.
	- `_extract_json_object()` to reliably parse a single JSON object from AI output.
- Payer selection cache: JSON file at `PAYER_CACHE_FILE`.
- Form-fill plan cache per payer: JSON file at `FORM_FILL_CACHE_FILE`.
- Processing loop: fetch next row → select payer → AI fill plan → submit → expand/report → AI parse → screenshot/upload → update sheet.
- Logging and metrics: rotating file in `logs/` + console; run summary at end.

Project files you should know:
- `bot.py` — main automation script
- `.env.example` — all supported configuration keys
- `requirements.txt` — Python deps (Playwright, google APIs, generativeai, gspread, dotenv)
- `.github/copilot-instructions.md` — this file
- `README.md` — user-facing instructions
- `.gitignore` — ignores credentials, tokens, profile, cookies, cache, logs, screenshots

## Runtime flow (high level)
1) Auth: Google OAuth using `client_secret.json` → stores `token.json`.
2) Browser: Launch persistent Chromium profile (or ephemeral + cookies) → verify session.
3) Sheets: Read next pending row with required 6 input columns and empty status.
4) Payer selection: Use cache or AI plan; robust waits/retries/scroll.
5) Form fill: AI produces selectors/values; fill; submit; wait for response.
6) Report parse: expand, sanitize HTML, prompt AI for strict JSON; parse with extractor.
7) Artifacts: screenshot element (fallback full-page); upload to Drive; write results back.
8) Loop or exit (RUN_ONCE).

## Environment variables (complete list)
- Google/Sheets/Drive: `SPREADSHEET_ID`, `SHEET_NAME`, `DRIVE_FOLDER_ID`, `PDF_DRIVE_FOLDER_ID` (optional)
- TriZetto: `TRIZETTO_USERNAME`, `TRIZETTO_PASSWORD`, `OTP_EMAIL_ADDRESS_TEXT`
- AI: `GEMINI_API_KEY_1`, `GEMINI_API_KEY_2`, `GEMINI_API_KEY_3` (per-run start rotation)
- Behavior: `HEADLESS`, `RUN_ONCE`, `CHECK_INTERVAL_SECONDS`, `OPERATION_TIMEOUT_SECONDS`, `LOG_LEVEL`
- Files/paths: `PAYER_CACHE_FILE`, `CHROME_PROFILE_DIR`, `COOKIES_FILE`

## Google Sheet schema
Inputs (columns A–F):
- A: Date of Service (DOS)
- B: First Name
- C: Last Name
- D: Date of Birth (DOB)
- E: Payer Name
- F: Member/Subscriber ID

Outputs (columns G–N):
- G: Status (e.g., Active Coverage)
- H: Policy Begin
- I: Policy End
- J: Screenshot Link (Drive)
- K: Copay (summary)
- L: Deductible (summary)
- M: Coinsurance (summary)
- N: Out of Pocket (summary)

Rows are considered “pending” when columns A–F are filled and column G is empty.

## AI prompt contracts
- Form filling (gemini-2.5-flash): returns JSON array of `{selector, value}` pairs for text inputs; ignore selects.
- Payer selection (gemini-2.5-flash): returns object `{"category_text", "payer_text"}`; exact text matches preferred.
- Report generation (gemini-2.5-pro): returns a single JSON object with keys:
	- `status` (e.g., "Active Coverage"), `policy_begin`, `policy_end`
	- `summaries` object: `copay`, `deductible`, `coinsurance`, `out_of_pocket`

Strict output rules: double quotes, no trailing commas/comments; if missing, use "Not Found".

## Selectors, waits, and retries
- Prefer IDs and stable attributes; verify visibility before clicking.
- Use `wait_for_load_state('networkidle')` before screenshots; small waits after expands.
- Payer selection: expand category with retries; wait for nested list; try exact link, then non-exact.
- Timeouts: gate long operations with `OPERATION_TIMEOUT_SECONDS` via `_check_timeout()`.

## Error handling and diagnostics
- Form errors: `#EligibilityValidationErrors` checked after submit.
- AI parse failures: retry once; on failure, write `logs/ai_report_fail_*.json` and `logs/report_html_*.html`.
- Screenshots: element-level then full-page fallback; only upload if file exists.
- Metrics: processed/ok/fail/retried counters logged at run end.

## Security and secrets (enforced)
- Do not commit `.env`, tokens (`token.json`), `client_secret.json`, cookies (`Trizetto_cookies.pkl`), persistent profile (`chrome-profile/`), payer cache, or logs.
- Update `.env.example` and docs when introducing new env keys.

## Testing and local runs
- Set `HEADLESS=false` to observe flows; use `RUN_ONCE=true` for targeted validation.
- Use a test worksheet and payer to avoid production data during development.
- If adding unit tests, stub AI calls but retain key rotation logic.

## Contribution notes
- Keep diffs minimal; avoid reformatting the entire file for small changes.
- Favor small, focused functions with pragmatic type hints.
- Document new behavior in README and Copilot instructions.
- Validate changes: lint/typecheck, one RUN_ONCE pass, and confirm artifacts/logs.
