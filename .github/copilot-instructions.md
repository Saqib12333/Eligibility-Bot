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

## Testing tips
- Set `HEADLESS=false` during debugging.
- Use a test worksheet to avoid production data.
- Mock or stub AI calls if adding unit tests; keep API key rotation logic intact.

## Security and secrets
- `.gitignore` excludes credentials, tokens, `.env`, and generated artifacts.
- If secrets were committed, remove with git history rewriting before pushing.

## Roadmap for changes

### Phase A (Reliability, Observability)
- Cache payer selection per payer to avoid repeated AI calls and variability.
- Add waits before screenshots `page.wait_for_load_state('networkidle')` and a brief pause post-expand.
- Implement per-row operation timeout and fail fast with clear sheet errors.
- Switch to logging (console + rotating file), INFO default, DEBUG via env.
- Log compact result summaries and classify errors; expose simple counters at run end.

### Phase B (Performance, Maintainability)
- Pre-trim AI prompt contexts (report HTML, payer list) before prompting.
- Keep reusing context; consider preloading eligibility page after login.
- Split into modules: `ai.py`, `sheets_drive.py`, `trizetto.py`, `main.py`.

### Phase C (Artifacts, Safety, UX)
- Save raw report HTML and upload with screenshots.
- Timestamp artifacts and include row indices.
- Add print-to-PDF export upload.
- Add a “Processing…” watchdog and graceful shutdown.
