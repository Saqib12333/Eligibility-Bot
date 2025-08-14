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
