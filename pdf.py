import os
import time
from typing import Optional

# NOTE: This module provides a helper to save the currently open eligibility report page to PDF.
# It uses Playwright's Chromium CDP to print the page. This file can be imported and called by bot.py
# after completing a row, without changing the main parsing logic yet.


def sanitize_filename(name: str) -> str:
    return "".join(c for c in name if c.isalnum() or c in ("_", "-", "."))[:128]


def save_current_page_pdf(page, base_dir: str, patient_name: str, subdir: str = "logs/PDFs") -> Optional[str]:
    """
    Save the current page as a PDF under base_dir/subdir/<patient_name>.pdf

    - page: Playwright Page (Chromium)
    - base_dir: project base directory (e.g., SCRIPT_DIR)
    - patient_name: suggested filename (e.g., Last_First)
    - subdir: relative directory for PDFs

    Returns the absolute file path on success; None on failure.
    """
    try:
        # Ensure directory exists
        pdf_dir = os.path.join(base_dir, subdir)
        os.makedirs(pdf_dir, exist_ok=True)

        # Avoid clicking any print button that could open a blocking dialog; use CDP printToPDF directly
        try:
            page.emulate_media(media="print")
        except Exception:
            pass

        # Use CDP session for printToPDF (Chromium only)
        client = page.context.new_cdp_session(page)
        # Prefer A4 portrait, scale down slightly to fit; print backgrounds true for styling
        # Keep options conservative to avoid large files and hangs
        result = client.send('Page.printToPDF', {
            'printBackground': True,
            'preferCSSPageSize': True,
            'scale': 0.95,
            'landscape': False,
            'transferMode': 'ReturnAsBase64'
        })

        pdf_base64 = result.get('data')
        if not pdf_base64:
            return None

        safe_name = sanitize_filename(patient_name or f"report_{time.strftime('%Y%m%d-%H%M%S')}")
        pdf_path = os.path.join(pdf_dir, f"{safe_name}.pdf")
        with open(pdf_path, 'wb') as f:
            f.write(__import__('base64').b64decode(pdf_base64))

        return pdf_path
    except Exception:
        return None
