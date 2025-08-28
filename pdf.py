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

        # Try to trigger the print view to ensure print CSS applies if needed
        try:
            # If there is an explicit print button present, click it; otherwise rely on printToPDF
            btn = page.locator('#exportPrint')
            if btn and btn.count() > 0:
                try:
                    btn.first.scroll_into_view_if_needed()
                    page.wait_for_timeout(150)
                    btn.first.click(timeout=2000)
                    page.wait_for_timeout(500)
                except Exception:
                    pass
        except Exception:
            pass

        # Use CDP session for printToPDF (Chromium only)
        client = page.context.new_cdp_session(page)
        # Prefer A4 portrait, scale down slightly to fit; print backgrounds true for styling
        result = client.send('Page.printToPDF', {
            'printBackground': True,
            'preferCSSPageSize': True,
            'scale': 0.95,
            'landscape': False,
        })

        pdf_base64 = result.get('data')
        if not pdf_base64:
            return None

        safe_name = sanitize_filename(patient_name or f"report_{time.strftime('%Y%m%d-%H%M%S')}")
        pdf_path = os.path.join(pdf_dir, f"{safe_name}.pdf")
        with open(pdf_path, 'wb') as f:
            f.write(bytes.fromhex(''))  # no-op to ensure file creation on some FS before write
            f.write(bytes.fromhex(''))
        # Write actual content
        with open(pdf_path, 'wb') as f:
            f.write(__import__('base64').b64decode(pdf_base64))

        return pdf_path
    except Exception:
        return None
