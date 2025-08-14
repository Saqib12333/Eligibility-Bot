# bot.py
# Version 5.3 - Final Corrected Version with Robust AI Payer Selection and Error Handling.

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

# Create an infinite cycle of the valid API keys
api_key_cycler = itertools.cycle(VALID_API_KEYS)

def get_next_api_key():
    """Returns the next API key from the rotation."""
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

# --- File/Path Configuration ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CLIENT_SECRETS_FILE = os.path.join(SCRIPT_DIR, 'client_secret.json') # Use OAuth secrets
TOKEN_FILE = os.path.join(SCRIPT_DIR, 'token.json') # Stores user's access token
STATE_FILE = os.path.join(SCRIPT_DIR, "login_state.json")
SCREENSHOT_DIR = os.path.join(SCRIPT_DIR, "Screenshots")

# --- AI AND AUTOMATION LOGIC ---

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
    You are an expert data structuring engine for U.S. healthcare insurance reports. Your task is to transform the provided HTML into a single, comprehensive JSON object.

    **CRITICAL INSTRUCTIONS:**
    1.  **Single JSON Output:** Your entire output MUST be a single, valid JSON object. Do not produce a truncated or incomplete response. Do not add any text before or after the JSON object.
    2.  **Required JSON Schema:** The JSON object MUST have the following top-level keys: `status`, `policy_begin`, `policy_end`, `summaries`, and `tables`.
        a.  `status`: Find the patient's coverage status. The value must be specific, like "Active Coverage" or "Inactive Coverage". **Do not shorten this to just "Active"**.
        b.  `policy_begin`, `policy_end`: Extract these from the patient information section.
        c.  `summaries`: This must be an object with four keys: `copay`, `deductible`, `coinsurance`, `out_of_pocket`. The value for each should be a single string summarizing all relevant benefits, separated by a newline character (`\\n`). Be concise.
        d.  `tables`: This must be an object where keys are section names (e.g., "Co-Insurance"). The value for each section key must be a list of row objects. **CRITICAL:** Create a **separate JSON object for every single benefit line item**. Do not group multiple service types into a single long string.
    """
}

def make_ai_call_with_retry(chat_session, prompt_text, max_retries=3):
    """
    Sends a message to the AI chat session with an automatic retry mechanism
    for 500-level server errors.
    """
    for attempt in range(max_retries):
        try:
            response = chat_session.send_message(prompt_text)
            return response
        except Exception as e:
            # Check if the error is a 500-level server error
            if "500" in str(e) or "internal error" in str(e).lower():
                print(f"   -!- AI server error (500), attempt {attempt + 1} of {max_retries}. Retrying in 5 seconds...")
                time.sleep(5)
                continue # Go to the next attempt
            else:
                # If it's a different error, raise it immediately
                raise e
    # If all retries fail, raise the last exception
    raise Exception(f"AI call failed after {max_retries} retries.")

def upload_file_to_drive(drive_service, folder_id, file_path, mimetype):
    """Uploads a file to a specific Google Drive folder and returns its shareable link."""
    try:
        file_name = os.path.basename(file_path)
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
        api_key = get_next_api_key()
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-2.5-flash')
        
        chat = model.start_chat()
        chat.send_message(SYSTEM_PROMPTS['form_filling'])
        
        data_prompt = f"**Patient Data:**\n```json\n{json.dumps(patient_data, indent=2)}\n```\n\n**Form HTML:**\n```html\n{page_html}\n```"
        # --- MODIFIED LINE: Using the retry function ---
        response = make_ai_call_with_retry(chat, data_prompt)

        cleaned_text = response.text.strip().replace("```json", "").replace("```", "").strip()
        plan = json.loads(cleaned_text)
        print("   - AI form-fill plan generated successfully.")
        return plan
    except Exception as e:
        print(f"   -!- CRITICAL: AI failed to generate a valid form-fill plan: {e}")
        return []

def get_all_report_data(html_content: str) -> dict:
    """Uses a Gemini chat session to get all structured data from the report."""
    print("   - Asking AI to generate all report data (Model: gemini-2.5-pro)...")
    try:
        api_key = get_next_api_key()
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-2.5-pro')
        
        chat = model.start_chat()
        chat.send_message(SYSTEM_PROMPTS['report_generation'])
        
        # --- MODIFIED LINE: Using the retry function ---
        response = make_ai_call_with_retry(chat, f"**HTML Content:**\n```html\n{html_content}\n```")

        raw_text = response.text
        start = raw_text.find('{')
        end = raw_text.rfind('}')
        if start != -1 and end != -1:
            cleaned_text = raw_text[start:end+1]
        else:
            cleaned_text = raw_text.strip().replace("```json", "").replace("```", "").strip()

        print("   - AI successfully generated consolidated report data.")
        return json.loads(cleaned_text)
    except Exception as e:
        print(f"   -!- AI failed to generate consolidated JSON: {e}")
        return {"Error": f"Failed to parse report: {e}"}

def select_payer_with_ai(page: Page, payer_name: str):
    """Uses a Gemini chat session to find and select the correct payer."""
    print("   - Starting AI-powered payer selection (Model: gemini-2.5-flash)...")
    try:
        page.goto("https://mytools.gatewayedi.com/ManagePatients/RealTimeEligibility/Index", wait_until="domcontentloaded")
        payer_list_container = page.locator("#InsurerAccordion")
        payer_list_container.wait_for(state="visible", timeout=30000)
        list_html = payer_list_container.inner_html()

        print("   - Asking AI to find the best payer match and create a plan...")
        api_key = get_next_api_key()
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-2.5-flash')
        
        chat = model.start_chat()
        chat.send_message(SYSTEM_PROMPTS['payer_selection'])

        data_prompt = f"The user wants to select: **\"{payer_name}\"**.\n\n**Payer List HTML:**\n```html\n{list_html}\n```"
        # --- MODIFIED LINE: Using the retry function ---
        response = make_ai_call_with_retry(chat, data_prompt)

        cleaned_text = response.text.strip().replace("```json", "").replace("```", "").strip()
        plan = json.loads(cleaned_text)

        print(f"   - AI Plan Received. Category: '{plan['category_text']}', Payer: '{plan['payer_text']}'")
        
        category_element = page.get_by_text(plan['category_text'], exact=True).first
        category_element.click()
        page.wait_for_timeout(1000)
        payer_list_container = page.locator(f"li[id='{plan['category_text']}'] ul.insurersDetail")
        payer_link = payer_list_container.get_by_text(plan['payer_text'], exact=True).first
        payer_link.click()
        page.wait_for_timeout(2000)
        print(f"   - AI Payer Selection for '{payer_name}' successful.")

    except Exception as e:
        print(f"   -!- CRITICAL: AI-driven payer selection failed: {e}")
        raise e
        
def process_patient(page: Page, drive_service, patient_data: dict) -> dict:
    """Handles all parsing, PDF generation, and uploads for a patient using a single AI call."""
    screenshot_link = "N/A"
    pdf_link = "N/A"
    try:
        # (Form filling logic is unchanged)
        form_html = page.locator("body").inner_html()
        fill_plan = generate_form_fill_plan(form_html, patient_data)
        if not fill_plan:
            raise ValueError("AI did not return a valid form-filling plan.")
        print("   - Executing AI form-filling plan...")
        for step in fill_plan:
            page.locator(step['selector']).fill(step['value'])
        print("   - Form filled according to AI plan.")
        page.locator("#btnUploadButton").click()
        page.wait_for_selector("#eligibilityRequestResponse, #EligibilityValidationErrors", timeout=90000)
        error_div = page.locator("#EligibilityValidationErrors")
        if error_div.is_visible() and len(error_div.inner_text().strip()) > 0:
            raise ValueError(f"Form submission error on page: {error_div.inner_text().strip()}")
        
        print("   - Expanding full benefit report...")
        page.locator("#tab_benefitinformation").click()
        page.wait_for_timeout(500)
        page.locator("#ExpandInfo > li > a").click()
        page.wait_for_timeout(1000)
        print("   - Report expanded.")

        report_container = page.locator("#eligibilityRequestResponse")
        report_html = report_container.inner_html()
        
        # --- NEW: SINGLE, CONSOLIDATED AI CALL ---
        all_data = get_all_report_data(report_html)
        if "Error" in all_data:
            raise ValueError(all_data["Error"])

        os.makedirs(SCREENSHOT_DIR, exist_ok=True)
        patient_name_str = f"{patient_data['last_name']}_{patient_data['first_name']}"
        



        # --- Upload Screenshot ---
        screenshot_path = os.path.join(SCREENSHOT_DIR, f"screenshot_{patient_name_str}.png")
        report_container.screenshot(path=screenshot_path)
        screenshot_link = upload_file_to_drive(drive_service, DRIVE_FOLDER_ID, screenshot_path, 'image/png')
        
        # Prepare results for Google Sheet update
        summaries = all_data.get("summaries", {})
        final_results = {
            "status": all_data.get("status", "Not Found"),
            "policy_begin": all_data.get("policy_begin", "Not Found"),
            "policy_end": all_data.get("policy_end", "Not Found"),
            "copay": summaries.get("copay", "Not Found"),
            "deductible": summaries.get("deductible", "Not Found"),
            "coinsurance": summaries.get("coinsurance", "Not Found"),
            "out_of_pocket": summaries.get("out_of_pocket", "Not Found"),
            "screenshot_link": screenshot_link,
            # "pdf_link": pdf_link
        }

        print(f"-> Check complete for {patient_data['first_name']}. Status: {final_results.get('status')}")
        return final_results

    except Exception as e:
        print(f"   -!- An error occurred during patient processing: {e}")
        os.makedirs(SCREENSHOT_DIR, exist_ok=True)
        screenshot_path = os.path.join(SCREENSHOT_DIR, f"ERROR_{patient_data.get('last_name', 'Unknown')}_{patient_data.get('first_name', 'Patient')}.png")
        if not page.is_closed():
            page.screenshot(path=screenshot_path)
        
        screenshot_link = upload_file_to_drive(drive_service, DRIVE_FOLDER_ID, screenshot_path, 'image/png')
        
        return {
            "status": f"Error: {type(e).__name__}", "policy_begin": f"{e}", "policy_end": "",
            "copay": "N/A", "deductible": "N/A", "coinsurance": "N/A", "out_of_pocket": "N/A",
            "screenshot_link": screenshot_link
        }


def get_oauth_credentials(scopes: list) -> Credentials:
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
    print("--- Eligibility Bot (AI Full Suite v5.5) Starting Up ---")

    print("-> Authenticating with Google Services...")
    creds = get_oauth_credentials(SCOPES)
    sheets_client = gspread.authorize(creds)
    drive_service = build('drive', 'v3', credentials=creds)
    sheet = sheets_client.open_by_key(SPREADSHEET_ID).worksheet(SHEET_NAME)
    print("-> Google Sheets & Drive authentication successful.")

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True, slow_mo=50)
        context, page = (None, None)

        if os.path.exists(STATE_FILE):
            try:
                print("-> Found saved session. Attempting to use...")
                context = browser.new_context(storage_state=STATE_FILE)
                page = context.new_page()
                # *** FIX: Removed Markdown from URL ***
                page.goto("https://mytools.gatewayedi.com/default.aspx", timeout=60000)
                page.locator("#NavCtrl_navHome").wait_for(timeout=15000)
                print("-> Session is valid. Login skipped.")
            except Exception as e:
                print(f"-> Session invalid: {e}. A new login is required.")
                if context: context.close()
                context = None

        if not context:
            print("-> Performing new login to Trizetto...")
            context = browser.new_context()
            page = context.new_page()
            # *** FIX: Removed Markdown from URL ***
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
            print("-> LOGIN SUCCESSFUL! Saving session...")
            context.storage_state(path=STATE_FILE)

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

                    select_payer_with_ai(page, current_payer)

                    patient_data = {
                        "dos": row_to_process[0], "first_name": row_to_process[1],
                        "last_name": row_to_process[2], "dob": row_to_process[3],
                        "payer_name": current_payer, "member_id": row_to_process[5]
                    }

                    results = process_patient(page, drive_service, patient_data)

                    print(f"-> Writing results back to row {row_index_to_process}...")
                    # Existing columns
                    sheet.update_cell(row_index_to_process, 7, results.get("status", "Error"))
                    sheet.update_cell(row_index_to_process, 8, results.get("policy_begin", ""))
                    sheet.update_cell(row_index_to_process, 9, results.get("policy_end", ""))
                    sheet.update_cell(row_index_to_process, 10, results.get("screenshot_link", "Upload Failed"))
                    # NEW: Additional benefit detail columns
                    sheet.update_cell(row_index_to_process, 11, results.get("copay", "Not Found"))
                    sheet.update_cell(row_index_to_process, 12, results.get("deductible", "Not Found"))
                    sheet.update_cell(row_index_to_process, 13, results.get("coinsurance", "Not Found"))
                    sheet.update_cell(row_index_to_process, 14, results.get("out_of_pocket", "Not Found"))
                    print("-> Sheet updated with comprehensive details and links.")

                else:
                    print(f"-> No new records found. Waiting {CHECK_INTERVAL_SECONDS} seconds...")
                    time.sleep(CHECK_INTERVAL_SECONDS)

            except gspread.exceptions.APIError as e:
                print(f"-!- GOOGLE SHEETS API ERROR: {e}. Waiting 5 minutes...")
                time.sleep(300)
            except Exception as e:
                print(f"-!- AN UNEXPECTED ERROR IN THE MAIN LOOP: {e}")
                print("-!- Resetting state. Will re-navigate on next record. Waiting 60 seconds...")
                time.sleep(60)

if __name__ == "__main__":
    main()