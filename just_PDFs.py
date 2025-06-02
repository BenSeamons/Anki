import google.generativeai as genai
import os
#import tkinter as tk
#from tkinter import filedialog
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
import time

# --- Configuration ---
# IMPORTANT: It's recommended to use environment variables or a secure config file for API keys.
GOOGLE_API_KEY = "AIzaSyAt-Vk0Y_elEgH1SKTaB0ZcHp2xBsVONtg"  # Replace with your actual API key
genai.configure(api_key=GOOGLE_API_KEY)

OUTPUT_DIR = "practice_tests_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# If modifying these scopes, delete token.json
SCOPES = [
    'https://www.googleapis.com/auth/documents',
    'https://www.googleapis.com/auth/drive.file'
]

# Your fixed prompt — no need to ask user every time
FIXED_PROMPT = (
    "Based on this document, generate 25 NBME Style multiple-choice questions with 4 options each. "
    "Also generate an answer key with the correct answer indicated for each numbered question. "
    "Focus on any learning objectives listed in the lecture."
)


def authenticate_google_api():
    creds = None
    if os.path.exists('token.json'):
        creds = Credentials.from_authorized_user_file('token.json', SCOPES)
    if not creds or not creds.valid:
        # Make sure 'credentials.json' is in the same directory or provide the correct path
        flow = InstalledAppFlow.from_client_secrets_file(
            'Credentials.json', SCOPES)
        creds = flow.run_local_server(port=0)
        with open('token.json', 'w') as token:
            token.write(creds.to_json())
    return creds


def create_google_doc(service, title):
    doc_body = {'title': title}
    try:
        doc = service.documents().create(body=doc_body).execute()
        print(f"Created document with ID: {doc['documentId']}")
        return doc['documentId']
    except Exception as e:
        print(f"Error creating Google Doc: {e}")
        return None


def append_text_to_doc(service, document_id, text):
    requests = [
        {
            'insertText': {
                # The 'location' field has been removed to avoid conflict with 'endOfSegmentLocation'
                'endOfSegmentLocation': {'segmentId': ''},  # This tells Docs API to insert at the end of the document body
                'text': text + '\n\n'
            }
        }
    ]
    try:
        result = service.documents().batchUpdate(documentId=document_id, body={'requests': requests}).execute()
        # print(f"Successfully appended text to Google Doc ID: {document_id}") # Optional: success message
        return result
    except Exception as e:
        print(f"Error appending text to Google Doc: {e}")
        return None

def generate_practice_test(pdf_path, prompt_text, output_prefix, initial_delay=20, max_retries=5):
    """
    Generates practice test questions from a PDF using the Gemini API by directly uploading the PDF.

    Args:
        pdf_path (str): The file path to the PDF document.
        prompt_text (str): The prompt to send to the Gemini model.
        output_prefix (str): The prefix for the output text file.
        initial_delay (int): The initial delay in seconds for API retries.
        max_retries (int): The maximum number of retries for the API call.
    """
    # Use 'gemini-1.5-flash' for faster processing or 'gemini-1.5-pro' for potentially higher quality.
    # Both support file uploads.
    model = genai.GenerativeModel('gemini-1.5-flash')
    print(f"Processing PDF: {pdf_path}")

    uploaded_file_resource = None  # To store the File object returned by upload_file
    try:
        print(f"Uploading {os.path.basename(pdf_path)} to Google AI...")
        # MIME type is usually inferred from the file extension by the API.
        # You can explicitly set it with mime_type='application/pdf' if needed.
        uploaded_file_resource = genai.upload_file(path=pdf_path,
                                                   display_name=os.path.basename(pdf_path))
        print(f"Successfully uploaded '{uploaded_file_resource.display_name}' as '{uploaded_file_resource.name}'")

        # IMPORTANT: The content for generate_content should be a list.
        # The prompt text should be a simple string.
        # The uploaded file resource is passed directly.
        content_parts = [prompt_text, uploaded_file_resource]

        full_generated_text = []
        retries = 0
        current_delay = initial_delay

        while retries < max_retries:
            try:
                print(
                    f"Sending request for '{uploaded_file_resource.display_name}' to Gemini API (Attempt {retries + 1}/{max_retries})...")
                # Increased timeout for potentially large PDF processing
                request_options = genai.types.RequestOptions(timeout=600)  # 10 minutes timeout
                response = model.generate_content(content_parts, request_options=request_options)

                if response.candidates:
                    generated_text = response.text
                    print(f"Received response for '{uploaded_file_resource.display_name}'.")
                    full_generated_text.append(generated_text)
                    break  # Success
                else:
                    block_reason = response.prompt_feedback.block_reason if response.prompt_feedback else "Unknown"
                    block_message = response.prompt_feedback.block_reason_message if response.prompt_feedback else "No additional message."
                    print(
                        f"No candidates found in response for '{uploaded_file_resource.display_name}'. Block Reason: {block_reason}. Message: {block_message}. Retrying...")
                    time.sleep(current_delay)
                    current_delay *= 2
                    retries += 1

            except genai.types.BlockedPromptException as e:
                print(f"Prompt was blocked for '{uploaded_file_resource.display_name}': {e}")
                break
            except genai.types.StopCandidateException as e:
                print(
                    f"Content generation stopped due to safety reasons for '{uploaded_file_resource.display_name}': {e}")
                if response and response.text:  # Check if any partial text was generated
                    print("Partial text might be available. Using it.")
                    full_generated_text.append(response.text)
                break
            except genai.types.generation_types.InternalServerError as e:
                print(
                    f"Internal Server Error (500) from Gemini API for '{uploaded_file_resource.display_name}': {e}. Retrying in {current_delay} seconds...")
                time.sleep(current_delay)
                current_delay *= 2
                retries += 1
            except genai.types.generation_types.DeadlineExceeded as e:
                print(
                    f"Deadline Exceeded (e.g., 504) from Gemini API for '{uploaded_file_resource.display_name}': {e}. Retrying in {current_delay} seconds...")
                time.sleep(current_delay)
                current_delay *= 2
                retries += 1
            except genai.types.GoogleAPIError as e:  # Catching broader Google API errors
                error_message = str(e).lower()
                if "429" in error_message or "rate limit" in error_message or "resource has been exhausted" in error_message:
                    print(
                        f"API capacity/rate limit hit for '{uploaded_file_resource.display_name}'. Retrying in {current_delay} seconds...")
                    time.sleep(current_delay)
                    current_delay *= 2
                    retries += 1
                elif "503" in error_message or "service unavailable" in error_message:  # Service unavailable
                    print(
                        f"Service Unavailable (503) for '{uploaded_file_resource.display_name}'. Retrying in {current_delay} seconds...")
                    time.sleep(current_delay)
                    current_delay *= 2
                    retries += 1
                else:
                    print(f"Unhandled Google API Error processing '{uploaded_file_resource.display_name}': {e}")
                    break
            except Exception as e:
                print(f"General unexpected error processing '{uploaded_file_resource.display_name}': {e}")
                break

        if retries == max_retries:
            print(
                f"Failed to generate practice test for '{uploaded_file_resource.display_name}' after {max_retries} attempts.")

        if full_generated_text:
            output_filename = os.path.join(OUTPUT_DIR, f"{output_prefix}.txt")
            with open(output_filename, "w", encoding="utf-8") as f:
                f.write("\n".join(full_generated_text))
            print(f"Saved practice test for {output_prefix} to: {output_filename}\n")
        else:
            print(f"No content generated for {pdf_path}.")

    except genai.types.GoogleAPIError as e:  # Catch errors during file upload specifically
        print(f"Failed to upload or process file {pdf_path} with Google AI: {e}")
    except Exception as e:
        print(f"An unexpected error occurred with {pdf_path} before or during API interaction: {e}")
    finally:
        if uploaded_file_resource:
            try:
                print(f"Attempting to delete uploaded file: {uploaded_file_resource.name}...")
                genai.delete_file(name=uploaded_file_resource.name)
                print(f"Successfully deleted {uploaded_file_resource.name}.")
            except Exception as e:
                print(f"Error deleting uploaded file {uploaded_file_resource.name}: {e}")
                print("You may need to manually delete it from Google AI Studio > API > Files.")


def cleanup_txt_files(folder):
    txt_files = [f for f in os.listdir(folder) if f.endswith('.txt')]
    if not txt_files:
        print(f"No .txt files found in {folder} to delete.")
        return
    print(f"Cleaning up .txt files in {folder}...")
    for txt_file in txt_files:
        try:
            os.remove(os.path.join(folder, txt_file))
            print(f"Deleted {txt_file}")
        except Exception as e:
            print(f"Error deleting {txt_file}: {e}")


def main():
    root = tk.Tk()
    root.withdraw()

    print("Please select the folder containing your lecture PDFs.")
    folder_path = filedialog.askdirectory(title="Select folder with PDFs")
    if not folder_path:
        print("No folder selected. Exiting.")
        return

    pdf_files = [f for f in os.listdir(folder_path) if f.lower().endswith('.pdf')]
    if not pdf_files:
        print("No PDF files found in the selected folder. Exiting.")
        return

    print(f"Found PDF files: {pdf_files}")

    for pdf_file in pdf_files:
        pdf_path = os.path.join(folder_path, pdf_file)
        output_prefix = os.path.splitext(pdf_file)[0]  # filename without extension
        # Call the updated generate_practice_test function
        generate_practice_test(pdf_path, FIXED_PROMPT, output_prefix)

    print("\nAttempting to authenticate with Google for Docs API...")
    creds = authenticate_google_api()
    if not creds:
        print("Failed to authenticate Google API. Cannot upload to Google Docs.")
        # Decide if you want to exit or just skip the Docs part
        print("Cleanup of local TXT files will be skipped if Docs upload fails or is skipped.")
        return  # Exiting if auth fails

    print("Authentication successful. Building Google Docs service...")
    docs_service = build('docs', 'v1', credentials=creds)

    doc_title = 'Consolidated Practice Tests - NBME Style'
    print(f"Creating/Reusing Google Doc titled: '{doc_title}'")
    document_id = create_google_doc(docs_service, doc_title)

    if not document_id:
        print("Failed to create Google Document. Aborting upload to Docs.")
        print("Cleanup of local TXT files will be skipped.")
        return

    txt_folder = OUTPUT_DIR  # Use the configured OUTPUT_DIR
    txt_files_to_upload = [f for f in os.listdir(txt_folder) if f.endswith('.txt')]

    if not txt_files_to_upload:
        print(f"No .txt files found in {txt_folder} to upload to Google Docs.")
    else:
        print(f"Found .txt files to upload: {txt_files_to_upload}")
        for txt_file in txt_files_to_upload:
            path = os.path.join(txt_folder, txt_file)
            print(f"Reading content from {txt_file} for Google Docs upload...")
            with open(path, 'r', encoding='utf-8') as f:
                content = f.read()
            print(f"Appending '{txt_file}' to Google Doc ID: {document_id}...")
            # Add a title for each appended section
            title_for_section = f"--- Content from: {txt_file} ---\n\n"
            append_text_to_doc(docs_service, document_id, title_for_section + content)
        print("All TXT files uploaded to Google Docs!")

    # Clean up local TXT files after successful processing (and potential upload)
    cleanup_txt_files(OUTPUT_DIR)
    print("All done!")


#if __name__ == "__main__":
    #main()
def generate_practice_test_return_text(pdf_path, prompt_text=FIXED_PROMPT, output_prefix=None):
    if output_prefix is None:
        output_prefix = os.path.splitext(os.path.basename(pdf_path))[0]
    generate_practice_test(pdf_path, prompt_text, output_prefix)
    output_file = os.path.join(OUTPUT_DIR, f"{output_prefix}.txt")
    if os.path.exists(output_file):
        with open(output_file, 'r', encoding='utf-8') as f:
            return f.read()
    return ""
