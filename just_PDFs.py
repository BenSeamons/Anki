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

def generate_practice_test(pdf_path, prompt_text=FIXED_PROMPT, initial_delay=20, max_retries=5):
    model = genai.GenerativeModel('gemini-1.5-flash')
    uploaded_file_resource = None
    full_generated_text = []

    try:
        uploaded_file_resource = genai.upload_file(path=pdf_path, display_name=os.path.basename(pdf_path))
        content_parts = [prompt_text, uploaded_file_resource]

        retries = 0
        current_delay = initial_delay

        while retries < max_retries:
            try:
                request_options = genai.types.RequestOptions(timeout=600)  # 10 minutes timeout
                response = model.generate_content(content_parts, request_options=request_options)

                if response.candidates:
                    full_generated_text.append(response.text)
                    break
                else:
                    time.sleep(current_delay)
                    current_delay *= 2
                    retries += 1

            except genai.types.BlockedPromptException as e:
                break
            except genai.types.StopCandidateException as e:
                if response and response.text:
                    full_generated_text.append(response.text)
                break
            except (genai.types.generation_types.InternalServerError, genai.types.generation_types.DeadlineExceeded) as e:
                time.sleep(current_delay)
                current_delay *= 2
                retries += 1
            except genai.types.GoogleAPIError as e:
                err = str(e).lower()
                if any(x in err for x in ["429", "rate limit", "resource has been exhausted", "503", "service unavailable"]):
                    time.sleep(current_delay)
                    current_delay *= 2
                    retries += 1
                else:
                    break
            except Exception as e:
                break

        if retries == max_retries:
            print(f"Failed to generate practice test for {pdf_path} after {max_retries} attempts.")

        return "\n".join(full_generated_text) if full_generated_text else ""

    finally:
        if uploaded_file_resource:
            try:
                genai.delete_file(name=uploaded_file_resource.name)
            except Exception as e:
                print(f"Failed to delete uploaded file: {e}")


#def main():
    #root = tk.Tk()
    #root.withdraw()

   # print("Please select the folder containing your lecture PDFs.")
    #folder_path = filedialog.askdirectory(title="Select folder with PDFs")
    #if not folder_path:
     #   print("No folder selected. Exiting.")
      #  return

    #pdf_files = [f for f in os.listdir(folder_path) if f.lower().endswith('.pdf')]
    #if not pdf_files:
     #   print("No PDF files found in the selected folder. Exiting.")
      #  return

    #print(f"Found PDF files: {pdf_files}")

    #for pdf_file in pdf_files:
     #   pdf_path = os.path.join(folder_path, pdf_file)
      #  output_prefix = os.path.splitext(pdf_file)[0]  # filename without extension
       # # Call the updated generate_practice_test function
        #generate_practice_test(pdf_path, FIXED_PROMPT, output_prefix)

    #print("\nAttempting to authenticate with Google for Docs API...")
    #creds = authenticate_google_api()
    #if not creds:
     #   print("Failed to authenticate Google API. Cannot upload to Google Docs.")
      #  # Decide if you want to exit or just skip the Docs part
       # print("Cleanup of local TXT files will be skipped if Docs upload fails or is skipped.")
        #return  # Exiting if auth fails

    #print("Authentication successful. Building Google Docs service...")
    #docs_service = build('docs', 'v1', credentials=creds)

    #doc_title = 'Consolidated Practice Tests - NBME Style'
    #print(f"Creating/Reusing Google Doc titled: '{doc_title}'")
    #document_id = create_google_doc(docs_service, doc_title)

    #if not document_id:
     #   print("Failed to create Google Document. Aborting upload to Docs.")
      #  print("Cleanup of local TXT files will be skipped.")
       # return

    #txt_folder = OUTPUT_DIR  # Use the configured OUTPUT_DIR
    #txt_files_to_upload = [f for f in os.listdir(txt_folder) if f.endswith('.txt')]

    #if not txt_files_to_upload:
     #   print(f"No .txt files found in {txt_folder} to upload to Google Docs.")
    #else:
     #   print(f"Found .txt files to upload: {txt_files_to_upload}")
      #  for txt_file in txt_files_to_upload:
       #     path = os.path.join(txt_folder, txt_file)
        #    print(f"Reading content from {txt_file} for Google Docs upload...")
         #   with open(path, 'r', encoding='utf-8') as f:
          #      content = f.read()
           # print(f"Appending '{txt_file}' to Google Doc ID: {document_id}...")
            ## Add a title for each appended section
            #title_for_section = f"--- Content from: {txt_file} ---\n\n"
            #append_text_to_doc(docs_service, document_id, title_for_section + content)
        #print("All TXT files uploaded to Google Docs!")

    # Clean up local TXT files after successful processing (and potential upload)
    #cleanup_txt_files(OUTPUT_DIR)
    #print("All done!")


#if __name__ == "__main__":
    #main()
def generate_practice_test_return_text(pdf_path, prompt_text=FIXED_PROMPT,):
    return generate_practice_test(pdf_path, prompt_text)
