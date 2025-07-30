# email_fetcher.py

import os
import base64
from email import message_from_bytes
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from google.auth.transport.requests import Request

from spam_detector import load_model, predict_spam
from email_cleaner import delete_spam_emails, log_emails_for_review

SCOPES = ['https://www.googleapis.com/auth/gmail.modify']

def authenticate_gmail():
    creds = None
    token_path = 'token.json'

    if os.path.exists(token_path):
        creds = Credentials.from_authorized_user_file(token_path, SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file('credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)
        with open(token_path, 'w') as token:
            token.write(creds.to_json())

    return build('gmail', 'v1', credentials=creds)

def fetch_latest_emails(service, model, max_results=100):
    results = service.users().messages().list(userId='me', labelIds=['INBOX'], maxResults=max_results).execute()
    messages = results.get('messages', [])

    email_data = []

    for msg in messages:
        msg_raw = service.users().messages().get(userId='me', id=msg['id'], format='raw').execute()
        raw_data = base64.urlsafe_b64decode(msg_raw['raw'].encode('ASCII'))
        mime_msg = message_from_bytes(raw_data)

        subject = mime_msg['Subject']
        sender = mime_msg['From']
        body = extract_body(mime_msg)

        label, confidence = predict_spam(model, body)

        email_data.append({
            'id': msg['id'],
            'subject': subject,
            'sender': sender,
            'body': body,
            'label': label,
            'confidence': round(confidence, 2)
        })

    return email_data

def extract_body(msg):
    if msg.is_multipart():
        for part in msg.walk():
            if part.get_content_type() == 'text/plain':
                return part.get_payload(decode=True).decode(errors='ignore')
    else:
        return msg.get_payload(decode=True).decode(errors='ignore')
    return ""

if __name__ == "__main__":
    model = load_model()
    service = authenticate_gmail()
    emails = fetch_latest_emails(service, model)

    for email in emails:
        print("\n=== EMAIL ===")
        print("From:", email['sender'])
        print("Subject:", email['subject'])
        print("Prediction:", email['label'], "| Confidence:", email['confidence'])

    # Save all for manual review
    log_emails_for_review(emails)

    # Delete high-confidence spam
    delete_spam_emails(service, emails, threshold=0.9)
