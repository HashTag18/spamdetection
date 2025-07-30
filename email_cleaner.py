# email_cleaner.py

import os
import csv
import json

def delete_spam_emails(service, emails, threshold=0.9):
    deleted_log = []

    for email in emails:
        if email['label'] == 'spam' and email['confidence'] >= threshold:
            # Move to Trash
            service.users().messages().trash(userId='me', id=email['id']).execute()
            print(f"[🗑] Deleted: {email['subject']} ({email['confidence']})")

            # Save locally
            folder = "deleted_emails"
            os.makedirs(folder, exist_ok=True)
            filename = os.path.join(folder, f"{email['id']}.txt")

            with open(filename, 'w', encoding='utf-8') as f:
                f.write(f"From: {email['sender']}\n")
                f.write(f"Subject: {email['subject']}\n")
                f.write(f"Confidence: {email['confidence']}\n\n")
                f.write(email['body'])

            deleted_log.append(email)

    # Save a JSON log
    os.makedirs("logs", exist_ok=True)
    with open("logs/deleted_log.json", "w") as f:
        json.dump(deleted_log, f, indent=2)

    print(f"[✔] {len(deleted_log)} spam emails deleted and stored locally.")

def clean_text_for_csv(text):
    """Clean newlines and quotes for CSV compatibility."""
    text = text.replace('\n', ' ').replace('\r', ' ')
    text = text.replace('"', "'")
    return text.strip()

def log_emails_for_review(emails, filename="email_review.csv"):
    """Save all emails with predictions for manual labeling."""
    fieldnames = ['id', 'subject', 'sender', 'body', 'predicted_label', 'confidence', 'manual_label']
    is_new_file = not os.path.exists(filename)

    with open(filename, 'a', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames, quoting=csv.QUOTE_MINIMAL)
        if is_new_file:
            writer.writeheader()

        for email in emails:
            writer.writerow({
                'id': email['id'],
                'subject': clean_text_for_csv(email['subject']),
                'sender': clean_text_for_csv(email['sender']),
                'body': clean_text_for_csv(email['body']),
                'predicted_label': email['label'],
                'confidence': email['confidence'],
                'manual_label': ''  # fill manually later
            })

    print(f"[📥] Logged {len(emails)} emails for manual review in '{filename}'.")
