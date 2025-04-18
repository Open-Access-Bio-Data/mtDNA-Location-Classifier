# ✅ Optimized mtDNA MVP UI with Faster Pipeline & Required Feedback

import gradio as gr
from collections import Counter
import csv
import os
from functools import lru_cache
from mtdna_classifier import classify_sample_location
import subprocess
import json

@lru_cache(maxsize=128)
def classify_sample_location_cached(accession):
    return classify_sample_location(accession)

# Count and suggest final location
def compute_final_suggested_location(rows):
    candidates = [
        row.get("Predicted Location", "").strip()
        for row in rows
        if row.get("Predicted Location", "").strip().lower() not in ["", "sample id not found", "unknown"]
    ] + [
        row.get("Inferred Region", "").strip()
        for row in rows
        if row.get("Inferred Region", "").strip().lower() not in  ["", "sample id not found", "unknown"]
    ]

    if not candidates:
        return Counter(), ("Unknown", 0)

    counts = Counter(candidates)
    top_location, count = counts.most_common(1)[0]
    return counts, (top_location, count)

# Store feedback (with required fields)
import gspread
from oauth2client.service_account import ServiceAccountCredentials

'''creds_dict = json.loads(os.environ["GCP_CREDS_JSON"])

scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_dict, scope)

def store_feedback_to_google_sheets(accession, answer1, answer2, contact=""):
    if not answer1.strip() or not answer2.strip():
        return "⚠️ Please answer both questions before submitting."

    try:
        # Define the scope and authenticate
        scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
        creds = ServiceAccountCredentials.from_json_keyfile_name("credentials.json", scope)
        client = gspread.authorize(creds)

        # Open the spreadsheet and worksheet
        sheet = client.open("feedback_mtdna").sheet1  # You can change the name
        sheet.append_row([accession, answer1, answer2, contact])
        return "✅ Feedback submitted. Thank you!"
    except Exception as e:
        return f"❌ Error submitting feedback: {str(e)}"'''

import os
import json
from oauth2client.service_account import ServiceAccountCredentials
import gspread

def store_feedback_to_google_sheets(accession, answer1, answer2, contact=""):
    if not answer1.strip() or not answer2.strip():
        return "⚠️ Please answer both questions before submitting."

    try:
        # ✅ Step: Load credentials from Hugging Face secret
        creds_dict = json.loads(os.environ["GCP_CREDS_JSON"])
        scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
        creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_dict, scope)

        # Connect to Google Sheet
        client = gspread.authorize(creds)
        sheet = client.open("feedback_mtdna").sheet1  # make sure sheet name matches

        # Append feedback
        sheet.append_row([accession, answer1, answer2, contact])
        return "✅ Feedback submitted. Thank you!"

    except Exception as e:
        return f"❌ Error submitting feedback: {e}"


def summarize_results(accession):
    try:
        output = classify_sample_location_cached(accession)
        print(output)
    except Exception as e:
        return [], f"❌ Error: {e}"

    if accession not in output:
        return [], "❌ Accession not found in results."

    isolate = next((k for k in output if k != accession), None)
    row_score = []
    rows = []

    for key in [accession, isolate]:
        if key not in output:
            continue
        sample_id_label = f"{key} ({'accession number' if key == accession else 'isolate of accession'})"
        for section, techniques in output[key].items():
            for technique, content in techniques.items():
                source = content.get("source", "")
                predicted = content.get("predicted_location", "")
                haplogroup = content.get("haplogroup", "")
                inferred = content.get("inferred_location", "")
                context = content.get("context_snippet", "")[:300] if "context_snippet" in content else ""

                row = {
                    "Sample ID": sample_id_label,
                    "Technique": technique,
                    "Source": f"The region of haplogroup is inferred\nby using this source: {source}" if technique == "haplogroup" else source,
                    "Predicted Location": "" if technique == "haplogroup" else predicted,
                    "Haplogroup": haplogroup if technique == "haplogroup" else "",
                    "Inferred Region": inferred if technique == "haplogroup" else "",
                    "Context Snippet": context
                }

                row_score.append(row)
                rows.append(list(row.values()))

    location_counts, (final_location, count) = compute_final_suggested_location(row_score)
    summary_lines = [f"### 🧭 Location Frequency Summary", "After counting all predicted and inferred locations:\n"]
    summary_lines += [f"- **{loc}**: {cnt} times" for loc, cnt in location_counts.items()]
    summary_lines.append(f"\n**Final Suggested Location:** 🗺️ **{final_location}** (mentioned {count} times)")
    summary = "\n".join(summary_lines)

    return rows, summary
# Gradio UI
with gr.Blocks() as interface:
    gr.Markdown("# 🧬 mtDNA Location Classifier (MVP)")
    gr.Markdown("Enter an accession number to infer geographic origin. You'll see predictions, confidence scores, and can submit feedback.")

    with gr.Row():
        accession = gr.Textbox(label="Enter Accession Number (e.g., KU131308)")
        run_button = gr.Button("🔍 Submit and Classify")
        reset_button = gr.Button("🔄 Reset")

    status = gr.Markdown(visible=False)
    headers = ["Sample ID", "Technique", "Source", "Predicted Location", "Haplogroup", "Inferred Region", "Context Snippet"]
    output_table = gr.Dataframe(headers=headers, interactive=False)
    output_summary = gr.Markdown()

    gr.Markdown("---")
    gr.Markdown("### 💬 Feedback (required)")
    q1 = gr.Textbox(label="1️⃣ Was the inferred location accurate or helpful? Please explain.")
    q2 = gr.Textbox(label="2️⃣ What would improve your experience with this tool?")
    contact = gr.Textbox(label="📧 Your email or institution (optional)")
    submit_feedback = gr.Button("✅ Submit Feedback")
    feedback_status = gr.Markdown()

    def classify_with_loading(accession):
        return gr.update(value="⏳ Please wait... processing...", visible=True)

    def classify_main(accession):
        table, summary = summarize_results(accession)
        return table, summary, gr.update(visible=False)

    def reset_fields():
        return "", "", "", "", "", [], "", gr.update(visible=False)

    run_button.click(fn=classify_with_loading, inputs=accession, outputs=status)
    run_button.click(fn=classify_main, inputs=accession, outputs=[output_table, output_summary, status])
    submit_feedback.click(fn=store_feedback_to_google_sheets, inputs=[accession, q1, q2, contact], outputs=feedback_status)
    reset_button.click(fn=reset_fields, inputs=[], outputs=[accession, q1, q2, contact, feedback_status, output_table, output_summary, status])

interface.launch(share=True)
