<<<<<<< HEAD
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
=======
# ✅ Optimized mtDNA MVP UI with Faster Pipeline & Required Feedback

import gradio as gr
from collections import Counter
import csv
import os
from functools import lru_cache
from mtdna_classifier import classify_sample_location 
import subprocess
import json
import pandas as pd
import io
import re
import tempfile
import gspread
from oauth2client.service_account import ServiceAccountCredentials

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

# helper function to extract accessions
def extract_accessions_from_input(file=None, raw_text=""):
    print(f"RAW TEXT RECEIVED: {raw_text}")
    accessions = []
    seen = set()
    if file:
        try:
            if file.name.endswith(".csv"):
                df = pd.read_csv(file)
            elif file.name.endswith(".xlsx"):
                df = pd.read_excel(file)
            else:
                return [], "Unsupported file format. Please upload CSV or Excel."
            for acc in df.iloc[:, 0].dropna().astype(str).str.strip():
                if acc not in seen:
                    accessions.append(acc)
                    seen.add(acc)
        except Exception as e:
            return [], f"Failed to read file: {e}"

    if raw_text:
        text_ids = [s.strip() for s in re.split(r"[\n,;\t]", raw_text) if s.strip()]
        for acc in text_ids:
            if acc not in seen:
                accessions.append(acc)
                seen.add(acc)

    return list(accessions), None

def summarize_results(accession):
    try:
        output, labelAncient_Modern, explain_label = classify_sample_location_cached(accession)
        print(output)
    except Exception as e:
        return [], f"Error: {e}"

    if accession not in output:
        return [], "Accession not found in results."

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

    return rows, summary, labelAncient_Modern, explain_label

# save the batch input in excel file
def save_to_excel(all_rows, summary_text, flag_text, filename):
    with pd.ExcelWriter(filename) as writer:
        # Save table
        df = pd.DataFrame(all_rows, columns=["Sample ID", "Technique", "Source", "Predicted Location", "Haplogroup", "Inferred Region", "Context Snippet"])
        df.to_excel(writer, sheet_name="Detailed Results", index=False)
        
        # Save summary
        summary_df = pd.DataFrame({"Summary": [summary_text]})
        summary_df.to_excel(writer, sheet_name="Summary", index=False)
        
        # Save flag
        flag_df = pd.DataFrame({"Flag": [flag_text]})
        flag_df.to_excel(writer, sheet_name="Ancient_Modern_Flag", index=False)

# save the batch input in JSON file
def save_to_json(all_rows, summary_text, flag_text, filename):
    output_dict = {
        "Detailed_Results": all_rows,
        "Summary_Text": summary_text,
        "Ancient_Modern_Flag": flag_text
    }
    with open(filename, "w") as f:
        json.dump(output_dict, f, indent=2)

# save the batch input in Text file
def save_to_txt(all_rows, summary_text, flag_text, filename):
    with open(filename, "w") as f:
        f.write("=== Detailed Results ===\n")
        for row in all_rows:
            f.write(", ".join(str(x) for x in row) + "\n")
        
        f.write("\n=== Summary ===\n")
        f.write(summary_text + "\n")
        
        f.write("\n=== Ancient/Modern Flag ===\n")
        f.write(flag_text + "\n")

def save_batch_output(all_rows, summary_text, flag_text, output_type):
    tmp_dir = tempfile.mkdtemp()

    if output_type == "Excel":
        file_path = f"{tmp_dir}/batch_output.xlsx"
        save_to_excel(all_rows, summary_text, flag_text, file_path)
    elif output_type == "JSON":
        file_path = f"{tmp_dir}/batch_output.json"
        save_to_json(all_rows, summary_text, flag_text, file_path)
    elif output_type == "TXT":
        file_path = f"{tmp_dir}/batch_output.txt"
        save_to_txt(all_rows, summary_text, flag_text, file_path)
    else:
        return None  # invalid option
    
    return file_path

# run the batch
def summarize_batch(file=None, raw_text=""):
    accessions, error = extract_accessions_from_input(file, raw_text)
    if error:
        return [], "", "", f"Error: {error}"

    all_rows = []
    all_summaries = []
    all_flags = []

    for acc in accessions:
        try:
            rows, summary, label, explain = summarize_results(acc)
            all_rows.extend(rows)
            all_summaries.append(f"**{acc}**\n{summary}")
            all_flags.append(f"**{acc}**: {label}\n_Explanation:_ {explain}")
        except Exception as e:
            all_summaries.append(f"**{acc}**: Failed - {e}")

    summary_text = "\n\n---\n\n".join(all_summaries)
    flag_text = "\n\n".join(all_flags)

    return all_rows, summary_text, flag_text, gr.update(visible=False)

# Gradio UI
with gr.Blocks() as interface:
    gr.Markdown("# 🧬 mtDNA Location Classifier (MVP)")

    inputMode = gr.Radio(choices=["Single Accession", "Batch Input"], value="Single Accession", label="Choose Input Mode")

    with gr.Group() as single_input_group:
        single_accession = gr.Textbox(label="Enter Single Accession (e.g., KU131308)")

    with gr.Group(visible=False) as batch_input_group:
        raw_text = gr.Textbox(label="🧬 Paste Accession Numbers")
        file_upload = gr.File(label="📁 Or Upload CSV/Excel File", file_types=[".csv", ".xlsx"], interactive=True, elem_id="file-upload-box")
        print(raw_text)
        # Make the file box smaller
        gr.HTML('<style>#file-upload-box { width: 200px; }</style>')

    with gr.Row():
        run_button = gr.Button("🔍 Submit and Classify")
        reset_button = gr.Button("🔄 Reset")

    status = gr.Markdown(visible=False)

    with gr.Group(visible=False) as results_group:
        with gr.Row():
            with gr.Column():
                output_summary = gr.Markdown()
            with gr.Column():
                output_flag = gr.Markdown()
        
        gr.Markdown("---")
        output_table = gr.Dataframe(
            headers=["Sample ID", "Technique", "Source", "Predicted Location", "Haplogroup", "Inferred Region", "Context Snippet"],
            interactive=False,
            row_count=(5, "dynamic")
        )

        with gr.Row():
            output_type = gr.Dropdown(choices=["Excel", "JSON", "TXT"], label="Select Output Format", value="Excel")
            download_button = gr.Button("⬇️ Download Output")
            download_file = gr.File(label="Download File Here")

        gr.Markdown("---")

        gr.Markdown("### 💬 Feedback (required)")
        q1 = gr.Textbox(label="1️⃣ Was the inferred location accurate or helpful? Please explain.")
        q2 = gr.Textbox(label="2️⃣ What would improve your experience with this tool?")
        contact = gr.Textbox(label="📧 Your email or institution (optional)")
        submit_feedback = gr.Button("✅ Submit Feedback")
        feedback_status = gr.Markdown()

    # Functions

    def toggle_input_mode(mode):
        if mode == "Single Accession":
            return gr.update(visible=True), gr.update(visible=False)
        else:
            return gr.update(visible=False), gr.update(visible=True)

    def classify_with_loading():
        return gr.update(value="⏳ Please wait... processing...", visible=True)

    def classify_dynamic(single_accession, file, text, mode):
        print(f"MODE: {mode} | RAW TEXT: {text}")
        if mode == "Single Accession":
            return classify_main(single_accession)
        else:
            return summarize_batch(file, text)

    def classify_main(accession):
        table, summary, labelAncient_Modern, explain_label = summarize_results(accession)
        flag_output = f"### 🏺 Ancient/Modern Flag\n**{labelAncient_Modern}**\n\n_Explanation:_ {explain_label}"
        return (
            table,
            summary,
            flag_output,
            gr.update(visible=True),
            gr.update(visible=False)
        )

    def reset_fields():
        return (
            gr.update(value=""),  # single_accession
            gr.update(value=""),  # raw_text
            gr.update(value=None), # file_upload
            gr.update(value="Single Accession"), # inputMode
            gr.update(value=[], visible=True), # output_table
            gr.update(value="", visible=True), # output_summary
            gr.update(value="", visible=True), # output_flag
            gr.update(visible=False), # status
            gr.update(visible=False)  # results_group
        )

    inputMode.change(fn=toggle_input_mode, inputs=inputMode, outputs=[single_input_group, batch_input_group])
    run_button.click(fn=classify_with_loading, inputs=[], outputs=status)
    run_button.click(
        fn=classify_dynamic,
        inputs=[single_accession, file_upload, raw_text, inputMode],
        outputs=[output_table, output_summary, output_flag, results_group, status]
    )
    reset_button.click(
        fn=reset_fields,
        inputs=[],
        outputs=[
            single_accession, raw_text, file_upload, inputMode,
            output_table, output_summary, output_flag,
            status, results_group
        ]
    )

    download_button.click(
        save_batch_output, [output_table, output_summary, output_flag, output_type], download_file
    )
    submit_feedback.click(
        fn=store_feedback_to_google_sheets, inputs=[single_accession, q1, q2, contact], outputs=feedback_status
    )

interface.launch(share=True)
>>>>>>> 597aa7c (WIP: Save local changes which mainly updated appUI  before moving to UpdateAppUI)
