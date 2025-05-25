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
from io import StringIO

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
    # Step 1: Combine into one string and split using regex to handle commas, line breaks, etc.
    tokens = []
    for item in candidates:
        # Split by comma, whitespace, and newlines
        parts = re.split(r'[\s,]+', item)
        tokens.extend(parts)

    # Step 2: Clean and normalize tokens
    tokens = [word.strip() for word in tokens if word.strip().isalpha()]  # Keep only alphabetic tokens

    # Step 3: Count
    counts = Counter(tokens)

    # Step 4: Get most common
    top_location, count = counts.most_common(1)[0]
    return counts, (top_location, count)

# Store feedback (with required fields)

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
        #print(output)
    except Exception as e:
        return [], f"Error: {e}", f"Error: {e}", f"Error: {e}"

    if accession not in output:
        return [], "Accession not found in results.", "Accession not found in results.", "Accession not found in results."

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
        "Detailed_Results": all_rows,  # <-- make sure this is a plain list, not a DataFrame
        "Summary_Text": summary_text,
        "Ancient_Modern_Flag": flag_text
    }

    # If all_rows is a DataFrame, convert it
    if isinstance(all_rows, pd.DataFrame):
        output_dict["Detailed_Results"] = all_rows.to_dict(orient="records")

    with open(filename, "w") as external_file:
        json.dump(output_dict, external_file, indent=2)

# save the batch input in Text file
def save_to_txt(all_rows, summary_text, flag_text, filename):
    if isinstance(all_rows, pd.DataFrame):
        detailed_results = all_rows.to_dict(orient="records")
    output = ""
    output += ",".join(list(detailed_results[0].keys())) + "\n\n"
    for r in detailed_results:
      output += ",".join([str(v) for v in r.values()]) + "\n\n"
    with open(filename, "w") as f:
        f.write("=== Detailed Results ===\n")
        f.write(output + "\n")

        f.write("\n=== Summary ===\n")
        f.write(summary_text + "\n")
        
        f.write("\n=== Ancient/Modern Flag ===\n")
        f.write(flag_text + "\n")

def save_batch_output(all_rows, summary_text, flag_text, output_type):
    tmp_dir = tempfile.mkdtemp()

    #html_table = all_rows.value  # assuming this is stored somewhere

    # Parse back to DataFrame
    #all_rows = pd.read_html(all_rows)[0]  # [0] because read_html returns a list
    all_rows = pd.read_html(StringIO(all_rows))[0]
    print(all_rows)

    if output_type == "Excel":
        file_path = f"{tmp_dir}/batch_output.xlsx"
        save_to_excel(all_rows, summary_text, flag_text, file_path)
    elif output_type == "JSON":
        file_path = f"{tmp_dir}/batch_output.json"
        save_to_json(all_rows, summary_text, flag_text, file_path)
        print("Done with JSON")
    elif output_type == "TXT":
        file_path = f"{tmp_dir}/batch_output.txt"
        save_to_txt(all_rows, summary_text, flag_text, file_path)
    else:
        return gr.update(visible=False)  # invalid option
    
    return gr.update(value=file_path, visible=True)

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
            all_flags.append(f"**{acc}**\n### 🏺 Ancient/Modern Flag\n**{label}**\n\n_Explanation:_ {explain}")
        except Exception as e:
            all_summaries.append(f"**{acc}**: Failed - {e}")

    """for row in all_rows:
          source_column = row[2]  # Assuming the "Source" is in the 3rd column (index 2)
          
          if source_column.startswith("http"):  # Check if the source is a URL
              # Wrap it with HTML anchor tags to make it clickable
              row[2] = f'<a href="{source_column}" target="_blank" style="color: blue; text-decoration: underline;">{source_column}</a>'"""
      

    summary_text = "\n\n---\n\n".join(all_summaries)
    flag_text = "\n\n---\n\n".join(all_flags)
    return all_rows, summary_text, flag_text, gr.update(visible=True), gr.update(visible=False)