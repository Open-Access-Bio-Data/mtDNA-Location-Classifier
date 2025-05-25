import gradio as gr
from mtdna_backend import *
import json
# Gradio UI
with gr.Blocks() as interface:
    gr.Markdown("# 🧬 mtDNA Location Classifier (MVP)")

    inputMode = gr.Radio(choices=["Single Accession", "Batch Input"], value="Single Accession", label="Choose Input Mode")

    with gr.Group() as single_input_group:
        single_accession = gr.Textbox(label="Enter Single Accession (e.g., KU131308)")

    with gr.Group(visible=False) as batch_input_group:
        raw_text = gr.Textbox(label="🧬 Paste Accession Numbers (e.g., MF362736.1,MF362738.1,KU131308,MW291678)")
        gr.HTML("""<a href="https://drive.google.com/file/d/1t-TFeIsGVu5Jh3CUZS-VE9jQWzNFCs_c/view?usp=sharing" download target="_blank">Download Example CSV Format</a>""")
        gr.HTML("""<a href="https://docs.google.com/spreadsheets/d/1lKqPp17EfHsshJGZRWEpcNOZlGo3F5qU/edit?usp=sharing&ouid=112390323314156876153&rtpof=true&sd=true" download target="_blank">Download Example Excel Format</a>""")
        file_upload = gr.File(label="📁 Or Upload CSV/Excel File", file_types=[".csv", ".xlsx"], interactive=True, elem_id="file-upload-box")
        


    with gr.Row():
        run_button = gr.Button("🔍 Submit and Classify")
        reset_button = gr.Button("🔄 Reset")

    status = gr.Markdown(visible=False)

    with gr.Group(visible=False) as results_group:
      with gr.Accordion("Open to See the Result", open=False) as results:  
          with gr.Row():
              output_summary = gr.Markdown(elem_id="output-summary")
              output_flag = gr.Markdown(elem_id="output-flag")
          
          gr.Markdown("---")

      with gr.Accordion("Open to See the Output Table", open=False) as table_accordion:    
          """output_table = gr.Dataframe(
              headers=["Sample ID", "Technique", "Source", "Predicted Location", "Haplogroup", "Inferred Region", "Context Snippet"],
              interactive=False,
              row_count=(5, "dynamic")
          )"""
          output_table = gr.HTML(render=True)


      with gr.Row():
          output_type = gr.Dropdown(choices=["Excel", "JSON", "TXT"], label="Select Output Format", value="Excel")
          download_button = gr.Button("⬇️ Download Output")
      download_file = gr.File(label="Download File Here",visible=False)

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
        return gr.update(value="⏳ Please wait... processing...",visible=True)  # Show processing message
         
    def classify_dynamic(single_accession, file, text, mode):
        if mode == "Single Accession":
            return classify_main(single_accession)  + (gr.update(visible=False),)
        else:
            #return summarize_batch(file, text) + (gr.update(visible=False),)  # Hide processing message
            return classify_mulAcc(file, text) + (gr.update(visible=False),)  # Hide processing message

    # for single accession
    def classify_main(accession):
        table, summary, labelAncient_Modern, explain_label = summarize_results(accession)
        flag_output = f"### 🏺 Ancient/Modern Flag\n**{labelAncient_Modern}**\n\n_Explanation:_ {explain_label}"
        return (
            #table,
            make_html_table(table),
            summary,
            flag_output,
            gr.update(visible=True),
            gr.update(visible=False)
        )
    # for batch accessions
    def classify_mulAcc(file, text):
        table, summary, flag_output, gr1, gr2 = summarize_batch(file, text)
        #flag_output = f"### 🏺 Ancient/Modern Flag\n**{labelAncient_Modern}**\n\n_Explanation:_ {explain_label}"
        return (
            #table,
            make_html_table(table),
            summary,
            flag_output,
            gr.update(visible=True),
            gr.update(visible=False)
        )

    def make_html_table(rows):
      html = """
      <div style='overflow-x: auto; padding: 10px;'>
          <div style='max-height: 400px; overflow-y: auto; border: 1px solid #444; border-radius: 8px;'>
              <table style='width:100%; border-collapse: collapse; table-layout: auto; font-size: 14px; color: #f1f1f1; background-color: #1e1e1e;'>
                  <thead style='position: sticky; top: 0; background-color: #2c2c2c; z-index: 1;'>
                      <tr>
      """
      headers = ["Sample ID", "Technique", "Source", "Predicted Location", "Haplogroup", "Inferred Region", "Context Snippet"]
      html += "".join(
          f"<th style='padding: 10px; border: 1px solid #555; text-align: left; white-space: nowrap;'>{h}</th>"
          for h in headers
      )
      html += "</tr></thead><tbody>"

      for row in rows:
          html += "<tr>"
          for i, col in enumerate(row):
              header = headers[i]
              style = "padding: 10px; border: 1px solid #555; vertical-align: top;"

              # For specific columns like Haplogroup, force nowrap
              if header in ["Haplogroup", "Sample ID", "Technique"]:
                  style += " white-space: nowrap; text-overflow: ellipsis; max-width: 200px; overflow: hidden;"

              if header == "Source" and isinstance(col, str) and col.strip().lower().startswith("http"):
                  col = f"<a href='{col}' target='_blank' style='color: #4ea1f3; text-decoration: underline;'>{col}</a>"

              html += f"<td style='{style}'>{col}</td>"
          html += "</tr>"

      html += "</tbody></table></div></div>"
      return html
  

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
    run_button.click(fn=classify_with_loading, inputs=[], outputs=[status])
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
      fn=save_batch_output, 
      inputs=[output_table, output_summary, output_flag, output_type], 
      outputs=[download_file])

    submit_feedback.click(
        fn=store_feedback_to_google_sheets, inputs=[single_accession, q1, q2, contact], outputs=feedback_status
    )
        # Custom CSS styles
    gr.HTML("""
    <style>
      /* Ensures both sections are equally spaced with the same background size */
      #output-summary, #output-flag {
          background-color: #f0f4f8; /* Light Grey for both */
          padding: 20px;
          border-radius: 10px;
          margin-top: 10px;
          width: 100%; /* Ensure full width */
          min-height: 150px; /* Ensures both have a minimum height */
          box-sizing: border-box; /* Prevents padding from increasing size */
          display: flex;
          flex-direction: column;
          justify-content: space-between;
      }
      
      /* Specific background colors */
      #output-summary {
          background-color: #434a4b; 
      }

      #output-flag {
          background-color: #141616; 
      }

      /* Ensuring they are in a row and evenly spaced */
      .gradio-row {
          display: flex;
          justify-content: space-between;
          width: 100%;
      }
    </style>
    """)


interface.launch(share=True,debug=True)