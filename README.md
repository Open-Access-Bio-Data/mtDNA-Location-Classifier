---
setup: bash setup.sh
title: MtDNALocation
emoji: 📊
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 5.25.0
app_file: app.py
pinned: false
license: mit
short_description: mtDNA Location Classification tool
---

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference

# Installation
## Set up environments and start GUI:
```bash
git clone https://github.com/Open-Access-Bio-Data/mtDNA-Location-Classifier.git
```
If installed using mamba (recommended):
```bash
mamba env create -f env.yaml
``` 
If not, check current python version in terminal and make sure that it is python version 3.11.12, then run
```bash
pip install -r requirements.txt
```
To start the programme, run this in terminal:
```bash
python app.py
```
Then follow its instructions
# Descriptions:
mtDNA-Location-Classifier uses [Gradio](https://www.gradio.app/docs) to handle the front-end interactions. 

The programme takes **an accession number** (an NCBI GenBank/nuccore identifier) as input and returns the likely origin of the sequence through `classify_sample_location_cached(accession=accession_number)`. This function wraps around a pipeline that proceeds as follow:
## Steps 1-3: Check and retrieve base materials: the Pubmed ID, isolate, DOI and text:
- Which are respectively:

### Step 1: pubmed_ids and isolates
        `get_info_from accession(accession=accession_number)`
    - Current input is a string of `accession_number` and output are two lists, one of PUBMED IDs and one of isolate(s).
    - Which look through the metadata of the sequence with `accession_number` and extract `PUBMED ID` if available or `isolate` information.
    - The presence of PUBMED ID is currently important for the retrieval of texts in the next steps, which are eventually used by method 4.1 (question-answering) and 4.2 (infer from haplogroup)
    - Some sequences might not have `isolate` info but its availibity is optional. (as they might be used by method 4.1 and 4.2 as alternative)

### Step 2: dois
        `get_doi_from_pubmed_id(pubmed_ids = pubmed_ids)`
    - Input is a list of PUBMED IDs of the sequence with `accession_number` (retrieved from previous step) and output is a dictionary with keys = PUBMED IDs and values = according DOIs.
    - The pubmed_ids are retrieved from the `get_info_from accession(accession=accession_number)` mentioned above.
    - The DOIs will be passed down to dependent functions to extract texts of publications to pass on to method 4.1 and 4.2

### Step 3: get text
        `get_paper_text(dois = dois)`
    - Input is currently a list of dois retrieved from previous step and output is a dictionary with keys = sources (doi links or file type) (We might improve this to have other inputs in addition to just doi links - maybe files); values = texts obtained from sources.
    - Output of this step is crucial to method 4.1 and 4.2


## Step 4: Prediction of origin:
### Method 4.0: 
    - The first method attempts to directly look in the metadata for information that was submitted along with the sequence. Thus, it does not require availability of PUBMED IDs/DOIs or isolates.
    - However, this information is not always available in the submission. Thus, we use other methods (4.1,4.2) to retrieve publications through which we can extract the information of the source of mtDNA

### Method 4.1:
    - 

### Method 4.2:
    - 

## More in the package
### extraction of text from HTML
### extraction of text from PDF
