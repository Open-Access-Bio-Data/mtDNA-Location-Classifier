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

The programme takes an accession number (an NCBI GenBank/nuccore identifier) as input and returns the likely origin of the sequence through `classify_sample_location_cached(accession=accession_number)`. This function wraps around a pipeline that proceeds as follow:
### Check and retrieve the Pubmed ID, isolate and DOI:
- Which are handled by 
        `get_info_from accession(accession=accession_number)`
    - Which look through the metadata of the sequence with `accession_number` and extract `PUBMED ID` if available or `isolate` information.

and 
``

