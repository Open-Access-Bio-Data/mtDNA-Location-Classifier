# mtDNA Location Classifier MVP (Google Colab)
# Accepts accession number → Fetches PubMed ID + isolate name → Gets abstract → Predicts location
import os
import subprocess
import re
from Bio import Entrez
import fitz
import spacy
from spacy.cli import download
from NER.PDF import pdf
from NER.WordDoc import wordDoc
from NER.html import extractHTML
from NER.word2Vec import word2vec
from transformers import pipeline
import urllib.parse, requests
from pathlib import Path
from upgradeClassify import filter_context_for_sample, infer_location_for_sample
# Set your email (required by NCBI Entrez)
#Entrez.email = "your-email@example.com"
import nltk

nltk.download("stopwords")
nltk.download("punkt")
nltk.download('punkt_tab')
# Step 1: Get PubMed ID from Accession using EDirect

'''def get_info_from_accession(accession):
    cmd = f'{os.environ["HOME"]}/edirect/esummary -db nuccore -id {accession} -format medline | egrep "PUBMED|isolate"'
    result = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    output = result.stdout
    pubmedID, isolate = "", ""
    for line in output.split("\n"):
      if len(line) > 0:
        if "PUBMED" in line:
          pubmedID = line.split()[-1]
        if "isolate" in line:  # Check for isolate information
          # Try direct GenBank annotation: /isolate="XXX"
          match1 = re.search(r'/isolate\s*=\s*"([^"]+)"', line)  # search on current line
          if match1:
            isolate = match1.group(1)
          else:
            # Try from DEFINITION line: ...isolate XXX...
            match2 = re.search(r'isolate\s+([A-Za-z0-9_-]+)', line) # search on current line
            if match2:
              isolate = match2.group(1)'''
from Bio import Entrez, Medline
import re

Entrez.email = "your_email@example.com"

def get_info_from_accession(accession):
    try:
        handle = Entrez.efetch(db="nuccore", id=accession, rettype="medline", retmode="text")
        text = handle.read()
        handle.close()

        # Extract PUBMED ID from the Medline text
        pubmed_match = re.search(r'PUBMED\s+(\d+)', text)
        pubmed_id = pubmed_match.group(1) if pubmed_match else ""

        # Extract isolate if available
        isolate_match = re.search(r'/isolate="([^"]+)"', text)
        if not isolate_match:
            isolate_match = re.search(r'isolate\s+([A-Za-z0-9_-]+)', text)
        isolate = isolate_match.group(1) if isolate_match else ""

        if not pubmed_id:
            print(f"⚠️ No PubMed ID found for accession {accession}")

        return pubmed_id, isolate

    except Exception as e:
        print("❌ Entrez error:", e)
        return "", ""
# Step 2: Get doi link to access the paper
'''def get_doi_from_pubmed_id(pubmed_id):
    cmd = f'{os.environ["HOME"]}/edirect/esummary -db pubmed -id {pubmed_id} -format medline | grep -i "AID"'
    result = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    output = result.stdout

    doi_pattern = r'10\.\d{4,9}/[-._;()/:A-Z0-9]+(?=\s*\[doi\])'
    match = re.search(doi_pattern, output, re.IGNORECASE)

    if match:
        return match.group(0)
    else:
        return None  # or raise an Exception with a helpful message'''

def get_doi_from_pubmed_id(pubmed_id):
    try:
        handle = Entrez.efetch(db="pubmed", id=pubmed_id, rettype="medline", retmode="text")
        records = list(Medline.parse(handle))
        handle.close()

        if not records:
            return None
        
        record = records[0]
        if "AID" in record:
            for aid in record["AID"]:
                if "[doi]" in aid:
                    return aid.split(" ")[0]  # extract the DOI

        return None

    except Exception as e:
        print(f"❌ Failed to get DOI from PubMed ID {pubmed_id}: {e}")
        return None


# Step 3: Extract Text: Get the paper (html text), sup. materials (pdf, doc, excel) and do text-preprocessing
# Step 3.1: Extract Text
# sub: download excel file
def download_excel_file(url, save_path="temp.xlsx"):
    if "view.officeapps.live.com" in url:
        parsed_url = urllib.parse.parse_qs(urllib.parse.urlparse(url).query)
        real_url = urllib.parse.unquote(parsed_url["src"][0])
        response = requests.get(real_url)
        with open(save_path, "wb") as f:
            f.write(response.content)
        return save_path
    elif url.startswith("http") and (url.endswith(".xls") or url.endswith(".xlsx")):
        response = requests.get(url)
        response.raise_for_status()  # Raises error if download fails
        with open(save_path, "wb") as f:
            f.write(response.content)
        return save_path
    else:
        print("URL must point directly to an .xls or .xlsx file\n or it already downloaded.")
        return url
def get_paper_text(doi,id,manualLinks=None):
  # create the temporary folder to contain the texts
  folder_path = Path("data/"+str(id))
  if not folder_path.exists():
      cmd = f'mkdir data/{id}'
      result = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
      print("data/"+str(id) +" created.")
  else:
      print("data/"+str(id) +" already exists.")
  saveLinkFolder = "data/"+id

  link = 'https://doi.org/' + doi
  '''textsToExtract = { "doiLink":"paperText"
                        "file1.pdf":"text1",
                        "file2.doc":"text2",
                        "file3.xlsx":excelText3'''
  textsToExtract = {}
  # get the file to create listOfFile for each id
  html = extractHTML.HTML("",link)
  jsonSM = html.getSupMaterial()
  text = ""
  links  = [link] + sum((jsonSM[key] for key in jsonSM),[])
  if manualLinks != None:
    links += manualLinks
  for l in links:
    # get the main paper
    name = l.split("/")[-1]
    file_path = folder_path / name
    if l == link:
      text = html.getListSection()
      textsToExtract[link] = text
    elif l.endswith(".pdf"):
      if file_path.is_file():
          l = saveLinkFolder + "/" + name
          print("File exists.")
      p = pdf.PDF(l,saveLinkFolder,doi)
      f = p.openPDFFile()
      pdf_path = saveLinkFolder + "/" + l.split("/")[-1]
      doc = fitz.open(pdf_path)
      text = "\n".join([page.get_text() for page in doc])
      textsToExtract[l] = text
    elif l.endswith(".doc") or l.endswith(".docx"):
      d = wordDoc.wordDoc(l,saveLinkFolder)
      text = d.extractTextByPage()
      textsToExtract[l] = text
    elif l.split(".")[-1].lower() in "xlsx":
      wc = word2vec.word2Vec()
      # download excel file if it not downloaded yet
      savePath = saveLinkFolder +"/"+ l.split("/")[-1]
      excelPath = download_excel_file(l, savePath)
      corpus = wc.tableTransformToCorpusText([],excelPath)
      text = ''
      for c in corpus:
        para = corpus[c]
        for words in para:
          text += " ".join(words)
      textsToExtract[l] = text
  # delete folder after finishing getting text
  #cmd = f'rm -r data/{id}'
  #result = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
  return textsToExtract
# Step 3.2: Extract context
def extract_context(text, keyword, window=500):
    # firstly try accession number
    idx = text.find(keyword)
    if idx == -1:
        return "Sample ID not found."
    return text[max(0, idx-window): idx+window]
def extract_relevant_paragraphs(text, accession, keep_if=None, isolate=None):
    if keep_if is None:
        keep_if = ["sample", "method", "mtdna", "sequence", "collected", "dataset", "supplementary", "table"]

    outputs = ""
    text = text.lower()

    # If isolate is provided, prioritize paragraphs that mention it
    # If isolate is provided, prioritize paragraphs that mention it
    if accession and accession.lower() in text:
        if extract_context(text, accession.lower(), window=700) != "Sample ID not found.":
            outputs += extract_context(text, accession.lower(), window=700)       
    if isolate and isolate.lower() in text:
        if extract_context(text, isolate.lower(), window=700) != "Sample ID not found.":
            outputs += extract_context(text, isolate.lower(), window=700)
    for keyword in keep_if:
        para = extract_context(text, keyword)
        if para and para not in outputs:
            outputs += para + "\n"
    return outputs
# Step 4: Classification for now (demo purposes)
# 4.1: Using a HuggingFace model (question-answering)
def infer_fromQAModel(context, question="Where is the mtDNA sample from?"):
    try:
        qa = pipeline("question-answering", model="distilbert-base-uncased-distilled-squad")
        result = qa({"context": context, "question": question})
        return result.get("answer", "Unknown")
    except Exception as e:
        return f"Error: {str(e)}"

# 4.2: Infer from haplogroup
# Load pre-trained spaCy model for NER
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

# Define the haplogroup-to-region mapping (simple rule-based)
import csv

def load_haplogroup_mapping(csv_path):
    mapping = {}
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            mapping[row["haplogroup"]] = [row["region"],row["source"]]
    return mapping

# Function to extract haplogroup from the text
def extract_haplogroup(text):
    match = re.search(r'\bhaplogroup\s+([A-Z][0-9a-z]*)\b', text)
    if match:
        submatch = re.match(r'^[A-Z][0-9]*', match.group(1))
        if submatch:
            return submatch.group(0)
        else:
            return match.group(1)  # fallback
    fallback = re.search(r'\b([A-Z][0-9a-z]{1,5})\b', text)
    if fallback:
        return fallback.group(1)
    return None


# Function to extract location based on NER
def extract_location(text):
    doc = nlp(text)
    locations = []
    for ent in doc.ents:
        if ent.label_ == "GPE":  # GPE = Geopolitical Entity (location)
            locations.append(ent.text)
    return locations

# Function to infer location from haplogroup
def infer_location_from_haplogroup(haplogroup):
  haplo_map = load_haplogroup_mapping("data/haplogroup_regions_extended.csv")
  return haplo_map.get(haplogroup, ["Unknown","Unknown"])

# Function to classify the mtDNA sample
def classify_mtDNA_sample_from_haplo(text):
    # Extract haplogroup
    haplogroup = extract_haplogroup(text)
    # Extract location based on NER
    locations = extract_location(text)
    # Infer location based on haplogroup
    inferred_location, sourceHaplo = infer_location_from_haplogroup(haplogroup)[0],infer_location_from_haplogroup(haplogroup)[1]
    return {
        "source":sourceHaplo,
        "locations_found_in_context": locations,
        "haplogroup": haplogroup,
        "inferred_location": inferred_location

    }
# 4.3 Get from available NCBI
def infer_location_fromNCBI(accession):
    try:
        handle = Entrez.efetch(db="nuccore", id=accession, rettype="medline", retmode="text")
        text = handle.read()
        handle.close()
        match = re.search(r'/(geo_loc_name|country|location)\s*=\s*"([^"]+)"', text)
        if match:
            return match.group(2), match.group(0)  # This is the value like "Brunei"
        return "Not found", "Not found"

    except Exception as e:
        print("❌ Entrez error:", e)
        return "Not found", "Not found"

### ANCIENT/MODERN FLAG
from Bio import Entrez
import re

def flag_ancient_modern(accession, textsToExtract, isolate=None):
    """
    Try to classify a sample as Ancient or Modern using:
    1. NCBI accession (if available)
    2. Supplementary text or context fallback
    """
    context = ""
    label, explain = "", ""

    try:
        # Check if we can fetch metadata from NCBI using the accession
        handle = Entrez.efetch(db="nuccore", id=accession, rettype="medline", retmode="text")
        text = handle.read()
        handle.close()

        isolate_source = re.search(r'/(isolation_source)\s*=\s*"([^"]+)"', text)
        if isolate_source:
            context += isolate_source.group(0) + " "

        specimen = re.search(r'/(specimen|specimen_voucher)\s*=\s*"([^"]+)"', text)
        if specimen:
            context += specimen.group(0) + " "

        if context.strip():
            label, explain = detect_ancient_flag(context)
            if label!="Unknown":
              return label, explain + " from NCBI\n(" + context + ")"

        # If no useful NCBI metadata, check supplementary texts
        if textsToExtract:
            labels = {"modern": [0, ""], "ancient": [0, ""], "unknown": 0}

            for source in textsToExtract:
                text_block = textsToExtract[source]
                context = extract_relevant_paragraphs(text_block, accession, isolate=isolate)  # Reduce to informative paragraph(s)
                label, explain = detect_ancient_flag(context)

                if label == "Ancient":
                    labels["ancient"][0] += 1
                    labels["ancient"][1] += f"{source}:\n{explain}\n\n"
                elif label == "Modern":
                    labels["modern"][0] += 1
                    labels["modern"][1] += f"{source}:\n{explain}\n\n"
                else:
                    labels["unknown"] += 1

            if max(labels["modern"][0],labels["ancient"][0]) > 0:
                if labels["modern"][0] > labels["ancient"][0]:
                    return "Modern", labels["modern"][1]
                else:
                    return "Ancient", labels["ancient"][1]
            else:
              return "Unknown", "No strong keywords detected"
        else:
            print("No DOI or PubMed ID available for inference.")
            return "", ""

    except Exception as e:
        print("Error:", e)
        return "", ""


def detect_ancient_flag(context_snippet):
    context = context_snippet.lower()

    ancient_keywords = [
        "ancient", "archaeological", "prehistoric", "neolithic", "mesolithic", "paleolithic",
        "bronze age", "iron age", "burial", "tomb", "skeleton", "14c", "radiocarbon", "carbon dating",
        "postmortem damage", "udg treatment", "adna", "degradation", "site", "excavation",
        "archaeological context", "temporal transect", "population replacement", "cal bp", "calbp", "carbon dated"
    ]

    modern_keywords = [
        "modern", "hospital", "clinical", "consent","blood","buccal","unrelated", "blood sample","buccal sample","informed consent", "donor", "healthy", "patient",
        "genotyping", "screening", "medical", "cohort", "sequencing facility", "ethics approval",
        "we analysed", "we analyzed", "dataset includes", "new sequences", "published data",
        "control cohort", "sink population", "genbank accession", "sequenced", "pipeline", 
        "bioinformatic analysis", "samples from", "population genetics", "genome-wide data"
    ]

    ancient_hits = [k for k in ancient_keywords if k in context]
    modern_hits = [k for k in modern_keywords if k in context]

    if ancient_hits and not modern_hits:
        return "Ancient", f"Flagged as ancient due to keywords: {', '.join(ancient_hits)}"
    elif modern_hits and not ancient_hits:
        return "Modern", f"Flagged as modern due to keywords: {', '.join(modern_hits)}"
    elif ancient_hits and modern_hits:
        if len(ancient_hits) >= len(modern_hits):
            return "Ancient", f"Mixed context, leaning ancient due to: {', '.join(ancient_hits)}"
        else:
            return "Modern", f"Mixed context, leaning modern due to: {', '.join(modern_hits)}"
    
    # Fallback to QA
    answer = infer_fromQAModel(context, question="Are the mtDNA samples ancient or modern? Explain why.")
    if answer.startswith("Error"):
        return "Unknown", answer
    if "ancient" in answer.lower():
        return "Ancient", f"Leaning ancient based on QA: {answer}"
    elif "modern" in answer.lower():
        return "Modern", f"Leaning modern based on QA: {answer}"
    else:
        return "Unknown", f"No strong keywords or QA clues. QA said: {answer}"

# STEP 5: Main pipeline: accession -> 1. get pubmed id and isolate -> 2. get doi -> 3. get text -> 4. prediction -> 5. output: inferred location + explanation + confidence score
def classify_sample_location(accession):
  outputs = {}
  keyword, context, location, qa_result, haplo_result = "", "", "", "", ""
  # Step 1: get pubmed id and isolate
  pubmedID, isolate = get_info_from_accession(accession)
  '''if not pubmedID:
    return {"error": f"Could not retrieve PubMed ID for accession {accession}"}'''
  if not isolate:
    isolate = "UNKNOWN_ISOLATE"
  # Step 2: get doi
  doi = get_doi_from_pubmed_id(pubmedID)
  '''if not doi:
    return {"error": "DOI not found for this accession. Cannot fetch paper or context."}'''
  # Step 3: get text
  '''textsToExtract = { "doiLink":"paperText"
                        "file1.pdf":"text1",
                        "file2.doc":"text2",
                        "file3.xlsx":excelText3'''
  if doi and pubmedID:                      
    textsToExtract = get_paper_text(doi,pubmedID)
  else: textsToExtract = {}  
  '''if not textsToExtract:
    return {"error": f"No texts extracted for DOI {doi}"}'''
  if isolate not in [None, "UNKNOWN_ISOLATE"]:
    label, explain = flag_ancient_modern(accession,textsToExtract,isolate)
  else: 
    label, explain = flag_ancient_modern(accession,textsToExtract)  
  # Step 4: prediction
  outputs[accession] = {}
  outputs[isolate] = {}
  # 4.0 Infer from NCBI
  location, outputNCBI = infer_location_fromNCBI(accession)
  NCBI_result = {
      "source": "NCBI",
      "sample_id": accession,
      "predicted_location": location,
      "context_snippet": outputNCBI}
  outputs[accession]["NCBI"]= {"NCBI": NCBI_result}
  if textsToExtract:
    long_text = ""
    for key in textsToExtract:
      text = textsToExtract[key]
      # try accession number first
      outputs[accession][key] = {}
      keyword = accession
      context = extract_context(text, keyword, window=500)
      # 4.1: Using a HuggingFace model (question-answering)
      location = infer_fromQAModel(context, question=f"Where is the mtDNA sample {keyword} from?")
      qa_result = {
          "source": key,
          "sample_id": keyword,
          "predicted_location": location,
          "context_snippet": context
      }
      outputs[keyword][key]["QAModel"] = qa_result
      # 4.2: Infer from haplogroup
      haplo_result = classify_mtDNA_sample_from_haplo(context)
      outputs[keyword][key]["haplogroup"] = haplo_result
      # try isolate
      keyword = isolate
      outputs[isolate][key] = {}
      context = extract_context(text, keyword, window=500)
      # 4.1.1: Using a HuggingFace model (question-answering)
      location = infer_fromQAModel(context, question=f"Where is the mtDNA sample {keyword} from?")
      qa_result = {
          "source": key,
          "sample_id": keyword,
          "predicted_location": location,
          "context_snippet": context
      }
      outputs[keyword][key]["QAModel"] = qa_result
      # 4.2.1: Infer from haplogroup
      haplo_result = classify_mtDNA_sample_from_haplo(context)
      outputs[keyword][key]["haplogroup"] = haplo_result
      # add long text
      long_text += text + ". \n"
    # 4.3: UpgradeClassify
    # try sample_id as accession number
    sample_id = accession
    if sample_id:
      filtered_context = filter_context_for_sample(sample_id.upper(), long_text, window_size=1)
      locations = infer_location_for_sample(sample_id.upper(), filtered_context)
      if locations!="No clear location found in top matches":
        outputs[sample_id]["upgradeClassifier"] = {}
        outputs[sample_id]["upgradeClassifier"]["upgradeClassifier"] = {
          "source": "From these sources combined: "+ ", ".join(list(textsToExtract.keys())),
          "sample_id": sample_id,
          "predicted_location": ", ".join(locations),
          "context_snippep": "First 1000 words: \n"+ filtered_context[:1000]
        }
    # try sample_id as isolate name
    sample_id = isolate
    if sample_id:
      filtered_context = filter_context_for_sample(sample_id.upper(), long_text, window_size=1)
      locations = infer_location_for_sample(sample_id.upper(), filtered_context)
      if locations!="No clear location found in top matches":
        outputs[sample_id]["upgradeClassifier"] = {}
        outputs[sample_id]["upgradeClassifier"]["upgradeClassifier"] = {
          "source": "From these sources combined: "+ ", ".join(list(textsToExtract.keys())),
          "sample_id": sample_id,
          "predicted_location": ", ".join(locations),
          "context_snippep": "First 1000 words: \n"+ filtered_context[:1000]
        }
  return outputs, label, explain
