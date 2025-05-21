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
        # Try to fetch text from the nucleotide core database in Medline format
        handle = Entrez.efetch(db="nuccore", id=accession, rettype="medline", retmode="text")
        text = handle.read()
        handle.close()

        # Extract PUBMED IDs from text
        pubmed_ids = re.findall(r'PUBMED\s+(\d+)', text)
        # Extract isolate from text
        isolate_matches = list(set([i.split()[0] for i in re.findall(r'(?:/isolate="|isolate\s+)([A-Za-z0-9 _-]+)"?', text)]))
        
        if not pubmed_ids:
            print(f"⚠️ No PubMed ID found for accession {accession}")
        if not isolate_matches:
            print(f"⚠️ No isolate info for accession {accession}")

        return pubmed_ids, isolate_matches

    except Exception as e:
        print("❌ Entrez error:", e)
        return [] , []
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
### pubmed_ids is a list from get_info_from_accession, list might be empty
def get_doi_from_pubmed_id(pubmed_ids):
    dois = {}
    for pubmed_id in pubmed_ids:
        try:
            handle = Entrez.efetch(db="pubmed", id=pubmed_id, rettype="medline", retmode="text")
            records = list(Medline.parse(handle))
            handle.close()
            if not records:
                print("Invalid PUBMED ID or no record for this ID or Entrez blocked")
            record = records[0]
            if "AID" in record:
                for aid in record["AID"]:
                    if "[doi]" in aid:
                        dois[pubmed_id] = ( aid.split(" ")[0]  )# extract the DOI
            else: print("No DOI in record")
    
        except Exception as e:
            print(f"❌ Failed to get DOI from PubMed ID {pubmed_id}: {e}")
    if dois:
        return dois
    else:
        return None


# Step 3: Extract Text: Get the paper (html text), sup. materials (pdf, doc, excel) and do text-preprocessing
# Step 3.1: Extract Text
# DOIs is a dict with index = id and value = doi or a NoneType
def get_paper_text(dois):
  # create the temporary folder to contain the texts
  if dois: 
      textsToExtract = {}
      '''textsToExtract = { "doiLink":"paperText"
                            "file1.pdf":"text1",
                            "file2.doc":"text2",
                            "file3.xlsx":excelText3'''
      for pubmed_id in dois:
          cmd = f'mkdir data/{pubmed_id}'
          result = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
          saveLinkFolder = "data/"+pubmed_id
        
          link = 'https://doi.org/' + dois[pubmed_id]
          
          # get the file to create listOfFile for each id
          html = extractHTML.HTML("",link)
          jsonSM = html.getSupMaterial()
          text = ""
          links  = [link] + sum((jsonSM[key] for key in jsonSM),[])
          #print(links)
          for l in links:
            # get the main paper
            if l == link:
              text = html.getListSection()
              textsToExtract[link] = text
            elif l.endswith(".pdf"):
              p = pdf.PDF(l,saveLinkFolder,dois[pubmed_id])
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
              corpus = wc.tableTransformToCorpusText([],l)
              text = ''
              for c in corpus:
                para = corpus[c]
                for words in para:
                  text += " ".join(words)
              textsToExtract[l] = text
          # delete folder after finishing getting text
          cmd = f'rm -r data/{pubmed_id}'
          result = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
      return textsToExtract
  else: print("⚠️ No DOI to get text")
# Step 3.2: Extract context
def extract_context(text, keyword, window=500):
    idx = text.find(keyword)
    if idx == -1:
        return "Sample ID not found."
    return text[max(0, idx-window): idx+window]
# Step 4: Classification for now (demo purposes)
# 4.1: Using a HuggingFace model (question-answering)
def infer_location_fromQAModel(context, question="Where is the mtDNA sample from?", qa = None):
    try:
        # qa = pipeline("question-answering", model="distilbert-base-uncased-distilled-squad")
        # result = qa({"context": context, "question": question})
        result = qa(question=question, context=context)
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
        match = re.findall(r'/(geo_loc_name|country|location)\s*=\s*"([^"]+)"', text)
        if match:
            return ",".join([value for key, value in match]), ",".join([key for key, value in match])  # This is the value like "Brunei"
        else:
            return "" , ""

    except Exception as e:
        print("❌ Entrez error:", e)
        return "", ""


# STEP 5: Main pipeline: accession -> 1. get pubmed id and isolate -> 2. get doi -> 3. get text -> 4. prediction -> 5. output: inferred location + explanation + confidence score
def classify_sample_location(accession):
    outputs = {}
    keyword, context, location, qa_result, haplo_result = "", "", "", "", ""
    
    ### First method (4.0): infer_location_fromNCBI(): Check direct info
    """
    Info directly searched in metadata by tags of /(geo_loc_name|country|location)
    """
    outputs[accession] = {}
    location, search_key = infer_location_fromNCBI(accession)
    NCBI_result = {
        "source": "NCBI",
        "sample_id": accession,
        "predicted_location": location,
        "context_snippet": search_key}
    outputs[accession]["NCBI"]= {"NCBI": NCBI_result}
    
    ### Other methods
    """
    Without direct info of the origin, more advanced methods needs retrieval of articles and from isolates info
    """
    # Step 1: get pubmed id and isolate
    pubmed_ids, isolates = get_info_from_accession(accession)
    if pubmed_ids: 
    # Step 2: get doi
        dois = get_doi_from_pubmed_id(pubmed_ids)
        if dois:
    # Step 3: get text
            '''textsToExtract = { "doiLink":"paperText"
                                  "file1.pdf":"text1",
                                  "file2.doc":"text2",
                                  "file3.xlsx":excelText3'''
            textsToExtract = get_paper_text(dois)
            if textsToExtract:
    # Step 4: prediction
                # Prepare: load QA model once and for all
                qa = pipeline("question-answering", model="distilbert-base-uncased-distilled-squad")
                
                for key in textsToExtract:
                    text = textsToExtract[key]
        # try accession number first
                    outputs[accession][key] = {}
                    keyword = accession
                    context = extract_context(text, keyword, window=500)
              # Method 4.1: Using a HuggingFace model (question-answering)
                    location = infer_location_fromQAModel(context, question=f"Where is the mtDNA sample {keyword} from?")
                    qa_result = {
                        "source": key,
                        "sample_id": keyword,
                        "predicted_location": location,
                        "context_snippet": context
                    }
                    outputs[keyword][key]["QAModel"] = qa_result
                      
                      
              # Method 4.2: Infer from haplogroup
                    haplo_result = classify_mtDNA_sample_from_haplo(context)
                    outputs[keyword][key]["haplogroup"] = haplo_result
                  
    # try isolate
                    if isolates: 
                        for isolate in isolates: 
                            outputs[isolate] = {}
                            keyword = isolate
                            outputs[isolate][key] = {}
                            context = extract_context(text, keyword, window=500)
                  # Method 4.1 again for isolate: Using a HuggingFace model (question-answering)
                            location = infer_location_fromQAModel(context, question=f"Where is the mtDNA sample {keyword} from?", qa=qa)
                            qa_result = {
                              "source": key,
                              "sample_id": keyword,
                              "predicted_location": location,
                              "context_snippet": context
                            }
                            outputs[keyword][key]["QAModel"] = qa_result
                  # Method 4.2 again for isolate: Infer from haplogroup
                            haplo_result = classify_mtDNA_sample_from_haplo(context)
                            outputs[keyword][key]["haplogroup"] = haplo_result
                    else: 
                      print("UNKNOWN_ISOLATE")
            else:
                print(f"error: No texts extracted for DOI {dois}")
        else:
            print("error: DOI not found for this accession. Cannot fetch paper or context.")
    else:
        print(f"error: Could not retrieve PubMed ID for accession {accession}")
    return outputs