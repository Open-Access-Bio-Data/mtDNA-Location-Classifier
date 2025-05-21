#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  2 13:48:15 2025

@author: h_k_linh
"""
import pandas as pd
import numpy as np
import pickle
import os
os.getcwd()
os.chdir('/home/h_k_linh/Desktop/CodingWithVy/mtDNA-Location-Classifier')
os.listdir()
from mtdna_classifier import *
#%%% Extract NC_ list from article
'''
 Turns out the list is not all human so I only create a list of 'human' down below. Naming NC_homo_list
'''
excelNC_ = pd.ExcelFile('/home/h_k_linh/OneDrive/Desktop/CodingWithVy/IDlists/41598_2022_9512_MOESM2_ESM.xlsx') 
sheetnames = excelNC_.sheet_names
listNC_ = []
for sheet in sheetnames:
    df = pd.read_excel(excelNC_, sheet_name=sheet) # skiprows = 1, header=[0, 1]
    # colname = [col for col in df.columns if col[1] == 'NCBI Ref ID']
    listNC_.extend(df.iloc[:,4].dropna().tolist())
listNC = list(set(listNC_))
repeated = list(set(pd.Series(listNC_)[pd.Series(listNC_).duplicated()]))

if 'NCBI Ref ID' in listNC:
    print('yes in list, remove it')
else:
    print('no, perfect!')
    
listNC.remove('NCBI Ref ID')
pd.Series(listNC).to_csv('/home/h_k_linh/OneDrive/Desktop/CodingWithVy/IDlists/NC_ids.txt', index=False, header=False)

repeated.remove('NCBI Ref ID')
pd.Series(repeated).to_csv('/home/h_k_linh/OneDrive/Desktop/CodingWithVy/IDlists/NC_ids_repeated.txt', index=False, header=False)

NC_homo_list = ['NC_homo_list', 'NC_012920', 'NC_013993', 'NC_011137.1'] # Homo heidelbergensis, Homo sapiens, Homo sapiens altai, Homo sapiens neanderthalensis
NC_homo = {}
for i in NC_homo_list:
    print(i)
    NC_homo[i] = infer_location_fromNCBI(i)

#%%
# test = pd.read_html('https://mitomaster.mitomap.org/cgi-bin/index_mitomap.cgi?pos=502&ref=C&alt=T')
import requests

# url = 'https://mitomaster.mitomap.org/cgi-bin/index_mitomap.cgi?pos=502&ref=C&alt=T'
# headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0 Safari/537.36'}

# response = requests.get(url, headers=headers)

# # Check status
# print(response.status_code)

# # Now read HTML from content
# tables = pd.read_html(response.text)
# #%%
# from selenium import webdriver
# from selenium.webdriver.chrome.service import Service
# from selenium.webdriver.common.by import By
# import pandas as pd
# import time

# # Setup Chrome driver
# service = Service('/usr/bin/google-chrome')  # Update with your chromedriver path
# driver = webdriver.Chrome(service=service)

# url = 'https://mitomaster.mitomap.org/cgi-bin/index_mitomap.cgi?pos=502&ref=C&alt=T'
# driver.get(url)

# # Wait for the page to load
# time.sleep(5)  # Increase if needed for slow network

# # Get the table HTML
# table_html = driver.find_element(By.TAG_NAME, 'table').get_attribute('outerHTML')

# # Close browser
# driver.quit()

# # Parse the table with pandas
# df = pd.read_html(table_html)[0]

#%% EBI
#%%% List from Mitobank
df = pd.read_csv('IDlists/MITOBANK_gbcontrol_ids.txt', header=0)
mitobank1 = {}
for i in df['genbank_id']:
    print(i)
    mitobank1[i] = infer_location_fromNCBI(i)

df2 = pd.read_csv('IDlists/MITOBANK_genbank_ids.txt', header=0)
# set(df) & set(df2)
mitobank2 = {}
for i in df2['genbank_id']:
    print(i)
    mitobank2[i] = infer_location_fromNCBI(i)

DQresults = {}    
for i in np.arange(686,962):
    print(f'DQ112{i}')
    DQresults[f'DQ112{i}'] = infer_location_fromNCBI(f'DQ112{i}')

#%%% Link from https://www.ebi.ac.uk/genomes/organelle.html
EBIlist1 = ['J01415', 'AM948965', 'FN673705', 'FR695060'] # Homo sapiens mitochondrion isolate HeLa, Homo sapiens neanderthalensis mitochondrion, 	Homo sapiens ssp. Denisova mitochondrion, Homo sapiens ssp. Denisova mitochondrion, isolate Denisova molar
EBIorganelle = {}
for i in EBIlist1:
    print(i)
    EBIorganelle[i] = infer_location_fromNCBI(i)
#%%% Link from https://www.ebi.ac.uk/ena/portal/api/search?result=sequence&query=tax_tree(33157)&fields=accession
EBIhmmmm = pd.read_csv('IDlists/EBI_ids.txt', header= None)
EBIhmmm = {}
for i in EBIhmmmm[0]:
    print(i)
    EBIhmmm[i] = infer_location_fromNCBI(i)
    
#%% To dataframe 
df_NC_homo = pd.DataFrame.from_dict(NC_homo, orient='index', columns=['loc_name','full_match_line'])

df_mitobank1 = pd.DataFrame.from_dict(mitobank1, orient='index', columns=['loc_name','full_match_line'])

df_mitobank2 = pd.DataFrame.from_dict(mitobank2, orient='index', columns=['loc_name','full_match_line'])

df_DQresults = pd.DataFrame.from_dict(DQresults, orient='index', columns = ['loc_name', 'full_match_line'])

df_EBIorganelle = pd.DataFrame.from_dict(EBIorganelle, orient='index', columns=['loc_name','full_match_line'])

df_EBIhmmm = pd.DataFrame.from_dict(EBIhmmm, orient='index', columns=['loc_name','full_match_line'])

# del df_NC_homo, df_mitobank1, df_mitobank2, df_DQresults, df_EBIorganelle, df_EBIhmmm

#%% Save dictionary before adding other results
os.chdir('/home/h_k_linh/OneDrive/Desktop/CodingWithVy/')
with open('IDlists/MITOBANK_gbcontrol_ids.pkl', 'wb') as f:
    pickle.dump(mitobank1, f)
    
with open('IDlists/MITOBANK_genbank_ids.pkl', 'wb') as f:
    pickle.dump(mitobank2, f)
    
with open('IDlists/DQ_article.pkl', 'wb') as f:
    pickle.dump(DQresults, f)
    
with open('IDlists/NC_homo.pkl', 'wb') as f:
    pickle.dump(NC_homo, f)
    
with open('IDlists/EBIorganelle.pkl', 'wb') as f:
    pickle.dump(EBIorganelle, f)
    
with open('IDlists/EBIhmmm.pkl', 'wb') as f:
    pickle.dump(EBIhmmm, f)
#%%% Load dictionary   
os.chdir('/home/h_k_linh/OneDrive/Desktop/CodingWithVy/')
with open('IDlists/MITOBANK_gbcontrol_ids.pkl', 'rb') as f:
    mitobank1 = pickle.load(f)
    
with open('IDlists/MITOBANK_genbank_ids.pkl', 'rb') as f:
    mitobank2 = pickle.load(f)
    
with open('IDlists/DQ_article.pkl', 'rb') as f:
    DQresults = pickle.load(f)
    
with open('IDlists/NC_homo.pkl', 'rb') as f:
    NC_homo = pickle.load(f)
    
with open('IDlists/EBIorganelle.pkl', 'rb') as f:
    EBIorganelle = pickle.load(f)
    
with open('IDlists/EBIhmmm.pkl', 'rb') as f:
    EBIhmmm = pickle.load(f)
    
#%% Merge data
from mergedeep import merge
all_list = [NC_homo, mitobank1, mitobank2, DQresults, EBIorganelle, EBIhmmm]
merged = merge(*all_list)
df_merged = pd.DataFrame.from_dict(merged, orient='index', columns=['loc_name','full_match_line'])

#%% Vy's data
Vy1_id = pd.read_excel('IDlists/1234Table.xlsx', header = 0)
Vy2_id = pd.read_excel('IDlists/4932Table.xlsx', header = 0)
countt=0 
for i in Vy2_id['AccessionNumber']:
    if i in df_merged.index:
        print(i)
print(countt)
#%% Do as Vy
ls = ['Reference', 'pubmedID', 'title', 'year', 'recent or ancient', 'Author(s)', 'AccessionNumber', 'name', 'Country', 'Isolate', 'Explanation', 'Ethnicity', 'Location', 'Location_Label', 'Language', 'Language', 'Language family', 'haplo', 'haplogroup1', 'haplogroup2', 'haplogroup3', 'Polymorphism', 'Location_reliability']

#%% Accession number with two publications:
# accession = "FJ383757.1" 
# accession = "DQ112870.2" # only one PUBMED
# accession = "NC_012920" # standard
# accession = "OP004741.1" # no PUBMED
accession = "KX456972" # one of Vy's
pubmed_ids, isolate_matches = get_info_from_accession(accession)
dois = get_doi_from_pubmed_id(pubmed_ids)
textToExtract = get_paper_text(dois)
context = extract_context(text['https://static-content.springer.com/esm/art%3A10.1007%2Fs00439-016-1742-y/MediaObjects/439_2016_1742_MOESM2_ESM.pdf'], accession, window=500)
infer_location_fromNCBI(accession)
outputs = classify_sample_location(accession)

#%% extractlocation()

nlp = spacy.load("en_core_web_sm")
text = textToExtract['https://doi.org/10.1007/s00439-016-1742-y']
doc = nlp(text)
locations = []
for ent in doc.ents:
    if ent.label_ == "GPE":  # GPE = Geopolitical Entity (location)
        locations.append(ent.text)

# infer_location_fromQAModel
test = infer_location_fromQAModel(text, question=f"Where is the mtDNA sample {accession} from?", qa=qa)
context = extract_context(text, accession, window=500)
testcontext = infer_location_fromQAModel(context, question=f"Where is the mtDNA sample {accession} from?", qa=qa)




#%% Dois that is bot blocked
"""
text returned was:
    Just a moment...Enable JavaScript and cookies to continue
doi:
    https://doi.org/10.1534/genetics.105.043901
of accession DQ112870.2
"""
htmlLink = "https://doi.org/10.1534/genetics.105.043901"

#%%%
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

options = Options()
options.add_argument("--headless=new") # Headless mode: no GUI
options.add_argument("--no-sandbox")  # Sometimes needed in cloud environments
options.add_argument("--disable-dev-shm-usage")  # Optional (needed for some systems)
driver = webdriver.Chrome(options=options)
driver.get(htmlLink)
try:
    WebDriverWait(driver, 100).until(EC.presence_of_element_located((By.TAG_NAME, "h2")))
except:
    print("⚠️ Timeout waiting for full page to load.")
html = driver.page_source
driver.quit()
soup = BeautifulSoup(html, 'html.parser')

# driver = wd.Chrome(options = options)
# driver.implicitly_wait(1000)
# driver.get(htmlLink)
# soup = BeautifulSoup(driver.page_source, 'html.parser')
soup.find_all('h2')
#%%%
import asyncio
import requests
from bs4 import BeautifulSoup
from playwright.async_api import async_playwright

async def fetch_rendered_html(url, from_doi=True, headless=False, wait_selector="h2", timeout=10000):
    """
    Fetch a fully rendered page using Playwright and return the parsed BeautifulSoup object.

    Args:
        url (str): A direct URL or a DOI link.
        from_doi (bool): Whether to resolve the URL from a DOI redirect.
        headless (bool): Whether to run browser in headless mode.
        wait_selector (str): CSS selector to wait for (e.g. 'h2').
        timeout (int): Max time to wait for selector in ms.

    Returns:
        BeautifulSoup object of the rendered page.
    """

    # Step 1: resolve DOI if needed
    if from_doi:
        try:
            response = requests.head(url, allow_redirects=True)
            final_url = response.url
        except Exception as e:
            raise RuntimeError(f"DOI resolution failed: {e}")
    else:
        final_url = url

    print(f"🌐 Loading: {final_url}")

    # Step 2: Playwright browser load
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=headless)
        page = await browser.new_page()
        await page.goto(final_url)

        # Optional wait for specific content to ensure full render
        try:
            await page.wait_for_selector(wait_selector, timeout=timeout)
        except Exception as e:
            print(f"⚠️ Warning: {wait_selector} not found within {timeout}ms: {e}")

        # Optional: scroll to help trigger lazy load
        await page.mouse.wheel(0, 2000)
        await page.wait_for_timeout(2000)

        html_content = await page.content()
        await browser.close()

    # Step 3: Parse and return soup
    soup = BeautifulSoup(html_content, 'html.parser')
    return soup

soup = await fetch_rendered_html("https://doi.org/10.1534/genetics.105.043901", from_doi=True) 
soup.find_all('h2')
