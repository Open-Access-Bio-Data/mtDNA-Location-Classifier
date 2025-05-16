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
os.chdir('/home/h_k_linh/Desktop/mtDNA-Location-Classifier')
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
accession = "DQ112870.2" # only one PUBMED
# accession = "NC_012920" # standard
# accession = "OP004741.1" # no PUBMED
pubmed_ids, isolate_matches = get_info_from_accession(accession)
dois = get_doi_from_pubmed_id(pubmed_ids)
text = get_paper_text(dois)
infer_location_fromNCBI(accession)
outputs = classify_sample_location(accession)
