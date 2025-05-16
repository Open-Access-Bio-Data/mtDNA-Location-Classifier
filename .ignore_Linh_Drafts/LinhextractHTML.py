#!pip install bs4
# reference: https://www.crummy.com/software/BeautifulSoup/bs4/doc/#for-html-documents
from bs4 import BeautifulSoup
import requests
from DefaultPackages import openFile, saveFile
from NER import cleanText
import pandas as pd
class HTML():
  def __init__(self, htmlFile = [], htmlLink = []):
    self.htmlLink = htmlLink
    self.htmlFile = htmlFile
  def openHTMLFile(self):
    soups = []
    if self.htmlLink:  # not an empty list
        for link in self.htmlLink:
            r = requests.get(link)
            soup = BeautifulSoup(r.content, 'html.parser')
            soups.append(soup)
    elif self.htmlFile:  # fallback to local files
        for file_path in self.htmlFile:
            with open(file_path, encoding='utf-8') as fp:
                soup = BeautifulSoup(fp, 'html.parser')
                soups.append(soup)
    else:
        print("⚠️ No HTML links or files provided.")
    return soups  # list of BeautifulSoup objects
    # if self.htmlLink != "None":
    #   r = requests.get(self.htmlLink)
    #   soup = BeautifulSoup(r.content, 'html.parser')
    # else:
    #   with open(self.htmlFile) as fp:
    #     soup = BeautifulSoup(fp, 'html.parser')
    # return soup
  def getText(self):
    soup = self.openHTMLFile()
    s = soup.find_all("html")
    for t in range(len(s)):
      text = s[t].get_text()
    cl = cleanText.cleanGenText()
    text = cl.removeExtraSpaceBetweenWords(text)
    return text
  def getListSection(self, scienceDirect=None):
    json = {}
    text = ""
    textJson, textHTML = "",""
    if scienceDirect == None:
      soup = self.openHTMLFile()
      # get list of section
      json = {}
      for h2Pos in range(len(soup.find_all('h2'))):
        if soup.find_all('h2')[h2Pos].text not in json:
          json[soup.find_all('h2')[h2Pos].text] = []
        if h2Pos + 1 < len(soup.find_all('h2')):
          content = soup.find_all('h2')[h2Pos].find_next("p")
          nexth2Content = soup.find_all('h2')[h2Pos+1].find_next("p")
          while content.text != nexth2Content.text:
            json[soup.find_all('h2')[h2Pos].text].append(content.text)
            content = content.find_next("p")
        else:
          content = soup.find_all('h2')[h2Pos].find_all_next("p",string=True)
          json[soup.find_all('h2')[h2Pos].text] = list(i.text for i in content)
      # format
      '''json = {'Abstract':[], 'Introduction':[], 'Methods'[],
        'Results':[], 'Discussion':[], 'References':[],
        'Acknowledgements':[], 'Author information':[], 'Ethics declarations':[],
        'Additional information':[], 'Electronic supplementary material':[],
        'Rights and permissions':[], 'About this article':[], 'Search':[], 'Navigation':[]}'''
    if scienceDirect!= None or len(json)==0:
      # Replace with your actual Elsevier API key
      api_key = "d0f25e6ae2b275e0d2b68e0e98f68d70"
      # ScienceDirect article DOI or PI (Example DOI)
      doi =  self.htmlLink.split("https://doi.org/")[-1]  #"10.1016/j.ajhg.2011.01.009"
      # Base URL for the Elsevier API
      base_url = "https://api.elsevier.com/content/article/doi/"
      # Set headers with API key
      headers = {
          "Accept": "application/json",
          "X-ELS-APIKey": api_key
      }
      # Make the API request
      response = requests.get(base_url + doi, headers=headers)
# Check if the request was successful
      if response.status_code == 200:
        data = response.json()
        supp_data = data["full-text-retrieval-response"]#["coredata"]["link"]
        if "originalText" in list(supp_data.keys()):
          if type(supp_data["originalText"])==str:
            json["originalText"] = [supp_data["originalText"]]
          if type(supp_data["originalText"])==dict:
            json["originalText"] = [supp_data["originalText"][key] for key in supp_data["originalText"]]
        else:
          if type(supp_data)==dict:
            for key in supp_data:
              json[key] = [supp_data[key]]

    textJson = self.mergeTextInJson(json)
    textHTML = self.getText()
    if len(textHTML) > len(textJson):
      text = textHTML
    else: text = textJson
    return text #json
  def getReference(self):
    # get reference to collect more next data
    ref = []
    json = self.getListSection()
    for key in json["References"]:
      ct = cleanText.cleanGenText(key)
      cleanText, filteredWord = ct.cleanText()
      if cleanText not in ref:
        ref.append(cleanText)
    return ref
  def getSupMaterial(self):
    # check if there is material or not
    json = {}
    soup = self.openHTMLFile()
    for h2Pos in range(len(soup.find_all('h2'))):
      if "supplementary" in soup.find_all('h2')[h2Pos].text.lower() or "material" in soup.find_all('h2')[h2Pos].text.lower() or "additional" in soup.find_all('h2')[h2Pos].text.lower() or "support" in soup.find_all('h2')[h2Pos].text.lower():
        #print(soup.find_all('h2')[h2Pos].find_next("a").get("href"))
        link, output = [],[]
        if soup.find_all('h2')[h2Pos].text not in json:
          json[soup.find_all('h2')[h2Pos].text] = []
        for l in soup.find_all('h2')[h2Pos].find_all_next("a",href=True):
            link.append(l["href"])
        if h2Pos + 1 < len(soup.find_all('h2')):
          nexth2Link = soup.find_all('h2')[h2Pos+1].find_next("a",href=True)["href"]
          if nexth2Link in link:
            link = link[:link.index(nexth2Link)]
        # only take links having "https" in that
        for i in link:
          if "https" in i:  output.append(i)
        json[soup.find_all('h2')[h2Pos].text].extend(output)
    return json
  def extractTable(self):
    soup = self.openHTMLFile()
    df = []
    try:
      df = pd.read_html(str(soup))
    except ValueError:
      df = []
      print("No tables found in HTML file")
    return df
  def mergeTextInJson(self,jsonHTML):
    cl = cleanText.cleanGenText()
    #cl = cleanGenText()
    htmlText = ""
    for sec in jsonHTML:
      # section is "\n\n"
      if len(jsonHTML[sec]) > 0:
        for i in range(len(jsonHTML[sec])):
          # same section is just a dot.
          text = jsonHTML[sec][i]
          if len(text)>0:
            #text = cl.removeTabWhiteSpaceNewLine(text)
            #text = cl.removeExtraSpaceBetweenWords(text)
            text, filteredWord = cl.textPreprocessing(text, keepPeriod=True)
          jsonHTML[sec][i] = text
          if i-1 >= 0:
            if len(jsonHTML[sec][i-1])>0:
              if jsonHTML[sec][i-1][-1] != ".":
                htmlText += ". "
          htmlText += jsonHTML[sec][i]
        if len(jsonHTML[sec][i]) > 0:
          if jsonHTML[sec][i][-1]!=".":
            htmlText += "."
        htmlText += "\n\n"
    return htmlText
  def removeHeaders(self):
    pass
  def removeFooters(self):
    pass
  def removeReferences(self):
    pass