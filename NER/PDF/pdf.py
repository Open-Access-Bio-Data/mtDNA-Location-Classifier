#!pip install pdfreader
import pdfreader
from pdfreader import PDFDocument, SimplePDFViewer
#!pip install bs4
from bs4 import BeautifulSoup
import requests
from NER import cleanText
#!pip install tabula-py

import tabula
class PDF(): # using PyPDF2
  def __init__(self, pdf, saveFolder, doi=None):
    self.pdf = pdf
    self.doi = doi
    self.saveFolder = saveFolder
  def openPDFFile(self):
    if "https" in self.pdf:
      name = self.pdf.split("/")[-1]
      name = self.downloadPDF(self.saveFolder)
      if name != "no pdfLink to download":
        fileToOpen = self.saveFolder + "/" + name
      else: fileToOpen = self.pdf
    else: fileToOpen = self.pdf
    return open(fileToOpen, "rb")
  def downloadPDF(self, saveFolder):
    pdfLink = ''
    if ".pdf" not in self.pdf and "https" not in self.pdf: # the download link is a general URL not pdf link
      r = requests.get(self.pdf)
      soup = BeautifulSoup(r.content, 'html.parser')
      links = soup.find_all("a")
      for link in links:
        if ".pdf" in link.get("href"):
          if self.doi in link.get("href"):
            pdfLink = link.get("href")
            break
    else:
      pdfLink = self.pdf
    if pdfLink != '':
      response = requests.get(pdfLink)
      name = pdfLink.split("/")[-1]
      pdf = open(saveFolder+"/"+name, 'wb')
      pdf.write(response.content)
      pdf.close()
      print("pdf downloaded")
      return name
    else:
      return "no pdfLink to download"
  def extractText(self):
    jsonPage = {}
    pdf = self.openPDFFile()
    doc = PDFDocument(pdf)
    viewer = SimplePDFViewer(pdf)
    all_pages = [p for p in doc.pages()]
    cl = cleanText.cleanGenText()
    for page in range(1,len(all_pages)):
      viewer.navigate(page)
      viewer.render()
      if str(page) not in jsonPage:
        jsonPage[str(page)] = {}
      # text
        text = "".join(viewer.canvas.strings)
      clean, filteredWord = cl.textPreprocessing(text) #cleanText.cleanGenText(text).cleanText()
      # save the text of filtered words which remove "a", the, "an", "is", etc.
      jsonPage[str(page)]["normalText"] = [text]
      jsonPage[str(page)]["cleanText"] = [' '.join(filteredWord)]
      #image
      image = viewer.canvas.images
      jsonPage[str(page)]["image"] = [image]
      #form
      form = viewer.canvas.forms
      jsonPage[str(page)]["form"] = [form]
      # content based on PDF adobe
      content = viewer.canvas.text_content
      jsonPage[str(page)]["content"] = [content]
      # inline_image:'''
      '''Inline images are aligned with the text,
      and are usually content images like photos, charts, or graphs.'''
      inline_image = viewer.canvas.inline_images
      jsonPage[str(page)]["inline_image"] = [inline_image]
    pdf.close()
    '''Output Format:
    jsonPage[str(page)]["normalText"]
    jsonPage[str(page)]["cleanText"]
    jsonPage[str(page)]["image"]
    jsonPage[str(page)]["form"]
    jsonPage[str(page)]["content"]'''
    return jsonPage
  def extractTable(self,pages,saveFile=None,outputFormat=None):
    '''pages (str, int, iterable of int, optional) –
      An optional values specifying pages to extract from. It allows str,`int`, iterable of :int. Default: 1
      Examples: '1-2,3', 'all', [1,2]'''
    df = []
    if "https" in self.pdf:
      name = self.pdf.split("/")[-1]
      name = self.downloadPDF(self.saveFolder)
      if name != "no pdfLink to download":
        fileToOpen = self.saveFolder + "/" + name
      else: fileToOpen = self.pdf
    else: fileToOpen = self.pdf
    try:
      df = tabula.read_pdf(fileToOpen, pages=pages)
    # saveFile: "/content/drive/MyDrive/CollectData/NER/PDF/tableS1.csv"
    # outputFormat: "csv"
    #tabula.convert_into(self.pdf, saveFile, output_format=outputFormat, pages=pages)
    except:# ValueError:
      df = []
      print("No tables found in PDF file")
    return df
  def mergeTextinJson(self,jsonPDF):
    # pdf
    #cl = cleanGenText()
    cl = cleanText.cleanGenText()
    pdfText = ""
    for page in jsonPDF:
      # page is "\n\n"
      if len(jsonPDF[page]["normalText"]) > 0:
        for i in range(len(jsonPDF[page]["normalText"])):
          text = jsonPDF[page]["normalText"][i]
          if len(text)>0:
            text = cl.removeTabWhiteSpaceNewLine(text)
            text = cl.removeExtraSpaceBetweenWords(text)
          jsonPDF[page]["normalText"][i] = text
          # same page is just a dot.
          if i-1 > 0:
            if jsonPDF[page]["normalText"][i-1][-1] != ".":
              pdfText += ". "
          pdfText += jsonPDF[page]["normalText"][i]
        if len(jsonPDF[page]["normalText"][i])>0:
          if jsonPDF[page]["normalText"][i][-1]!=".":
            pdfText += "."
          pdfText += "\n\n"
    return pdfText
  def getReference(self):
    pass
  def getSupMaterial(self):
    pass
  def removeHeaders(self):
    pass
  def removeFooters(self):
    pass
  def removeReference(self):
    pass