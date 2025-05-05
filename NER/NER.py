import spacy
import json
import random
from CollectData.DefaultPackages import openFile, saveFile
from CollectData.NER import cleanText
from CollectData.NER.html import extractHTML
from CollectData.NER.PDF import pdf
from CollectData.NER.excel import excel
from CollectData.NER.WordDoc import wordDoc
from spacy.lang.en import English
from spacy.pipeline import EntityRuler
from CollectData.NER.word2Vec import word2vec
import nltk
#nltk.download()
from nltk.corpus import stopwords

class NER():
  def __init__(self, jsonFile=None):
    self.jsonFile = jsonFile
  # try doing entityruler or rule-based NER
  def openFile(self):
    return openFile.openJsonFile(self.jsonFile)
  def createNewCustomizedData(self):
    json = self.openFile()
    #json = openFile.openJsonFile(jsonFile)
    newJson = {}
    '''- remove punctuation
    - remove stopwords
    - remove: whitespace(strip()); "\t","\n"
    - add more the interchangable name of that same location such as
    Viet Nam (NCBI) = Vietnam (different sources)
    - for some special cases such as "Viet Nam", instead of splitting "Viet", "Nam", let them tgt
    "Viet Nam". The more special cases I know to add to my char list, the better'''
    cl = cleanText.cleanGenText()
    #cl = cleanGenText()
    for key in json:
      newJson[key] = []
      for word in json[key]:
        # A word inside parenthesis should be a separated words
        # ex: Czechia (Czech Republic): ["Czechia","Czech Republic"]
        if " (" in word:
          words = []
          w_s = word.split(" (")
          for w in w_s:
            words += cl.removeLowercaseBetweenUppercase(w)
        else:
          words = cl.removeLowercaseBetweenUppercase(word)
        for w in words:
          clean, filtered = cl.cleanText(w)
          if len(filtered)>0:
            newJson[key].append(" ".join(filtered))
      # filter to only take one unique value
      newJson[key] = list(set(newJson[key]))
      newJson[key].sort()
    return newJson

  def createPatternsTrainingData(self, type):
    json = openFile.openJsonFile(self.jsonFile)
    patterns = []
    for key in json:
      for word in json[key]:
        pattern = {
            "label": type,
            "pattern": word
        }
        patterns.append(pattern)
    return patterns

  def generateRules(self, patterns, saveFolder):
    '''nlp = English()
    ruler = EntityRuler(nlp)
    ruler.add_patterns(patterns)
    nlp.add_pipe(ruler)'''
    nlp = spacy.load("en_core_web_sm")
    ruler = nlp.add_pipe("entity_ruler")
    ruler.add_patterns(patterns)
    # save to load later the json file after adding to pipe
    nlp.to_disk(saveFolder)
    # return nlp

  def test_modelNLP(self, model, text):
    # test for a single text
    doc = model(text)
    results = []
    entities = []
    for ent in doc.ents:
      entities.append((ent.text, ent.start_char, ent.end_char, ent.label_))
    if len(entities)>0:
      results = [text, {"entities": entities}]
    return results
    
  def getCleanTableDiffFormat(self,file,saveFolder=None,pdfPages="all",doi=None):
    wc = word2vec.word2Vec()
    #wc = word2Vec()
    output = ""
    excelFile = None
    df = []
    # get df from html
    if file.split(".")[-1].lower() in "html":
      if doi!=None: doi = 'https://doi.org/' + doi
      html = extractHTML.HTML(file, doi)
      #html = HTML(file, doi)
      df = html.extractTable()
    elif file.split(".")[-1].lower() in "pdf":
      p = pdf.PDF(file, saveFolder, doi)
      #p = PDF(file, saveFolder, doi)
      df = p.extractTable(pdfPages)
    elif file.split(".")[-1].lower() in "csv":
      ex = excel.excel(file)
      df = ex.extractCSV()
    elif file.split(".")[-1].lower() in "xlsx":
      df = []
      excelFile = file
    elif file.split(".")[-1].lower() in "docx":
      w = wordDoc.wordDoc(file, saveFolder)
      #w = wordDoc(file,saveFolder)
      nameFile = w.extractTableAsExcel()
      if nameFile != "No table found on word doc":
        ex = excel.excel(nameFile)
        df = ex.extractXLSX("all")
        excelFile = nameFile
      else:
        df = []
    jsonCorpus = wc.tableTransformToCorpusText(df,excelFile)
    for key in jsonCorpus:
      if len(jsonCorpus[key])>0:
        for sub in jsonCorpus[key]:
          if len(sub)>0:
            output += " ".join(sub) + ". "
    if saveFolder != None and len(output)>0:
      name = file.split("/")[-1].split(".")[0]
      saveFile.saveFile(saveFolder+"/"+name+"_CleanTextFormat.txt",output)
    return output
  def getCleanTextFromDifferentFormats(self,file,saveFolder=None,doi=None):
    # to get the good labeled, maybe should not remove filler words like "and", "the", etc.
    cl = cleanText.cleanGenText()
    #cl = cleanGenText()
    output = ""
    allText = ""
    # get text from HTML file
    if file.split(".")[-1].lower() in "html":
      if doi!=None: doi = 'https://doi.org/' + doi
      html = extractHTML.HTML(file, doi)
      #html = HTML(file, doi)
      json = html.getListSection()
      allText = html.mergeTextInJson(json)
    # get text from PDF file
    elif file.split(".")[-1].lower() in "pdf":
      p = pdf.PDF(file, saveFolder, doi)
      #p = PDF(file, saveFolder, doi)
      jsonPDF = p.extractText()
      allText = p.mergeTextinJson(jsonPDF)
    # get text from Excel
    elif file.split(".")[-1].lower() in "xlsx":
      #ex = excel.excel(file)
      #df = ex.extractXLSX("all")
      allText = ""
    elif file.split(".")[-1].lower() in "csv":
      #ex = excel.excel(file)
      #df = ex.extractCSV()
      allText = ""
    # get text from Word doc
    elif file.split(".")[-1].lower() in "docx":
      w = wordDoc.wordDoc(file,saveFolder)
      #w = wordDoc(file,saveFolder)
      allText = w.extractTextByPage()
    # get Text from allText after cleaning from the different format
    if len(allText)>0:
      for text in allText.split("\n\n"):
        # text is a page, segment is a paragraph in that page
        if len(text) > 0:
          # remove punctuation except period
          text = cl.removePunct(text,True)
          newText = []
          for word in text.split(" "):
            if len(word) > 0:
              #newText.append(cl.splitStickWords(word))
              newText.append(word)
          output += " ".join(newText) + "\n\n"
    if saveFolder != None and len(output)>0:
      name = file.split("/")[-1].split(".")[0]
      saveFile.saveFile(saveFolder+"/"+name+"_CleanTextFormat.txt",output)
    return output
  def createTrainDataNLP(self, modelFile, text, saveFolder=None, fileType="Text"):
    TRAIN_DATA = []
    #text = ""
    '''if fileType != "Text": # means it is table
      text = self.getCleanTableDiffFormat(file)
    else:  
      text = self.getCleanTextFromDifferentFormats(file)'''
    nlp = spacy.load(modelFile)
    for segment in text.split("\n\n"):
      if len(segment) > 0:
      # text is a page, segment is a paragraph in that page
        results = self.test_modelNLP(nlp,segment) # model here is NLP
      if len(results) > 0:
        TRAIN_DATA.append(results)
    if saveFolder != None: # save train_data list in json format
      saveFile.saveJsonFile(saveFolder,TRAIN_DATA)    
    return TRAIN_DATA

  def trainSpacy(self, trainData, iterations, modelNLP="default"):
    '''In order to use languages that don’t yet come with a trained pipeline,
    you have to import them directly, or use spacy.blank.
    A blank pipeline is typically just a tokenizer'''
    # TRAIN_DATA = [(text, {"entities": [(word, start, end, label)]})]
    # reference: https://medium.com/@hirthicksofficial/building-a-custom-named-entity-recognition-ner-model-with-spacy-8dca839d8abc
    # create a blank model or load a pre-trained model from en_core_web_sm
    # create a blank model
    if modelNLP == "default":
      nlp = spacy.blank("en")
    # let's train and add training data to small spacy by using "en_core_web_sm", sm means small
    else:
      #nlp = spacy.load("en_core_web_sm")
      #nlp = spacy.load("/content/drive/MyDrive/CollectData/NER/Training_Model/GPE_PatternsRule_NER")
      nlp = spacy.load(modelNLP)
    if "ner" not in nlp.pipe_names:
      ner = nlp.create_pipe("ner")
      nlp.add_pipe(ner, last=True)
    for result in trainData:
      text = result[0]
      entities = result[1]
      for ent in entities["entities"]:
        ner.add_label(ent[-1]) # the last word in the list is a label
    otherPipes = [pipe for pipe in nlp.pipe_names if pipe!='ner']
    # train model parameter
    # reference: https://github.com/wjbmattingly/ner_youtube/blob/d06e2f7dfed51db8658e9e3ab13f3ff46aa005d2/lessons/04_03_customizing_spacy.py
    with nlp.disable_pipes(*otherPipes):
      optimizer = nlp.begin_training()
      for itn in range(iterations):
        print("Starting iteration " + str(itn))
        random.shuffle(trainData)
        losses = {}
        for Text, Annotations in trainData:
          nlp.update(
                      [Text],
                      [Annotations],
                      drop=0.2,
                      sgd=optimizer,
                      losses=losses
                  )
          print(losses)
    return nlp