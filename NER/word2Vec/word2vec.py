<<<<<<< HEAD
'''WORD TO VECTOR'''
import pandas as pd
import json
import gensim
import spacy
from DefaultPackages import openFile, saveFile
from NER import cleanText
from gensim.models.keyedvectors import KeyedVectors
from gensim.test.utils import common_texts
from gensim.models.word2vec import Word2Vec
from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.test.utils import datapath, get_tmpfile
import sys
import subprocess
# can try multiprocessing to run quicker
import multiprocessing
import copy
sys.setrecursionlimit(1000)
# creat folder word2Vec
#! mkdir /content/drive/MyDrive/CollectData/NER/word2Vec
# create word2vec model
#model = KeyedVectors.load_word2vec_format('/content/drive/MyDrive/CollectData/NER/word2Vec', binary=True)
'''Some notes for this model
sometimes when we do the corpus, there are some adverbs which are unnecessary but might be seen as
a similar word to the word we are finding, so can we try to preprocess text so that
we make the corpus more effective and only contains the important words. Then when we
train the model, the important words will be seen as important. Or
when we already have the similar list of words, we can remove the words in there
that are stopwords/unnecessary words.'''
### For more complex analysis, consider using sentence embedding models like "Doc2Vec" to represent the meaning of entire sentences instead of just individual words
class word2Vec():
  def __init__(self, nameFile=None, modelName=None):
    self.nameFile = nameFile
    self.modelName = modelName
  def spacy_similarity(self, word):
    # when use word2vec, try medium or large is better
    # maybe try odc similarity?
    nlp = spacy.load("en_core_web_lg")
    doc = nlp(word)
    for token1 in doc:
      for token2 in doc:
        print(token1.text, token2.text, token1.similarity(token2))
    pass
  # clean text before transform to corpus
  def cleanTextBeforeCorpus(self,oriText, doi=None):
    cl = cleanText.cleanGenText()
    #cl = cleanGenText()
    output = ""
    alreadyRemoveDoi = False
    for word in oriText.split(" "):
      # remove DOI
      if doi != None and doi in oriText:
        if alreadyRemoveDoi == False:
          newWord = cl.removeDOI(word,doi)
          if len(newWord) > 0 and newWord != word:
            alreadyRemoveDoi = True
            word = newWord
      # remove punctuation
      # split the sticked words
      #word = cl.splitStickWords(word)
      # remove punctuation
      word = cl.removePunct(word,True)
      # remove URL
      word = cl.removeURL(word)
      # remove HTMLTag
      word = cl.removeHTMLTag(word)
      # remove tab, white space, newline
      word = cl.removeTabWhiteSpaceNewLine(word)
      # optional: remove stopwords
      #word = cl.removeStopWords(word)
      if len(word)>0:
        output += word + " "
    return output
  def cleanAllTextBeforeCorpus(self, allText, doi=None):
    cleanOutput = ""
    remove = "Evaluation Warning: The document was created with Spire.Doc for Python."
    if len(allText) > 0:
      corpusText = allText
      for pos in range(len(corpusText.split("\n\n"))):
        if len(corpusText.split("\n\n")[pos]) > 0:
          lines = corpusText.split("\n\n")[pos]
          for line in lines.split("\n"):
            if remove in line:  line = line.replace(remove, "")
            clean_text = self.cleanTextBeforeCorpus(line, doi)
            cleanOutput += clean_text + "\n"
          cleanOutput += "\n\n"
    return cleanOutput
  def tableTransformToCorpusText(self, df, excelFile=None):
    # PDF, Excel, WordDoc
    #cl = cleanText.cleanGenText()
    corpus = {}
      # PDF or df
    if excelFile == None:
      if len(df) > 0:
        try:
          for i in range(len(df)):
            # each new dimension/page is considered to be a sentence which ends with the period.
            # each new line is a new list, and each new df is a new corpus
            outputDF = []
            text = df[i].values.tolist()
            if len(text) > 0:
              outputRowDF = self.helperRowTableToCorpus(text)
              #outputColDF = self.helperColTableToCorpus(text)
              outputDF.extend(outputRowDF)
              #outputDF.extend(outputColDF)
            if len(outputDF) > 0:
              corpus["corpus" + str(i)] = outputDF
        except:
          outputDF = []
          text = df.values.tolist()
          if len(text) > 0:
            outputRowDF = self.helperRowTableToCorpus(text)
            #outputColDF = self.helperColTableToCorpus(text)
            outputDF.extend(outputRowDF)
            #outputDF.extend(outputColDF)
          if len(outputDF) > 0:
            corpus["corpus0"] = outputDF
    else:
      try:
          df = pd.ExcelFile(excelFile)
      except:
          if filepath.endswith('.xls'):
            df = pd.read_excel(filepath, engine='xlrd')
          else:
            df = pd.read_excel(filepath, engine='openpyxl')
      sheetNames = df.sheet_names
      output = []
      if len(sheetNames) > 0:
        for s in range(len(sheetNames)):
          outputDF = []
          with pd.ExcelFile(excelFile) as xls:
            data = pd.read_excel(xls, sheetNames[s])
          if sheetNames[s] != 'Evaluation Warning':
            text = data.values.tolist()
            if len(text) > 0:
              outputRowDF = self.helperRowTableToCorpus(text)
              #outputColDF = self.helperColTableToCorpus(text)
              outputDF.extend(outputRowDF)
              #outputDF.extend(outputColDF)
          if len(outputDF) > 0:
            corpus["corpus" + str(s)] = outputDF
    return corpus
  def helperRowTableToCorpus(self, textList):
    #cl = cleanGenText()
    cl = cleanText.cleanGenText()
    stopWords = ["NaN","Unnamed:","nan"]
    outputDF = []
    for line in textList:
      outputLine = []
      for words in line:
        words = str(words)
        if len(words) > 0:
          for word in words.split(" "):
            # remove specific stopwords for table: "NaN", "Unnamed: 0", row index: if the number appears first, it's just a row index; keep "KM1"
            if str(word) not in stopWords: # remove "NaN", "Unnamed:","nan"
              #word = cl.splitStickWords(word)
              word = cl.removePunct(word)
              word = " ".join(cl.removeStopWords(word))
              word = cl.removeTabWhiteSpaceNewLine(word)
              if len(word) > 1:
                if len(word.split(" ")) > 1:
                  for x in word.split(" "):
                    if len(x) > 1 and x.isnumeric()==False:
                      outputLine.append(x.lower())
                else:
                  if word.isnumeric() == False:
                    outputLine.append(word.lower())
      if len(outputLine) > 0:
        outputDF.append(outputLine)
    return outputDF
  def helperColTableToCorpus(self, dfList):
    #cl = cleanGenText()
    cl = cleanText.cleanGenText()
    stopWords = ["NaN","Unnamed:","nan"]
    outputDF = []
    # use the first length line as the column ref
    for pos in range(len(dfList[0])):
      outputLine = []
      for line in dfList:
        if pos < len(line):
          words = line[pos]
          words = str(words)
        else: words = ""
        if len(words) > 0:
          for word in words.split(" "):
            # remove specific stopwords for table: "NaN", "Unnamed: 0", row index: if the number appears first, it's just a row index; keep "KM1"
            if str(word) not in stopWords: # remove "NaN", "Unnamed:","nan"
              #word = cl.splitStickWords(word)
              word = cl.removePunct(word)
              word = " ".join(cl.removeStopWords(word))
              word = cl.removeTabWhiteSpaceNewLine(word)
              if len(word) > 1:
                if len(word.split(" ")) > 1:
                  for x in word.split(" "):
                    if len(x) > 1 and x.isnumeric()==False:
                      outputLine.append(x.lower())
                else:
                  if word.isnumeric() == False:
                    outputLine.append(word.lower())
      if len(outputLine) > 0:
        outputDF.append(outputLine)
    return outputDF
  # create a corpus
  def createCorpusText(self, corpusText):
    '''ex: "Tom is cat. Jerry is mouse."
    corpus = [["Tom", "is", "cat"], ["Jerry", "is", "mouse"]]'''
    # the output should be like this:
    '''texts = {
      "Paragraph 1": [["Cat", "is", "an","animal], ["Tom", "is", "cat"]],
      "Paragraph 2": [["Mouse", "is", "an", "animal"], ["Jerry", "is", "mouse"]]
    }
    '''
    # separate paragraph
    '''Ex: Cat is an animal. Tom is cat.

    Mouse is an animal.
    Jerry is mouse.'''
    texts = {}
    cl = cleanText.cleanGenText()
    #cl = cleanGenText()
    for pos in range(len(corpusText.split("\n\n"))):
      if len(corpusText.split("\n\n")[pos]) > 0:
        texts["Paragraph "+str(pos)] = []
        lines = corpusText.split("\n\n")[pos]
        for line in lines.split("\n"):
          for l in line.split("."):
            if len(l) > 0:
              cl.removeTabWhiteSpaceNewLine(l)
              l = l.lower()
              newL = []
              for word in l.split(" "):
                if len(word) > 0:
                  word = cl.removeStopWords(word)
                  for w in word:
                    if len(w) > 0 and w.isnumeric()==False:
                      newL.append(w)
              if len(newL)>0:
                texts["Paragraph "+str(pos)].append(newL)
        if len(texts["Paragraph "+str(pos)]) == 0:
          del texts["Paragraph "+str(pos)]
    return texts
  def selectParaForWC(self,corpus):
    ''' corpus should be in the format:
    corpus = [["Tom", "is", "cat"], ["Jerry", "is", "mouse"]]'''
    corSize, window, vector_size, sample, negative, epochs, sg = None, None, None, None, None, None, None
    corSize = len(corpus)
    # less than 2000
    if 0 < corSize < 2000:
      window=3.5
      vector_size=75
      sample=1e-3
      negative=10
      epochs=10
      sg=1
    # 2000 - 100000
    elif 2000 <= corSize < 100000:
      window=3.5
      vector_size=75
      sample=1e-5
      negative=10
      epochs=10
      sg=1
    elif 100000 <=corSize < 1000000:
      window=7.5
      vector_size=150
      sample=1e-5
      negative=10
      epochs=6
      sg=0
    return window, vector_size, sample, negative, epochs, sg
  def trainWord2Vec(self,nameFile,modelName,saveFolder,window=3.5,
                    vector_size=75,sample=1e-3,negative=10,epochs=10,sg=1):
    # if you dont have backup file, you can use again the nameFile just to increase the lenght of corpus
    jsonFile = ""
    jsonFile = openFile.openJsonFile(nameFile) # this is a corpus json file from an article
    cores = multiprocessing.cpu_count()
    combinedCorpus = []
    window, vector_size, sample, negative, epochs, sg = None, None, None, None, None, None
    if len(jsonFile) > 0:
      for key in jsonFile:
        combinedCorpus.extend(jsonFile[key])
      window, vector_size, sample, negative, epochs, sg = self.selectParaForWC(combinedCorpus)
      # # min_count=1 ensures all words are included
      '''w2vModel = Word2Vec(
                          min_count=1,
                          window=window,
                          vector_size=vector_size,
                          sample=sample,
                          alpha=0.03,
                          min_alpha=0.0007,
                          negative=negative,
                          workers=cores-1,
                          epochs = epochs,
                          sg=sg)'''
      #w2vModel = Word2Vec(vector_size=150, window=10, min_count=1, workers=4)
      accept = False
      while not accept:
        if window!=None and vector_size!=None and sample!=None and negative!=None and epochs!=None and sg!=None:
          try:
            w2vModel = Word2Vec(
                            min_count=1,
                            window=window,
                            vector_size=vector_size,
                            sample=sample,
                            alpha=0.03,
                            min_alpha=0.0007,
                            negative=negative,
                            workers=cores-1,
                            epochs = epochs,
                            sg=sg)
            w2vModel.build_vocab(combinedCorpus)
            w2vModel.train(combinedCorpus, total_examples=w2vModel.corpus_count, epochs=30)
            accept = True
          except:
            for key in jsonFile:
              combinedCorpus.extend(jsonFile[key])
            window, vector_size, sample, negative, epochs, sg = self.selectParaForWC(combinedCorpus)
            print("next is " + str(len(combinedCorpus)))
        else:
          print("no parameter to train")
          break
      #w2vModel.build_vocab(combinedCorpus)
      #w2vModel.train(combinedCorpus, total_examples=w2vModel.corpus_count, epochs=30)
      #w2vModel.save("/content/drive/MyDrive/CollectData/NER/word2Vec/TestExamples/models/wordVector_"+modelName+".model")
      #w2vModel.wv.save_word2vec_format("/content/drive/MyDrive/CollectData/NER/word2Vec/TestExamples/models/wordVector_"+modelName+".txt")
      w2vModel.save(saveFolder+"/"+modelName+".model")
      w2vModel.wv.save_word2vec_format(saveFolder+"/"+modelName+".txt")
      print("done w2v")
    else: print("no corpus to train")
    #return combinedCorpus
  def genSimilar(self,word,modelFile,n=10, cos_thres=0.7):
    # might not be a meaningful keyword
    #stopWords = ["show"]
    # same word but just plural nouns, tense
    simWords = [word+"s",word+"es",word+"ing",word+"ed"]
    model = KeyedVectors.load_word2vec_format(modelFile, binary = False) # model file in format txt
    results = model.most_similar(positive=[word],topn=n)
    #removeIndex = []
    #currN = copy.deepcopy(n)
    '''for r in range(len(results)):
      if len(results[r][0]) < 2:
        removeIndex.append(results[r])
      # remove the same word but just plural and singular noun and lower than the cos_thres
      elif results[r][0] == word:
        removeIndex.append(results[r])
      elif results[r][0] in simWords or float(results[r][1]) < cos_thres or results[r][0] in stopWords:
        removeIndex.append(results[r])
    for rem in removeIndex:
      results.remove(rem)
    while len(results)!=n and len(results) != 0:
      moreNewResult = model.most_similar(positive=[word],topn=currN+1)[-1]
      if moreNewResult not in results and len(moreNewResult[0])>1:
        if moreNewResult[0] not in stopWords and results[0] != word:
          results.append(moreNewResult)
      currN +=1'''
    return results
  # adding our model into spacy
  # this deals with command line; but instead of using it, we write python script to run command line
  def loadWordVec(self,modelName,wordVec):
    # modelName is the name you want to save into spacy
    # wordVec is the trained word2vec in txt format
    subprocess.run([sys.executable,
                    "-m",
                    "spacy",
                    "init-model",
                    "en",
                    modelName, # this modelName comes from the saved modelName of function trainWord2Vec
                    "--vectors-loc",
                    wordVec])
=======
'''WORD TO VECTOR'''
import pandas as pd
import json
import gensim
import spacy
from DefaultPackages import openFile, saveFile
from NER import cleanText
from gensim.models.keyedvectors import KeyedVectors
from gensim.test.utils import common_texts
from gensim.models.word2vec import Word2Vec
from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.test.utils import datapath, get_tmpfile
from gensim.models import Phrases
from gensim.models.phrases import Phraser
import sys
import subprocess
import os
# can try multiprocessing to run quicker
import multiprocessing
import copy
sys.setrecursionlimit(1000)
# creat folder word2Vec
#! mkdir /content/drive/MyDrive/CollectData/NER/word2Vec
# create word2vec model
#model = KeyedVectors.load_word2vec_format('/content/drive/MyDrive/CollectData/NER/word2Vec', binary=True)
'''Some notes for this model
sometimes when we do the corpus, there are some adverbs which are unnecessary but might be seen as
a similar word to the word we are finding, so can we try to preprocess text so that
we make the corpus more effective and only contains the important words. Then when we
train the model, the important words will be seen as important. Or
when we already have the similar list of words, we can remove the words in there
that are stopwords/unnecessary words.'''
### For more complex analysis, consider using sentence embedding models like "Doc2Vec" to represent the meaning of entire sentences instead of just individual words
class word2Vec():
  def __init__(self, nameFile=None, modelName=None):
    self.nameFile = nameFile
    self.modelName = modelName
    #self.nlp = spacy.load("en_core_web_lg")
    self.cl = cleanText.cleanGenText()
  def spacy_similarity(self, word):
    # when use word2vec, try medium or large is better
    # maybe try odc similarity?
    doc = self.nlp(word)
    for token1 in doc:
      for token2 in doc:
        print(token1.text, token2.text, token1.similarity(token2))
    pass
  # clean text before transform to corpus
  def cleanTextBeforeCorpus(self,oriText, doi=None):
    #cl = cleanText.cleanGenText()
    #cl = cleanGenText()
    output = ""
    alreadyRemoveDoi = False
    for word in oriText.split(" "):
      # remove DOI
      if doi != None and doi in oriText:
        if alreadyRemoveDoi == False:
          newWord = self.cl.removeDOI(word,doi)
          if len(newWord) > 0 and newWord != word:
            alreadyRemoveDoi = True
            word = newWord
      # remove punctuation
      # split the sticked words
      #word = cl.splitStickWords(word)
      # remove punctuation
      word = self.cl.removePunct(word,True)
      # remove URL
      word = self.cl.removeURL(word)
      # remove HTMLTag
      word = self.cl.removeHTMLTag(word)
      # remove tab, white space, newline
      word = self.cl.removeTabWhiteSpaceNewLine(word)
      # optional: remove stopwords
      #word = cl.removeStopWords(word)
      if len(word)>0:
        output += word + " "
    return output
  def cleanAllTextBeforeCorpus(self, allText, doi=None):
    cleanOutput = ""
    remove = "Evaluation Warning: The document was created with Spire.Doc for Python."
    if len(allText) > 0:
      corpusText = allText.split("\n\n")
      for pos in range(len(corpusText)):
        lines = corpusText[pos]
        if len(lines) > 0:
          for line in lines.split("\n"):
            if remove in line:  line = line.replace(remove, "")
            clean_text = self.cleanTextBeforeCorpus(line, doi)
            cleanOutput += clean_text + "\n"
          cleanOutput += "\n\n"
    return cleanOutput
  import urllib.parse, requests

  def tableTransformToCorpusText(self, df, excelFile=None):
    # PDF, Excel, WordDoc
    #cl = cleanText.cleanGenText()
    corpus = {}
      # PDF or df
    if excelFile == None:
      if len(df) > 0:
        try:
          for i in range(len(df)):
            # each new dimension/page is considered to be a sentence which ends with the period.
            # each new line is a new list, and each new df is a new corpus
            outputDF = []
            text = df[i].values.tolist()
            if len(text) > 0:
              outputRowDF = self.helperRowTableToCorpus(text)
              #outputColDF = self.helperColTableToCorpus(text)
              outputDF.extend(outputRowDF)
              #outputDF.extend(outputColDF)
            if len(outputDF) > 0:
              corpus["corpus" + str(i)] = outputDF
        except:
          outputDF = []
          text = df.values.tolist()
          if len(text) > 0:
            outputRowDF = self.helperRowTableToCorpus(text)
            #outputColDF = self.helperColTableToCorpus(text)
            outputDF.extend(outputRowDF)
            #outputDF.extend(outputColDF)
          if len(outputDF) > 0:
            corpus["corpus0"] = outputDF
    else:
      try:
          df = pd.ExcelFile(excelFile)
      except:
          if excelFile.endswith('.xls'):
            df = pd.read_excel(excelFile, engine='xlrd')
          else:
            df = pd.read_excel(excelFile, engine='openpyxl')    
      sheetNames = df.sheet_names
      output = []
      if len(sheetNames) > 0:
        for s in range(len(sheetNames)):
          outputDF = []
          with pd.ExcelFile(excelFile) as xls:
            data = pd.read_excel(xls, sheetNames[s])
          if sheetNames[s] != 'Evaluation Warning':
            text = data.values.tolist()
            if len(text) > 0:
              outputRowDF = self.helperRowTableToCorpus(text)
              #outputColDF = self.helperColTableToCorpus(text)
              outputDF.extend(outputRowDF)
              #outputDF.extend(outputColDF)
          if len(outputDF) > 0:
            corpus["corpus" + str(s)] = outputDF
    return corpus
  def helperRowTableToCorpus(self, textList):
    #cl = cleanGenText()
    #cl = cleanText.cleanGenText()
    stopWords = ["NaN","Unnamed:","nan"]
    outputDF = []
    for line in textList:
      outputLine = []
      for words in line:
        words = str(words)
        if len(words) > 0:
          for word in words.split(" "):
            # remove specific stopwords for table: "NaN", "Unnamed: 0", row index: if the number appears first, it's just a row index; keep "KM1"
            if str(word) not in stopWords: # remove "NaN", "Unnamed:","nan"
              #word = cl.splitStickWords(word)
              word = self.cl.removePunct(word)
              word = " ".join(self.cl.removeStopWords(word))
              word = self.cl.removeTabWhiteSpaceNewLine(word)
              if len(word) > 1:
                if len(word.split(" ")) > 1:
                  for x in word.split(" "):
                    if len(x) > 1 and x.isnumeric()==False:
                      outputLine.append(x.lower())
                else:
                  if word.isnumeric() == False:
                    outputLine.append(word.lower())
      if len(outputLine) > 0:
        outputDF.append(outputLine)
    return outputDF
  def helperColTableToCorpus(self, dfList):
    #cl = cleanGenText()
    #cl = cleanText.cleanGenText()
    stopWords = ["NaN","Unnamed:","nan"]
    outputDF = []
    # use the first length line as the column ref
    for pos in range(len(dfList[0])):
      outputLine = []
      for line in dfList:
        if pos < len(line):
          words = line[pos]
          words = str(words)
        else: words = ""
        if len(words) > 0:
          for word in words.split(" "):
            # remove specific stopwords for table: "NaN", "Unnamed: 0", row index: if the number appears first, it's just a row index; keep "KM1"
            if str(word) not in stopWords: # remove "NaN", "Unnamed:","nan"
              #word = cl.splitStickWords(word)
              word = self.cl.removePunct(word)
              word = " ".join(self.cl.removeStopWords(word))
              word = self.cl.removeTabWhiteSpaceNewLine(word)
              if len(word) > 1:
                if len(word.split(" ")) > 1:
                  for x in word.split(" "):
                    if len(x) > 1 and x.isnumeric()==False:
                      outputLine.append(x.lower())
                else:
                  if word.isnumeric() == False:
                    outputLine.append(word.lower())
      if len(outputLine) > 0:
        outputDF.append(outputLine)
    return outputDF
  # create a corpus
  def createCorpusText(self, corpusText):
    '''ex: "Tom is cat. Jerry is mouse."
    corpus = [["Tom", "is", "cat"], ["Jerry", "is", "mouse"]]'''
    # the output should be like this:
    '''texts = {
      "Paragraph 1": [["Cat", "is", "an","animal], ["Tom", "is", "cat"]],
      "Paragraph 2": [["Mouse", "is", "an", "animal"], ["Jerry", "is", "mouse"]]
    }
    '''
    # separate paragraph
    '''Ex: Cat is an animal. Tom is cat.

    Mouse is an animal.
    Jerry is mouse.'''
    texts = {}
    #cl = cleanText.cleanGenText()
    #cl = cleanGenText()
    corpus = corpusText.split("\n\n")
    for pos in range(len(corpus)):
      if len(corpus[pos]) > 0:
        texts["Paragraph "+str(pos)] = []
        lines = corpus[pos]
        for line in lines.split("\n"):
          for l in line.split("."):
            if len(l) > 0:
              l = self.cl.removeTabWhiteSpaceNewLine(l)
              l = l.lower()
              newL = []
              for word in l.split(" "):
                if len(word) > 0:
                  word = self.cl.removeStopWords(word)
                  for w in word:
                    if len(w) > 0 and w.isnumeric()==False:
                      newL.append(w)
              if len(newL)>0:
                texts["Paragraph "+str(pos)].append(newL)
        if len(texts["Paragraph "+str(pos)]) == 0:
          del texts["Paragraph "+str(pos)]
    return texts

  def selectParaForWC(self, corpus):
    """
    corpus = [["Tom", "is", "cat"], ["Jerry", "is", "mouse"]]
    Heuristically determine Word2Vec parameters.
    """
    corSize = len(corpus)
    
    if corSize == 0:
        return None, None, None, None, None, None

    # Adjust parameters based on corpus size
    if corSize < 2000:
        # Small corpus — need high generalization
        window = 3
        vector_size = 100
        sample = 1e-3
        negative = 5
        epochs = 20
        sg = 1  # Skip-gram preferred for rare words
    elif corSize < 10000:
        window = 5
        vector_size = 150
        sample = 1e-4
        negative = 10
        epochs = 20
        sg = 1
    elif corSize < 100000:
        window = 7
        vector_size = 200
        sample = 1e-5
        negative = 15
        epochs = 15
        sg = 1
    elif corSize < 500000:
        window = 10
        vector_size = 250
        sample = 1e-5
        negative = 15
        epochs = 10
        sg = 0  # CBOW is okay when data is large
    else:
        # Very large corpus
        window = 12
        vector_size = 300
        sample = 1e-6
        negative = 20
        epochs = 5
        sg = 0

    return window, vector_size, sample, negative, epochs, sg
  

  def trainWord2Vec(self,nameFile,modelName,saveFolder,window=None,
                    vector_size=None,sample=None,negative=None,epochs=None,sg=None):
    jsonFile = ""
    jsonFile = openFile.openJsonFile(nameFile) # this is a corpus json file from an article
    if not jsonFile:
        print("No corpus to train")
        return
    cores = multiprocessing.cpu_count()
    combinedCorpus = []
    for key in jsonFile:
      combinedCorpus.extend(jsonFile[key])
    # detect phrase before choosing parameters
    phrases = Phrases(combinedCorpus, min_count=2, threshold=10)
    bigram = Phraser(phrases)
    combinedCorpus = [bigram[sent] for sent in combinedCorpus]

    if window==None and vector_size==None and sample==None and negative==None and epochs==None and sg==None:   
      window, vector_size, sample, negative, epochs, sg = self.selectParaForWC(combinedCorpus)
    # # min_count=1 ensures all words are included
    #w2vModel = Word2Vec(vector_size=150, window=10, min_count=1, workers=4)
    accept = False
    # add retry limit because if training keeps failing (bad corpus or corrupted input), it’ll keep retrying without limit.
    retries = 0
    while not accept and retries < 3:
      if window!=None and vector_size!=None and sample!=None and negative!=None and epochs!=None and sg!=None:
        try:
          w2vModel = Word2Vec(
                          min_count=1,
                          window=window,
                          vector_size=vector_size,
                          sample=sample,
                          alpha=0.03,
                          min_alpha=0.0007,
                          negative=negative,
                          workers=cores-1,
                          epochs = epochs,
                          sg=sg)
          w2vModel.build_vocab(combinedCorpus)
          w2vModel.train(combinedCorpus, total_examples=w2vModel.corpus_count, epochs=epochs)
          accept = True
        except Exception as e:
          print(f"Retry #{retries+1} failed: {e}")
          retries +=1
      else:
        print("no parameter to train")
        break
    #w2vModel.build_vocab(combinedCorpus)
    #w2vModel.train(combinedCorpus, total_examples=w2vModel.corpus_count, epochs=30)
    #w2vModel.save("/content/drive/MyDrive/CollectData/NER/word2Vec/TestExamples/models/wordVector_"+modelName+".model")
    #w2vModel.wv.save_word2vec_format("/content/drive/MyDrive/CollectData/NER/word2Vec/TestExamples/models/wordVector_"+modelName+".txt")
    w2vModel.save(saveFolder+"/"+modelName+".model")
    w2vModel.wv.save_word2vec_format(saveFolder+"/"+modelName+".txt")
    print("done w2v")
    #return combinedCorpus
  def updateWord2Vec(self, modelPath, newCorpus, saveFolder=None):
    if not newCorpus:
        raise ValueError("New corpus is empty!")

    model = Word2Vec.load(modelPath)

    # Phrase detection on new data
    phrases = Phrases(newCorpus, min_count=2, threshold=10)
    bigram = Phraser(phrases)
    newCorpus = [bigram[sent] for sent in newCorpus]

    # Update vocab & retrain
    model.build_vocab(newCorpus, update=True)
    model.train(newCorpus, total_examples=len(newCorpus), epochs=model.epochs)

  def genSimilar(self,word,modelFile,n=10, cos_thres=0.7):
    # might not be a meaningful keyword
    #stopWords = ["show"]
    # same word but just plural nouns, tense
    simWords = [word+"s",word+"es",word+"ing",word+"ed"]
    model = KeyedVectors.load_word2vec_format(modelFile, binary = False) # model file in format txt
    results = model.most_similar(positive=[word],topn=n)
    #removeIndex = []
    #currN = copy.deepcopy(n)
    '''for r in range(len(results)):
      if len(results[r][0]) < 2:
        removeIndex.append(results[r])
      # remove the same word but just plural and singular noun and lower than the cos_thres
      elif results[r][0] == word:
        removeIndex.append(results[r])
      elif results[r][0] in simWords or float(results[r][1]) < cos_thres or results[r][0] in stopWords:
        removeIndex.append(results[r])
    for rem in removeIndex:
      results.remove(rem)
    while len(results)!=n and len(results) != 0:
      moreNewResult = model.most_similar(positive=[word],topn=currN+1)[-1]
      if moreNewResult not in results and len(moreNewResult[0])>1:
        if moreNewResult[0] not in stopWords and results[0] != word:
          results.append(moreNewResult)
      currN +=1'''
    return results
  # add more data to existing word2vec model
  def updateWord2Vec(self, modelPath, newCorpus, saveFolder=None):
    if not newCorpus:
        raise ValueError("New corpus is empty!")

    model = Word2Vec.load(modelPath)

    # Phrase detection on new data
    phrases = Phrases(newCorpus, min_count=2, threshold=10)
    bigram = Phraser(phrases)
    newCorpus = [bigram[sent] for sent in newCorpus]

    # Update vocab & retrain
    model.build_vocab(newCorpus, update=True)
    model.train(newCorpus, total_examples=len(newCorpus), epochs=model.epochs)

    # Save updated model
    if saveFolder:
        os.makedirs(saveFolder, exist_ok=True)
        name = os.path.basename(modelPath).replace(".model", "_updated.model")
        model.save(f"{saveFolder}/{name}")
        print(f"🔁 Model updated and saved to {saveFolder}/{name}")
    else:
        model.save(modelPath)
        print(f"🔁 Model updated and overwritten at {modelPath}")
  
  # adding our model into spacy
  # this deals with command line; but instead of using it, we write python script to run command line
  def loadWordVec(self,modelName,wordVec):
    # modelName is the name you want to save into spacy
    # wordVec is the trained word2vec in txt format
    subprocess.run([sys.executable,
                    "-m",
                    "spacy",
                    "init-model",
                    "en",
                    modelName, # this modelName comes from the saved modelName of function trainWord2Vec
                    "--vectors-loc",
                    wordVec])
>>>>>>> 597aa7c (WIP: Save local changes which mainly updated appUI  before moving to UpdateAppUI)
    print("done")