# reference:
# https://ayselaydin.medium.com/1-text-preprocessing-techniques-for-nlp-37544483c007
import re
import nltk
#nltk.download('stopwords')
#nltk.download()
from CollectData.DefaultPackages import openFile, saveFile
import json
from nltk.corpus import stopwords
from nltk.corpus.reader.api import wordpunct_tokenize
from nltk.tokenize import word_tokenize
#from wordsegment import load, segment
from wordsegment import load, segment
class cleanGenText():
  def __init__(self):
    #self.text = text
    load()
    pass
  def removePunct(self,text,KeepPeriod=False):
    punctuation = r'[^\w\s]'
    if KeepPeriod==True:
      punctuation = r'[^\w\s\.]' 
    return re.sub(punctuation, '', text)
  def removeURL(self,text):
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    return url_pattern.sub(r'', text)
  def removeHTMLTag(self,text):
    html_tags_pattern = r'<.*?>'
    return re.sub(html_tags_pattern, '', text)
  def removeTabWhiteSpaceNewLine(self,text):
    # remove \n or \t and unnecessary white space
    cleanText = text.replace("\n\n","")
    cleanText = text.replace("\n","")
    cleanText = cleanText.replace("\t","")
    cleanText = cleanText.strip()
    return cleanText
  def removeExtraSpaceBetweenWords(self,text):
    return re.sub(r'\s+', ' ',text).strip()  
  def removeStopWords(self,text):
    #extraUnwantedWords = ["resource","groups","https","table","online","figure","frequency","aslo","fig","shows","respectively"]
    filteredWord = []
    stopWords = set(list(set(stopwords.words('english'))))# + extraUnwantedWords)
    textWords = word_tokenize(text)
    for word in textWords:
      if word.lower() not in stopWords:
        filteredWord.append(word) # and w.isalpha()==True]
    return filteredWord
  def removeLowercaseBetweenUppercase(self,segment):
    # segment such as "Myanmar (formerly Burma)"
    # but not change anything for "Viet Nam"
    # for special cases:
        # the capital letter:
        # When there is a lowercase word between:
        # e.g: "Myanmar (formerly Burma)" can be "Myanmar", "Burma" instead of "myanmar formerly burma"        
        # When there is no lowercase word or uppercase words in a row:
        # e.g: "Viet Nam" can be "Viet Nam" or "viet nam", instead of "Viet", "Nam"
    outputUp = []
    segment = self.removeTabWhiteSpaceNewLine(segment)
    segments = segment.split(" ")
    for w in range(len(segments)):
      word = segments[w]
      cleanWord = self.removePunct(word)
      cleanWord = self.removeTabWhiteSpaceNewLine(cleanWord) 
      prevWord = ""
      if w > 0:
        prevWord = segments[w-1]
        cleanPreWord = self.removePunct(prevWord)
        cleanPreWord = self.removeTabWhiteSpaceNewLine(cleanPreWord)
      if cleanWord[0].isupper() == True: # check isupper of first letter of capital word
        if len(prevWord)>0 and prevWord[0].isupper() == True:
          outputUp[-1] += " " + cleanWord 
        else:
          outputUp.append(cleanWord)
    return outputUp    
  def textPreprocessing(self, text, keepPeriod=False):
    # lowercase
    #lowerText = self.text.lower()
    # remove punctuation & special characacters
    cleanText = self.removePunct(text, KeepPeriod=keepPeriod)
    # removal of URLs in text
    cleanText = self.removeURL(cleanText)
    # removal of HTML Tags
    cleanText = self.removeHTMLTag(cleanText)
    # remove \n or \t and unnecessary white space
    cleanText = self.removeTabWhiteSpaceNewLine(cleanText)
    # stop-words removal
    filteredWord = self.removeStopWords(cleanText)
    # a sentence or the capital word behind a period "."
    return cleanText, filteredWord
  #generateNewChar = textPreprocessing("/content/drive/MyDrive/CollectData/NER/CountriesNameNCBI.json")
  #saveFile.saveFile("/content/drive/MyDrive/CollectData/NER/NewCharCountriesNameNCBI.json", json.dumps(generateNewChar))
  def splitStickWords(self,word):
    #output = []
    split_words = segment(word)
    '''for w in split_words:
      pos = word.lower().find(w)
      if word[pos].isupper() == True:
        output.append(w[0].upper() + w[1:])   
      else:
        output.append(w)
      if pos >=0:
        if pos+len(w)<len(word):
          if word[pos+len(w)] == ".":   
            output[-1] = output[-1] + "."  '''
    return " ".join(split_words)
  def removeDOI(self, word, doiLink=None):
    # if they have the word DOI in that: ex: 1368598DOI after general clean
    if "DOI" in word:
      word = word.replace(word,"")
    # if they have the link DOI in that: ex: 10.1007s004390161742yORIGINAL, but we still split the word
    if doiLink != None:
      w = self.splitStickWords(word)
      cleanDOI = self.removePunct(doiLink)
      if cleanDOI in w:
        word = w.replace(cleanDOI,"")
    return word