from CollectData.DefaultPackages import openFile, saveFile
import json
class convertToJson():
  def __init__(self,nameFile):
    self.nameFile = nameFile
  # json location of all countries
  def convertTextCommaToJson(self, countryJson):
    file = openFile.openFile(self.nameFile)
    if "worldcountries" in self.nameFile.lower():
      countryJson["worldCountries"] = []
      for country in file.split(","):
        if "\n" in country: country = country.split("\n")[0]
        countryJson["worldCountries"].append(country)
    elif "historycountries" in self.nameFile.lower():
      countryJson["historyCountries"] = []
      for country in file.split(","):
        if "\n" in country: country = country.split("\n")[0]
        countryJson["historyCountries"].append(country)
    return countryJson
    #countryJson = {}
    #for name in ["worldCountriesNameNCBI","historyCountriesNameNCBI"]:
      #countryJson = convertTextCommaToJson("/content/drive/MyDrive/Customers/HovhannesSahakyanProject/others/"+name+".txt", countryJson)
    #saveFile.saveFile("/content/drive/MyDrive/CollectData/NER/CountriesNameNCBI.json", json.dumps(countryJson))