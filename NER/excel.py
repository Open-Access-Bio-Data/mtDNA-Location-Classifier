import pandas as pd
class excel(): # using pandas to read table?
  def __init__(self, excelFile):
    self.excelFile = excelFile
  def extractCSV(self):
    df = pd.read_csv(self.excelFile)
    return df
  def getSheetNames(self):
    df = pd.ExcelFile(self.excelFile)
    sheetNames = df.sheet_names
    return sheetNames  
  def extractXLSX(self, sheet_name=None):
    df = pd.ExcelFile(self.excelFile)
    sheetNames = df.sheet_names
    output = ""
    if sheet_name == None: # default will be first sheet
      #df = pd.read_excel(self.excelFile)
      with pd.ExcelFile(self.excelFile) as xls:
        output = pd.read_excel(xls, sheetNames[0])
    elif sheet_name == "all":
      for i in range(len(sheetNames)):
        with pd.ExcelFile(self.excelFile) as xls:
          if len(output) == 0:
            output = pd.read_excel(xls, sheetNames[i])
          else:
            df = pd.read_excel(xls, sheetNames[i])
            output = pd.concat([output, df], ignore_index=True)
    else:
      with pd.ExcelFile(self.excelFile) as xls:
        output = pd.read_excel(xls, sheet_name)
    return output