def openFile(file):
  with open(file) as f:
    openFile = f.read()
  return openFile

def openJsonFile(file):
  import json
  # Opening JSON file
  with open(file, 'r') as openfile:
    # Reading from json file
    json_object = json.load(openfile)
  return json_object
