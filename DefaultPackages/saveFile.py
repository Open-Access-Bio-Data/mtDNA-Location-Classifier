import json
def saveFile(name,content):
  Name = name
  fi = open(Name, "a")
      # Add new change in to saved file
  with open(Name, "w") as external_file:
    add_text = content
    print(add_text, file=external_file)
    external_file.close()
def saveJsonFile(name,content):
  saveFile(name,json.dumps(content))    