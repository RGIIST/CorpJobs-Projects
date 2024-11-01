import json
import os

files = os.listdir("invoice_2")

text = ""
for file in files:
    if file.split('.')[-1]=='json':
        with open(f'invoice_2/{file}', 'rb') as f:
            data = json.load(f)
        text += data['content'] + "\n\n"

with open("DexNinja_dataset.txt", 'w') as f:
    f.write(text)