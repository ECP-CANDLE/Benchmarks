import json

with open('config.json') as f:
   data = json.load(f)

print(data['data_loading']['data_path'])

