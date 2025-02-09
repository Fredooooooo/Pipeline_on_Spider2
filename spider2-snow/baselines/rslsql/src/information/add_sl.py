import json

with open('src/information/ppl_dev.json', 'r') as f:
    ppls = json.load(f)

with open('src/information/schema.json', 'r') as f:
    datas = json.load(f)

# for i in range(len(ppls)):
# Try first 128 questions for now
for i in range(128):
    data = datas[i]
    ppls[i]['tables'] = data['tables']
    ppls[i]['columns'] = data['columns']

with open('src/information/ppl_dev.json', 'w') as f:
    json.dump(ppls, f, indent=4, ensure_ascii=False)


