import json
import pandas as pd
with open('../DataModels/VerbsPairs/NVpair3.txt') as json_data:
    d = json.load(json_data)
count = 0
a = []
for key, elem in enumerate(d):
    # print key
    if count % 2 == 0:
        s = d[key]["word"]+ " "  + d[key + 1]["word"]
        a.append(s)
    count = count + 1

print len(a)
b = set(a)
print len(b)

def df_to_json(df, filename=''):
    x = df.reset_index().T.to_dict().values()
    if filename:
        with open(filename, 'w+') as f: f.write(json.dumps(x))
    return x
counts  = []
for item in b:
    count = {}
    x = 0
    for i in a:
        if item == i:
            x = x + 1
            count["c"] = x
            count["s"]= i
    counts.append(count)
print counts

df = pd.DataFrame(counts)
print df.info()
noun = []
df = df.sort(['c'], ascending=False)[:40]
d =  df.head(40)
df_to_json(df, 'nvc1.json')

def df_to_json(df, filename=''):
    x = df.reset_index().T.to_dict().values()
    if filename:
        with open(filename, 'w+') as f: f.write(json.dumps(x))
    return x