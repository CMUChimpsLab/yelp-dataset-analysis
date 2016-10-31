import json
import pandas as pd
with open('NVpair16.txt') as json_data:
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

df = df.sort(['c'], ascending=False)
print df

