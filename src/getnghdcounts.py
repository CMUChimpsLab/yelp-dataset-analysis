import json
from collections import defaultdict
from csv import DictWriter, DictReader


words_per_nghd = json.load(open('count.json'))
# for i in range(5):
#     print words_per_nghd['count'][str(i)]
#     print words_per_nghd['neighborhood'][str(i)]
#     print words_per_nghd['categories'][str(i)]


# for i in range(580):
#     target_dict['neighborhood'] = words_per_nghd['neighborhood'][str(i)]
count = {}
full = []
# for line in DictReader(open('nghd_central_point.csv')):
#     print line['nghd']

for line in DictReader(open('nghd_central_point.csv')):
    empty = {}
    nghd = line['nghd']
    categories = []
    for i in range(582):
        if words_per_nghd['neighborhood'][str(i)] == nghd:
            category= {}
            category['name'] = nghd;
            category['count'] = words_per_nghd['count'][str(i)]
            category['latitude'] = line['lat']
            category['longitude'] = line['lon']
            category['category'] = " ".join(str(x) for x in words_per_nghd['categories'][str(i)])
            categories.append(category)
            empty['neighborhoods'] = category
            if category:
                full.append(category)


count['items'] = full
print json.dumps(count, indent=1)
# with open('outputs/tweets_per_nghd_words.json','w') as outfile:
#         json.dump(tweets_per_word,outfile, indent=2)
