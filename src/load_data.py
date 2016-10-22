'''
Script to load data from file. The exact columns to load should
be specified.
----------
'''
from gensim import corpora, models, similarities
import pandas as pd
import simplejson as json
from datetime import datetime
from csv import DictReader
import sys
import util
import nltk
from nltk.corpus import stopwords
import itertools
from sklearn.cross_validation import train_test_split

print '**Loading data...'

# LOAD DATA FOR TYPE = dataset_type
fileheading = '/Users/Neeraj/Downloads/yelp_dataset_challenge_academic_dataset/yelp_academic_dataset_'

class Getnghd():
    def __init__(self):
        self.bins_to_nghds = {}
        self.nghdName = {}
        for line in DictReader(open('../xls/point_map.csv')):
            self.bins_to_nghds[(float(line['lat']), float(line['lon']))] = line['nghd']
            if line['nghd'] not in self.nghdName:
                self.nghdName[line['nghd']] = 0
    def getnghd(self, coordinate):
        bin = util.round_latlon(coordinate[0], coordinate[1])
        if bin in self.bins_to_nghds:
            return self.bins_to_nghds[bin]
        else:
            return 'Outside Neeraj'
    def getnghdName(self):
        return self.nghdName.keys()



def get_data(line, cols):
    d = json.loads(line)
    return dict((key, d[key]) for key in cols)

tt = Getnghd()
def applyneighborhood(row):
    return tt.getnghd((row['latitude'],row['longitude']))

# Load business data
cols = ('business_id', 'name', 'categories', 'city', 'latitude', 'longitude')
with open(fileheading + 'business.json') as f:
    df_business = pd.DataFrame(get_data(line, cols) for line in f)
df_business = df_business.sort('business_id')
df_business.index = range(len(df_business))
df_business.info()
pittsburgh = df_business[df_business.city == 'Pittsburgh']
pittsburgh['neighborhood'] = pittsburgh.apply(applyneighborhood, axis =1)
print pittsburgh.info()
# print pittsburgh['new_col']
# print business_ids.values
# print df_business[['latitude','longitude']]
print pittsburgh.groupby('neighborhood').size().sort_values(ascending=False)
print pittsburgh[['name', 'neighborhood', 'longitude', 'latitude']]




# Load review data
cols = ( 'business_id','text')
with open(fileheading + 'review.json') as f:
    df_review = pd.DataFrame(get_data(line, cols) for line in f)
    print df_review.info
df_review = df_review[df_review.business_id.isin(df_review.business_id)]

def get_review():
    return df_review

# print(df_review[df_review.business_id == df_business.business_id[(df_business.city == 'Pittsburgh')|(df_business.city == 'Carnegie')]])
df_pitt = (df_review[df_review['business_id'].isin(df_business.business_id[(df_business.city == 'Pittsburgh')|(df_business.city == 'Carnegie')])])
# documents = dfList
# texts = [[word for word in document.lower().split() if word not in stopwords] for document in documents]
# dictionary = corpora.Dictionary(texts)
# corpus = [dictionary.doc2bow(text) for text in texts]
# #
# lda = models.ldamodel.LdaModel(corpus=corpus, id2word=dictionary, num_topics=100, update_every=1, chunksize=10000, passes=5)
# print lda.print_topics(5)
data_load_time = datetime.now()
print 'Data was loaded at ' + data_load_time.time().isoformat()
