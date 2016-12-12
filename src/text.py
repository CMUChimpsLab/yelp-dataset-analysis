import json
import pandas as pd
import seaborn as sns
from csv import DictReader
import util
import funcy as fp
import re
import gensim
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from gensim import corpora, models, similarities
from gensim.corpora import Dictionary, MmCorpus

import nltk
import pyLDAvis.gensim as gensimvis
import pyLDAvis




'''
load_data(filepath)
Given a filepath to a JSON file, loads in the file and formats the JSON
'''

fileheading = '../yelp_dataset/yelp_academic_dataset_'

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

tt = Getnghd()

def applyneighborhood(row):
    return tt.getnghd((row['latitude'],row['longitude']))



def load_data(filepath):
    data = []

    # Open file and read in line by line
    with open(filepath) as file:
        for line in file:
            # Strip out trailing whitespace at the end of the line
            data.append(json.loads(line.rstrip()))

    return data

def get_data(line, cols):
    d = json.loads(line.rstrip())
    return dict((key, d[key]) for key in cols)

data = load_data('/Users/Neeraj/PycharmProjects/Yelp_data_mining/yelp_dataset/yelp_academic_dataset_business.json')

business_df = pd.DataFrame.from_dict(data)

business_df = business_df[business_df.city == 'Las Vegas']
business_df['neighborhood'] = business_df.apply(applyneighborhood, axis =1)
business_df['neighborhood'] = business_df['neighborhood'].astype('str')
print business_df[business_df.business_id== 'QoDa50dc7g62xciFygXB9w']


print business_df.info()

print business_df.head()

reviews = load_data('/Users/Neeraj/PycharmProjects/Yelp_data_mining/yelp_dataset/yelp_academic_dataset_review.json')
review_df = pd.DataFrame.from_dict(reviews)

review_df = review_df[review_df.business_id.isin(business_df.business_id)]
print review_df.head()
print review_df.info()



for ind, val in review_df.iterrows():
    with open('corpus.txt', 'a+') as f:
        f.write(val['text'].encode("utf-8"))