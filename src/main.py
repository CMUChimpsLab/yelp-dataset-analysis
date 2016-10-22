import pandas as pd
import numpy as np
import matplotlib as plt

df = pd.read_csv('..//yelp_dataset//yelp_academic_dataset_business.csv', low_memory=False) #Reading the dataset in a dataframe using Pandas
a = (df.business_id[(df.city == 'Pittsburgh')|(df.city == 'Carnegie')])


np = np.array(a, dtype=pd.Series)

print(type(np))


dfs = pd.DataFrame()

data = pd.read_table('..//yelp_dataset//yelp_academic_dataset_review.csv', sep='|', chunksize=4)
for chunk in data:
    print(chunk.columns)
    dfs.append(chunk)
    print dfs.info()
    print type(dfs)
    break



