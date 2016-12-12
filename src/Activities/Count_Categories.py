import json
import pandas as pd
from csv import DictReader
import util
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

business_df = business_df[business_df.city == 'Pittsburgh']
business_df['neighborhood'] = business_df.apply(applyneighborhood, axis =1)
business_df['neighborhood'] = business_df['neighborhood'].astype('str')
print business_df[business_df.business_id== 'QoDa50dc7g62xciFygXB9w']
print business_df[business_df.neighborhood== 'Shadyside']

print business_df.info()

print business_df.head()

reviews = load_data('/Users/Neeraj/PycharmProjects/Yelp_data_mining/yelp_dataset/yelp_academic_dataset_review.json')
review_df = pd.DataFrame.from_dict(reviews)

review_df = review_df[review_df.business_id.isin(business_df.business_id)]
print review_df.head()
print review_df.info()





# col = 'business_id'
# count = 0
# for ind, val in business_df.business_id.iteritems():
#     print ind, val
# print business_df.business_id.count()
# Format the attributes as a list of dict objects
attributes_dict = [{'attributes': x} for x in business_df['attributes'].values]

# Create a DataFrame with json_normalize
attributes_df = pd.io.json.json_normalize(attributes_dict)

# Convert objects to a numeric datatype if possible
attributes_df = attributes_df.convert_objects(convert_numeric=True)

non_numeric_attributes = attributes_df.select_dtypes(include=['object']).columns
numeric_attributes = attributes_df.select_dtypes(exclude=['object']).columns
attributes_df[non_numeric_attributes].head()

# Create dummy variables for non-numeric attributes
dummy_vars = pd.get_dummies(attributes_df[non_numeric_attributes])

# Drop non-numeric attributes from attributes_df
attributes_df = attributes_df.drop(non_numeric_attributes, axis=1)

# Add the dummy variables to attributes_df
attributes_df = pd.merge(attributes_df, dummy_vars, left_index=True, right_index=True)

# Save the list of attributes for future use
attributes = attributes_df.columns.values

# Merge it with our original dataframe
business_df = pd.merge(business_df, attributes_df, left_index = True, right_index = True)

# Drop our original attributes column that is no longer needed
business_df = business_df.drop('attributes', axis=1)

# Create dummy variables for categories
categories_df = business_df['categories'].str.join(sep=',').str.get_dummies(sep=',')

# Save the list of categories for future use
categories = categories_df.columns.values
# np.set_printoptions(threshold='nan')

# print categories
# Merge it with our original dataframe
business_df = pd.merge(business_df, categories_df, left_index = True, right_index = True)

business_df['categories'] = business_df['categories'].apply(lambda x: tuple(x))

# pittsburgh_df = business_df[business_df.city == 'Pittsburgh']

# print business_df['Chinese'].sum()

# Count the number of restaurants are in each category
# restaurant_category_counts = business_df[business_df['Restaurants'] == 1][categories].sum()
#
#
# # Sort the category counts
# sorted_categories = restaurant_category_counts.order(ascending=False)
#
# # Print the top 20
# print sorted_categories[:20]

final_df = business_df.groupby(['neighborhood', 'categories']).size().reset_index(name='count')
print final_df.head()


with open("count.json", "w+") as output_file:
    output_file.write(final_df.to_json())

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
