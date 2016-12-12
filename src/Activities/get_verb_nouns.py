from nltk.corpus import wordnet as wordnet
from nltk.corpus import sentiwordnet as swn
from nltk.stem import WordNetLemmatizer
import nltk
import json
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet as wn
from nltk.stem.porter import PorterStemmer
stopwords = {}
filetype = 'txt'
count = 0
wordnet_lemmatizer = WordNetLemmatizer()
with open('stopwords.txt', 'rU') as f:
    for line in f:
        stopwords[line.strip()] = 1
for i in range(0, 1):
    count = count + 1
    sentences = []
    tagged_tuples = []
    NVpairs = []
    with open('../DataModels/Reviews/review14.txt', 'r') as f:
        my_list = [line.decode('unicode-escape').rstrip(u'\n') for line in f]
        '''POS tagging of a sentence.'''
    print 'read file' ,my_list[:1]
    for list in my_list:
        tokens = nltk.tokenize.sent_tokenize(list)
        for tok in tokens:
            sentences.append(tok)
    print 'token',tokens[:1]
    for sent in sentences:
        tokens = nltk.word_tokenize(sent)
        tag_tuples = nltk.pos_tag(tokens)
        tagged_words = []
        for (string, tag) in tag_tuples:
            token = {'word': string, 'pos': tag}
            tagged_words.append(token)
        for i, tag_words in enumerate(tagged_words):
            pair = []
            if i + 1 < len(tagged_words) - 1 and tag_words['pos'].startswith('V') and len(tag_words['word']) > 3:
                if tagged_words[i + 1]['pos'].startswith('V'):
                    continue
                if i + 1 < len(tagged_words) - 1 and tagged_words[i + 1]['pos'].startswith('N') and len(
                        tagged_words[i + 1]['word']) > 3:
                    pair.append(tagged_words[i])
                    pair.append(tagged_words[i + 1])
                    NVpairs.append(tagged_words[i])
                    NVpairs.append(tagged_words[i + 1])
                    continue
                if i + 2 < len(tagged_words) - 1 and tagged_words[i + 2]['pos'].startswith('V'):
                    continue
                if i + 2 < len(tagged_words) - 1 and tagged_words[i + 2]['pos'].startswith('N') and len(
                        tagged_words[i + 2]['word']) > 3:
                    pair.append(tagged_words[i])
                    pair.append(tagged_words[i + 2])
                    NVpairs.append(tagged_words[i])
                    NVpairs.append(tagged_words[i + 2])
                    continue
                if i + 3 < len(tagged_words) - 1 and tagged_words[i + 3]['pos'].startswith('V'):
                    continue
                if i + 3 < len(tagged_words) - 1 and tagged_words[i + 3]['pos'].startswith('N') and len(
                        tagged_words[i + 3]['word']) > 3:
                    pair.append(tagged_words[i])
                    pair.append(tagged_words[i + 3])
                    # NVpairs.append(pair)
                    NVpairs.append(tagged_words[i])
                    NVpairs.append(tagged_words[i + 3])
                    continue
        porter_stemmer = PorterStemmer()
    print 'NVpairs', NVpairs[:5]
    for pair in NVpairs:
                pair['word'] = wordnet_lemmatizer.lemmatize(pair['word'])
                # pair['word'] = porter_stemmer.stem(pair['word'])
    with open('NVpair' + str(count) +'.txt', 'w') as f:
        json.dump(NVpairs, f)
    print 'wrote'

# for sent in sentences
#     tag_tuples = nltk.pos_tag(sent)
#     for (string, tag) in tag_tuples:
#         token = {'word':string, 'pos':tag}

tokens = nltk.sent_tokenize(my_list[1])

tagged_tuples = []
NVpairs = []

# for sent in sentences:
#     tokens = nltk.word_tokenize(sent)
#     tag_tuples = nltk.pos_tag(tokens)
#     tagged_words =[]
#     for (string, tag) in tag_tuples:
#         token = {'word':string, 'pos':tag}
#         tagged_words.append(token)
#     for i,tag_words in enumerate(tagged_words):
#         pair = []
#         if i+1 < len(tagged_words) - 1 and tag_words['pos'].startswith('V') and len(tag_words['word']) >3:
#             if tagged_words[i + 1]['pos'].startswith('V'):
#                 continue
#             if i+1 < len(tagged_words) - 1 and tagged_words[i+1]['pos'].startswith('N') and len(tagged_words[i+1]['word']) >3:
#                 pair.append(tagged_words[i])
#                 pair.append(tagged_words[i+1])
#                 NVpairs.append(tagged_words[i])
#                 NVpairs.append(tagged_words[i + 1])
#                 continue
#             if i+2 < len(tagged_words) - 1 and tagged_words[i + 2]['pos'].startswith('V'):
#                 continue
#             if i+2 < len(tagged_words) - 1 and tagged_words[i + 2]['pos'].startswith('N') and len(tagged_words[i + 2]['word']) >3:
#                 pair.append(tagged_words[i])
#                 pair.append(tagged_words[i + 2])
#                 NVpairs.append(tagged_words[i])
#                 NVpairs.append(tagged_words[i + 2])
#                 continue
#             if i+3 < len(tagged_words) - 1 and tagged_words[i + 3]['pos'].startswith('V'):
#                 continue
#             if i+3 < len(tagged_words) - 1 and tagged_words[i + 3]['pos'].startswith('N') and len(tagged_words[i + 3]['word']) >3:
#                 pair.append(tagged_words[i])
#                 pair.append(tagged_words[i + 3])
#                 # NVpairs.append(pair)
#                 NVpairs.append(tagged_words[i])
#                 NVpairs.append(tagged_words[i + 3])
#                 continue



print NVpairs[:50]


#     for i, tag_words in enumerate(sent):
#     pair = []
#     if tag_words['pos'].startswith('V'):
#         if tagged_words[i+1]['pos'].startswith('N'):
#             pair.append(tagged_words[i])
#             pair.append(tagged_words[i+1])
#             NVpairs.append(pair)
#             continue
#         if tagged_words[i + 2]['pos'].startswith('N'):
#             pair.append(tagged_words[i])
#             pair.append(tagged_words[i + 2])
#             NVpairs.append(pair)
#             continue
#         if tagged_words[i + 3]['pos'].startswith('N'):
#             pair.append(tagged_words[i])
#             pair.append(tagged_words[i + 3])
#             NVpairs.append(pair)
#             continue
#
# print NVpairs[:50]
#
#
#


# from nltk.corpus import wordnet as wordnet
# from nltk.corpus import sentiwordnet as swn
# import nltk
#
# stopwords = {}
# with open('stopwords.txt', 'rU') as f:
#     for line in f:
#         stopwords[line.strip()] = 1
#
# with open('review1.txt', 'r') as f:
#     my_list = [line.decode('unicode-escape').rstrip(u'\n') for line in f]
#
#     '''POS tagging of a sentence.'''
# tagged_words = []
# for list in my_list:
#     tokens = nltk.word_tokenize(list)
#     filteredWords = [word for word in tokens if word not in stopwords]
#     tag_tuples = nltk.pos_tag(filteredWords)
#     for (string, tag) in tag_tuples:
#         token = {'word':string, 'pos':tag}
#         tagged_words.append(token)
#
#
# print my_list[:2]
# print tagged_words[:2]
# NVpairs = []
# for i,tag_words in enumerate(tagged_words):
#     pair = []
#     if tag_words['pos'].startswith('V'):
#         if tagged_words[i+1]['pos'].startswith('N'):
#             pair.append(tagged_words[i])
#             pair.append(tagged_words[i+1])
#             NVpairs.append(pair)
#             continue
#         if tagged_words[i + 2]['pos'].startswith('N'):
#             pair.append(tagged_words[i])
#             pair.append(tagged_words[i + 2])
#             NVpairs.append(pair)
#             continue
#         if tagged_words[i + 3]['pos'].startswith('N'):
#             pair.append(tagged_words[i])
#             pair.append(tagged_words[i + 3])
#             NVpairs.append(pair)
#             continue
#
# print NVpairs[]
#


