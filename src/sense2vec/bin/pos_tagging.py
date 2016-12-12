from spacy.en import English
import spacy
import os
import io

import re

LABELS = {
    'ENT': 'ENT',
    'PERSON': 'ENT',
    'NORP': 'ENT',
    'FAC': 'ENT',
    'ORG': 'ENT',
    'GPE': 'ENT',
    'LOC': 'ENT',
    'LAW': 'ENT',
    'PRODUCT': 'ENT',
    'EVENT': 'ENT',
    'WORK_OF_ART': 'ENT',
    'LANGUAGE': 'ENT',
    'DATE': 'DATE',
    'TIME': 'TIME',
    'PERCENT': 'PERCENT',
    'MONEY': 'MONEY',
    'QUANTITY': 'QUANTITY',
    'ORDINAL': 'ORDINAL',
    'CARDINAL': 'CARDINAL'
}



nlp = spacy.load('en')

def whatisthis(s):
    if isinstance(s, str):
        print "ordinary string"
    elif isinstance(s, unicode):
        print "unicode string"
    else:
        print "not a string"

def transform_doc(doc):
    for ent in doc.ents:
        if isinstance(ent, unicode):
            ent.merge(ent.root.tag_, ent.text, LABELS[ent.label_])
    for np in doc.noun_chunks:
        while len(np) > 1 and np[0].dep_ not in ('advmod', 'amod', 'compound'):
            np = np[1:]
        np.merge(np.root.tag_, np.text, np.root.ent_type_)
    strings = []
    for sent in doc.sents:
        if sent.text.strip():
            strings.append(' '.join(represent_word(w) for w in sent if not w.is_space))
            # print(strings)
    if strings:
        return ''.join(strings) + '\n'
    else:
        return ''


def represent_word(word):
    if word.like_url:
        return '%%URL|X'
    text = re.sub(r'\s', '_', word.text)
    tag = LABELS.get(word.ent_type_, word.pos_)
    if not tag:
        tag = '?'
    return text + '|' + tag

def print_fine_pos(token):
    return (token.tag_)

def pos_tags(sentence):
    tokens = nlp(sentence)
    # print transform_doc(tokens)
    tags = []
    for tok in tokens:
        tags.append((tok,print_fine_pos(tok)))
    return tags

count = 0
with open('corpus.txt', 'r') as f:
    for str1 in f:
        str1 = str1.decode('utf-8')
    # print pos_tags(line)
        tokens = nlp(str1)
    # print transform_doc(tokens)
    # print "hey"
        with open('ps.txt', 'a+') as f:
            f.write(transform_doc(tokens).encode('utf-8'))




#
# print my_list[8]
# # print pos_tags(my_list[8])
#         for line in f:
#             if (count < 10):
#                 count = count + 1
#                 print pos_tags(line)
            # print doc.ents


# a = "When people asked me what I had at Meat and Potatoes it was really easy for me to respond, meat and potatoes!  That answer was lame the minute I heard myself respond that way.  I'm just gonna have to describe to you how good my meal was there!"
# print pos_tags(a)
