# yelp-dataset-analysis
Integrating Yelp Dataset in building our Neighbourhood guides

Instructions
Run requirements.txt to install all dependencies

src/Activities/Count_Categories.py 
Gives business categories of all the neighborhood using Yelp data

src/Activities/get_verb_noun.py
Create noun verb pairs for reviews

src/Activities/CountNounVerbPair.py
Gives you the top 40 activities for a place using the file generated in Activities/get_verb_noun.py

src/Activities/extract_reviews.py
Extract all the review to create corpus for word2vec model


Sense2vec
src/Sense2vec/bin/pos_tagging.py
This script pre-processes text and pos tags data using spaCy, so that the sense2vec model can be trained using Gensim.

src/Sense2vec/bin/train_word2vec.py
This script reads a directory of text files, and then trains a word2vec model using Gensim. The script includes its own vocabulary counting code, because Gensim's vocabulary count is a bit slow for our large, sparse vocabulary.

src/Sense2vec/bin/gensim2sense.py
This script converts the gensim model to a vector store for efficient querying

src/Sense2vec/bin/similar.py
to get similar words based on a word using our model