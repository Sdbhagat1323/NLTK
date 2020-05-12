# Text classifier
# can be apply for any label text as long as they have to categories
# spam msg classifier

import nltk
import random
from nltk.corpus import movie_reviews

##documents  = []
##for category in movie_reviews.categories():
##         for fileid in movie_reviews.fileids(category):
##                  documents.append(list(movie_reviews.words(fileid)), category)
##
# document is here to create trainning and test sets.
documents = [(list(movie_reviews.words(fileid)), category)
             for category in movie_reviews.categories()
             for fileid in movie_reviews.fileids(category)]


random.shuffle(documents)
#print(documents[2])
# complie all words in all the reviews. All positive as well as negetive.

all_words = []

for w in movie_reviews.words():
         all_words.append(w.lower())

# find out which ones are more frequently used, for that we use nltk frequency distribusion.

all_words = nltk.FreqDist(all_words)
print(all_words.most_common(15))
print(all_words["like"])
