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

##print(all_words.most_common(15))
##print(all_words["like"])

#processing for naive_bays algorithms
#limit on amount of words

word_features = list(all_words.keys())[:3000]

# if one of this 3000 words are in this document return true else false.
def find_feature(document):
         words = set(document)# all the words not the amount of words.
         features = {}
         for w in word_features:
                  features[w]  = (w in words)
         return features


##print((find_feature(movie_reviews.words("neg/cv000_29416.txt"))))

featuresets = [(find_feature(rev), category) for (rev, category) in documents]
                  

# Now we build a naive bayes algorithms for categories data as negative and positive data.

training_set = featuresets[:1900]
testing_set = featuresets[1900:]

# posterior =  prior occurence * liklihood / evidence

classifier = nltk.NaiveBayesClassifier.train(training_set)
print("NaiveBayes algo accuracy percentage:", (nltk.classify.accuracy(classifier, testing_set))* 100)
classifier.show_most_informative_features(15)

























