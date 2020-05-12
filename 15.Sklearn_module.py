# Text classifier
# can be apply for any label text as long as they have to categories
# spam msg classifier

import nltk
import random
from nltk.corpus import movie_reviews
from nltk.classify.scikitlearn import SklearnClassifier
import pickle

from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC




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

#classifier = nltk.NaiveBayesClassifier.train(training_set)

#for load saved classifier
classifier_p = open("naivebayes.pickle", "rb")
classifier = pickle.load(classifier_p)
classifier_p.close()
print("Original NaiveBayes algo accuracy percentage:", (nltk.classify.accuracy(classifier, testing_set))* 100)
#classifier.show_most_informative_features(15)

# MultinomialNB, GaussianNB, BernouliNB

MNB_classifier = SklearnClassifier(MultinomialNB())
MNB_classifier.train(training_set)


print("Multinomial classifier algo accuracy percentage:", (nltk.classify.accuracy(MNB_classifier, testing_set)) * 100)
#MNBclassifier.show_informative_features(15)

##
##GNB_classifier = SklearnClassifier(GaussianNB())
##GNB_classifier.train(training_set)
##
##print("GaussionNn Classifier Algo acccuracy percentage:", (nltk.classify.accuracy(GNB_classifier, testing_set))*100)
###GNB_classifier.show_informative_features(15)



BNB_classifier = SklearnClassifier(BernoulliNB())
BNB_classifier.train(training_set)

print("BernoulliNB Classifier Algo accuracy percentage:",(nltk.classify.accuracy(BNB_classifier, testing_set))*100 )


##LogisticRegression, SGDClassifier
## SVC, LinearSVC, NuSVC


log_classifier = SklearnClassifier(LogisticRegression())
log_classifier.train(training_set)

print("LogisticRegression Classifier Algo accuracy percentage:",(nltk.classify.accuracy(log_classifier, testing_set))*100 )



SGD_classifier = SklearnClassifier(SGDClassifier())
SGD_classifier.train(training_set)

print("SGDClassifier Algo accuracy percentage:",(nltk.classify.accuracy(SGD_classifier, testing_set))*100 )


SVC_classifier = SklearnClassifier(SVC())
SVC_classifier.train(training_set)

print("SVC Classifier Algo accuracy percentage:",(nltk.classify.accuracy(SVC_classifier, testing_set))*100 )


LinearSVC_classifier = SklearnClassifier(LinearSVC())
LinearSVC_classifier.train(training_set)

print("LinearSVC Classifier Algo accuracy percentage:",(nltk.classify.accuracy(LinearSVC_classifier, testing_set))*100 )


NuSVC_classifier = SklearnClassifier(NuSVC())
NuSVC_classifier.train(training_set)

print("NuSVC Classifier Algo accuracy percentage:",(nltk.classify.accuracy(NuSVC_classifier, testing_set))*100 )





















































