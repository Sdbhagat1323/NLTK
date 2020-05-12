# Stopword
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# stopwords are the most most common words that we used in sentences to make meaning out of it.
# But, we don't need those word in NLP, so we  removed them.

example_sentence = "This is an example showing off stopword filteration."

# first tokenization
# then we remove stopwors.

words = word_tokenize(example_sentence)

corpus = []
for w in words:
         if w not in set(stopwords.words("english")):
                  corpus.append(w)
         

print(corpus)

stop_word = set(stopwords.words("english"))

example_words = [w for w in words if not w in stop_word]

print(example_words)
