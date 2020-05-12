# STEMMING
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

example_words = ["Stunning", "pythoned", "python", "pythoning", "pythoner", "pythonly"]

ps = PorterStemmer()

for w in example_words:
         print(ps.stem(w))
         
print("--------------------xx--------------------")

example_text = "It is very  important to be pythonly while you are pythoning with python. All pythoners have pythoned poorly at least once"

stop_words = set(stopwords.words("english"))

tokenized_words = word_tokenize(example_text)
print(tokenized_words)

##corpus = []
##
##for w in tokenized_words:
##         if w not in stop_words:
##                  corpus.append(w)
                  

corpus = [word for word in tokenized_words if not word in stop_words]

print(corpus)

# steeming words

for w in corpus:
         print(ps.stem(w))
