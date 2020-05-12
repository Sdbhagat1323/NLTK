from nltk.stem import WordNetLemmatizer
# lemmating is more powerful than stemming.

lemmatizer = WordNetLemmatizer()

print(lemmatizer.lemmatize("better"))
print(lemmatizer.lemmatize("better", pos="a")) # pos means  part of speech tag
print(lemmatizer.lemmatize("running", pos="v"))

print(lemmatizer.lemmatize("toying", pos="v"))

