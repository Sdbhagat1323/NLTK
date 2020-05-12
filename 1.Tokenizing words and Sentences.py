#\Tokenizing words and Sentence.

from nltk import sent_tokenize, word_tokenize 


example_text = "Hello there Mr. Smith, how are you doing today? The weather is good and python is awesome.The sky is pinkish-blue. you should not eat cardboard."


# separation by sentences.
print(sent_tokenize(example_text))
print(word_tokenize(example_text))

# now you want iterate throught sentences.

for i in sent_tokenize(example_text):
         print(i)
#now you want iterate throught words in sentences.

for i in word_tokenize(example_text):
         print(i)
# Now list comprihasion, with single line for loop, something to show off.

words = [word for word in word_tokenize(example_text)]

print(words)




