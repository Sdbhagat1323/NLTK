from nltk.corpus import wordnet

syns = wordnet.synsets("program")

#synset
print(syns)

# just the word
print(syns[0].lemmas()[0].name())

#defination
print(syns[0].definition())

#example
print(syns[0].examples())


print ("----------------------------------------------------------------")

synonyms = []
antonyms = []

for syn in wordnet.synsets("good"):
         for l in syn.lemmas():
                  synonyms.append(l.name())
                  if l.antonyms():
                           antonyms.append(l.antonyms()[0].name())

print(set(synonyms))
print(set(antonyms))

print ("----------------------------------------------------------------")

                      
                           

synonyms = []
antonyms = []

for syn in wordnet.synsets("good"):
         for l in syn.lemmas():
                  print("l:", l)
                  synonyms.append(l.name())
                  if l.antonyms():
                           antonyms.append(l.antonyms()[0].name())
##
##print(set(synonyms))
##print(set(antonyms))
##
##                                           
##
