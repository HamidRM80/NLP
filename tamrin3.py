import nltk
nltk.download('wordnet')
from nltk.corpus import wordnet
synonyms = []
antonyms = []
for s in wordnet.synsets('courageous'):
    for l in s.lemmas():
        synonyms.append(l.name())
        if l.antonyms():
            antonyms.append(l.antonyms()[0].name)

print('antonyms of nice = ', set(antonyms))


print('------------------------------------------------------------')


for s in wordnet.synsets('courageous'):
    for l in s.lemmas():
        synonyms.append(l.name())
        if l.antonyms():
            antonyms.append(l.antonyms()[1].name)

print('antonyms of nice = ', set(antonyms))