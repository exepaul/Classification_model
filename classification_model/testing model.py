import spacy

nlp=spacy.load('/Users/exepaul/')

text=nlp(u'What People are talking about Bitcoin and Apple is looking at buying U.K. startup for $1 billion')


for i in text.ents:
    print(i.label_,i.text)