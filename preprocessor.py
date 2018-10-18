import os
import sys
import csv
import pickle
from stopwords import stopwords
import unicodedata
import re

tbl = dict.fromkeys(i for i in range(sys.maxunicode)
                      if unicodedata.category(chr(i)).startswith('P'))

def removePunctuations(sentence):
    return sentence.translate(tbl)

def removeExtraSpaces(sentence):
    return re.sub(' +',' ',sentence)

def removeStopwords(sentence):
    filtered_sentence = ""
    for word in sentence.split(" "):
        if word not in stopwords:
            filtered_sentence += word + " "
    return filtered_sentence.strip()

def retainAsciiOnly(sentence):
    filtered_sentence = ""
    for c in sentence:
        if ord(c) < 128:
            filtered_sentence += c
    return filtered_sentence

def cleanSentence(sentence):
    sentence = sentence.strip().lower()
    sentence = retainAsciiOnly(sentence)
    # sentence = removeStopwords(sentence)
    sentence = removePunctuations(sentence)
    sentence = removeExtraSpaces(sentence)
    return sentence

data = pickle.load(open(os.path.join('news', 'data.pkl'), 'rb'))
titles = data[0]
articles = data[1]

titles_filtered = []
articles_filtered = []

for title in titles:
    titles_filtered.append(cleanSentence(title))

for article in articles:
    articles_filtered.append(cleanSentence(article))

pickle.dump([ titles_filtered, articles_filtered ], open(os.path.join('news', 'data_filtered.pkl'), 'wb'))
