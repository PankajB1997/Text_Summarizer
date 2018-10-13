import os
import sys
import csv
import pickle

data = pickle.load(open(os.path.join('news', 'data_filtered.pkl'), 'rb'))

titles = data[0]
articles = data[1]

print(titles[0])
print(articles[0])
print(len(titles))
print(len(articles))
