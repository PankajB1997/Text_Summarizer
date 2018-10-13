import os
import sys
import csv
import pickle
from pprint import pprint

maxInt = sys.maxsize
decrement = True

while decrement:
    # decrease the maxInt value by factor 10
    # as long as the OverflowError occurs.

    decrement = False
    try:
        csv.field_size_limit(maxInt)
    except OverflowError:
        maxInt = int(maxInt/10)
        decrement = True

csvs = [ 'articles1.csv', 'articles2.csv', 'articles3.csv' ]
titles = []
articles = []

for file in csvs:
    with open(os.path.join('news', file), encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            titles.append(row['title'])
            articles.append(row['content'])

print(titles[0])
print(articles[0])
print(len(titles))
print(len(articles))

pickle.dump([ titles, articles ], open(os.path.join('news', 'data.pkl'), 'wb'))
