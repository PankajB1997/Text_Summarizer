import os
import sys
import csv
import pickle
import gensim
import random
import numpy as np
from collections import Counter
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler, QuantileTransformer, Normalizer
from sklearn.utils.class_weight import compute_sample_weight
import keras
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout, RepeatVector
from keras.layers.wrappers import TimeDistributed
from keras.layers.recurrent import LSTM
from keras.layers.embeddings import Embedding
from keras.regularizers import l2
from keras.models import load_model, Model
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint
from keras.preprocessing import sequence
from keras.utils import np_utils

from preprocessor import cleanSentence

def train(articles, titles, word2vec_model):
    return None

def predict(articles, model, word2vec_model):
    return []

def score(actual_title, predicted_title, model, word2vec_model):
    actual_title = cleanSentence(actual_title)
    predicted_title = cleanSentence(predicted_title)
    
    return 0.0

# Load Google's pre-trained Word2Vec model
word2vec = gensim.models.KeyedVectors.load_word2vec_format(os.path.join('word_embedding', 'GoogleNews-vectors-negative300.bin'), binary=True)
print(word2vec['man'])

# Load cleaned dataset of news articles and their respective headlines
data = pickle.load(open(os.path.join('news', 'data_filtered.pkl'), 'rb'))
headlines = data[0]
articles = data[1]

def get_vocab(list):
    vocab = Counter([w for text in list for w in text.split()])
    return zip(*[ (x, vocab[x]) for x in vocab if vocab[x] > 1 ])

vocab, vocabCount = get_vocab(headlines + articles)
vocab = list(vocab)
vocabCount = list(vocabCount)
print(vocab[:50])
print(len(vocab))
print(vocabCount[:50])
print(len(vocabCount))

word_embeddings =

# Initialize parameters for Encoder-Decoder Model
maxlen_articles = 25 # 0 - if we dont want to use articles at all
maxlen_headlines = 25
maxlen = maxlen_articles + maxlen_headlines
rnn_size = 512 # must be same as 160330-word-gen
rnn_layers = 3  # match FN1
batch_norm = False
activation_rnn_size = 40 if maxlend else 0

# Initialize training parameters
p_W, p_U, p_dense, p_emb, weight_decay = 0, 0, 0, 0, 0
optimizer = 'adam'
LR = 1e-4
batch_size=64
nflips=10

def build_model(embeddings):
    model = Sequential()
    model.add(Embedding(weights=[embeddings], name='embedding_1'))
    for i in range(3):
        lstm = LSTM(rnn_size, name='lstm_%d' % (i+1))
        model.add(lstm)
        model.add(Dropout(p_dense, name='dropout_%d' % (i+1)))
    model.add(Dense())
    model.add(Activation('softmax', name='activation'))
    return model

encoder = build_model(word_embeddings)
