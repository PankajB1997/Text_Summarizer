import pickle
import traceback
from gensim.models.KeyedVectors import load_word2vec_format

import text_summarizer_helper as api

class TextSummarizer:

    articles = None
    titles = None

    word2vec_model = None
    text_summarizer_model = None

    '''
    Initialize Text Summarizer by providing the dataset of articles and titles as parameters.
    '''
    def __init__(self, articles, titles):
        self.articles = articles
        self.titles = titles

    '''
    Load a pretrained Word2Vec model from a given filepath.
    Preconditions:
    1. Word2Vec model is stored as a binary file.
    '''
    def load_word2vec_model(self, pretrained_model_filepath):
        try:
            self.word2vec_model = load_word2vec_format(str(pretrained_model_filepath), binary=True)
        except:
            traceback.print_exc()
            print("Error in loading Word2Vec model!")

    '''
    Load pretrained text summarizer model.
    Preconditions:
    1. The pretrained text summarizer model is stored as a pickle file.
    '''
    def load_pretrained_text_summarizer_model(self, text_summarizer_model_filepath):
        try:
            self.text_summarizer_model = pickle.load(open(str(text_summarizer_model_filepath), 'rb'))
        except:
            traceback.print_exc()
            print("Error in loading pretrained Text Summarizer model!")

    '''
    Train the text summarizer model using given dataset of articles and titles.
    Preconditions:
    1. A valid list of articles and an equal sized list of corresponding titles have been entered while
       declaring an instance of this class.
    '''
    def train_text_summarizer_model(self):
        try:
            articles = [ str(article) for article in articles ]
            titles = [ str(title) for title in titles ]
            self.text_summarizer_model = api.train(articles, titles, word2vec_model = self.word2vec_model)
        except:
            traceback.print_exc()
            print("Error in training the text summarizer model!")

    '''
    Summarize a set of given input articles and return the list of predicted titles.
    Preconditions:
    1. The text summarizer model has already been trained before calling this method.
    2. The input list of articles should be similar to the ones the model was trained with, to give accurate results.
    '''
    def summarize_articles(self, articles):
        try:
            articles = [ str(article) for article in articles ]
            return api.predict(articles, model = self.text_summarizer_model, word2vec_model = self.word2vec_model)
        except:
            traceback.print_exc()
            print("Error in summarizing given input of articles!")

    '''
    Summarize a given input article and return the predicted title for the same.
    Preconditions:
    1. The text summarizer model has already been trained before calling this method.
    2. The input article should be similar to the ones the model was trained with, to give accurate results.
    '''
    def summarize_article(self, article):
        try:
            return api.predict([ str(article) ], model = self.text_summarizer_model, word2vec_model = self.word2vec_model)[0]
        except:
            traceback.print_exc()
            print("Error in summarizing given article!")

    '''
    Score a predicted title by comparing with the actual title and measuring for similarity.
    Returns a score between 0 to 100, with 100 indicating very similar and 0 indicating not similar at all.
    Preconditions:
    1. The text summarizer model has already been trained before calling this method.
    '''
    def score_predicted_title(self, actual_title, predicted_title):
        try:
            return api.score(str(actual_title), str(predicted_title), model = self.text_summarizer_model, word2vec_model = self.word2vec_model)
        except:
            traceback.print_exc()
            print("Error in scoring the predicted title against the actual title!")
