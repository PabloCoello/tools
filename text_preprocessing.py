import unidecode
import re
import string
from nltk.corpus import stopwords
import pandas as pd
from preprocessor.api import clean


def clean_text(row):
    '''
    Function for apply. Performs preprocess.api clean function to raw tweet text.
    
    args:
        -row: row from df (apply).
    '''
    clean_tweet = unidecode.unidecode(str(row))
    clean_tweet = clean_tweet.translate(str
                                            .maketrans('',
                                                        '',
                                                        string.punctuation))
    clean_tweet = clean(clean_tweet)
    clean_tweet = clean_tweet.lower()  # All lowercase
    clean_tweet = re.sub(r'(\s)http\w+', r'\1', clean_tweet)
    clean_tweet = re.sub(' +', ' ', clean_tweet)  # Remove double spaces
    clean_tweet = clean_tweet.lstrip()  # Remove leading space
    clean_tweet = clean_tweet.rstrip()
    return clean_tweet


def lemmatization(row,nlp,allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    """https://spacy.io/api/annotation"""
    doc = nlp(row)
    tokenized_text = [token.lemma_ for token in doc if token.pos_ in allowed_postags]
    return tokenized_text

'''

                  mix_list=['covid19', 'coronavirus', 'sars-cov',
                            'covid', 'covid-19', 'sarscov2', 'sars-cov-2'],
                  merge=u'coronavirus'
    for token in doc:
        if(token.text in mix_list):
'''
def remove_stopwords(row):
    '''
    '''
    esp_stopwords = [unidecode.unidecode(
        elem) for elem in stopwords.words('spanish')]
    toret = [[word for word in simple_preprocess(
        str(doc)) if word not in stopwords.words('english')] for doc in row['tokenized_text']]
    toret = [[word for word in simple_preprocess(
        str(doc)) if word not in esp_stopwords] for doc in toret]
    return toret


def sent_to_words(self, sentences):
    '''
    '''
    for sentence in sentences:
        # deacc=True removes punctuations
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))

