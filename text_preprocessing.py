import unidecode
import re
import string
from nltk.corpus import stopwords
import pandas as pd
from preprocessor.api import clean

def unicode_to_ascii(text):
    '''
    Performs transformation from unicode format to ascii using Unidecode library.
    '''
    return unidecode.unidecode(str(text))


def remove_punctuation(text):
    '''
    Removes punctuation sings from a string.
    '''
    return text.translate(str.maketrans('', '', string.punctuation))


def remove_urls(text):
    '''
    Remuve all url from a string.
    '''
    return re.sub(r'(\s)http\w+', r'\1', text)


def remove_double_whitespaces(text):
    '''
    Remove double whitespaces from a string.
    '''
    return re.sub(' +', ' ', text)


def remove_lspace(text):
    '''
    Remove initial whitespace.
    '''
    return text.lstrip()


def remove_rspace(text):
    '''
    Remove final whitespace.
    '''
    return text.rstrip()


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

