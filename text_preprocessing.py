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


def clean_pipeline(text):
    '''
    Function for apply. Performs preprocess.api clean function to raw text.
    
    args:
        -text: string.
    '''
    toret = unicode_to_ascii(text)
    toret = remove_punctuation(toret)
    toret = clean(toret)
    toret = toret.lower()  # All lowercase
    toret = remove_urls(toret)
    toret = remove_double_whitespaces(toret)  # Remove double spaces
    toret = remove_lspace(toret)  # Remove leading space
    toret = remove_rspace(toret)
    return toret


def lemmatization(row, nlp, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    """https://spacy.io/api/annotation"""
    doc = nlp(row)
    tokenized_text = [
        token.lemma_ for token in doc if token.pos_ in allowed_postags]
    return tokenized_text


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

