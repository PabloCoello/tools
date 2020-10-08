from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from afinn import Afinn
from labMTsimple.storyLab import *
import numpy as np
import rpy2.robjects as ro
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter
import pandas as pd
import unidecode

def labmt_proc(row, labMT, labMTvector, labMTwordList):
    '''
    Function for apply. Performs labmt sentiment analysis. Returns columns with results.
    
    args:
        -row: row from df (apply).
        -labMT: list, Output from emotionFileReader function (labMT).
        -labMTvector: list, Output from emotionFileReader function (labMT).
        -labMTwordlist: list, Output from emotionFileReader function (labMT).
    '''
    t_res = extract_sentiment(
        row, labMT, labMTvector, labMTwordList)
    toret = {}
    toret['labmt_freq'] = sum(t_res[1])
    toret['labmt_n_words'] = len(row.split(' '))
    toret['labmt_ratio'] = float(
        toret['labmt_freq'])/float(toret['labmt_n_words'])
    if toret['labmt_ratio'] == 0:
        toret['labmt_sent'] = np.nan
    else:
        toret['labmt_sent'] = t_res[0]
    return toret

def extract_sentiment( row, labMT, labMTvector, labMTwordList):
    '''
    Performs labmt sentiment analysis.
    
    args:
        -row: row from df (apply).
        -labMT: list, Output from emotionFileReader function (labMT).
        -labMTvector: list, Output from emotionFileReader function (labMT).
        -labMTwordlist: list, Output from emotionFileReader function (labMT).
    '''
    sent, freq = emotion(row, labMT, shift=True, happsList=labMTvector)
    freq_stopped = stopper(freq, labMTvector, labMTwordList, stopVal=1.0)
    sent_stopped = emotionV(freq_stopped, labMTvector)
    return(sent_stopped, freq_stopped)


def vader_proc(row):
    '''
    Function for apply. Performs vader sentiment analysis. Returns columns with results.
    
    args:
        -row: row from df (apply).
    '''
    vader_analyzer = SentimentIntensityAnalyzer()
    vs = vader_analyzer.polarity_scores(str(row))
        
    toret = {}   
    toret['vader_neg'] = vs['neg']
    toret['vader_neu'] = vs['neu']
    toret['vader_pos'] = vs['pos']
    toret['vader_compound'] = vs['compound']
    return toret


def afinn_proc(row):
    '''
    Function for apply. Performs afinn sentiment analysis. Returns columns with results.
    
    args:
        -row: row from df (apply).
    '''
    afinn = Afinn()
    toret = {}
    toret['afinn_score'] = afinn.score(str(row))
    return toret


# R function for NRC sentiment analysis.
import rpy2.robjects as ro
def get_nrc_rfunc(path):
    '''
    '''
    with open(path) as f:
        rfunc = ro.r(f.read())
    return rfunc


def apply_sentiment(df, conf, lang):
    '''
    Function for dask environment. Apply sentiment analysis to df.

    args:
        -df: df, df to be analysed.
        -conf: dict, conf dict from conf.json.
        -lang: str, language
    '''
    # Set LabMT lexicon for given language
    labMT, labMTvector, labMTwordList = emotionFileReader(
        stopval=0.0,
        lang=conf['matches'][lang],
        returnVector=True
    )
    index = [(df['lang'][row] in conf['matches'].keys())
            for row in df.index]
    df = df[index]

    df = df.apply(labmt_proc, args=(labMT, labMTvector, labMTwordList), axis=1)
    df = df.apply(vader_proc, axis=1)
    df = df.apply(afinn_proc, axis=1)
    return df


def R_sentiment_analysis(df, conf, lang):
    '''
    Perform multiprocessign nlp operations on df using R code.

    args:
        -df: df, dataframe to be formatted.
    '''
    with localconverter(ro.default_converter + pandas2ri.converter):
        r_data = ro.conversion.py2rpy(df) # Transform pandas dataframe to R dataframe.

    rfunc = get_nrc_rfunc('./nrc.R')
    r_df = rfunc(df=r_data, lang=conf['matches'][lang]) # Perform R function.

    with localconverter(ro.default_converter + pandas2ri.converter):
        data = ro.conversion.rpy2py(r_df) # Transform R dataframe to pandas dataframe.
    data.drop(['id_str','proc_text'], axis=1, inplace=True)
    return data


def get_scores_dict(lang):
    '''
    '''
    reference = {
        'en': './Hedonometer_scores/en_scores.xlsx',
        'es': './Hedonometer_scores/es_scores.xlsx'
    }

    score = pd.read_excel(reference.get(lang), skiprows=[0])
    score.rename(
        columns={
            'Word': 'word', 
            'Happiness Score': 'score'
            },
        inplace=True)
    score.word.apply(unidecode.unidecode)
    return dict(zip(score.word, score.score))


def get_matching_array(text, score_dict):
    '''
    '''
    array = [score_dict.get(word) for word in text.split(' ') if score_dict.get(word) != None]
    return array


def score_matcher(row, score_dict):
    '''
    '''
    toret = {}
    match = get_matching_array(str(row), score_dict)
    
    toret['total_words'] = len(row.split(' '))
    toret['match_words'] = len(match)
    toret['freq'] = toret['match_words'] / toret['total_words']
    toret['hedonometer_sent'] = np.mean(match)
    return toret
