import rpy2.robjects as ro
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter
import rpy2.robjects as robjects
import json

def pandas2Rconverter(df):
    '''
    '''
    with localconverter(ro.default_converter + pandas2ri.converter):
        r_data = ro.conversion.py2rpy(df)
    return r_data


def R2pandasconverter(df):
    '''
    '''
    with localconverter(ro.default_converter + pandas2ri.converter):
        data = ro.conversion.rpy2py(df)
    return data


def get_rfunc(path):
    '''
    '''
    with open(path) as f:
        rfunc = ro.r(f.read())
    return rfunc


def ols(data, formula):
    '''
    '''
    rfunc = get_rfunc('./R/lm.R')
    toret = str(
        robjects.StrVector(
            rfunc(
                data=pandas2Rconverter(data),
                formula=formula
            )
        )[0]
    )
    return json.loads(toret)

if __name__ == '__main__':
    data = df
    formula = 'sepal_length ~ sepal_width'
    res = ols(df, 'sepal_length ~ sepal_width')
