import sys
import os
import dask
from dask.distributed import Client, LocalCluster
from dask.diagnostics import ProgressBar
import dask.dataframe as dd

def dask_multi(df, meta, threads, function, **kwargs):
    '''
    Perform multiprocessign nlp operations on df.

    args:
        -df: df, dataframe to be formatted.
    '''
    args = dict(**kwargs)
    args["meta"] = meta
    ddf = dd.from_pandas(df, npartitions=threads/2)
    df_out = ddf.map_partitions(
        function,
        **args)
    df = df_out.compute(scheduler='processes')
    return df


def dask_apply(df, **argv):
    '''
    '''
    for func, args in argv["execution"].items():
        #func= list(params.funcs())[0]
        df = df.apply(func, **args, axis=1)
    return df


def _getThreads():
    '''
    Returns the number of available threads on a posix/win based system.
    '''
    if sys.platform == 'win32':
        return (int)(os.environ['NUMBER_OF_PROCESSORS'])
    else:
        return (int)(os.popen('grep -c cores /proc/cpuinfo').read())


import requests
import multiprocessing
import time
import functools

session = None

def set_global_session():
    global session
    if not session:
        session = requests.Session()

def multiproc_func(rows, func, **kwargs):
    with multiprocessing.Pool(initializer=set_global_session) as pool:
        return pool.map(functools.partial(func,**kwargs), rows)




if __name__ == "__main__":
    rows = df['text'].to_list()
    start_time = time.time()
    res= []
    res.append(download_all_sites(rows))
    duration = time.time() - start_time
    print(f"Downloaded {len(rows)} in {duration} seconds")