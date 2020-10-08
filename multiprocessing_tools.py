import sys
import os
import requests
import multiprocessing
import time
import functools


def _getThreads():
    '''
    Returns the number of available threads on a posix/win based system.
    '''
    if sys.platform == 'win32':
        return (int)(os.environ['NUMBER_OF_PROCESSORS'])
    else:
        return (int)(os.popen('grep -c cores /proc/cpuinfo').read())

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