import datetime

def convert(seconds): 
    '''
    Function to format time en print messages.

    args:
        -seconds: float, number of seconds.
    '''
    seconds = seconds % (24 * 3600) 
    hour = seconds // 3600
    seconds %= 3600
    minutes = seconds // 60
    seconds %= 60
    
    return "%d:%02d:%02d" % (hour, minutes, seconds)

def extract_date(row):
    '''
    Function for apply. Returns properly formated date without time.

    args:
        -row: row from df (apply).
    '''
    date = str(row).split(' ')[0]
    return date


def get_min_max_count_date(date_from, date_to):
    '''
    '''
    min_date = datetime.datetime.strptime(date_from, "%Y-%m-%d")
    max_date = datetime.datetime.strptime(date_to, "%Y-%m-%d")
    day_count = (max_date - min_date).days + 1
    return (min_date, max_date, day_count)