def build_query(conf, location):
    '''
    '''
    date_from = conf['date_from'].split('-')
    date_to = conf['date_to'].split('-')
    query = {
        "geometry": {"$geoWithin": {'$geometry': location[0]}},
        "created_at": {'$lt': datetime.datetime(int(date_to[0]), int(date_to[1]), int(date_to[2]), 23, 59, 59),
                       '$gt': datetime.datetime(int(date_from[0]), int(date_from[1]), int(date_from[2]), 0, 0, 0)
                       }
    }
    return query