from paramiko import SSHClient, AutoAddPolicy
from pymongo import MongoClient, GEOSPHERE
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine, MetaData, Table, select
import pandas as pd
import json
import re
from collections import defaultdict
import geopandas as gpd


class SQL_bertamirans():
    '''
    '''
    def __init__(self, database):
        '''
        Creates connection engine for a given database.

        args:
            -database: str, database name.
        '''
        databases = {
            'covid19': 'data2.globalsentiment.org',
            'datos_sociales': 'global.codery.net'
        }
        try:
            self.engine = create_engine(
                'postgresql://pcoello:pcoello_20@@' +
                databases[database]+':5432/'+database
            )
        except:
            print("Connection Failed")


    def get_data(self, query):
        '''
        Returns a df with the data for a given query from bertamirans SQL database.

        args:
            -query: str, sql query.
        '''
        df = pd.read_sql(query, self.engine, )
       
        return(df)


    def list_tables(self):
        '''
        Lists all the tables in a given database.
        '''
        print(self.engine.table_names())


    def list_table_columns(self, table):
        '''
        Lists all the columns for a given table.

        args:
            -table: str, name of the table.
        '''
        conn = self.engine.connect()
        metadata = MetaData(conn)
        t = Table(table, metadata, autoload=True)
        columns = [m.key for m in t.columns]
        print(columns)


    def close(self):
        '''
        '''
        self.engine.dispose()



class MDB_remote():
    '''
    '''

    def __init__(self, conf):
        '''
        '''
        self.conf = conf
        if conf['ssh']:
            self.ssh_client = SSHClient()
            self.ssh_client.set_missing_host_key_policy(AutoAddPolicy())
            self.ssh_client.connect(
                conf['ssh_server'], username=conf['ssh_user'], password=conf['ssh_password'])
            
        # Set mongodb conexion.
        if re.search(' ', conf['database']) is None and len(conf['database']) < 20:
            if re.search(' ', conf['collection']) is None and len(conf['collection']) < 20:

                self.mg_client = MongoClient('mongodb://localhost:27017/')
                db = eval('self.mg_client.' + conf['database'])
                self.collection = eval('db.' + conf['collection'])
                # Allow for geolocation index.
                self.collection.create_index([("geometry", GEOSPHERE)])

            else:
                print('invalid collection name')
        else:
            print('invalid database name')

    def insert_data_mongodb(self, data):
        '''
        '''
        self.collection.insert_many(data)
        self.mg_client.close()  # Closing mongodb client.

    def update_data_mongodb(self, df, var):
        '''
        '''
        counter = 0
        for id_ in df.id_str.unique():
            try:
                data1 = self.collection.find({'id_str': id_})[0]
                if var not in data1.keys():
                    data2 = {var: df.set_index('id_str').loc[id_][var]}
                    insert = {**data1, **data2}
                    self.collection.update({'id_str': str(id_)}, insert)
                    counter += 1
            except:
                pass
        self.mg_client.close()
        return counter

    def get_first_date(self, query={}):
        '''
        '''
        return self.collection.find(query).sort("date", 1).limit(1)[0]['date']

    def get_last_date(self, query={}):
        '''
        '''
        return self.collection.find(query).sort("date", -1).limit(1)[0]['date']
    
    def get_number_of_records(self, query={}):
        '''
        '''
        return self.collection.find(query).count()

    def get_number_of_users(self, query={}):
        '''
        '''
        return len(self.collection.distinct('owner'))

    def get_users_list(self):
        '''
        '''
        return self.collection.distinct('owner')

    def clean_geometry(self, row):
        row['x'] = row['geometry']['coordinates'][0]
        row['y'] = row['geometry']['coordinates'][1]
        return row

    def get_query(self, query={}):
        '''
        '''
        cursor = self.collection.find(query)
        toret = [elem for elem in cursor]
        if len(toret)>0:
            toret = pd.DataFrame(toret).apply(self.clean_geometry, axis=1)
            return gpd.GeoDataFrame(toret, geometry=gpd.points_from_xy(toret.x, toret.y))
        else:
            return 'no data'
    
    def close(self):
        '''
        '''
        self.mg_client.close()
        if self.conf['ssh']:
            self.ssh_client.close()
