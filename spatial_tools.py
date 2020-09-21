import numpy as np
import geopandas as gpd
import shapely
import pandas as pd

def format_geometry(gdf, geom_var):
    '''
    Return geojson geometry from gdf geometry column.
    '''
    gdf['geo_json'] = gdf[geom_var].apply(
        lambda x: shapely.geometry.mapping(x))
    return gdf


def get_location(conf):
    '''
    Return the coordinates for a given location.
    '''
    gdf = gpd.read_file(conf['loc_path'])
    gdf = format_geometry(gdf, 'geometry')
    location = gdf[gdf[conf['ref_var']] == conf['country']]['geo_json'].values
    return location


def spatial_join(df, countries):
    '''
    Perform spatial operations to obtain standarised country names for each tweet.

    args:
        -df: df, DataFrame with geolocated tweets.
        -countries: gdf, geodataframe with poligons for world countries.
    '''
    
    df[['x','y']] = pd.DataFrame(df.coordinates.tolist(), index= df.index)
    df = df[df.x.notnull()]
      # Remove rows with null coordinates.
    # convert df to gdf.
    df = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.x, df.y))
    df = df.set_crs(epsg=4326)  # set crs to WGS84.
    df.drop(['x', 'y'], axis = 1, inplace = True) 
    # Spatial join with intersection between df and countries.
    df = gpd.sjoin(df, countries, how='inner', op='intersects')
    df['geometry'] = df['geometry'].apply(  # set coordinates to geojson mongodb friendly format.
        lambda x: shapely.geometry.mapping(x))
    return df


def extract_coordinates(row):
    '''
    Function for apply. Returns properly formated coordinates splited in columns.

    args:
        -row: row from df (apply).
    '''
    try:
        l = row.replace('(', '').replace(')', '')
        x = float(l.split(',')[1])
        y = float(l.split(',')[0])
    except:
        x = np.nan
        y = np.nan
    return [x, y]