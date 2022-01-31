from sentinelsat import SentinelAPI, geojson_to_wkt
import json
import pandas as pd
from os.path import exists
from time import sleep
from IPython.display import clear_output

def getProductList(startdate='20210325', enddate='20210405', LatMin=51.25, LatMax=51.75, LngMin=-0.6, LngMax=0.28):
    '''Provides the list of Sentinel5P data products using SentinelAPI which contain the area specified by the coordinates between the start and end dates.
       By default the area and date range are as provided in the BlueSkyAbove Challenge.

       Parameters:
       >>> startdate: YYYYMMDD format string of the required starting date of data.
       >>> enddate: YYYYMMDD format string of the required ending date of data.
       >>> LatMin, LatMax, LngMin, LngMax: Floating point numbers specifying the minimum and maximum latitudes and longitudes in degrees.

       Returns:
       >>> products_found: Ordered dictionary provided by SentinelAPI containing the Sentinel5P products having the data requested. 
       >>> api: SentinelAPI to be used for downloading necessary products.
    '''
    
    geojsonstring='{{"type":"FeatureCollection","features":[{{"type":"Feature","properties":{{}},"geometry":{{"type":"Polygon","coordinates":\
        [[[{LongiMin},{LatiMin}],[{LongiMax},{LatiMin}],[{LongiMax},{LatiMax}],[{LongiMin},{LatiMax}],[{LongiMin},{LatiMin}]]]}}}}]}}'.format(LongiMin=LngMin,\
            LatiMin=LatMin,LongiMax=LngMax,LatiMax=LatMax)

    #username and password 's5pguest' datahubs url
    api = SentinelAPI('s5pguest', 's5pguest' , 'https://s5phub.copernicus.eu/dhus')
    footprint = geojson_to_wkt(json.loads(geojsonstring))

    products_found = api.query(footprint, date = (startdate,enddate), producttype = 'L2__NO2___' )

    return products_found

def saveProductNames(products_found):
    '''Saves the list of products found in the same orderred sequence in a CSV file called 'satellite_data_filenames'.

       Parameters:
       >>> products_found: Ordered dictionary returned by SentinelAPI contaiting a list of data products.

       Returns:
       >>> None    
    '''
    
    #creating an empty dataframe
    names = list()
    
    for key in products_found.keys():
        names.append(products_found[key]['title'])

    names = pd.Series(names, name='ProductFiles')
    names.to_csv('data/functional_data/satellite_data_filenames.csv', mode='w+', index=False) #overwite on any existing satellie_data_filenames.csv file
    print('satellite_data_filenames.csv has been updated.')

    return None

def downloadNewProducts(products_found):
    '''Checks the data/L2 folder and downloads the product files which are not already present.

       Parameters: 
       >>> products_found: Ordered dictionary returned by SentinelAPI contaiting a list of data products.

       Returns:
       >>> None          
    '''
    products_to_download = products_found.copy()

    for key in products_found.keys():
        if exists('data/L2/%s.nc'%products_found[key]['title']):
            print(products_found[key]['title'], 'already exists.')
            del products_to_download[key]
            sleep(0.5)
        else:
            print(products_found[key]['title'], 'will be downloaded.')
            sleep(0.5)

    api = SentinelAPI('s5pguest', 's5pguest' , 'https://s5phub.copernicus.eu/dhus')
    api.download_all(products_to_download, directory_path='data\L2')

    clear_output(wait = True)
    print('Available data products are saved in data/L2!')

    return None