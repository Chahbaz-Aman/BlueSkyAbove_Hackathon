import pandas as pd
import numpy as np
import smopy
import re
import cv2
from geopy.geocoders import Nominatim
import geopy.distance
geolocator = Nominatim(user_agent='myapplication')

def get_area(place,km):
    '''Returns an km x km square area with the specified place at its centre as a numpy_array image.
    
       Parameters:
       >>> place: Name of the place of interest as a string.
       >>> km: Side length of the area required in kilometres. 
       
       Returns:
       interest_area: image of the area as a numpy_array of dimensions (x,x,3)
    '''
    
    if km < 5:
        km = 5
    #geolocator = Nominatim(user_agent='myapplication')

    zoom_level = 12
    C = 40075016.686 # Earth circumference at the equator in metres
    location = geolocator.geocode(place)
    
    dLat = 1/110*5/2 #_1deg latitude = 110km    
    dLng = 1/(110*np.round(np.cos(location.latitude*np.pi/180), decimals = 4))*5/2 #_1deg Longitude = 110*cos(latitude)

    LatMin=location.latitude-dLat
    LatMax=location.latitude+dLat
    LngMin=location.longitude-dLng
    LngMax=location.longitude+dLng
    
    for i in range(1,5):
        try:
            map = smopy.Map((LatMin,LngMin, LatMax,LngMax), z= zoom_level);
        except:
            pass

    map_array = map.to_numpy();
    
    centre_pixel = (int(map_array.shape[0]/2),int(map_array.shape[1]/2))
    gp = int((km*1000)/np.round(C*np.round(np.cos(location.latitude*np.pi/180), decimals = 4)/2**(zoom_level+8), decimals = 0)/2)
    sl = int((km*1000)/np.round(C*np.round(np.cos(location.latitude*np.pi/180), decimals = 4)/2**(zoom_level+8), decimals = 0)/2)
    
    interest_area = map_array[(centre_pixel[0]-sl):(centre_pixel[0]+sl), (centre_pixel[1]-gp):(centre_pixel[1]+gp)] 
    
    return interest_area

def prep_distances(area):
    '''Returns a numpy array of squared Eucledian Distances of each element from the centre element.
       If centre element is at (mc,nc), squared distance of element at (x,y) is
           squared_distance = (x - mc)**2 + (y - nc)**2
       
       Parameters:
       >>>area: image of the area of interest as a 3D-numpy_array
       
       Returns:
       >>>distances: 2D-numpy-array of squared distances of the elements from the centre element.
    '''
            
    m = area[:,:,0].shape
    distances = np.zeros(m)
    for i in range(0,m[0]):
        for j in range(0,m[1]):
            distances[i,j] = (i - m[0]/2)**2 + (j - m[1]/2)**2  
    distances[distances == 0] = 1
    return distances

def get_landUse(color_range, area):
    '''Returns a dataframe of the inverse distance weighted sum of various land uses.
       
       Parameters:
       >>>color_range: A dataframe with a categories and their respective color ranges on the area.
       >>>area: numpy_array image of the area of interest.
       
       Returns:
       >>>elements: Dataframe of land uses in the area.
    '''
    
    elements = pd.DataFrame(columns = ['Land_Use','Coverage'])
    distances = prep_distances(area)
    for idx in color_range.index:
        mask = extract_feature_mask(color_range['Low'][idx], color_range['High'][idx], area)
        mask = mask/distances/255
        cov = mask.sum()
        elements = elements.append({'Land_Use': color_range['Land_Use'][idx],
                                    'Coverage': cov}, ignore_index = True)
    return elements

def extract_feature_mask(low, high, image):
    '''Returns a mask of the image with pixels in the color range set to 255 and rest 0.
    
       Parameters:
       >>>low: Lower color limit in the color range in HSV space.
       >>>high: Upper color limit in the color range in HSV space.
       >>>image: RBG numpy_array image.
       
       Returns:
       >>>mask: 2D numpy_array with elements in the range    
    '''
    
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    mask = cv2.inRange(hsv, low, high)
    return mask

def generate_dataset(sites,km,color_range):
    ''' Creates 'Site_wise_LandUse.csv' file storing land-uses for a list of places/sites.
    
        Parameters:
        >>> sites: DataFrame have names of pollution monitoring sites in 'Site_Name' column.
        >>> km: Side length of the area required in kilometres. 
        
        Returns:
        No return.    
    '''
    sites_lu = pd.DataFrame(columns = ['Place','Highways','Local_Roads','Forests','Waterbodies','Settlements','RailRoads'])
    for site in sites['Site_Name']:
        area = get_area(site,km);

        land_use = get_landUse(color_range,area);
        land_use.set_index(land_use['Land_Use'],drop=True,inplace=True)
        land_use.drop(columns = ['Land_Use'], inplace = True)
        land_use = land_use.transpose()
        land_use['Place']=site

        sites_lu = sites_lu.append(land_use,ignore_index=True)
    return sites_lu

def get_area_coords(location,km):
    '''Returns an km x km square area with the specified coordinates at its centre as a numpy_array image.
    
       Parameters:
       >>> location: Coordinates as a tuple of the place of interest.
       >>> km: Side length of the area required in kilometres. 
       
       Returns:
       interest_area: image of the area as a numpy_array of dimensions (x,x,3)
    '''

    if km < 5:
        km = 5

    zoom_level = 12
    C = 40075016.686 # Earth circumference at the equator in metres
    
    dLat = 1/110*5/2 #_1deg latitude = 110km    
    dLng = 1/(110*np.round(np.cos(location[0]*np.pi/180), decimals = 4))*5/2 #_1deg Longitude = 110*cos(latitude)

    LatMin=location[0]-dLat
    LatMax=location[0]+dLat
    LngMin=location[1]-dLng
    LngMax=location[1]+dLng
    
    for i in range(1,5): #5 attempts to get the map
        try:
            map = smopy.Map((LatMin,LngMin, LatMax,LngMax), z= zoom_level);
            break
        except:
            pass
        
    map_array = map.to_numpy();
    centre_pixel = (int(map_array.shape[0]/2),int(map_array.shape[1]/2))
    gp = int((km*1000)/np.round(C*np.round(np.cos(location[0]*np.pi/180), decimals = 4)/2**(zoom_level+8), decimals = 0)/2)
    sl = int((km*1000)/np.round(C*np.round(np.cos(location[0]*np.pi/180), decimals = 4)/2**(zoom_level+8), decimals = 0)/2)
    
    interest_area = map_array[(centre_pixel[0]-sl):(centre_pixel[0]+sl), (centre_pixel[1]-gp):(centre_pixel[1]+gp)] 
    
    return interest_area

def getLandUseTerms(location,color_range,km=7):
    sites_lu = pd.DataFrame(columns = ['Highways','Local_Roads','Forests','Waterbodies','Settlements','RailRoads'])
    area = get_area_coords(location,km);
    land_use = get_landUse(color_range,area);
    land_use.set_index(land_use['Land_Use'],drop=True,inplace=True)
    land_use.drop(columns = ['Land_Use'], inplace = True)
    land_use = land_use.transpose()
    sites_lu = sites_lu.append(land_use,ignore_index=True)
    
    return sites_lu

