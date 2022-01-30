from unicodedata import decimal
import xarray as xr
import numpy as np
from tqdm import tqdm
import re
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, losses
from tensorflow.keras.models import Model
from os.path import exists
from datetime import datetime

class Autoencoder(Model):
    def __init__(self, latent_dim,m):
        super(Autoencoder, self).__init__()
        self.latent_dim = latent_dim   
        self.encoder = tf.keras.Sequential([
          layers.Flatten(), layers.Dense((m*m)//3, activation='relu'),
          layers.Dense(latent_dim, activation='relu'),
        ])
        self.decoder = tf.keras.Sequential([
          layers.Dense((m*m)//3, activation='relu'),
          layers.Dense(m*m, activation='relu'),
          layers.Reshape((m, m))
        ])

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

def generateDataCompressorTrainingData(data_files, kernel_size):
    ''' Generates training data for the satellite data compressor.

        Parameters:
        >>> data_files: List of names of Sentinel 5P L2 product files
        >>> kernel_size: Grid size of satellte data to be considered

        Returns:
        >>> data_list: a stacked array of satellite measurements split into blocks of size (kernel_size x kernel_size)
    '''
    
    data_list = list() #empty list to store extracted subsets from satellite data
    for file in tqdm(data_files):
        if exists('data/L2/%s.nc'%file):
            s5p_PRD = xr.open_dataset('data/L2/' + file + '.nc', group = 'PRODUCT')       
            no2 = s5p_PRD['nitrogendioxide_tropospheric_column']
            qav = s5p_PRD['qa_value']

            for sl in range(0, kernel_size*(s5p_PRD.scanline.shape[0]//kernel_size), kernel_size):
                for gp in range(0, kernel_size*(s5p_PRD.ground_pixel.shape[0]//kernel_size), kernel_size):
                    if(np.min(qav[0][sl:(sl+kernel_size), gp:(gp+kernel_size)].values) > 0.5):
                        kernel = no2[0][sl:(sl+kernel_size), gp:(gp+kernel_size)].values
                        if(sum(sum(np.isnan(kernel)))>0):
                            try: 
                                kernel = np.apply_along_axis(pad, 0, kernel)
                            except:
                                kernel = np.apply_along_axis(pad, 1, kernel)
                        data_list.append(kernel)
        else:
            pass

    return np.stack(data_list)

def makeDataScaler(x_train, x_test, scaler):
    ''' Trains a MinMaxScaler() object on the provided x_train dataset. Additionally it returns the scaled x_train and x_test datasets.

       Parameters:
       >>> x_train: Training data for the scaler. 
       >>> x_test: Test set corresponding to the given training data.
       >>> scaler: Untrained MinMaxScaler() object

       Returns:
       >>> x_train: Scaled input x_train
       >>> x_test: Scaled input x_test
       >>> scaler: Trained MinMaxScaler() object    
    '''

    scaler.fit(x_train.reshape(x_train.shape[0],x_train.shape[1]*x_train.shape[2]))
    x_train = scaler.transform(x_train.reshape(x_train.shape[0],\
                                               x_train.shape[1]*x_train.shape[2])).reshape(x_train.shape[0],x_train.shape[1],x_train.shape[2])
    x_test = scaler.transform(x_test.reshape(x_test.shape[0],\
                                             x_test.shape[1]*x_test.shape[2])).reshape(x_test.shape[0],x_test.shape[1],x_test.shape[2])
    return x_train, x_test, scaler

def trainNewEncoder(x_train, x_test, encd_out_size, M, epochs):
    '''Trains a new autoencoder model using Tensorflow.

       Parameters:
       >>> x_train: Training dataset
       >>> x_test: Validation dataset
       >>> encd_out_size: Required length of compressed data-array (1D)
       >>> M: Number of rows (= number of columns) in the input data-array
       >>> epochs: Number of epochs to train the model for.

       Returns:
       >>> autoencoder: Trained autoencoder model
       >>> history: Training history of the autoencoder    
    '''
    
    autoencoder = Autoencoder(encd_out_size,M) 
    autoencoder.compile(optimizer='adam', loss=losses.MeanSquaredError())
    history = autoencoder.fit(x_train, x_train, epochs=epochs, shuffle=True, validation_data=(x_test, x_test), verbose = 0)
    return autoencoder, history

def scaleNshape(data,scaler):
    '''Scales the values in a 2D array to the [0,1] range using a trained sklearn MinMaxScaler() and flattens to a 1D array.

    Parameters:
    >>> data: 2D numpy array to be reshaped.
    >>> scaler: trained sklearn MinMaxScaler.

    Returns:
    >>> data: scaled and reshaped data    
    '''
    
    data = data.reshape((1,data.shape[0], data.shape[1]))
    data = scaler.transform(data.reshape(data.shape[0],data.shape[1]*data.shape[2])).reshape(data.shape[0],data.shape[1],data.shape[2])
    return data

def pad(data):
    '''Interpolates the given 2D array.'''

    bad_indexes = np.isnan(data)
    good_indexes = np.logical_not(bad_indexes)
    good_data = data[good_indexes]
    interpolated = np.interp(bad_indexes.nonzero()[0], good_indexes.nonzero()[0], good_data)
    data[bad_indexes] = interpolated
    return data

def kernel_extract(filename, location, kernel_size, autoencoder, scaler):
    '''Finds a submatrix of NO2_trophospheric_column data centred at the ground_pixel closest to the given location.

       Parameters:
       >>> filename: Name of the Sentinel 5P L2 product file.
       >>> location: Tuple containing latitude and longitude of the ground site whose measurement data needs to be extracted from the file.
       >>> kernel_size: Grid size of satellte data to be considered
       >>> autoencoder: Trained autoencoder network for convering (kernel_size x kernel_size) data array into (1 x encd_outsize) data array
       >>> scaler: trained sklearn MinMaxScaler.

       Returns:
       >>> date: Date of measurement.
       >>> time: Time of measurement.
       >>> kernel: 1-D numpy array with the encoded data of the measurement over area defined by kernel_size and location.
       >>> qa: the lowest qa_value in the requested data grid.  
    '''

    s5p_PRD = xr.open_dataset('data/L2/' + filename + '.nc', group = 'PRODUCT')       
    no2 = s5p_PRD['nitrogendioxide_tropospheric_column']
    qav = s5p_PRD['qa_value']
    
    gp_index = np.unravel_index(np.abs((no2.longitude[0].values - location[1])**2 \
                                       + (no2.latitude[0].values - location[0])**2).argmin(), no2.longitude[0].values.shape)
    
    M = int(kernel_size/2)
    kernel = no2[0][(gp_index[0]-M):(gp_index[0]+M+1) , (gp_index[1]-M):(gp_index[1]+M+1)].values
    
    qa = qav[0][(gp_index[0]-M):(gp_index[0]+M+1) , (gp_index[1]-M):(gp_index[1]+M+1)].values
    qa[qa == 0] = np.nan
    
    dt = s5p_PRD.time_utc.values[0][gp_index[0]]
    date = re.findall('([\d-]+)T',dt)[0]
    time = re.findall('T([\d:]+).',dt)[0]

    num_of_nans = sum(sum(np.isnan(kernel))) #save the number of missing measurements for evaluation of the kernel

    if kernel.shape == (kernel_size,kernel_size) and num_of_nans<30: 
        
        if(num_of_nans>0):
            try:
                kernel = np.apply_along_axis(pad, 0, kernel) #interpolate missing values along axis 0
            except:
                kernel = np.apply_along_axis(pad, 1, kernel) #interpolate missing values along axis 1

        kernel = scaleNshape(kernel, scaler)
        kernel = autoencoder.encoder(kernel)[0].numpy().tolist()
        qa = np.nanmean(qa)
    else:
        kernel, qa = [np.nan], np.nan
    
    return date, time, kernel, qa

def generateRegressorTrainingData(ground_data_sites, data_files, kernel_size, encd_outsize, autoencoder, scaler):
    ''' Generates a Dataframe of reliable satellite measurements (Sentinel5 QA > 0.5) over an area of fixed size centred at the locations of interest.
        The outputs are encoded by given encoder into a 1D array of dimensions 1 x encd_out_size.

        Parameters:
        >>> ground_data_sites: Iterable containing the names of the locations where measured data of NO2 concentrations is available
        >>> data_files: List of names of Sentinel 5P L2 product files
        >>> kernel_size: Grid size of satellte data to be considered
        >>> encd_outsize: length of the encoded satellite data array
        >>> autoencoder: trained autoencoder network for convering (kernel_size x kernel_size) data array into (1 x encd_outsize) data array 
        >>> scaler: trained sklearn MinMaxScaler.

        Returns:
        >>> satellite_data: Dataframe of satellite measurements.    
    '''
    
    cols = ['Place','Latitude', 'Longitude', 'Date','Time','QA']
    for i in range(1, encd_outsize+1):
        cols.append('S'+str(i))
    
    satellite_data = list()
    
    for idx in ground_data_sites.index:
        for file in data_files:
            if exists('data/L2/%s.nc'%file):
                date, time, kernel, qav = kernel_extract(file,(ground_data_sites['Latitude'][idx], ground_data_sites['Longitude'][idx]), kernel_size, autoencoder, scaler)
                data = [ground_data_sites['Site_Name'][idx],
                        ground_data_sites['Latitude'][idx],
                        ground_data_sites['Longitude'][idx],
                        date,
                        time,
                        qav]
                data.extend(kernel)
                satellite_data.append(data)
            else:
                pass
    
    satellite_data = pd.DataFrame(satellite_data, columns = cols)
    satellite_data = satellite_data.dropna()
    satellite_data = satellite_data[(satellite_data['QA']>=0.5)]
    
    return satellite_data

def getMostRecentSatelliteData(location, datetime, data_files, autoencoder, scaler):
    '''Finds the most recent reliable satellite data for the given location and time. As satellite data is subject to atmospheric conditions,
       the measurements can be unreliable. The Sentinel5P TROPOMI products provide a qa_value for every measurement. As per the User Manual of Sentinel data products,
       measurements with qa_value < 0.5 must be ignored. As a consequence multiple measurements may have to be rejected when searching the data_files in reverse chronology.

       Parameters:
       >>> location: (latitude, longitude) tuple of the location of interest
       >>> datetime: datetime object storing the date and time of interest
       >>> data_files: List of names of Sentinel 5P L2 product files
       >>> autoencoder: trained autoencoder network for convering 2D data array into a compressed 1D data array
       >>> scaler: trained sklearn MinMaxScaler.

       Returns:
       >>> sat_datetime: datetime object storing the date and time  of the satellite measurement data
       >>> kernel: compressed satellite data as a list of 10 floating point numbers   
    '''
    
    sat_datetime, kernel, search_status = 0,0,0

    for filename in data_files:
        if exists('data/L2/%s.nc'%filename):
            file_date = datetime.strptime(filename[36:44], "%Y%m%d")
            if (file_date.date() <= datetime.date()):
                date, time, kernel, qa = kernel_extract(filename, location, 17, autoencoder, scaler)
                if qa > 0.5:
                    search_status = 1
                    sat_datetime = date+' '+time
                    sat_datetime = datetime.strptime(sat_datetime, "%Y-%m-%d %H:%M:%S")
                    if sat_datetime <= datetime:
                        break
                else:
                    continue
        else:
            pass
    
    if search_status == 1:
        minutes_elapsed = (datetime - sat_datetime).seconds/60

        cols = list()
        for i in range(1, 10+1):
            cols.append('S'+str(i))

        cols.append('minutes_elapsed')
        kernel.append(minutes_elapsed)  

        data = pd.DataFrame([kernel], columns=cols)
        data.drop(columns = ['minutes_elapsed']).to_csv('Compressed_Satellite_Data.csv', index=False)
        
        return data
    else:
        raise Exception('No Data')
