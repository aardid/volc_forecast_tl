import os, sys
sys.path.insert(0, os.path.abspath('..'))
from whakaari import TremorData, ForecastModel, load_dataframe, datetimeify
from datetime import timedelta, datetime
from matplotlib import pyplot as plt
import numpy as np
import time, math
from functools import partial
from multiprocessing import Pool
import pandas as pd
import seaborn as sns
import glob
import scipy.cluster.hierarchy as sch
from scipy.spatial.distance import squareform
#
from combined_forecaster import TremorDataCombined, ForecastModelCombined

# tsfresh and sklearn dump a lot of warnings - these are switched off below, but should be
# switched back on when debugging
import logging
logger = logging.getLogger("tsfresh")
logger.setLevel(logging.ERROR)
import warnings
from sklearn.exceptions import FitFailedWarning
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=FitFailedWarning)

# constants
month = timedelta(days=365.25/12)
day = timedelta(days=1)
textsize = 13.
# dictionary of eruption names 
erup_dict = {'WIZ_1': 'Whakaari 2012',
            'WIZ_2': 'Whakaari 2013a',
            'WIZ_3': 'Whakaari 2013b',
            'WIZ_4': 'Whakaari 2016',
            'WIZ_5': 'Whakaari 2019',
            'FWVZ_1': 'Ruapehu 2006',
            'FWVZ_2': 'Ruapehu 2007',
            'FWVZ_3': 'Ruapehu 2009',
            'KRVZ_1': 'Tongariro 2012',
            'KRVZ_2': 'Tongariro 2013',
            'BELO_1': 'Bezymiany 2007a',
            'BELO_2': 'Bezymiany 2007b',
            'BELO_3': 'Bezymiany 2007c',
            'PVV_1': 'Pavlof 2014a',
            'PVV_2': 'Pavlof 2014b',
            'PVV_3': 'Pavlof 2016',
            'VNSS_1': 'Veniaminof 2013',
            'VNSS_2': 'Veniaminof 2018'
            }

##########################################################
# Auxiliary functions
def datetimeify(t):
    """ Return datetime object corresponding to input string.

        Parameters:
        -----------
        t : str, datetime.datetime
            Date string to convert to datetime object.

        Returns:
        --------
        datetime : datetime.datetime
            Datetime object corresponding to input string.

        Notes:
        ------
        This function tries several datetime string formats, and raises a ValueError if none work.
    """
    from pandas._libs.tslibs.timestamps import Timestamp
    if type(t) in [datetime, Timestamp]:
        return t
    fmts = ['%Y-%m-%d %H:%M:%S', '%Y-%m-%d', '%Y %m %d %H %M %S',]
    for fmt in fmts:
        try:
            return datetime.strptime(t, fmt)
        except ValueError:
            pass
    raise ValueError("time data '{:s}' not a recognized format".format(t))
def find_nearest(array, value):
    """ Find nearest to 'value' in array. 
    Return index in 'array' and closest value.   
    """
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx, array[idx]
def moving_average(a, n=3):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n
#
def volc_mom_mag(): 
    '''
    Volcanic moment magnitud (introducing)
    Mv = a*log(Ea)-b
    '''
    # constants
    a = 1.1
    b = 5.
    pre_time = 5*day
    post_time = 5*day
    # dictionary of eruption names 
    erup_dict_vei = {'WIZ_1': ['Whakaari 2012',2],
                'WIZ_2': ['Whakaari 2013a',-1],
                'WIZ_3': ['Whakaari 2013b',-1],
                'WIZ_4': ['Whakaari 2016',1],
                'WIZ_5': ['Whakaari 2019',2],
                'FWVZ_1': ['Ruapehu 2006',1],
                'FWVZ_2': ['Ruapehu 2007',1],
                'FWVZ_3': ['Ruapehu 2009',-1],
                'KRVZ_1': ['Tongariro 2012',2], 
                'KRVZ_2': ['Tongariro 2013',-1],
                'BELO_1': ['Bezymiany 2007a',3],
                'BELO_2': ['Bezymiany 2007b',1],
                'BELO_3': ['Bezymiany 2007c',2],
                'PVV_1': ['Pavlof 2014a',3],
                'PVV_2': ['Pavlof 2014b',1],
                'PVV_3': ['Pavlof 2016',2],
                'VNSS_1': ['Veniaminof 2013',3],
                'VNSS_2': ['Veniaminof 2018',1],
                }
    ## Select eruptions to calculate 
    ## stations (volcanoes)
    ss = ['WIZ','FWVZ','KRVZ','PVV','VNSS','BELO'] # ,'SSLW'
    ## data streams
    ds = ['zsc2_rsamF']
    # file to save 
    with open('vei_vs_vmm.txt', 'w') as f:
        f.write('eruption \tvei \tvmm \n')
        # loop over volcanoes 
        for s in ss:
            fm = ForecastModel(window=2., overlap=1., station = s,
                look_forward=2., data_streams=ds, savefile_type='csv')
            # loop over eruptions (eruption times: fm.data.tes)
            for j in range(len(fm.data.tes)):
                ## Select period (pre and post eruption)
                t0 = fm.data.tes[j] - pre_time
                t1 = fm.data.tes[j] + post_time
                ## Calculate acustic energy
                data = fm.data.get_data(ti=t0, tf=t1)
                # time diferences between points (vector)
                dts = [data.T.columns[i+1]-data.T.columns[i] for i in range(len(data.T.columns)-1)]
                dts = np.asarray([dts[i].total_seconds() for i in range(len(dts))])
                #
                data = data[ds].values[:-1]
                data = [data[i][0] for i in range(len(data))]
                ## Calculate moment base on constants a and b
                Ea = np.dot(data,dts)
                Mv = a * np.log10(Ea) - b
                f.write(erup_dict_vei[s+'_'+str(j+1)][0]+'\t'+
                        str(erup_dict_vei[s+'_'+str(j+1)][1])+'\t'+
                        str(round(Mv,1))+'\n')

def main():
    #
    volc_mom_mag()


if __name__ == "__main__":
    main()
