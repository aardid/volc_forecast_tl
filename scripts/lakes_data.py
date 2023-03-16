import os, sys
sys.path.insert(0, os.path.abspath('..'))
from whakaari import TremorData, ForecastModel, load_dataframe, datetimeify
from datetime import timedelta, datetime
from matplotlib import pyplot as plt
import numpy as np
import time, math
import random
from functools import partial
from multiprocessing import Pool
from scipy.stats import kstest
import pandas as pd
import seaborn as sns
import glob
import csv
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
year = timedelta(days=365)
month = timedelta(days=365.25/12)
day = timedelta(days=1)
hour = day/24
textsize = 12.
# list of eruptions
erup_list = ['WIZ_1','WIZ_2','WIZ_3','WIZ_4','WIZ_5',
            'FWVZ_1','FWVZ_2','FWVZ_3',
            'KRVZ_1','KRVZ_2',
            'BELO_1','BELO_2','BELO_3',
            'PVV_1','PVV_2','PVV_3','VNSS_1','VNSS_2'
            ]           
# dictionary of eruption names 
erup_dict = {'WIZ_1': 'Whakaari 2012',
            'WIZ_2': 'Whakaari 2013a',
            'WIZ_3': 'Whakaari 2013b',
            'WIZ_4': 'Whakaari 2016',
            'WIZ_5': 'Whakaari 2019',
            'FWVZ_1': 'Ruapehu 2006',
            'FWVZ_2': 'Ruapehu 2007',
            'FWVZ_3': 'Ruapehu 2009',
            'KRVZ_1': 'Tongariro 2012a',
            'KRVZ_2': 'Tongariro 2012b',
            'BELO_1': 'Bezymianny 2007a',
            'BELO_2': 'Bezymianny 2007b',
            'BELO_3': 'Bezymianny 2007c',
            'PVV_1': 'Pavlof 2014a',
            'PVV_2': 'Pavlof 2014b',
            'PVV_3': 'Pavlof 2016',
            'VNSS_1': 'Veniaminof 2013',
            'VNSS_2': 'Veniaminof 2018',
            'TBTN_1': 'Telica 2011',
            'TBTN_2': 'Telica 2013',
            'MEA01_1': 'Merapi 2014a',
            'MEA01_2': 'Merapi 2014b',
            'MEA01_3': 'Merapi 2014c',
            'MEA01_4': 'Merapi 2018a',
            'MEA01_5': 'Merapi 2018b',
            'MEA01_6': 'Merapi 2018c',
            'MEA01_7': 'Merapi 2018d',
            'MEA01_8': 'Merapi 2019a',
            'GOD_1' : 'Eyjafjallajökull 2010a',
            'GOD_2' : 'Eyjafjallajökull 2010b'
            }            
# dictionary of eruption VEI 
erup_vei_dict = {'WIZ_1': '1',
            'WIZ_2': '0',
            'WIZ_3': '0-1',
            'WIZ_4': '1',
            'WIZ_5': '2',
            'FWVZ_1': '0',
            'FWVZ_2': '2',
            'FWVZ_3': '0',
            'KRVZ_1': '2',
            'KRVZ_2': '2',
            'BELO_1': '3',
            'BELO_2': '2',
            'BELO_3': '3',
            'PVV_1': '3',
            'PVV_2': '2',
            'PVV_3': '3',
            'VNSS_1': '2',
            'VNSS_2': '3',
            'TBTN_1': ' ',
            'TBTN_2': ' ',
            'GOD_1' : ' ',
            'GOD_2' : ' '
            }
# station code dic
sta_code = {'WIZ': 'Whakaari',
            'FWVZ': 'Ruapehu',
            'KRVZ': 'Tongariro',
            'BELO': 'Bezymiany',
            'PVV': 'Pavlof',
            'VNSS': 'Veniaminof',
            'IVGP': 'Vulcano',
            'AUS': 'Agustine',
            'TBTN': 'Telica',
            'OGDI': 'Reunion',
            'TBTN': 'Telica',
            'MEA01': 'Merapi',
            'GOD' : 'Eyjafjallajökull',
            'ONTA' : 'Ontake',
            'REF' : 'Redoubt',
            'POS' : 'Kawa Ijen',
            'DAM' : 'Kawa Ijen'
            }
# eruption times
erup_times = {'WIZ_1': '2012 08 04 16 52 00',
            'WIZ_2': '2013 08 19 22 23 00',
            'WIZ_3': '2013 10 11 07 09 00',
            'WIZ_4': '2016 04 27 09 37 00',
            'WIZ_5': '2019 12 09 01 11 00',
            'FWVZ_1': '2006 10 04 09 30 00',
            'FWVZ_2': '2007 09 25 08 20 00',
            'FWVZ_3': '2009 07 13 06 30 00',
            'FWVZ_4': '2021 09 09 12 00 00',
            'KRVZ_1': '2012 08 06 11 50 00',
            'KRVZ_2': '2012 11 21 00 20 00',
            'BELO_1': '2007 09 25 08 30 00',
            'BELO_2': '2007 10 14 14 27 00',
            'BELO_3': '2007 11 05 08 43 00',
            'PVV_1': '3',
            'PVV_2': '2',
            'PVV_3': '3',
            'VNSS_1': '2013 06 13 00 00 00',
            'VNSS_2': '2018 09 04 00 00 00',
            'TBTN_1': '2011 03 07 12 00 00',
            'TBTN_2': '2013 09 25 12 00 00',
            'MEA01_1': '2014 03 09 12 00 00',
            'MEA01_2': '2014 03 27 12 00 00',
            'MEA01_3': '2014 04 19 12 00 00',
            'MEA01_4': '2018 05 11 12 00 00',
            'MEA01_5': '2018 05 21 12 00 00',
            'MEA01_6': '2018 05 22 12 00 00',
            'MEA01_7': '2018 05 24 12 00 00',
            'MEA01_8': '2019 10 14 12 00 00',
            'GOD_1' : '2010 03 20 12 00 00',
            'GOD_2' : '2010 04 14 12 00 00',
            'ONTA_1' : '2014 09 27 11 52 00',
            'REF_1' : '2009 03 15 21 05 00',
            'POS_1' : '2013 04 01 00 00 00',
            'DAM_1' : '2013 03 20 00 00 00'
            }

# missed eruption times in ruapehu
erup_times_missed_ruap = {
            'FWVZ_1': '2009 07 13 06 30 00',
            'FWVZ_2': '2021 09 09 12 00 00'
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
    try:
        if '+' in t:
            t = t.split('+')[0]
    except:
        pass
    from pandas._libs.tslibs.timestamps import Timestamp
    if type(t) in [datetime, Timestamp]:
        return t
    fmts = ['%Y-%m-%d %H:%M:%S', '%Y-%m-%d', '%Y %m %d %H %M %S','%d/%m/%Y %H:%M', '%Y%m%d:%H%M']
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
def to_nztimezone(t):
    """ Routine to convert UTC to NZ time zone.
    """
    from dateutil import tz
    utctz = tz.gettz('UTC')
    nztz = tz.gettz('Pacific/Auckland')
    return [ti.replace(tzinfo=utctz).astimezone(nztz) for ti in pd.to_datetime(t)]
    def conv(at, x):
        y = ((x-np.mean(x))/np.std(x)*at.values).mean()
        return y
def chqv(y):
    '''
    quick calc of change quantile variance .4-.6
    for calling:
        df[dt].rolling(N*24*6).apply(chqv)[N*24*6:]
    '''
    y0,y1 = np.percentile(y, [40,60])
    # return y1-y0
    inds = np.where((y>y0)&(y<y1))
    return np.var(np.diff(y, prepend=0)[inds])
def conv(at, x):
    y = ((x-np.mean(x))/np.std(x)*at.values).mean()
    return y
#
##########################################################
# if both activate, results are saved in auto_picked folder

auto_picked = True
man_picked = False 

def plot_temp_data():
    '''
    plot temperature data from file 
    '''
    from obspy import UTCDateTime
    # convert to UTC 0
    utc_0 = True
    #
    t0 = "2022-01-01"#"2011-01-15"#"2021-07-01"#"2016-04-01"#"2021-08-09"
    t1 = "2022-04-08"#"2011-05-01"#"2021-12-30"#"2016-08-30"#"2021-09-09"
    #
    sta = 'FWVZ'#'POS'#'FWVZ'
    if sta == 'POS':
        t0 = "2012-01-01"
        t1 = "2013-06-01"
    if sta == 'COP':
        t0 = "2020-03-06"
        t1 = "2020-11-25"
    #
    plot_erup = False
    if plot_erup:
        te = datetimeify('2009 07 13 06 30 00')
    #
    nrow = 1
    ncol = 1
    fig, ax = plt.subplots(nrows=nrow, ncols=ncol,figsize=(10,2))#(14,4))
    #
    col = ['r','g','b']
    alpha = [.5, 1., 1.]
    thick_line = [1., 3., 3.]
    ## plot other data
    temp = True
    level = False
    rainfall = False
    #
    mov_avg = True # moving average for temp and level data
        
    # plot temp data
    if temp:
        if sta == 'FWVZ':
            ti_e1 = t0
            tf_e1 = t1
            # import temp data
            path = '..'+os.sep+'data'+os.sep+"RU001_temp_data.csv"
            pd_temp = pd.read_csv(path, index_col=1)
            if utc_0:
                pd_temp.index = [datetimeify(pd_temp.index[i])-6*hour for i in range(len(pd_temp.index))]
            else:
                pd_temp.index = [datetimeify(pd_temp.index[i]) for i in range(len(pd_temp.index))]
            # plot data in axis twin axis
            # Trim the data
            temp_e1_tim = pd_temp[ti_e1: tf_e1].index.values

            #temp_e1_tim=to_nztimezone(temp_e1_tim)
            #
            temp_e1_val = pd_temp[ti_e1: tf_e1].loc[:,' t (C)'].values
            # ax2
            #ax2b = ax2.twinx()   
            if mov_avg: # plot moving average
                #n=10
                #v_plot = (temp_e1_val-np.min(temp_e1_val))/np.max((temp_e1_val-np.min(temp_e1_val)))
                v_plot = temp_e1_val
                ax.plot(temp_e1_tim, v_plot, '-', color='g', label='lake temperature', alpha = 1.)
                #
                temp_e1_val = pd_temp[ti_e1: tf_e1].loc[:,' t (C)'].rolling(window=20).mean()
                ax.plot(temp_e1_tim, temp_e1_val, '--', color='k', alpha = 1.)
                #ax.plot(temp_e1_tim[:-n+1], moving_average(temp_e1_val, n=n), '--', color='k')#, label='temp. mov. avg.')
                #ax.plot(temp_e1_tim[n-1-20:-20], moving_average(v_plot[::-1], n=n)[::-1], '--', color='k')#, label='lake temperature')
            else:
                #ax2b.plot(temp_e1_tim, temp_e1_val, '-', color='g', label='temperature')
                #ax2.set_ylim([-40,40])
                #plt.show()
                v_plot = temp_e1_val
                ax.plot(temp_e1_tim, v_plot, '-', color='g', label='lake temperature', alpha = 1.)
            ax.set_ylabel('Temperature °C')
            #ax.set_ylim([32,40])
            #ax2b.set_ylabel('temperature C')   
            #ax.legend(loc = 2)         
        if sta == 'POS':
            ti_e1 = t0
            tf_e1 = t1
            # import temp data
            path = '..'+os.sep+'data'+os.sep+"POS_temp_data.csv"
            pd_temp = pd.read_csv(path, index_col=1)
            if utc_0:
                pd_temp.index = [datetimeify(pd_temp.index[i])-6*hour for i in range(len(pd_temp.index))]
            else:
                pd_temp.index = [datetimeify(pd_temp.index[i]) for i in range(len(pd_temp.index))]
            # plot data in axis twin axis
            # Trim the data
            temp_e1_tim = pd_temp[ti_e1: tf_e1].index.values

            #temp_e1_tim=to_nztimezone(temp_e1_tim)
            #
            temp_e1_val = pd_temp[ti_e1: tf_e1].loc[:,' t (C)'].values
            # ax2
            #ax2b = ax2.twinx()   
            if mov_avg: # plot moving average
                n=50
                #v_plot = (temp_e1_val-np.min(temp_e1_val))/np.max((temp_e1_val-np.min(temp_e1_val)))
                v_plot = temp_e1_val
                ax.plot(temp_e1_tim, v_plot, '-', color='g', label='lake temperature', alpha = 1.)
                #
                #ax.plot(temp_e1_tim[:-n+1], moving_average(temp_e1_val, n=n), '--', color='k')#, label='temp. mov. avg.')
                ax.plot(temp_e1_tim[n-1-20:-20], moving_average(v_plot[::-1], n=n)[::-1], '--', color='k')#, label='lake temperature')
            else:
                #ax2b.plot(temp_e1_tim, temp_e1_val, '-', color='g', label='temperature')
                #ax2.set_ylim([-40,40])
                #plt.show()
                v_plot = temp_e1_val
                ax.plot(temp_e1_tim, v_plot, '-', color='g', label='lake temperature', alpha = 1.)
            ax.set_ylabel('Temperature °C')
            #ax.set_ylim([32,40])
        if sta == 'COP':
            ti_e1 = t0
            tf_e1 = t1
            # import temp data
            path = '..'+os.sep+'data'+os.sep+"COP_temp_data.csv"
            pd_temp = pd.read_csv(path, index_col=1)
            if utc_0:
                pd_temp.index = [datetimeify(pd_temp.index[i])-6*hour for i in range(len(pd_temp.index))]
            else:
                pd_temp.index = [datetimeify(pd_temp.index[i]) for i in range(len(pd_temp.index))]
            # plot data in axis twin axis
            # Trim the data
            temp_e1_tim = pd_temp[ti_e1: tf_e1].index.values

            #temp_e1_tim=to_nztimezone(temp_e1_tim)
            #
            temp_e1_val = pd_temp[ti_e1: tf_e1].loc[:,' t (C)'].values
            # ax2
            #ax2b = ax2.twinx()   
            mov_avg = False
            if mov_avg: # plot moving average
                n=50
                #v_plot = (temp_e1_val-np.min(temp_e1_val))/np.max((temp_e1_val-np.min(temp_e1_val)))
                v_plot = temp_e1_val
                ax.plot(temp_e1_tim, v_plot, '-', color='g', label='lake temperature', alpha = 1.)
                #
                #ax.plot(temp_e1_tim[:-n+1], moving_average(temp_e1_val, n=n), '--', color='k')#, label='temp. mov. avg.')
                ax.plot(temp_e1_tim[n-1-20:-20], moving_average(v_plot[::-1], n=n)[::-1], '--', color='k')#, label='lake temperature')
            else:
                #ax2b.plot(temp_e1_tim, temp_e1_val, '-', color='g', label='temperature')
                #ax2.set_ylim([-40,40])
                #plt.show()
                v_plot = temp_e1_val
                ax.plot(temp_e1_tim, v_plot, '-', color='g', label='lake temperature', alpha = 1.)
            ax.set_ylabel('Temperature °C')

            if True: # plot so2
                ax2 = ax.twinx()
                # import temp data
                path = '..'+os.sep+'data'+os.sep+"COP_so2_data.csv"
                pd_temp = pd.read_csv(path, index_col=0)
                if utc_0:
                    pd_temp.index = [datetimeify(pd_temp.index[i])-6*hour for i in range(len(pd_temp.index))]
                else:
                    pd_temp.index = [datetimeify(pd_temp.index[i]) for i in range(len(pd_temp.index))]
                # plot data in axis twin axis
                # Trim the data
                temp_e1_tim = pd_temp[ti_e1: tf_e1].index.values

                #temp_e1_tim=to_nztimezone(temp_e1_tim)
                #
                temp_e1_val = pd_temp[ti_e1: tf_e1].loc[:,'SO2_column_number_density_15km'].values
                # ax2
                #ax2b = ax2.twinx()   
                mov_avg = False
                if mov_avg: # plot moving average
                    n=50
                    #v_plot = (temp_e1_val-np.min(temp_e1_val))/np.max((temp_e1_val-np.min(temp_e1_val)))
                    v_plot = temp_e1_val
                    ax2.plot(temp_e1_tim, v_plot, '-', color='r', label='lake temperature', alpha = 1.)
                    #
                    #ax.plot(temp_e1_tim[:-n+1], moving_average(temp_e1_val, n=n), '--', color='k')#, label='temp. mov. avg.')
                    ax2.plot(temp_e1_tim[n-1-20:-20], moving_average(v_plot[::-1], n=n)[::-1], '--', color='r')#, label='lake temperature')
                else:
                    #ax2b.plot(temp_e1_tim, temp_e1_val, '-', color='g', label='temperature')
                    #ax2.set_ylim([-40,40])
                    #plt.show()
                    v_plot = temp_e1_val
                    ax2.plot(temp_e1_tim, v_plot, '-', color='r', label='lake temperature', alpha = 1.)
                ax2.set_ylabel('SO2')
            #ax.set_ylim([32,40])
    # plot lake level data

    if level:
        axb = ax.twinx()
        if sta == 'FWVZ':
            # plot lake level data
            try:
                ti_e1 = t0 
                tf_e1 = t1
                #ax2b = ax2.twinx()
                #
                # import temp data
                if True:#t0.year > 2015:
                    path = '..'+os.sep+'data'+os.sep+"RU001A_level_data_full.csv"
                else:
                    pass
                    #path = '..'+os.sep+'data'+os.sep+"RU001_level_data.csv"         
                #    
                pd_temp = pd.read_csv(path, index_col=1)
                if utc_0:
                    pd_temp.index = [datetimeify(pd_temp.index[i])-6*hour for i in range(len(pd_temp.index))]
                    pd_temp.index = [datetimeify(pd_temp.index[i])-0*hour for i in range(len(pd_temp.index))]
                else:
                    pd_temp.index = [datetimeify(pd_temp.index[i]) for i in range(len(pd_temp.index))]

                if True: # rolling median over data
                    N = 2
                    pd_temp = pd_temp[:].rolling(60).median()#[N*24*6:]

                # plot data in axis twin axis
                # Trim the data
                temp_e1_tim = pd_temp[ti_e1: tf_e1].index.values
                #temp_e1_tim=to_nztimezone(temp_e1_tim)
                temp_e1_val = pd_temp[ti_e1: tf_e1].loc[:,' z (m)'].values
                # ax2
                #ax2b = ax2.twinx()
                if False:#mov_avg: # plot moving average
                    n=30
                    v_plot = temp_e1_val
                    axb.plot(temp_e1_tim, v_plot, '-', color='gray')
                    axb.plot([], [], '-', color='gray', label='lake level')
                    #ax2b.plot(temp_e1_tim[:-n+1], moving_average(temp_e1_val, n=n), '--', color='k', label='temp. mov. avg.')
                    #v_plot = (temp_e1_val-np.min(temp_e1_val))/np.max((temp_e1_val-np.min(temp_e1_val)))
                    axb.plot(temp_e1_tim[n-1-20:-20], moving_average(v_plot[::-1], n=n)[::-1], '--', color='k')#, label='lake level')
                    #ax2b.plot(temp_e1_tim, v_plot, '-', color='b', alpha = 0.3)
                else:
                    #ax2b.plot(temp_e1_tim, temp_e1_val, '-', color='b', label='level')
                    #ax2.set_ylim([-40,40])
                    #plt.show()
                    #v_plot = (temp_e1_val-np.min(temp_e1_val))/np.max((temp_e1_val-np.min(temp_e1_val)))
                    v_plot = temp_e1_val
                    axb.plot(temp_e1_tim, v_plot, '-', color='gray', label='lake level')
                #
                axb.set_ylabel('Lake level cm') 

                if False: # plot vertical lines
                    te = datetimeify(erup_time)#fm_e1.data.tes[int(erup[-1:])-1]
                    axb.axvline(te, color='k',linestyle='--', linewidth=2, zorder = 4)
                    #ax2.plot([], color='k', linestyle='--', linewidth=2, label = 'eruption')
            except:
                pass

    # plot rainfall data
    if rainfall:
        ti_e1 = t0#datetimeify(t0)
        tf_e1 = t1#datetimeify(t1)
        if sta == 'FWVZ':
            # import temp data
            path = '..'+os.sep+'data'+os.sep+"ruapehu_chateau_rainfall_data.csv"
            path = '..'+os.sep+'data'+os.sep+"ruapehu_chateau_rainfall_data_2021_09.csv"
            pd_rf = pd.read_csv(path, index_col=1)
            pd_rf.index = pd.to_datetime(pd_rf.index, format='%Y%m%d:%H%M')
            #pd_rf.index = [datetimeify(pd_temp.index[i]) for i in range(len(pd_temp.index))]
            if utc_0:
                pd_rf.index = [datetimeify(pd_rf.index[i])-6*hour for i in range(len(pd_rf.index))]
            else:
                pd_rf.index = [datetimeify(pd_rf.index[i]) for i in range(len(pd_rf.index))]
            # plot data in axis twin axis
            # Trim the data
            rf_e2_tim = pd_rf[ti_e1: tf_e1].index.values
            rf_e2_val = pd_rf[ti_e1: tf_e1].loc[:,'Amount(mm)'].values /4
            # ax2
            #ax2b = ax2.twinx()
            v_plot = (rf_e2_val-np.min(rf_e2_val))/np.max((rf_e2_val-np.min(rf_e2_val)))
            v_plot = v_plot*3 + 32
            ax.plot(rf_e2_tim, v_plot, '-', color='c', label='rain fall', alpha = 0.8)
            #ax2b.set_ylabel('temperature C')
            #ax2b.legend(loc = 1)
    
    if False: # not implemented
        # plot lake ph data
        if ph:
            if e1[:-2] == 'FWVZ':
                pass
            if e2[:-2] == 'FWVZ':
                # import temp data
                path = '..'+os.sep+'data'+os.sep+"RU001_ph_data.csv"
                pd_temp = pd.read_csv(path, index_col=1)
                pd_temp.index = [datetimeify(pd_temp.index[i]) for i in range(len(pd_temp.index))]
                # plot data in axis twin axis
                # Trim the data
                temp_e1_tim = pd_temp[ti_e2: tf_e2].index.values
                temp_e1_val = pd_temp[ti_e2: tf_e2].loc[:,' ph (-)'].values
                # ax2
                #ax2b = ax2.twinx()
                if False: # plot moving average
                    n=40
                    #ax2b.plot(temp_e1_tim[:-n+1], moving_average(temp_e1_val, n=n), '--', color='k', label='temp. mov. avg.')
                    ax2b.plot(temp_e1_tim[n-1-20:-20], moving_average(temp_e1_val[::-1], n=n)[::-1], '-', color='b', label='level mov. avg.')
                    ax2b.plot(temp_e1_tim, temp_e1_val, '-', color='y', label='ph', alpha = 0.5)
                else:
                    ax2b.plot(temp_e1_tim, temp_e1_val/np.max(temp_e1_val) - np.min(temp_e1_val), '-', color='y', label='ph')
                #ax2b.set_ylabel('temperature C')
                ax2.set_ylim([-40,40])
                ax2b.legend(loc = 3)            
        # plot displacement data (from Chateau Observatory)
        if u:
            if e1[:-2] == 'FWVZ':
                # import temp data
                path = '..'+os.sep+'data'+os.sep+"VGOB_u_disp_abs_data.csv"
                pd_temp = pd.read_csv(path, index_col=1)
                pd_temp.index = [datetimeify(pd_temp.index[i]) for i in range(len(pd_temp.index))]
                # plot data in axis twin axis
                # Trim the data
                temp_e1_tim = pd_temp[ti_e1: tf_e1].index.values
                temp_e1_val = pd_temp[ti_e1: tf_e1].loc[:,' u (mm)'].values
                # ax2
                #ax2b = ax2.twinx()
                v_plot = (temp_e1_val-np.min(temp_e1_val))/np.max((temp_e1_val-np.min(temp_e1_val)))
                ax1b.plot(temp_e1_tim, v_plot, '-', color='y', label='Chateau displacement')
                if False: # error bars 
                    temp_e1_val_err = pd_temp[ti_e2: tf_e2].loc[:,' error (mm)'].values
                    v_plot_er = (temp_e1_val_err-np.min(temp_e1_val_err))/np.max((temp_e1_val_err-np.min(temp_e1_val_err)))
                    #ax1b.errorbar(temp_e1_tim, v_plot, v_plot_er/3, color='y')
                #ax1b.set_ylabel('temperature C')
                #ax1b.legend(loc = 3)  
            if e2[:-2] == 'FWVZ':
                # import temp data
                path = '..'+os.sep+'data'+os.sep+"VGOB_u_disp_abs_data.csv"
                pd_temp = pd.read_csv(path, index_col=1)
                pd_temp.index = [datetimeify(pd_temp.index[i]) for i in range(len(pd_temp.index))]
                # plot data in axis twin axis
                # Trim the data
                temp_e1_tim = pd_temp[ti_e2: tf_e2].index.values
                temp_e1_val = pd_temp[ti_e2: tf_e2].loc[:,' u (mm)'].values
                # ax2
                #ax2b = ax2.twinx()
                v_plot = (temp_e1_val-np.min(temp_e1_val))/np.max((temp_e1_val-np.min(temp_e1_val)))
                ax2b.plot(temp_e1_tim, v_plot, '-', color='y', label='Chateau displacement')
                if False: # error bars 
                    temp_e1_val_err = pd_temp[ti_e2: tf_e2].loc[:,' error (mm)'].values
                    v_plot_er = (temp_e1_val_err-np.min(temp_e1_val_err))/np.max((temp_e1_val_err-np.min(temp_e1_val_err)))
                    #ax2b.errorbar(temp_e1_tim, v_plot, v_plot_er/3, color='y')
                #ax2b.set_ylabel('temperature C')
                #ax2b.legend(loc = 3)            
        # plot chloride data 
        if cl:
            if e1[:-2] == 'FWVZ':
                # import temp data
                path = '..'+os.sep+'data'+os.sep+"RU001_cl_data.csv"
                pd_temp = pd.read_csv(path, index_col=1)
                pd_temp.index = [datetimeify(pd_temp.index[i]) for i in range(len(pd_temp.index))]
                # plot data in axis twin axis
                # Trim the data
                temp_e1_tim = pd_temp[ti_e1: tf_e1].index.values
                temp_e1_val = pd_temp[ti_e1: tf_e1].loc[:,' Cl-w (mg/L)'].values
                # ax2
                #ax2b = ax2.twinx()
                v_plot = (temp_e1_val-np.min(temp_e1_val))/np.max((temp_e1_val-np.min(temp_e1_val)))
                ax1b.plot(temp_e1_tim, v_plot, '--', color='k', label='Lake Cl concentration')
                if False: # error bars 
                    temp_e1_val_err = pd_temp[ti_e2: tf_e2].loc[:,' error (mm)'].values
                    v_plot_er = (temp_e1_val_err-np.min(temp_e1_val_err))/np.max((temp_e1_val_err-np.min(temp_e1_val_err)))
                    #ax1b.errorbar(temp_e1_tim, v_plot, v_plot_er/3, color='y')
                #ax1b.set_ylabel('temperature C')
                #ax1b.legend(loc = 3)  
            if e2[:-2] == 'FWVZ':
                # import temp data
                path = '..'+os.sep+'data'+os.sep+"RU001_cl_data.csv"
                pd_temp = pd.read_csv(path, index_col=1)
                pd_temp.index = [datetimeify(pd_temp.index[i]) for i in range(len(pd_temp.index))]
                # plot data in axis twin axis
                # Trim the data
                temp_e1_tim = pd_temp[ti_e2: tf_e2].index.values
                temp_e1_val = pd_temp[ti_e2: tf_e2].loc[:,' Cl-w (mg/L)'].values
                # ax2
                #ax2b = ax2.twinx()
                v_plot = (temp_e1_val-np.min(temp_e1_val))/np.max((temp_e1_val-np.min(temp_e1_val)))
                ax2b.plot(temp_e1_tim, v_plot, '--', color='k', label='Lake Cl concentration')
                if False: # error bars 
                    temp_e1_val_err = pd_temp[ti_e2: tf_e2].loc[:,' error (mm)'].values
                    v_plot_er = (temp_e1_val_err-np.min(temp_e1_val_err))/np.max((temp_e1_val_err-np.min(temp_e1_val_err)))
                    #ax2b.errorbar(temp_e1_tim, v_plot, v_plot_er/3, color='y')
                #ax2b.set_ylabel('temperature C')
                #ax2b.legend(loc = 3)       
        # plot chloride data 
        if so4:
            if e1[:-2] == 'FWVZ':
                # import temp data
                path = '..'+os.sep+'data'+os.sep+"RU001_so4_data.csv"
                pd_temp = pd.read_csv(path, index_col=1)
                pd_temp.index = [datetimeify(pd_temp.index[i]) for i in range(len(pd_temp.index))]
                # plot data in axis twin axis
                # Trim the data
                temp_e1_tim = pd_temp[ti_e1: tf_e1].index.values
                temp_e1_val = pd_temp[ti_e1: tf_e1].loc[:,' SO4-w (mg/L)'].values
                # ax2
                #ax2b = ax2.twinx()
                v_plot = (temp_e1_val-np.min(temp_e1_val))/np.max((temp_e1_val-np.min(temp_e1_val)))
                ax1b.plot(temp_e1_tim, v_plot, '--', color='k', label='Lake SO4 concentration')
                if False: # error bars 
                    temp_e1_val_err = pd_temp[ti_e2: tf_e2].loc[:,' error (mm)'].values
                    v_plot_er = (temp_e1_val_err-np.min(temp_e1_val_err))/np.max((temp_e1_val_err-np.min(temp_e1_val_err)))
                    #ax1b.errorbar(temp_e1_tim, v_plot, v_plot_er/3, color='y')
                #ax1b.set_ylabel('temperature C')
                #ax1b.legend(loc = 3)  
            if e2[:-2] == 'FWVZ':
                # import temp data
                path = '..'+os.sep+'data'+os.sep+"RU001_so4_data.csv"
                pd_temp = pd.read_csv(path, index_col=1)
                pd_temp.index = [datetimeify(pd_temp.index[i]) for i in range(len(pd_temp.index))]
                # plot data in axis twin axis
                # Trim the data
                temp_e1_tim = pd_temp[ti_e2: tf_e2].index.values
                temp_e1_val = pd_temp[ti_e2: tf_e2].loc[:,' SO4-w (mg/L)'].values
                # ax2
                #ax2b = ax2.twinx()
                v_plot = (temp_e1_val-np.min(temp_e1_val))/np.max((temp_e1_val-np.min(temp_e1_val)))
                ax2b.plot(temp_e1_tim, v_plot, '--', color='k', label='Lake SO4 concentration')
                if False: # error bars 
                    temp_e1_val_err = pd_temp[ti_e2: tf_e2].loc[:,' error (mm)'].values
                    v_plot_er = (temp_e1_val_err-np.min(temp_e1_val_err))/np.max((temp_e1_val_err-np.min(temp_e1_val_err)))
                    #ax2b.errorbar(temp_e1_tim, v_plot, v_plot_er/3, color='y')
                #ax2b.set_ylabel('temperature C')
                #ax2b.legend(loc = 3)
    #
    if False:
        te = datetimeify("2021 09 07 22 10 00")#fm_e1.data.tes[int(erup[-1:])-1]
        #te = datetimeify("2009 07 13 06 30 00")#fm_e1.data.tes[int(erup[-1:])-1]
        ax.axvline(te, color='k',linestyle='--', linewidth=2, zorder = 4)
        ax.plot([], color='k', linestyle='--', linewidth=2, label = 'rsam peak')
    #
    ax.legend(loc = 2, framealpha = 1.0)
    if False:
        axb.set_xlim([datetimeify("2021 08 09 00 00 00"),datetimeify("2021 09 10 00 00 00")])
    #
    if plot_erup:
        ax.axvline(te, color='k',linestyle='--', linewidth=3, zorder = 4)
        ax.plot([], color='k', linestyle='--', linewidth=3, label = 'event') 
    #
    ax.grid()
    #ax.set_ylabel('feature value')
    #ax.set_xticks([datetimeify(t_aux) - 7*day*i for i in range(int(30/7)+1)])#[dat.index.values[0],dat.index.values[-1]])#, ]np.arange(0, len(x)+1, 5))
    #
    plt.show()

## seismic data 

def cc_over_record():
    '''
    '''
    def conv(at, x):
        y = ((x-np.mean(x))/np.std(x)*at.values).mean()
        return y
    def chqv(y):
        '''
        quick calc of change quantile variance .4-.6
        for calling:
            df[dt].rolling(N*24*6).apply(chqv)[N*24*6:]
        '''
        y0,y1 = np.percentile(y, [40,60])
        # return y1-y0
        inds = np.where((y>y0)&(y<y1))
        return np.var(np.diff(y, prepend=0)[inds])
    #
    feat = 'median'
    #feat = 'rate_variance'
    # select archetype
    if True: # dsar median WIZ 2019
        sta_arch = 'WIZ'
        dt = 'zsc2_dsarF'
        day = timedelta(days=1)
        fm = ForecastModel(window=2., overlap=1., station=sta_arch,
            look_forward=2., data_streams=[dt], 
            data_dir=r'C:\Users\aar135\codes_local_disk\volc_forecast_tl\volc_forecast_tl\data'
            )
        te = fm.data.tes[-1]
        # rolling median and signature length window
        N, M = [2,30]
        # time
        j = fm.data.df.index
        # construct signature
        if feat == 'median':
            df = fm.data.df[(j>(te-(M+N)*day))&(j<te+4*day)]
            archtype = df[dt].rolling(N*24*6).median()[N*24*6:]
        if feat == 'rate_variance':
            df = fm.data.df[(j>(te-(M+N)*day))&(j<te+2*day)]
            archtype = df[dt].rolling(N*24*6).apply(chqv)[N*24*6:]
    # convolve over the data (select record)
    if True:
        sta_rec = 'COP'#'FWVZ'#'POS'#'FWVZ'
        day = timedelta(days=1)
        fm = ForecastModel(window=2., overlap=1., station=sta_rec,
            look_forward=2., data_streams=[dt], 
            data_dir=r'C:\Users\aar135\codes_local_disk\volc_forecast_tl\volc_forecast_tl\data'
            )
        #
        df = fm.data.df[:]
        if feat == 'median':
            test = df[dt].rolling(N*24*6).median()[N*24*6:]
        if feat == 'rate_variance':
            test = df[dt].rolling(N*24*6).apply(chqv)[N*24*6:]
        #
        out = test.rolling(archtype.shape[0]).apply(partial(conv, (archtype-archtype.mean())/archtype.std()))       
        out = out.resample('1D').ffill()
    #
    _inds = out.index
    _cc = out.values
    #
    # write output
    path = 'C:\\Users\\aar135\\codes_local_disk\\volc_forecast_tl\\volc_forecast_tl\\features\\lake_data\\corr_dsar_ratevar\\'
    with open(path+sta_arch+'_'+str(te.year)+'_over_'+sta_rec+'_'+dt+'_'+feat+'.txt', 'w') as f:
        f.write('endtime,cc'+'\n')
        for i in range(len(_cc)):
            if not math.isnan(_cc[i]):
                f.write(str(_inds[i])+','+str(_cc[i])+'\n')
    ##

def locate_missed_events_seismic():
    '''
    Search in Ruapehu using DSAR and RATE VAR for previous eruptions to locate missed events. 
    '''
    sta = 'FWVZ'#'COP'#'FWVZ''POS'

    if sta == 'FWVZ':
        _from = datetimeify('2006-01-01')
        dsar_median =  True
        dsar_rv =  False
    if sta == 'POS':
        _from = datetimeify('2010-10-24')
        dsar_median =  True
        dsar_rv =  False
    if sta == 'COP':
        _from = datetimeify('2020-03-09')
        dsar_median =  True
        dsar_rv =  False
    pass
    path = 'C:\\Users\\aar135\\codes_local_disk\\volc_forecast_tl\\volc_forecast_tl\\features\\lake_data\\corr_dsar_ratevar\\'
    #
    if dsar_median:
        cc_threshold_median = 0.2#0.35#0.25#.5
        cc_threshold = cc_threshold_median#0.25#.5
    if dsar_rv:
        cc_threshold_rv = 0.3#0.1#.2
        cc_threshold = cc_threshold_rv#0.1#.2
    if dsar_median and dsar_rv:
        cc_threshold = 0.65#0.35
    #
    # import dsar median WIZ 2019 cc's over FWVZ
    if dsar_median:
        if False: # import cc from previus estimations (corr ana)
            path1 = 'WIZ_5_zsc2_dsarF__median_over_'+sta+'.csv'
            pd_cc_dsar_median = pd.read_csv(path + path1, index_col=0)
            pd_cc_dsar_median.index = [datetimeify(pd_cc_dsar_median.index[i]) for i in range(len(pd_cc_dsar_median.index))]
        #
        if True: # import cc from local estimation
            path1 = 'WIZ_2019_over_'+sta+'_zsc2_dsarF_median.txt'
            pd_cc_dsar_median = pd.read_csv(path + path1, index_col=0, sep = ',')
            pd_cc_dsar_median.index = [datetimeify(pd_cc_dsar_median.index[i]) for i in range(len(pd_cc_dsar_median.index))]# if datetimeify(pd_cc_dsar_median.index[i]) > _from]
    # import dsar rate var FWVZ 2006 cc's over FWVZ
    if dsar_rv:
        if False: # import cc from previus estimations (corr ana)
            path2 = 'FWVZ_2_zsc2_dsarF__change_quantiles__f_agg_-var-__isabs_False__qh_0.6__ql_0.4_over_FWVZ.csv'
            pd_cc_dsar_rv = pd.read_csv(path + path2, index_col=0)
            pd_cc_dsar_rv.index = [datetimeify(pd_cc_dsar_rv.index[i]) for i in range(len(pd_cc_dsar_rv.index))]
        if True: # import cc from local estimation
            path1 = 'WIZ_2019_over_FWVZ_zsc2_dsarF_rate_variance.txt'
            pd_cc_dsar_rv = pd.read_csv(path + path1, index_col=0, sep = ',')
            pd_cc_dsar_rv.index = [datetimeify(pd_cc_dsar_rv.index[i]) for i in range(len(pd_cc_dsar_rv.index))]
    #
    if False: # plot histograms
        #
        fig, (ax1, ax2) = plt.subplots(nrows = 1, ncols = 2, figsize = (8,4))
        colors = ['b', 'r', 'g', 'm']
        
        # select lists 
        #ax.axvline(x=0.05, color = 'k', ls='--', linewidth=1, label = '0.05 threshold')
        multi = [pd_cc_dsar_median['cc'], pd_cc_dsar_rv['cc']]
        colors = ['r', 'b']
        labels = ['dsarF_median', 'dsarF_rv']
        bins = 20#np.linspace(0, 1, 13)
        #bins = np.linspace(0, 1, 20)
        ax1.hist(multi, bins, color = colors, label=labels, density = True)
        #xlim = [0, 7] #12.5]

        ax1.set_title(' ')
        ax1.set_xlabel('cc')
        ax1.set_ylabel('pdf')
        ax1.legend()

        ##
        multi = [pd_cc['cc'], pd_cc['cc']]#, dT_in_rate_nonerup]
        colors = ['m', 'm']
        labels = ['sum']
        bins = 20#np.linspace(0, 1, 13)
        #bins = np.linspace(0, 1, 20)
        ax2.hist(multi, bins, color = colors, label=labels, density = False)
        xlim = None #[0, 4.2]#5] 8]

        ax2.set_title(' ')
        ax2.set_xlabel('cc')
        ax2.set_xlim(xlim)
        #ax2.set_xscale('log')
        
        plt.legend(loc='upper right')
        plt.show()
    
    if True: # explore dates of maximum cc
        if dsar_median:
            pass
            #pd_cc_dsar_median = pd_cc_dsar_median.sort_values('cc', ascending=False)
        if dsar_rv:
            pass
            #pd_cc_dsar_rv = pd_cc_dsar_rv.sort_values('cc', ascending=False)
        #
        filter_by_cont = False # non-physical
        filter_by_mag = True
        filter_by_mech = True
        filter_by_peak = True
        filter_by_exp_tremor = False
        #
        if sta == 'POS':
            pass
            #filter_by_mech = True 
        #
        if dsar_median:
            if filter_by_cont: # continuity: cc's that traspasses the threshold for more than 3 days in a row
                pass
                _a=[]
                _b=[]
                _c=[]
                for index, row in pd_cc_dsar_median.iterrows():
                    day = timedelta(days=1)
                    try:
                        _cc1= row['cc']
                        _cc2= pd_cc_dsar_median.loc[index+1*day, 'cc']
                        _cc3= pd_cc_dsar_median.loc[index+2*day, 'cc']
                        #_cc4= pd_cc_dsar_median.loc[index+3*day, 'cc']
                        if _cc1 > cc_threshold_median and _cc2 > cc_threshold_median and _cc3 > cc_threshold_median:# and _cc4 > cc_threshold_median:
                            _a.append(index)#*day)
                            _b.append(max(_cc1,_cc2,_cc3))#pd_cc_dsar_median.loc[index+2*day, 'cc'])
                    except:
                        pass
                d = {'cc':_b}#, 'max':_c}
                pd_cc_dsar_median = pd.DataFrame(data=d, index=_a)                
            if filter_by_mag: # filter dates that dsar median cycle is < 1 (in magnitud)
                if True:
                    dt = 'zsc2_dsarF'
                    day = timedelta(days=1)
                    fm = ForecastModel(window=2., overlap=1., station=sta,
                        look_forward=2., data_streams=[dt], 
                        data_dir=r'C:\\Users\\aar135\\codes_local_disk\\volc_forecast_tl\\volc_forecast_tl\\data\\'
                        )
                        # rolling median and signature length window
                    N, M = [2,30]
                    # time
                    j = fm.data.df.index
                    # construct signature
                    df = fm.data.df[:]#(j>(te-(M+N)*day))&(j<te)]
                    dsar_med = df[dt].rolling(N*24*6).median()[N*24*6:]
                    ##
                    # loop over dates
                    _a=[]
                    _b=[]
                    _c=[]
                    count = 0
                    for index, row in pd_cc_dsar_median.iterrows():

                        if index > _from:
                            #print(count)
                            count += 1
                            try:
                                l_back = 30
                                t0 = index - l_back*day
                                t1 = index  + 0*day
                                _dsar_med = dsar_med[t0:t1].values
                                if max(_dsar_med) >= 1:
                                    _a.append(index)
                                    _b.append(row['cc'])
                                    _c.append(max(_dsar_med))
                            except:
                                pass
                        d = {'cc':_b, 'max':_c}
                    pd_cc_dsar_median = pd.DataFrame(data=d, index=_a)
                    #pd_cc_dsar_median = pd.DataFrame(data=_b, index=_a)
                    #pd_cc = pd.DataFrame(data=_b, index=_a, columns=['cc'])
            if filter_by_mech: # filter by eruption mechanism: dsar median increas 
                #
                dt1 = 'zsc2_mfF'
                dt2 = 'zsc2_hfF'
                day = timedelta(days=1)
                fm = ForecastModel(window=2., overlap=1., station=sta,
                    look_forward=2., data_streams=[dt1,dt2], 
                    data_dir=r'C:\\Users\\aar135\\codes_local_disk\\volc_forecast_tl\\volc_forecast_tl\\data\\'
                    )
                    # rolling median and signature length window
                N, M = [2,30]
                # time
                j = fm.data.df.index
                # construct signature
                df = fm.data.df[:]#(j>(te-(M+N)*day))&(j<te)]
                mf_med = df[dt1].rolling(N*24*6).median()[N*24*6:]
                hf_med = df[dt2].rolling(N*24*6).median()[N*24*6:]
                ##
                # loop over dates
                _a=[]
                _b=[]
                _c=[]
                count = 0
                for index, row in pd_cc_dsar_median.iterrows():
                    if index == datetimeify('2009-07-07 00:00:00'):
                        a= 2
                    #_ = datetimeify('2013-03-30')
                    if index > _from:
                        #print(count)
                        count += 1
                        try:
                            l_back = 2
                            t0 = index - l_back*day
                            t1 = index  + 0*day
                            #
                            _mf_mx_b = max(mf_med[t0:t1].values)
                            _hf_mx_b = max(hf_med[t0:t1].values)
                            #
                            #_mf_mx_f = max(mf_med[t1:t1+l_back*day].values)
                            #_hf_mx_f = max(hf_med[t1:t1+l_back*day].values)
                            #
                            if _mf_mx_b > _hf_mx_b:#or _mf_mx_f > _hf_mx_f:
                                _a.append(index)
                                _b.append(row['cc'])
                                _c.append(max(_dsar_med))
                        except:
                            pass
                    d = {'cc':_b, 'max':_c}
                pd_cc_dsar_median = pd.DataFrame(data=d, index=_a)  
            if filter_by_peak: # in raw seismic data
                #
                dt1 = 'rsamF'
                #dt2 = 'zsc2_hfF'
                day = timedelta(days=1)
                fm = ForecastModel(window=2., overlap=1., station=sta,
                    look_forward=2., data_streams=[dt1], 
                    data_dir=r'C:\\Users\\aar135\\codes_local_disk\\volc_forecast_tl\\volc_forecast_tl\\data\\'
                    )
                    # rolling median and signature length window
                #N, M = [2,30]
                # time
                j = fm.data.df.index
                # 
                df = fm.data.df[:]#(j>(te-(M+N)*day))&(j<te)]
                rsam = df[dt1]#.rolling(N*24*6).median()[N*24*6:]

                ##
                # loop over dates
                _a=[]
                _b=[]
                _c=[]
                count = 0
                for index, row in pd_cc_dsar_median.iterrows():
                    idx_erup = index
                    if index == datetimeify('2020-08-18 00:00:00'):
                        a= 2
                    #_ = datetimeify('2013-03-30')
                    if index > _from:
                        #print(count)
                        count += 1
                        #
                        try:
                            t0 = index - 7*day
                            t1 = index + 7*day
                            #
                            _vals = rsam[t0:t1].values
                            _times = rsam[t0:t1].index
                            #
                            _rsam_mx = max(_vals)
                            if True: #move index to rsam peak
                                #
                                _idx = np.where(_vals == np.amax(_vals))[0][0]
                                idx_erup = _times[_idx]
                            #
                            _comp = 100
                            if sta == 'COP':
                                mu = np.mean(rsam) 
                                sigma = np.std(rsam) 
                                _comp = mu + 2*sigma #0.1
                            if sta == 'POS':
                                _comp = 60
                                mu = np.mean(rsam) 
                                sigma = np.std(rsam) 
                                _comp = mu + 2*sigma #0.1
                            if sta == 'FWVZ':
                                _comp = 60
                                mu = np.mean(rsam) 
                                sigma = np.std(rsam) 
                                _comp = mu + 3*sigma #0.1

                            if _rsam_mx > _comp:#or _mf_mx_f > _hf_mx_f:
                                _a.append(idx_erup)
                                _b.append(row['cc'])
                                _c.append(max(_dsar_med))
                        except:
                            pass
                    d = {'cc':_b, 'max':_c}
                pd_cc_dsar_median = pd.DataFrame(data=d, index=_a)  
            if filter_by_exp_tremor: # in raw seismic data
                #
                dt1 = 'rsamF'
                #dt2 = 'zsc2_hfF'
                day = timedelta(days=1)
                fm = ForecastModel(window=2., overlap=1., station=sta,
                    look_forward=2., data_streams=[dt1], 
                    data_dir=r'C:\\Users\\aar135\\codes_local_disk\\volc_forecast_tl\\volc_forecast_tl\\data\\'
                    )
                    # rolling median and signature length window
                #N, M = [2,30]
                # time
                j = fm.data.df.index
                # 
                df = fm.data.df[:]#(j>(te-(M+N)*day))&(j<te)]
                rsam = df[dt1]#.rolling(N*24*6).median()[N*24*6:]

                ##
                # loop over dates
                _a=[]
                _b=[]
                _c=[]
                tau_l=[]
                count = 0
                for index, row in pd_cc_dsar_median.iterrows():
                    idx_erup = index
                    #print(count)
                    count += 1
                    #
                    try:
                        t0 = index + 0.1*day
                        t1 = index + 1.1*day
                        #
                        _vals = rsam[t0:t1].values
                        _times = rsam[t0:t1].index
                        #
                        y = np.log10(_vals)
                        x = np.arange(y.shape[0])/6
                        z = np.polyfit(x, y, 1)
                        p = np.poly1d(z)
                        #
                        tau = -(x[-1]-x[0])/(y[-1]-y[0])
                        tau_l.append([index,tau])
                        #
                        _comp = 500
                        # if sta == 'COP':
                        #     mu = np.mean(rsam) 
                        #     sigma = np.std(rsam) 
                        #     _comp = mu + 2*sigma #0.1
                        # if sta == 'POS':
                        #     _comp = 60
                        #     mu = np.mean(rsam) 
                        #     sigma = np.std(rsam) 
                        #     _comp = mu + 2*sigma #0.1

                        if True:#tau > 1:#or _mf_mx_f > _hf_mx_f:
                            _a.append(idx_erup)
                            _b.append(row['cc'])
                            _c.append(max(_dsar_med))
                    except:
                        pass
                    d = {'cc':_b, 'max':_c}
                pd_cc_dsar_median = pd.DataFrame(data=d, index=_a)  
        # sum of both cc
        if dsar_median:
            pd_cc = pd_cc_dsar_median
        if dsar_rv:
            pd_cc = pd_cc_dsar_rv
        if dsar_median and dsar_rv: # sum cc values
            #pd_cc = pd_cc_dsar_median +  pd_cc_dsar_rv
            _idx_l = []
            _cc_l = []
            _max_l = []
            for index, row in pd_cc_dsar_median.iterrows():
                #a = pd_cc_dsar_rv.loc[[index][0]].values
                _idx_l.append(index)
                _cc_l.append(row['cc'] + pd_cc_dsar_rv.loc[[index][0]].values[0])
                _max_l.append(row['max'] )
                d = {'cc':_cc_l, 'max':_max_l}
                pd_cc = pd.DataFrame(data=d, index=_idx_l)
        #
        #pd_cc = pd_cc.sort_values('cc', ascending=False)
        #pd_cc.to_csv('test.csv', sep=',')
        #filtering close values 
        l_dates = []
        l_dates_rej = []
        l_cc = []
        l_max = []
        #
        for index, row in pd_cc.iterrows():
            if index.year >= _from.year:#2006:
                # first entry
                if not l_dates:
                    l_dates.append(index)
                    #for i in range(60):
                    #    _ = index-30*day + int(i)*day
                    #[l_dates_rej.append(index-30*day+ i*day) for i in range(60)]
                    [l_dates_rej.append(index-7*day+ i*day) for i in range(14)]
                    l_cc.append(row.cc)
                # 
                if index not in l_dates_rej:
                    l_dates.append(index)
                    #[l_dates_rej.append(index-30*day + i*day) for i in range(60)]
                    [l_dates_rej.append(index-7*day + i*day) for i in range(14)]
                    l_cc.append(row.cc)
                    l_max.append(row.values[1])
                #
        max_dates = int(1.*len(l_cc))#
        # write output
        with open(path+sta+'_dates_missed_events_from_dsar_median_rv_cc.txt', 'w') as f:
            for i, date in enumerate(l_dates[0:max_dates]):
                if l_cc[i] >= cc_threshold:
                    f.write(str(date)+'\n')
        with open(path+sta+'_dates_max_CC_missed_events_from_dsar_median_rv.txt', 'w') as f:
            for i, date in enumerate(l_dates[0:max_dates-1]):
                if l_cc[i] >= cc_threshold:
                    f.write(str(date)+','+str(round(l_max[i],2))+','+str(round(l_cc[i],2))+'\n')
        if filter_by_exp_tremor:
            with open(path+sta+'_dates_max_CC_missed_events_from_dsar_median_rv_tau.txt', 'w') as f:
                for i, date in enumerate(l_dates[0:max_dates-1]):
                    if l_cc[i] >= cc_threshold:
                        f.write(str(date)+','+str(round(l_max[i],2))+','+str(round(l_cc[i],2))+','+str(round(tau_l[i][1],2))+'\n')

def plot_located_events_compress(): 
    '''
    '''
    sta = 'FWVZ'
    # import list
    path_ap = 'C:\\Users\\aar135\\codes_local_disk\\volc_forecast_tl\\volc_forecast_tl\\features\\lake_data\\corr_dsar_ratevar\\'
    path = path_ap
    dates =[]
    path_dates_ap = path_ap+sta+'_dates_max_CC_missed_events_from_dsar_median_rv_with_filt_peak.txt'
    #path_dates_ap = path_ap+'FWVZ_dates_max_CC_missed_events_from_dsar_median_rv_with_filt_peak.txt'
    with open(path_dates_ap,'r') as fp:
        dates = [datetimeify(ln.rstrip()) for ln in fp.readlines()]
    
    # create figure
    if sta == 'FWVZ': 
        nrow = 5
        ncol = 3
        fig, axs = plt.subplots(nrows=nrow, ncols=ncol, figsize=(24,8))#(14,4))
        _log  = False
        lake_data = True
    if sta == 'COP': 
        nrow = 2
        ncol = 2
        fig, axs = plt.subplots(nrows=nrow, ncols=ncol, figsize=(24,8))#(14,4))
        _log  = False
        lake_data = False
    if sta == 'POS': 
        nrow = 2
        ncol = 2
        fig, axs = plt.subplots(nrows=nrow, ncols=ncol, figsize=(24,8))#(14,4))
        _log  = False
        lake_data = True

    # loop over axis and plot RSAM, DSAR, and temp
    l_forw = 7
    N, M = [2,15]
    col = ['b','k','r','g']
    alpha = [1., 1., 1., 1,]
    thick_line = [1., 1., 1., 1.]
    #
    for i, ax in enumerate(axs.reshape(-1)): 
        if sta == 'POS': 
            i = i+1
        if True:#try:
            if True: # RSAM
                ## DSAR median 
                day = timedelta(days=1)
                #sta_arch = 'WIZ'
                dt = 'zsc2_rsamF'
                fm = ForecastModel(window=2., overlap=1., station=sta,
                    look_forward=2., data_streams=[dt], 
                    data_dir=r'C:\Users\aar135\codes_local_disk\volc_forecast_tl\volc_forecast_tl\data'
                    )
                #
                te = dates[i]#fm.data.tes[erup] 
                # rolling median and signature length window
                #N, M = [2,15]
                #l_forw = 0
                # time
                j = fm.data.df.index
                # median 
                df = fm.data.df[(j>(te-(M+N)*day))&(j<te+l_forw*day)]
                #archtype = df[dt].rolling(N*24*6).median()[N*24*6:]
                archtype = df[dt]#.rolling(N*24*6).median()[N*24*6:]
                #
                _times = archtype.index
                _val = archtype.values
                _val_max = max(_val)
                #
                ft = 'nRSAM median'
                ax.plot(_times, _val, '-', color='k', alpha = 0.8, linewidth=thick_line[0], label=' '+ ft,zorder=1)
                #ax.plot([], [], '-', color='k', alpha = alpha[0], linewidth=thick_line[0], label=' '+ ft,zorder=1)
                # lim
                ax.set_ylim([0,np.mean(_val)+3*np.std(_val)])
            #
            if True: # DSAR
                ax2 = ax.twinx()
                ## DSAR median 
                day = timedelta(days=1)
                #sta_arch = 'WIZ'
                dt = 'zsc2_dsarF'
                fm = ForecastModel(window=2., overlap=1., station=sta,
                    look_forward=2., data_streams=[dt], 
                    data_dir=r'C:\Users\aar135\codes_local_disk\volc_forecast_tl\volc_forecast_tl\data'
                    )
                #
                te = dates[i]#fm.data.tes[erup] 
                # rolling median and signature length window
                #N, M = [2,15]
                #l_forw = 0
                # time
                j = fm.data.df.index
                # median 
                df = fm.data.df[(j>(te-(M+N)*day))&(j<te+l_forw*day)]
                archtype = df[dt].rolling(N*24*6).median()[N*24*6:]
                #
                _times = archtype.index
                _val = archtype.values
                _val_max = max(_val)
                #
                ft = 'DSAR median'
                ax2.plot(_times, _val, '-', color='b', alpha = alpha[0], linewidth=thick_line[0], label=' '+ ft,zorder=1)
                ax.plot([], [], '-', color='b', alpha = alpha[0], linewidth=thick_line[0], label=' '+ ft,zorder=1)
                #ax.plot([], [], '-', color='w', alpha = 0.1, linewidth=thick_line[0], label=str(te.year)+' '+str(te.month)+' '+str(te.day),zorder=1)

            # plot eruption 
            #ax.axvline(te, color='r',linestyle='--', linewidth=3,alpha = 0.7, zorder = 0)
            #ax.plot([], color='r', linestyle='-', linewidth=3, label = 'event')
            legend = ax.legend(loc=2)
            #legend.get_frame().set_facecolor('white')
            legend.set_alpha(1)
            #if sta == 'POS' and i == 2: 
            #    ax.set_ylim([0,10])
            #plt.text(60, .025, str(te.year)+' '+str(te.month)+' '+str(te.day))
            #ax.set_xticklabels([te - 7*day*i for i in range(int(len(l_forw+M)/7)+1)])#[dat.index.values[0],dat.index.values[-1]])#, ]np.arange(0, len(x)+1, 5))
            #ax2.set_xticklabels([te - 7*day*i for i in range(int(len(l_forw+M)/7)+1)])#[dat.index.values[0],dat.index.values[-1]])#, ]np.arange(0, len(x)+1, 5))

            #ax.set_xticks([])#[dat.index.values[0],dat.index.values[-1]])#, ]np.arange(0, len(x)+1, 5))
            # ax.set_xticks([te-7*day,te,te+7*day])
            # ax2.set_xticks([te-7*day,te,te+7*day])
            #ax.set_xticklabels(['Geeks', 'for', 'geeks', '!'])
            #matplotlib.pyplot.xticks(color='w')

            if lake_data: # lake data 
                temp = True
                level = True
                rainfall = True
                #
                mov_avg = True # moving average for temp and level data
                # convert to UTC 0
                utc_0 = True
                if utc_0:
                    _utc_0 = 0#-13 # hours
                # plot temp data
                if sta == 'FWVZ':
                    # plot temperature data 
                    if temp:
                        ax3 = ax.twinx()
                        try:
                            if te.year >= 2009:
                                ti_e1 = te - M*day
                                tf_e1 = te + l_forw*day
                                # import temp data
                                path = '..'+os.sep+'data'+os.sep+"RU001_temp_data.csv"
                                pd_temp = pd.read_csv(path, index_col=1)
                                if utc_0:
                                    #pd_temp.index = [datetimeify(pd_temp.index[i])-6*hour for i in range(len(pd_temp.index))]
                                    pd_temp.index = [datetimeify(pd_temp.index[i])+_utc_0*hour for i in range(len(pd_temp.index))]
                                else:
                                    pd_temp.index = [datetimeify(pd_temp.index[i]) for i in range(len(pd_temp.index))]
                                
                                # plot data in axis twin axis
                                # Trim the data
                                temp_e1_tim = pd_temp[ti_e1: tf_e1].index.values
                                #temp_e1_tim=to_nztimezone(temp_e1_tim)
                                #
                                temp_e1_val = pd_temp[ti_e1: tf_e1].loc[:,' t (C)'].values
                                # ax2
                                #ax2b = ax2.twinx()   
                                if mov_avg: # plot moving average
                                    n=30
                                    #v_plot = (temp_e1_val-np.min(temp_e1_val))/np.max((temp_e1_val-np.min(temp_e1_val)))
                                    v_plot = temp_e1_val
                                    #ax3.plot(temp_e1_tim, v_plot, '-', color='g', label='lake temperature', alpha = 1.)
                                    
                                    #
                                    #ax.plot(temp_e1_tim[:-n+1], moving_average(temp_e1_val, n=n), '--', color='k')#, label='temp. mov. avg.')
                                    ax3.plot(temp_e1_tim[n-1-20:-20], moving_average(v_plot[::-1], n=n)[::-1], '--', color='g')#, label='lake temperature')
                                else:
                                    v_plot = temp_e1_val
                                    ax3.plot(temp_e1_tim, v_plot, '-', color='g', label='lake temperature', alpha = 1.)
                                #ax3.set_ylabel('Temperature °C')
                                #
                                temp_min = min(temp_e1_val)
                                temp_max = max(temp_e1_val)
                                temp_mu = np.mean(temp_e1_tim)
                                temp_sigma = np.std(temp_e1_tim)
                                #ax3.set_ylim([temp_mu-3*temp_sigma,temp_mu+3*temp_sigma])
                                #ax2.set_ylabel('temperature C')   
                                # ax.set_xticks([te-7*day,te,te+7*day])
                                # ax2.set_xticks([te-7*day,te,te+7*day])
                                # ax3.set_xticks([te-7*day,te,te+7*day])
                        except:
                            pass

                    # plot lake level data
                    if False:#level:
                        #ax4 = ax.twinx()
                        #try:
                        if False:
                            #
                            # import temp data
                            if False:#t0.year > 2015:
                                path = '..'+os.sep+'data'+os.sep+"RU001A_level_data_full.csv"
                            else:
                                pass
                                path = '..'+os.sep+'data'+os.sep+"RU001_level_data.csv"         
                            #    
                            pd_temp = pd.read_csv(path, index_col=1)
                            if utc_0:
                                #pd_temp.index = [datetimeify(pd_temp.index[i])-6*hour for i in range(len(pd_temp.index))]
                                #pd_temp.index = [datetimeify(pd_temp.index[i])-0*hour for i in range(len(pd_temp.index))]
                                pd_temp.index = [datetimeify(pd_temp.index[i])+_utc_0*hour for i in range(len(pd_temp.index))]
                            else:
                                pd_temp.index = [datetimeify(pd_temp.index[i]) for i in range(len(pd_temp.index))]

                            if te.year>2010 and te.year<2016: # rolling median over data
                                N = 2
                                pd_temp = pd_temp[:].rolling(40).median()#[N*24*6:]

                            # plot data in axis twin axis
                            # Trim the data
                            temp_e1_tim = pd_temp[ti_e1: tf_e1].index.values
                            #temp_e1_tim=to_nztimezone(temp_e1_tim)
                            temp_e1_val = pd_temp[ti_e1: tf_e1].loc[:,' z (m)'].values
                            # ax2
                            #ax2b = ax2.twinx()
                            if False:#mov_avg: # plot moving average
                                n=10
                                v_plot = temp_e1_val
                                ax4.plot(temp_e1_tim, v_plot, '-', color='royalblue', alpha = 1.)
                                ax4.plot([], [], '-', color='royalblue', label='lake level')
                                #ax2b.plot(temp_e1_tim[:-n+1], moving_average(temp_e1_val, n=n), '--', color='k', label='temp. mov. avg.')
                                #v_plot = (temp_e1_val-np.min(temp_e1_val))/np.max((temp_e1_val-np.min(temp_e1_val)))
                                ax4.plot(temp_e1_tim[n-1-10:-10], moving_average(v_plot[::-1], n=n)[::-1], '--', color='royalblue', label='lake level')
                                #ax2b.plot(temp_e1_tim, v_plot, '-', color='b', alpha = 0.3)
                            else:
                                v_plot = temp_e1_val
                                ax4.plot(temp_e1_tim, v_plot, '-', color='royalblue', label='lake level')
                            #
                            ax4.set_ylabel('Lake level cm') 
                            ax4.plot([], [], '-', color='royalblue', label='lake level')

                            if False:#plot_erup: # plot vertical lines
                                te = datetimeify(erup_time)#fm_e1.data.tes[int(erup[-1:])-1]
                                ax2.axvline(te, color='k',linestyle='--', linewidth=2, zorder = 4)
                                #ax2.plot([], color='k', linestyle='--', linewidth=2, label = 'eruption')
                        #except:
                        #    pass
                    
                    # plot rainfall data
                    if False:#rainfall:
                        try:
                            ti_e1 = te - M*day
                            tf_e1 = te - l_forw*day
                            #
                            # import temp data
                            path = '..'+os.sep+'data'+os.sep+"_chateau_rain.csv"
                            pd_rf = pd.read_csv(path, index_col=1)
                            pd_rf.index = pd.to_datetime(pd_rf.index, format='%Y%m%d:%H%M')
                            if utc_0:
                                pd_rf.index = [pd_rf.index[i]+_utc_0*hour for i in range(len(pd_rf.index))]

                            # Trim the data
                            rf_e2_tim = pd_rf[ti_e1: tf_e1].index#.values
                            rf_e2_val = pd_rf[ti_e1: tf_e1].loc[:,'Amount(mm)'].values /3
                            # ax2
                            #ax2b = ax2.twinx()
                            v_plot = (rf_e2_val-np.min(rf_e2_val))/np.max((rf_e2_val-np.min(rf_e2_val)))
                            #v_plot = v_plot*5 + 14
                            if temp_max:
                                v_plot = v_plot*(temp_max-temp_min)*0.6 + temp_min
                            ax2.plot(rf_e2_tim, v_plot, '-', color='c', label='rain fall', alpha = 0.6)
                            #ax2b.set_ylabel('temperature C')
                            #ax2b.legend(loc = 1)
                        except:
                            pass

                if sta == 'DAM' or sta == 'POS':
                    ax3 = ax.twinx()
                    lake = False # no data
                    rainfall = False # no data
                    try:
                        if temp:
                            if te.year >= 2013:
                                ti_e1 = te - M*day
                                tf_e1 = te + l_forw*day
                                # import temp data
                                path = '..'+os.sep+'data'+os.sep+"DAM_temp_data.csv"
                                pd_temp = pd.read_csv(path, index_col=1)

                                if utc_0:
                                    pd_temp.index = [datetimeify(pd_temp.index[i])-6*hour for i in range(len(pd_temp.index))]
                                else:
                                    pd_temp.index = [datetimeify(pd_temp.index[i]) for i in range(len(pd_temp.index))]
                                # plot data in axis twin axis
                                # Trim the data
                                temp_e1_tim = pd_temp[ti_e1: tf_e1].index.values
                                #temp_e1_tim=to_nztimezone(temp_e1_tim)
                                #
                                temp_e1_val = pd_temp[ti_e1: tf_e1].loc[:,' t (C)'].values
                                # ax2
                                #ax2b = ax2.twinx()   
                                if mov_avg: # plot moving average
                                    n=50
                                    #v_plot = (temp_e1_val-np.min(temp_e1_val))/np.max((temp_e1_val-np.min(temp_e1_val)))
                                    v_plot = temp_e1_val
                                    ax3.plot(temp_e1_tim, v_plot, '-', color='g', label='lake temperature', alpha = 1.)
                                    #
                                    #ax.plot(temp_e1_tim[:-n+1], moving_average(temp_e1_val, n=n), '--', color='k')#, label='temp. mov. avg.')
                                    _x = temp_e1_tim[n-1-20:-20]
                                    _y = moving_average(v_plot[::-1], n=n)[::-1]
                                    ax3.plot(_x, _y, '--', color='k')#, label='lake temperature')
                                else:
                                    v_plot = temp_e1_val
                                    ax3.plot(temp_e1_tim, v_plot, '-', color='g', label='lake temperature', alpha = 1.)
                                ax3.set_ylabel('Temperature °C')
                                
                                _ylim = [min(_y)-1,max(_y)+1] 
                                ax3.set_ylim(_ylim)
                                #ax2.set_ylabel('temperature C')   
                                ax.set_xticks([te-7*day,te,te+7*day])
                                ax2.set_xticks([te-7*day,te,te+7*day])
                                ax3.set_xticks([te-7*day,te,te+7*day])
                    except:
                        pass
                    if False:#plot_erup: # plot vertical lines
                        te = datetimeify(erup_time)#fm_e1.data.tes[int(erup[-1:])-1]
                        ax2.axvline(te, color='k',linestyle='--', linewidth=2, zorder = 4)
                        #ax2.plot([], color='k', linestyle='--', linewidth=2, label = 'eruption')

        #except:
        #    pass

        ax.set_xticks([])
        ax2.set_xticks([])
        if lake_data:
            ax3.set_xticks([])
        ax.set_xticks([te-7*day,te,te+7*day])
        #
        ax2.set_yticks([])
        if lake_data:
            ax3.set_yticks([])
        ax.set_ylabel('nRSAM_F')
        #
    plt.show()
    asdf

def plot_seismic_temp_data():
    '''
    plot dsar, temp data, raw seismic data 
    '''
    def plot_seismic_temp_data(sta = None, erup_time = None, look_back = None, look_front = None, temp = True, level = True, rainfall = True, save_png_path = None,  date_line = None, plot_erup = None):
        #
        if sta == 'FWVZ':
            ffm = False
            server = False # files imported from server to local pc 
            server2 = False # server at uni 
            plot_erup = False
        if sta == 'COP':
            ffm = False
            server = False # files imported from server to local pc 
            server2 = False # server at uni 
            #plot_erup = False
        if sta == 'POS' or sta == 'DAM':
            ffm = False
            server = False # files imported from server to local pc 
            server2 = False # server at uni 
            plot_erup = False
        #
        day = timedelta(days=1)
        t0 = erup_time - look_back*day#30*day
        t1 = erup_time + look_front*day#hour
        #
        if sta == 'DAM' or sta == 'POS':
            server = False
        #
        ## plot other data
        temp = temp
        level = level
        rainfall = rainfall
        ## 
        # figure
        if False:
            nrow = 5
            ncol = 1
            fig, (ax1, ax5, ax2, ax3, ax4) = plt.subplots(nrows=nrow, ncols=ncol, figsize=(24,12))#(14,4))
            _log  = False
        else:
            nrow = 4
            ncol = 1
            fig, (ax1, ax5, ax2, ax4) = plt.subplots(nrows=nrow, ncols=ncol, figsize=(24,8))#(14,4))
            _log  = False
        #
        #####################################################
        # subplot one: MF, HF, DSAR medians (DSAR yaxis left; MF, HF yaxis rigth). 1/RSAM (normalized)
        #ax1
        if True:
            # features
            fts_yleft = ['zsc2_dsarF__median','zsc2_rmarF__median']#,'zsc2_lrarF__median','zsc2_lrarF__median','zsc2_vlarF__median']
            fts_yrigth = ['zsc2_lrarF__median','zsc2_vlarF__median']#['zsc2_mfF__median','zsc2_hfF__median','zsc2_rsamF__median']
            #
            fts_yleft = ['zsc2_dsarF__median']
            fts_yrigth = ['zsc2_mfF__median','zsc2_hfF__median']

            col = ['b','m','r']
            alpha = [1., 1., 1.]
            thick_line = [2., 2., 1.]
            #
            # try: 
            for i, ft in enumerate(fts_yleft):
                if False: # load feature (else: cal feature. median or rv)
                    if 'zsc2_dsarF' in ft:
                        ds = ['zsc2_dsarF'] 
                    if server:
                        path_feat_serv = 'C:\\Users\\aar135\\codes_local_disk\\volc_forecast_tl\\features_bkp\\features_server\\'
                        fm_e1 = ForecastModel(window=2., overlap=1., station =  sta,
                            look_forward=2., data_streams=ds, 
                            feature_dir=path_feat_serv, 
                            savefile_type='pkl') 
                    elif server2:
                        path_feat_serv = 'U:\\Research\\EruptionForecasting\\eruptions\\features\\'
                        fm_e1 = ForecastModel(window=2., overlap=1., station =  sta,
                            look_forward=2., data_streams=ds, 
                            feature_dir=path_feat_serv, 
                            savefile_type='pkl') 
                    else:
                        try:
                            fm_e1 = ForecastModel(window=2., overlap=1., station = sta,
                                look_forward=2., data_streams=ds, savefile_type='csv')
                        except:
                            fm_e1 = ForecastModel(window=2., overlap=1., station = sta,
                                look_forward=2., data_streams=ds, savefile_type='pkl')                    
                    ##  
                    ft = ft.replace("-",'"')
                    # adding multiple Axes objects
                    ft_e1 = fm_e1.get_features(ti=t0, tf=t1, n_jobs=1, compute_only_features=[ft])
                    # extract values to plot 
                    ft_e1_t = ft_e1[0].index.values
                    ft_e1_v = ft_e1[0].loc[:,ft]
                    #
                    v_plot = ft_e1_v

                    #v_plot = ft_e1_v-np.min(ft_e1_v)/np.max((ft_e1_v-np.min(ft_e1_v)))
                    #v_plot = ft_e1_v/np.max(ft_e1_v)
                else: 
                    #
                    if 'zsc2_dsarF' in ft:
                        ds = 'zsc2_dsarF'
                    if 'zsc2_mfF' in ft:
                        ds = 'zsc2_mfF' 
                    if 'zsc2_hfF' in ft:
                        ds = 'zsc2_hfF' 
                    if 'zsc2_rmarF' in ft:
                        ds = 'zsc2_rmarF' 
                    if 'zsc2_lrarF' in ft:
                        ds = 'zsc2_lrarF' 
                    # 
                    #
                    day = timedelta(days=1)
                    fm = ForecastModel(window=2., overlap=1., station=sta,
                        look_forward=2., data_streams=[ds], 
                        data_dir=r'C:\Users\aar135\codes_local_disk\volc_forecast_tl\volc_forecast_tl\data'
                        )
                    #
                    N, M = [2,30]
                    df = fm.data.df[t0:t1]
                    if 'median' in ft:
                        test = df[ds].rolling(N*24*6).median()[N*24*6:]
                    if 'rate_variance' in ft:
                        test = df[ds].rolling(N*24*6).apply(chqv)[N*24*6:]
                    #
                    #out = test.rolling(archtype.shape[0]).apply(partial(conv, (archtype-archtype.mean())/archtype.std()))       
                    #out = out.resample('1D').ffill()
                    #
                    ft_e1_t = test.index
                    v_plot = test.values
                #
                if ft == 'zsc2_dsarF__median':
                    ft = 'nDSAR median'
                if ft == 'zsc2_rmarF__median':
                    ft = 'nRMAR median'
                if ft == 'zsc2_lrarF__median':
                    ft = 'nLRAR median'
                #
                if i == 0:
                    _max= max(v_plot)
                    ax1.plot(ft_e1_t, v_plot, '-', color=col[i], linewidth=thick_line[0], alpha = alpha[i],label=' '+ ft)
                    #_max = 1.
                else:
                    ax1.plot(ft_e1_t, v_plot/max(v_plot) *.95*_max, '-', color=col[i], linewidth=thick_line[0], alpha = alpha[i],label=' '+ ft)
                #
                #
                if ffm: # ffm 
                    if i == 0:
                        #ax1b = ax1.twinx() 
                        #v_plot = data[data_stream].loc[inds]
                        inv_rsam = fm.data.get_data(ti=t0, tf=t1)['rsamF']#.loc[ft_e1_t]
                        inv_rsam_t = inv_rsam.index
                        inv_rsam = 1./inv_rsam.values
                        # normalized it to yaxis rigth 
                        #inv_rsam = inv_rsam/max(inv_rsam)
                        inv_rsam = inv_rsam*_max*8
                        #
                        ax1.plot(inv_rsam_t, inv_rsam, '-', color= 'gray', linewidth=1., markersize=0.5, alpha = 1.)
                        ax1.plot([], [], '-', color= 'gray', markersize=1, label='1/RSAM', alpha = 1.0)
                        ax1.set_ylim([0,800])
                        #ax1.set_yticks([])
                #
                if plot_erup: # plot vertical lines
                    te = datetimeify(erup_time)#fm_e1.data.tes[int(erup[-1:])-1]
                    ax1.axvline(te, color='k',linestyle='--', linewidth=2, zorder = 4)
                    ax1.plot([], color='k', linestyle='--', linewidth=2, label = 'eruption')
                #
                
                #
                te = datetimeify(erup_time)#fm_e1.data.tes[int(erup[-1:])-1]
                #ax1.set_xticks([ft_e1[0].index[-1] - 7*day*i for i in range(int(30/7)+1)])#[dat.index.values[0],dat.index.values[-1]])#, ]np.arange(0, len(x)+1, 5))
                
                #ax1b.set_yticks([])
                #ax1.set_yscale('log')
                ax1.grid()
                ax1.set_ylabel('nDSAR value')        #ax.set_xticks([ft_e1[0].index[-1]-7*day*i +day for i in range(5)])
                #ax.set_xticks([ft_e1[0].index[-1] - 7*day*i for i in range(int(30/7)+1)])#[dat.index.values[0],dat.index.values[-1]])#, ]np.arange(0, len(x)+1, 5))
                #ax1.set_xticks([t1 - 5*day*i for i in range(int(look_back/5)+1)])#[dat.index.values[0],dat.index.values[-1]])#, ]np.arange(0, len(x)+1, 5))
            #     #
            # except:
            #     pass
            if fts_yrigth:
                ax1b = ax1.twinx() 
                col = ['m','r','g','c']
                alpha = [.5, .5, .5]
                thick_line = [1.,1.,1.]
                #try: 
                for i, ft in enumerate(fts_yrigth):
                    if 'zsc2_dsarF' in ft:
                        ds = 'zsc2_dsarF'
                    if 'zsc2_mfF' in ft:
                        ds = 'zsc2_mfF' 
                    if 'zsc2_hfF' in ft:
                        ds = 'zsc2_hfF'
                    if 'zsc2_rsamF' in ft:
                        ds = 'zsc2_rsamF' 
                    if 'zsc2_lrarF' in ft:
                        ds = 'zsc2_lrarF' 
                    if 'zsc2_vlarF' in ft:
                        ds = 'zsc2_vlarF'
                    if 'zsc2_rmarF' in ft:
                        ds = 'zsc2_rmarF' 
                    # 
                    if False: # look feature in the prev cacl features (else: calculate feat from data; only for median and rv)
                        if server:
                            path_feat_serv = 'C:\\Users\\aar135\\codes_local_disk\\volc_forecast_tl\\features_bkp\\features_server\\'
                            fm_e1 = ForecastModel(window=2., overlap=1., station =  sta,
                                look_forward=2., data_streams=ds, 
                                feature_dir=path_feat_serv, 
                                savefile_type='pkl') 
                        elif server2:
                            path_feat_serv = 'U:\\Research\\EruptionForecasting\\eruptions\\features\\'
                            fm_e1 = ForecastModel(window=2., overlap=1., station =  sta,
                                look_forward=2., data_streams=ds, 
                                feature_dir=path_feat_serv, 
                                savefile_type='pkl') 
                        else:
                            try:
                                fm_e1 = ForecastModel(window=2., overlap=1., station = sta,
                                    look_forward=2., data_streams=ds, savefile_type='csv')
                            except:
                                fm_e1 = ForecastModel(window=2., overlap=1., station = sta,
                                    look_forward=2., data_streams=ds, savefile_type='pkl')                    
                        ##  
                        ft = ft.replace("-",'"')
                        
                        ft_e1 = fm_e1.get_features(ti=t0, tf=t1, n_jobs=1, compute_only_features=[ft])
                        # adding multiple Axes objects

                        # extract values to plot 
                        ft_e1_t = ft_e1[0].index.values
                        ft_e1_v = ft_e1[0].loc[:,ft]
                        #
                        v_plot = ft_e1_v

                    else:
                        day = timedelta(days=1)
                        fm = ForecastModel(window=2., overlap=1., station=sta,
                            look_forward=2., data_streams=[ds], 
                            data_dir=r'C:\Users\aar135\codes_local_disk\volc_forecast_tl\volc_forecast_tl\data'
                            )
                        #
                        N, M = [2,30]
                        df = fm.data.df[t0:t1]
                        if 'median' in ft:
                            test = df[ds].rolling(N*24*6).median()[N*24*6:]
                        if 'rate_variance' in ft:
                            test = df[ds].rolling(N*24*6).apply(chqv)[N*24*6:]
                        #
                        #out = test.rolling(archtype.shape[0]).apply(partial(conv, (archtype-archtype.mean())/archtype.std()))       
                        #out = out.resample('1D').ffill()
                        #
                        ft_e1_t = test.index
                        v_plot = test.values

                    #v_plot = ft_e1_v-np.min(ft_e1_v)/np.max((ft_e1_v-np.min(ft_e1_v)))
                    #v_plot = ft_e1_v/np.max(ft_e1_v)
                    #
                    if ft == 'zsc2_mfF__median':
                        ft = 'nMF median'
                    if ft == 'zsc2_hfF__median':
                        ft = 'nHF median'
                    if ft == 'zsc2_rsamF__median':
                        ft = 'nRSAM median'
                    if ft == 'zsc2_lrarF__median':
                        ft = 'nLRAR median'
                    if ft == 'zsc2_vlarF__median':
                        ft = 'nVLAR median'
                    #
                    if i == 0:
                        ax1b.plot(ft_e1_t, v_plot, '-', color=col[i], alpha = alpha[i],label=' '+ ft)
                        _max = max(v_plot)
                        #_max = 1.
                    else:
                        v_plot = v_plot/max(v_plot)*0.9*_max
                        ax1b.plot(ft_e1_t, v_plot, '-', color=col[i], alpha = alpha[i],label=' '+ ft)
                    #
                    ax1b.legend(loc = 3)
                    #
                    te = datetimeify(erup_time)#fm_e1.data.tes[int(erup[-1:])-1]
                    #ax1.set_xticks([ft_e1[0].index[-1] - 7*day*i for i in range(int(30/7)+1)])#[dat.index.values[0],dat.index.values[-1]])#, ]np.arange(0, len(x)+1, 5))
                    
                    #ax1b.set_yticks([])
                    ax1b.grid()
                    ax1b.set_ylabel('Feature value')        #ax.set_xticks([ft_e1[0].index[-1]-7*day*i +day for i in range(5)])
                    #ax.set_xticks([ft_e1[0].index[-1] - 7*day*i for i in range(int(30/7)+1)])#[dat.index.values[0],dat.index.values[-1]])#, ]np.arange(0, len(x)+1, 5))
                    #ax1.set_xticks([t1 - 5*day*i for i in range(int(look_back/5)+1)])#[dat.index.values[0],dat.index.values[-1]])#, ]np.arange(0, len(x)+1, 5))
                    #
                #except:
                #    pass
            #
            ax1.legend(loc = 2)
            if True:    
                ax1.set_xlim([t0+2*day,t1])
                if _log:
                    ax1.set_yscale('log')
                    if fts_yrigth:
                        ax1b.set_yscale('log')
        # subplot two: temp data (if any: level and rainfall)
        #ax2
        if True:
            mov_avg = True # moving average for temp and level data
            # convert to UTC 0
            utc_0 = True
            if utc_0:
                _utc_0 = 0#-13 # hours
            # plot temp data
            if sta == 'FWVZ':
                # plot temperature data 
                if temp:
                    try:
                        ti_e1 = t0
                        tf_e1 = t1
                        # import temp data
                        path = '..'+os.sep+'data'+os.sep+"RU001_temp_data.csv"
                        pd_temp = pd.read_csv(path, index_col=1)
                        if utc_0:
                            #pd_temp.index = [datetimeify(pd_temp.index[i])-6*hour for i in range(len(pd_temp.index))]
                            pd_temp.index = [datetimeify(pd_temp.index[i])+_utc_0*hour for i in range(len(pd_temp.index))]
                        else:
                            pd_temp.index = [datetimeify(pd_temp.index[i]) for i in range(len(pd_temp.index))]
                        
                        # plot data in axis twin axis
                        # Trim the data
                        temp_e1_tim = pd_temp[ti_e1: tf_e1].index.values
                        #temp_e1_tim=to_nztimezone(temp_e1_tim)
                        #
                        temp_e1_val = pd_temp[ti_e1: tf_e1].loc[:,' t (C)'].values
                        # ax2
                        #ax2b = ax2.twinx()   
                        if mov_avg: # plot moving average
                            n=30
                            #v_plot = (temp_e1_val-np.min(temp_e1_val))/np.max((temp_e1_val-np.min(temp_e1_val)))
                            v_plot = temp_e1_val
                            ax2.plot(temp_e1_tim, v_plot, '-', color='g', label='lake temperature', alpha = .5)
                            
                            #
                            #ax.plot(temp_e1_tim[:-n+1], moving_average(temp_e1_val, n=n), '--', color='k')#, label='temp. mov. avg.')
                            ax2.plot(temp_e1_tim[n-1-20:-20], moving_average(v_plot[::-1], n=n)[::-1], '--', color='g')#, label='lake temperature')
                        else:
                            v_plot = temp_e1_val
                            ax2.plot(temp_e1_tim, v_plot, '-', color='g', label='lake temperature', alpha = 1.)
                        ax2.set_ylabel('Temperature °C')
                        #
                        temp_min = min(temp_e1_val)
                        temp_max = max(temp_e1_val)
                        temp_mu = np.mean(temp_e1_tim)
                        temp_sigma = np.std(temp_e1_tim)
                        ax2.set_ylim([temp_mu-3*temp_sigma,temp_mu+3*temp_sigma])
                        #ax2.set_ylabel('temperature C')   
                    except:
                        pass

                # plot lake level data
                if level:
                    try:
                        ax2b = ax2.twinx()
                        #
                        # import temp data
                        if t0.year >= 2022:
                            path = '..'+os.sep+'data'+os.sep+"RU001A_level_data.csv"
                        else:
                            path = '..'+os.sep+'data'+os.sep+"RU001A_level_data_full.csv"
                            pass
                            #path = '..'+os.sep+'data'+os.sep+"RU001_level_data.csv"         
                        #    
                        pd_temp = pd.read_csv(path, index_col=1)
                        if utc_0:
                            #pd_temp.index = [datetimeify(pd_temp.index[i])-6*hour for i in range(len(pd_temp.index))]
                            #pd_temp.index = [datetimeify(pd_temp.index[i])-0*hour for i in range(len(pd_temp.index))]
                            pd_temp.index = [datetimeify(pd_temp.index[i])+_utc_0*hour for i in range(len(pd_temp.index))]
                        else:
                            pd_temp.index = [datetimeify(pd_temp.index[i]) for i in range(len(pd_temp.index))]

                        if t0.year>2010 and t1.year<2016: # rolling median over data
                            N = 2
                            pd_temp = pd_temp[:].rolling(40).median()#[N*24*6:]

                        # plot data in axis twin axis
                        # Trim the data
                        temp_e1_tim = pd_temp[ti_e1: tf_e1].index.values
                        #temp_e1_tim=to_nztimezone(temp_e1_tim)
                        temp_e1_val = pd_temp[ti_e1: tf_e1].loc[:,' z (m)'].values
                        # ax2
                        #ax2b = ax2.twinx()
                        if False:#mov_avg: # plot moving average
                            n=10
                            v_plot = temp_e1_val
                            ax2b.plot(temp_e1_tim, v_plot, '-', color='gray', alpha = 0.5)
                            ax2.plot([], [], '-', color='gray', label='lake level')
                            #ax2b.plot(temp_e1_tim[:-n+1], moving_average(temp_e1_val, n=n), '--', color='k', label='temp. mov. avg.')
                            #v_plot = (temp_e1_val-np.min(temp_e1_val))/np.max((temp_e1_val-np.min(temp_e1_val)))
                            ax2b.plot(temp_e1_tim[n-1-10:-10], moving_average(v_plot[::-1], n=n)[::-1], '--', color='gray', label='lake level')
                            #ax2b.plot(temp_e1_tim, v_plot, '-', color='b', alpha = 0.3)
                        else:
                            v_plot = temp_e1_val
                            ax2b.plot(temp_e1_tim, v_plot, '-', color='gray', label='lake level')
                        #
                        ax2b.set_ylabel('Lake level cm') 
                        ax2.plot([], [], '-', color='gray', label='lake level')

                        if plot_erup: # plot vertical lines
                            te = datetimeify(erup_time)#fm_e1.data.tes[int(erup[-1:])-1]
                            ax2.axvline(te, color='k',linestyle='--', linewidth=2, zorder = 4)
                            #ax2.plot([], color='k', linestyle='--', linewidth=2, label = 'eruption')
                    except:
                        pass
                # plot rainfall data
                
                if rainfall:
                    try:
                        ti_e1 = t0#datetimeify(t0)
                        tf_e1 = t1#datetimeify(t1)
                        #
                        # import temp data
                        #path = '..'+os.sep+'data'+os.sep+"chateau_rain_all.csv"
                        path = '..'+os.sep+'data'+os.sep+'_chateau_rain.csv'
                        pd_rf = pd.read_csv(path, index_col=1)
                        pd_rf.index = pd.to_datetime(pd_rf.index, format='%Y%m%d:%H%M')
                        if False:#utc_0:
                            pd_rf.index = [pd_rf.index[i]+_utc_0*hour for i in range(len(pd_rf.index))]

                        # Trim the data
                        rf_e2_tim = pd_rf[ti_e1: tf_e1].index#.values
                        rf_e2_val = pd_rf[ti_e1: tf_e1].loc[:,'Amount(mm)'].values /3
                        # ax2
                        #ax2b = ax2.twinx()
                        v_plot = (rf_e2_val-np.min(rf_e2_val))/np.max((rf_e2_val-np.min(rf_e2_val)))
                        #v_plot = v_plot*5 + 14
                        if temp_max:
                            v_plot = v_plot*(temp_max-temp_min)*0.6 + temp_min
                        ax2.plot(rf_e2_tim, v_plot, '-', color='c', label='rain fall', alpha = 0.6)
                        #ax2b.set_ylabel('temperature C')
                        #ax2b.legend(loc = 1)
                    except:
                        pass

            if sta == 'DAM' or sta == 'POS':
                lake = False # no data
                rainfall = False # no data
                try:
                    if temp:
                        ti_e1 = t0
                        tf_e1 = t1
                        # import temp data
                        path = '..'+os.sep+'data'+os.sep+"DAM_temp_data.csv"
                        pd_temp = pd.read_csv(path, index_col=1)

                        if utc_0:
                            pd_temp.index = [datetimeify(pd_temp.index[i])-6*hour for i in range(len(pd_temp.index))]
                        else:
                            pd_temp.index = [datetimeify(pd_temp.index[i]) for i in range(len(pd_temp.index))]
                        # plot data in axis twin axis
                        # Trim the data
                        temp_e1_tim = pd_temp[ti_e1: tf_e1].index.values
                        #temp_e1_tim=to_nztimezone(temp_e1_tim)
                        #
                        temp_e1_val = pd_temp[ti_e1: tf_e1].loc[:,' t (C)'].values
                        # ax2
                        #ax2b = ax2.twinx()   
                        if mov_avg: # plot moving average
                            n=50
                            #v_plot = (temp_e1_val-np.min(temp_e1_val))/np.max((temp_e1_val-np.min(temp_e1_val)))
                            v_plot = temp_e1_val
                            ax2.plot(temp_e1_tim, v_plot, '-', color='g', label='lake temperature', alpha = 1.)
                            #
                            #ax.plot(temp_e1_tim[:-n+1], moving_average(temp_e1_val, n=n), '--', color='k')#, label='temp. mov. avg.')
                            _x = temp_e1_tim[n-1-20:-20]
                            _y = moving_average(v_plot[::-1], n=n)[::-1]
                            ax2.plot(_x, _y, '--', color='k')#, label='lake temperature')
                        else:
                            v_plot = temp_e1_val
                            ax2.plot(temp_e1_tim, v_plot, '-', color='g', label='lake temperature', alpha = 1.)
                        ax2.set_ylabel('Temperature °C')
                        
                        _ylim = [min(_y)-1,max(_y)+1] 
                        ax2.set_ylim(_ylim)
                        #ax2.set_ylabel('temperature C')   
                except:
                    pass
                if plot_erup: # plot vertical lines
                    te = datetimeify(erup_time)#fm_e1.data.tes[int(erup[-1:])-1]
                    ax2.axvline(te, color='k',linestyle='--', linewidth=2, zorder = 4)
                    #ax2.plot([], color='k', linestyle='--', linewidth=2, label = 'eruption')

            ax2.legend(loc = 2)   
            ax2.grid()
            if True:  
                ax2.set_xlim([t0+2*day,t1])
            #ax2.set_xticks([t1 - 5*day*i for i in range(int(look_back/5)+1)])#[dat.index.values[0],dat.index.values[-1]])#, ]np.arange(0, len(x)+1, 5))
        # subplot three: filtered  RSAM, MF, HF datastreams
        #ax3
        if False:
            #
            td = TremorData(station = sta)
            #td.update(ti=t0, tf=t1)
            data_streams = ['rsam']#,'hf', 'mf']#, 'dsarF']
            label = ['nRSAM']#,'nHF','nMF','nDSAR']
            #label = ['1/RSAM']
            inv = False
            if False:
                data_streams = ['hf', 'mf', 'rsam', 'lf', 'vlf']
                label = ['HF','MF','RSAM','LF', 'VLF']

            if type(data_streams) is str:
                data_streams = [data_streams,]
            if any(['_' in ds for ds in data_streams]):
                td._compute_transforms()
            #ax.set_xlim(*range)
            # plot data for each year
            norm= False
            _range = [t0,t1]
            log =False
            col_def = None
            data = td.get_data(*_range)
            xi = datetime(year=1,month=1,day=1,hour=0,minute=0,second=0)
            cols = ['gray','g','r','m',[0.5,0.5,0.5],[0.75,0.75,0.75]]
            if inv:
                cols = ['gray','g','r','m',[0.5,0.5,0.5],[0.75,0.75,0.75]]
            inds = (data.index>=datetimeify(_range[0]))&(data.index<=datetimeify(_range[1]))
            i=0
            for data_stream, col in zip(data_streams,cols):
                v_plot = data[data_stream].loc[inds]
                if inv:
                    v_plot = 1/v_plot
                if log:
                    v_plot = np.log10(v_plot)
                if norm:
                    v_plot = (v_plot-np.min(v_plot))/np.max((v_plot-np.min(v_plot)))
                if label:
                    if col_def:
                        ax3.plot(data.index[inds], v_plot, '-', color=col_def, label=label[i], alpha = 1.0, linewidth=0.5)
                    else:
                        ax3.plot(data.index[inds], v_plot, '-', color=col, label=label[i], alpha = 1.0, linewidth=0.5)
                else:
                    ax3.plot(data.index[inds], v_plot, '-', color=col, label=data_stream, alpha = 1.0, linewidth=0.5)
                i+=1
            for te in td.tes:
                if [te>=datetimeify(_range[0]) and te<=datetimeify(_range[1])]:
                    pass
                    #ax.axvline(te, color='k', linestyle='--', linewidth=2, zorder = 0)
            #
            #ax.plot([], color='k', linestyle='--', linewidth=2, label = 'eruption')
            #ax3.set_xlim(_range)
            ax3.legend(loc = 2)
            ax3.grid()
            if log:
                ax3.set_ylabel(' ')
            else:
                ax3.set_ylabel('\u03BC m/s')
            #ax3.set_xlabel('Time [month-day hour]')
            #ax3.title.set_text('Station '+td.station+' ('+sta_code[td.station]+'): Tremor data')
            #
            if plot_erup: # plot vertical lines
                te = datetimeify(erup_time)#fm_e1.data.tes[int(erup[-1:])-1]
                ax3.axvline(te, color='k',linestyle='--', linewidth=2, zorder = 4)
                ax3.plot([], color='k', linestyle='--', linewidth=2, label = 'event')
            
            #
            te = datetimeify(erup_time)#fm_e1.data.tes[int(erup[-1:])-1]
            #ax3.set_xticks([te - 5*day*i for i in range(int(look_back/5)+1)])#[dat.index.values[0],dat.index.values[-1]])#, ]np.arange(0, len(x)+1, 5))

            ax3.set_yscale('log')
            lims = [10e-2,10e2]#[np.mean(np.log(v_plot))-3*np.std(np.log(v_plot)), np.mean(np.log(v_plot))+6*np.std(np.log(v_plot))]
            if sta == 'COP':
                lims = None
            ax3.set_ylim(lims)
            #
            ax3.set_ylim([1e0,1e4])
            if True:  
                ax3.set_xlim([t0+2*day,t1])
        # subplot four: non filtered  RSAM, MF, HF datastreams
        #ax4
        if True:
            #
            td = TremorData(station = sta)
            #td.update(ti=t0, tf=t1)
            data_streams = ['rsam','hf', 'mf']#, 'dsarF']
            label = ['RSAM','HF','MF','DSAR']
            #label = ['1/RSAM']
            inv = False
            if False:
                data_streams = ['hf', 'mf', 'rsam', 'lf', 'vlf']
                label = ['HF','MF','RSAM','LF', 'VLF']

            if type(data_streams) is str:
                data_streams = [data_streams,]
            if any(['_' in ds for ds in data_streams]):
                td._compute_transforms()
            #ax.set_xlim(*range)
            # plot data for each year
            norm= False
            _range = [t0,t1]
            log =False
            col_def = None
            data = td.get_data(*_range)
            xi = datetime(year=1,month=1,day=1,hour=0,minute=0,second=0)
            cols = ['r','g','gray','m',[0.5,0.5,0.5],[0.75,0.75,0.75]]
            if inv:
                cols = ['gray','g','r','m',[0.5,0.5,0.5],[0.75,0.75,0.75]]
            inds = (data.index>=datetimeify(_range[0]))&(data.index<=datetimeify(_range[1]))
            i=0
            for data_stream, col in zip(data_streams,cols):
                v_plot = data[data_stream].loc[inds]
                if inv:
                    v_plot = 1/v_plot
                if log:
                    v_plot = np.log10(v_plot)
                if norm:
                    v_plot = (v_plot-np.min(v_plot))/np.max((v_plot-np.min(v_plot)))
                if label:
                    if col_def:
                        ax4.plot(data.index[inds], v_plot, '-', color=col_def, label=label[i], alpha = 1.0)
                    else:
                        ax4.plot(data.index[inds], v_plot, '-', color=col, label=label[i], alpha = 1.0)
                else:
                    ax4.plot(data.index[inds], v_plot, '-', color=col, label=data_stream, alpha = 1.0)
                i+=1
            for te in td.tes:
                if [te>=datetimeify(_range[0]) and te<=datetimeify(_range[1])]:
                    pass
                    #ax.axvline(te, color='k', linestyle='--', linewidth=2, zorder = 0)
            #
            #ax.plot([], color='k', linestyle='--', linewidth=2, label = 'eruption')
            #ax4.set_xlim(_range)
            ax4.legend(loc = 2)
            ax4.grid()
            if log:
                ax4.set_ylabel(' ')
            else:
                ax4.set_ylabel('\u03BC m/s')
            #ax4.set_xlabel('Time [month-day hour]')
            #ax4.title.set_text('Station '+td.station+' ('+sta_code[td.station]+'): Tremor data')
            te = datetimeify(erup_time)#fm_e1.data.tes[int(erup[-1:])-1]
            if True: 
                ax4.set_xlim([t0+2*day,t1])
            #ax4.set_xticks([te - 5*day*i for i in range(int(look_back/5)+1)])#[dat.index.values[0],dat.index.values[-1]])#, ]np.arange(0, len(x)+1, 5))
            #ax4.set_ylim([1e9,1e13])
            #ax4.set_yscale('log')
            
            ## twin axis 
            if True:
                ax4b = ax4.twinx()
                data_streams = ['dsar']
                label = ['DSAR']
                #label = ['1/RSAM']

                if type(data_streams) is str:
                    data_streams = [data_streams,]
                if any(['_' in ds for ds in data_streams]):
                    td._compute_transforms()
                #ax.set_xlim(*range)
                # plot data for each year
                norm= False
                _range = [t0,t1]
                log =False
                col_def = None
                data = td.get_data(*_range)
                xi = datetime(year=1,month=1,day=1,hour=0,minute=0,second=0)
                cols = ['b','g','gray','m',[0.5,0.5,0.5],[0.75,0.75,0.75]]
                if inv:
                    cols = ['b','g','r','m',[0.5,0.5,0.5],[0.75,0.75,0.75]]
                inds = (data.index>=datetimeify(_range[0]))&(data.index<=datetimeify(_range[1]))
                i=0
                for data_stream, col in zip(data_streams,cols):
                    v_plot = data[data_stream].loc[inds]
                    if inv:
                        v_plot = 1/v_plot
                    if log:
                        v_plot = np.log10(v_plot)
                    if norm:
                        v_plot = (v_plot-np.min(v_plot))/np.max((v_plot-np.min(v_plot)))
                    if label:
                        if col_def:
                            ax4b.plot(data.index[inds], v_plot, '-', color=col_def, label=label[i], alpha = 1.0)
                        else:
                            ax4b.plot(data.index[inds], v_plot, '-', color=col, label=label[i], alpha = 1.0)
                    else:
                        ax4b.plot(data.index[inds], v_plot, '-', color=col, label=data_stream, alpha = 1.0)
                    i+=1
                for te in td.tes:
                    if [te>=datetimeify(_range[0]) and te<=datetimeify(_range[1])]:
                        pass
                        #ax.axvline(te, color='k', linestyle='--', linewidth=2, zorder = 0)
                #
                #ax.plot([], color='k', linestyle='--', linewidth=2, label = 'eruption')
                #ax4.set_xlim(_range)
                ax4b.legend(loc = 3)
                #ax4b.set_yscale('log')
                ax4b.grid()
                if log:
                    ax4b.set_ylabel(' ')
                else:
                    ax4b.set_ylabel('\u03BC m/s')

        # subplot extra: MF, RSAM, RMAR medians 
        #ax5
        if True:
            # features
            fts_yleft = ['zsc2_rmarF__median']#,'zsc2_lrarF__median','zsc2_lrarF__median','zsc2_vlarF__median']
            fts_yrigth = ['zsc2_rsamF__median','zsc2_mfF__median']

            col = ['m','m','r']
            alpha = [1., 1., 1.]
            thick_line = [2., 2., 1.]
            #
            # try: 
            for i, ft in enumerate(fts_yleft):
                if False: # load feature (else: cal feature. median or rv)
                    if 'zsc2_dsarF' in ft:
                        ds = ['zsc2_dsarF'] 
                    if server:
                        path_feat_serv = 'C:\\Users\\aar135\\codes_local_disk\\volc_forecast_tl\\features_bkp\\features_server\\'
                        fm_e1 = ForecastModel(window=2., overlap=1., station =  sta,
                            look_forward=2., data_streams=ds, 
                            feature_dir=path_feat_serv, 
                            savefile_type='pkl') 
                    elif server2:
                        path_feat_serv = 'U:\\Research\\EruptionForecasting\\eruptions\\features\\'
                        fm_e1 = ForecastModel(window=2., overlap=1., station =  sta,
                            look_forward=2., data_streams=ds, 
                            feature_dir=path_feat_serv, 
                            savefile_type='pkl') 
                    else:
                        try:
                            fm_e1 = ForecastModel(window=2., overlap=1., station = sta,
                                look_forward=2., data_streams=ds, savefile_type='csv')
                        except:
                            fm_e1 = ForecastModel(window=2., overlap=1., station = sta,
                                look_forward=2., data_streams=ds, savefile_type='pkl')                    
                    ##  
                    ft = ft.replace("-",'"')
                    # adding multiple Axes objects
                    ft_e1 = fm_e1.get_features(ti=t0, tf=t1, n_jobs=1, compute_only_features=[ft])
                    # extract values to plot 
                    ft_e1_t = ft_e1[0].index.values
                    ft_e1_v = ft_e1[0].loc[:,ft]
                    #
                    v_plot = ft_e1_v

                    #v_plot = ft_e1_v-np.min(ft_e1_v)/np.max((ft_e1_v-np.min(ft_e1_v)))
                    #v_plot = ft_e1_v/np.max(ft_e1_v)
                else: 
                    #
                    if 'zsc2_dsarF' in ft:
                        ds = 'zsc2_dsarF'
                    if 'zsc2_mfF' in ft:
                        ds = 'zsc2_mfF' 
                    if 'zsc2_hfF' in ft:
                        ds = 'zsc2_hfF' 
                    if 'zsc2_rmarF' in ft:
                        ds = 'zsc2_rmarF' 
                    if 'zsc2_lrarF' in ft:
                        ds = 'zsc2_lrarF' 
                    # 
                    #
                    day = timedelta(days=1)
                    fm = ForecastModel(window=2., overlap=1., station=sta,
                        look_forward=2., data_streams=[ds], 
                        data_dir=r'C:\Users\aar135\codes_local_disk\volc_forecast_tl\volc_forecast_tl\data'
                        )
                    #
                    N, M = [2,30]
                    df = fm.data.df[t0:t1]
                    if 'median' in ft:
                        test = df[ds].rolling(N*24*6).median()[N*24*6:]
                    if 'rate_variance' in ft:
                        test = df[ds].rolling(N*24*6).apply(chqv)[N*24*6:]
                    #
                    #out = test.rolling(archtype.shape[0]).apply(partial(conv, (archtype-archtype.mean())/archtype.std()))       
                    #out = out.resample('1D').ffill()
                    #
                    ft_e1_t = test.index
                    v_plot = test.values
                #
                if ft == 'zsc2_dsarF__median':
                    ft = 'nDSAR median'
                if ft == 'zsc2_rmarF__median':
                    ft = 'nRMAR median'
                if ft == 'zsc2_lrarF__median':
                    ft = 'nLRAR median'
                #
                if i == 0:
                    _max= max(v_plot)
                    ax5.plot(ft_e1_t, v_plot, '-', color=col[i], linewidth=thick_line[0], alpha = alpha[i],label=' '+ ft)
                    #_max = 1.
                else:
                    ax5.plot(ft_e1_t, v_plot/max(v_plot) *.95*_max, '-', color=col[i], linewidth=thick_line[0], alpha = alpha[i],label=' '+ ft)
                #
                #
                if ffm: # ffm 
                    if i == 0:
                        #ax1b = ax1.twinx() 
                        #v_plot = data[data_stream].loc[inds]
                        inv_rsam = fm.data.get_data(ti=t0, tf=t1)['rsamF']#.loc[ft_e1_t]
                        inv_rsam_t = inv_rsam.index
                        inv_rsam = 1./inv_rsam.values
                        # normalized it to yaxis rigth 
                        #inv_rsam = inv_rsam/max(inv_rsam)
                        inv_rsam = inv_rsam*_max*8
                        #
                        ax5.plot(inv_rsam_t, inv_rsam, '-', color= 'gray', linewidth=0.5, markersize=0.5, alpha = 1.)
                        ax5.plot([], [], '-', color= 'gray', markersize=1, label='1/RSAM', alpha = 1.0)
                        ax5.set_ylim([0,30])
                        #ax1.set_yticks([])
                #
                if plot_erup: # plot vertical lines
                    te = datetimeify(erup_time)#fm_e1.data.tes[int(erup[-1:])-1]
                    ax5.axvline(te, color='k',linestyle='--', linewidth=2, zorder = 4)
                    ax5.plot([], color='k', linestyle='--', linewidth=2, label = 'eruption')
                #
                
                #
                te = datetimeify(erup_time)#fm_e1.data.tes[int(erup[-1:])-1]
                #ax1.set_xticks([ft_e1[0].index[-1] - 7*day*i for i in range(int(30/7)+1)])#[dat.index.values[0],dat.index.values[-1]])#, ]np.arange(0, len(x)+1, 5))
                
                #ax1b.set_yticks([])
                #ax1.set_yscale('log')
                ax5.grid()
                ax5.set_ylabel('nRMAR value')        #ax.set_xticks([ft_e1[0].index[-1]-7*day*i +day for i in range(5)])
                #ax.set_xticks([ft_e1[0].index[-1] - 7*day*i for i in range(int(30/7)+1)])#[dat.index.values[0],dat.index.values[-1]])#, ]np.arange(0, len(x)+1, 5))
                #ax1.set_xticks([t1 - 5*day*i for i in range(int(look_back/5)+1)])#[dat.index.values[0],dat.index.values[-1]])#, ]np.arange(0, len(x)+1, 5))
            #     #
            # except:
            #     pass
            if fts_yrigth:
                ax5b = ax5.twinx() 
                col = ['gray','r','r']
                alpha = [.5, .5, .5]
                thick_line = [1.,1.,1.]
                #try: 
                for i, ft in enumerate(fts_yrigth):
                    if 'zsc2_dsarF' in ft:
                        ds = 'zsc2_dsarF'
                    if 'zsc2_mfF' in ft:
                        ds = 'zsc2_mfF' 
                    if 'zsc2_hfF' in ft:
                        ds = 'zsc2_hfF'
                    if 'zsc2_rsamF' in ft:
                        ds = 'zsc2_rsamF' 
                    if 'zsc2_lrarF' in ft:
                        ds = 'zsc2_lrarF' 
                    if 'zsc2_vlarF' in ft:
                        ds = 'zsc2_vlarF' 
                    # 
                    if False: # look feature in the prev cacl features (else: calculate feat from data; only for median and rv)
                        if server:
                            path_feat_serv = 'C:\\Users\\aar135\\codes_local_disk\\volc_forecast_tl\\features_bkp\\features_server\\'
                            fm_e1 = ForecastModel(window=2., overlap=1., station =  sta,
                                look_forward=2., data_streams=ds, 
                                feature_dir=path_feat_serv, 
                                savefile_type='pkl') 
                        elif server2:
                            path_feat_serv = 'U:\\Research\\EruptionForecasting\\eruptions\\features\\'
                            fm_e1 = ForecastModel(window=2., overlap=1., station =  sta,
                                look_forward=2., data_streams=ds, 
                                feature_dir=path_feat_serv, 
                                savefile_type='pkl') 
                        else:
                            try:
                                fm_e1 = ForecastModel(window=2., overlap=1., station = sta,
                                    look_forward=2., data_streams=ds, savefile_type='csv')
                            except:
                                fm_e1 = ForecastModel(window=2., overlap=1., station = sta,
                                    look_forward=2., data_streams=ds, savefile_type='pkl')                    
                        ##  
                        ft = ft.replace("-",'"')
                        
                        ft_e1 = fm_e1.get_features(ti=t0, tf=t1, n_jobs=1, compute_only_features=[ft])
                        # adding multiple Axes objects

                        # extract values to plot 
                        ft_e1_t = ft_e1[0].index.values
                        ft_e1_v = ft_e1[0].loc[:,ft]
                        #
                        v_plot = ft_e1_v

                    else:
                        day = timedelta(days=1)
                        fm = ForecastModel(window=2., overlap=1., station=sta,
                            look_forward=2., data_streams=[ds], 
                            data_dir=r'C:\Users\aar135\codes_local_disk\volc_forecast_tl\volc_forecast_tl\data'
                            )
                        #
                        N, M = [2,30]
                        df = fm.data.df[t0:t1]
                        if 'median' in ft:
                            test = df[ds].rolling(N*24*6).median()[N*24*6:]
                        if 'rate_variance' in ft:
                            test = df[ds].rolling(N*24*6).apply(chqv)[N*24*6:]
                        #
                        #out = test.rolling(archtype.shape[0]).apply(partial(conv, (archtype-archtype.mean())/archtype.std()))       
                        #out = out.resample('1D').ffill()
                        #
                        ft_e1_t = test.index
                        v_plot = test.values

                    #v_plot = ft_e1_v-np.min(ft_e1_v)/np.max((ft_e1_v-np.min(ft_e1_v)))
                    #v_plot = ft_e1_v/np.max(ft_e1_v)
                    #
                    if ft == 'zsc2_mfF__median':
                        ft = 'nMF median'
                    if ft == 'zsc2_hfF__median':
                        ft = 'nHF median'
                    if ft == 'zsc2_rsamF__median':
                        ft = 'nRSAM median'
                    if ft == 'zsc2_lrarF__median':
                        ft = 'nLRAR median'
                    if ft == 'zsc2_vlarF__median':
                        ft = 'nVLAR median'
                    #
                    if i == 0:
                        ax5b.plot(ft_e1_t, v_plot, '-', color=col[i], alpha = alpha[i],label=' '+ ft)
                        _max = max(v_plot)
                        #_max = 1.
                    else:
                        v_plot = v_plot/max(v_plot)*0.9*_max
                        ax5b.plot(ft_e1_t, v_plot, '-', color=col[i], alpha = alpha[i],label=' '+ ft)
                    #
                    ax5b.legend(loc = 3)
                    #
                    te = datetimeify(erup_time)#fm_e1.data.tes[int(erup[-1:])-1]
                    #ax1.set_xticks([ft_e1[0].index[-1] - 7*day*i for i in range(int(30/7)+1)])#[dat.index.values[0],dat.index.values[-1]])#, ]np.arange(0, len(x)+1, 5))
                    
                    #ax1b.set_yticks([])
                    ax5b.grid()
                    #ax5b.set_ylim([1e0,1e4])
                    ax5b.set_ylabel('Feature value')        #ax.set_xticks([ft_e1[0].index[-1]-7*day*i +day for i in range(5)])
                    #ax.set_xticks([ft_e1[0].index[-1] - 7*day*i for i in range(int(30/7)+1)])#[dat.index.values[0],dat.index.values[-1]])#, ]np.arange(0, len(x)+1, 5))
                    #ax1.set_xticks([t1 - 5*day*i for i in range(int(look_back/5)+1)])#[dat.index.values[0],dat.index.values[-1]])#, ]np.arange(0, len(x)+1, 5))
                    #
                #except:
                #    pass
            #
            ax5.legend(loc = 2)
            if True:  
                ax5.set_xlim([t0+2*day,t1])
                if _log:
                    ax5.set_yscale('log')
                    if fts_yrigth:
                        ax5b.set_yscale('log')
        #
        if date_line: # plot vertical lines
            te = datetimeify(erup_time)#fm_e1.data.tes[int(erup[-1:])-1]
            ax1.axvline(te, color='k',linestyle='--', linewidth=2, zorder = 4)
            ax2.axvline(te, color='k',linestyle='--', linewidth=2, zorder = 4)
            ax4.axvline(te, color='k',linestyle='--', linewidth=2, zorder = 4)
            ax1.plot([], color='k', linestyle='--', linewidth=2, label = 'event')
        #
        if save_png_path:
            dat = erup_time.strftime('%Y-%m-%d')
            title =  sta+'_'+dat+'_'+'look_back'+str(look_back)
            ax1.set_title(title)
            plt.savefig(save_png_path+sta+'_'+dat+'_'+'look_back'+str(look_back)+'.png')
        #
        # ax1.set_xticks([t1 - 2*day*i for i in range(int((look_back+look_front)/2)+1)])
        # ax2.set_xticks([t1 - 2*day*i for i in range(int((look_back+look_front)/2)+1)])
        # ax3.set_xticks([t1 - 2*day*i for i in range(int((look_back+look_front)/2)+1)])
        # ax4.set_xticks([t1 - 2*day*i for i in range(int((look_back+look_front)/2)+1)])
        
        #plt.show()
        plt.close('all')

    # define station and time
    if False: # on plot
        sta = 'FWVZ' #'FWVZ'#'WIZ'# 'DAM'
        erup = 1
        erup_time = datetimeify(erup_times[sta+'_'+str(erup)])
        #
        erup_time = datetimeify('2022-05-17 00:00:00')
        #
        #erup_time = datetimeify('2020-06-16 04:40:00')
        #erup_time = datetimeify('2020-07-16 06:50:00')
        #erup_time = datetimeify('2020-08-05 18:50:00')
        #
        look_back = 120#365*2.75
        look_front = 0
        #t0 = erup_time - look_back*day#30*day
        #t1 = erup_time + 20*day#hour
        save_png_path = 'C:\\Users\\aar135\\codes_local_disk\\volc_forecast_tl\\volc_forecast_tl\\features\\lake_data\\FWVZ\\'
        #
        plot_seismic_temp_data(sta = sta, erup_time = erup_time, look_back = look_back, look_front =look_front, save_png_path = None, plot_erup = True)
        asdf

    if False: # plot several (loop)
        sta = 'POS'#'FWVZ'
        #final_time = '2021 09 30 12 00 00'
        years = ['2013']#['2021','2020','2019','2018','2017','2016','2015','2014','2013','2012','2011','2010','2009'] # '2021','2020','2019','2018',
        for year in years:
            final_time = year+' 12 30 12 00 00'
            if year == '2021' and sta == 'FWVZ':
                final_time = year+' 09 12 12 00 00'
            if year == '2013' and sta == 'DAM':
                final_time = year+' 06 14 00 00 00'
            if year == '2013' and sta == 'POS':
                final_time = year+' 06 14 00 00 00'
            #final_time = '2020 12 30 12 00 00'
            final_time = datetimeify(final_time)
            time_lapse = 30 # days
            ite = 12 #
            #end_time = final_time - time_lapse*day *ite
            # list of reference times
            ref_times = [final_time - i*time_lapse*day for i in range(ite)]
            # iterate 
            save_png_path = 'C:\\Users\\aar135\\codes_local_disk\\volc_forecast_tl\\volc_forecast_tl\\features\\lake_data\\'+sta+'\\'
            print(year)
            for i, r_time in enumerate(ref_times):
                print(str(i+1)+'/'+str(len(ref_times)))
                plot_seismic_temp_data(sta = sta, erup_time = r_time, look_back = time_lapse, save_png_path = save_png_path)

    if True: # plot selected dates (ref for events)
        #auto_picked = True # dates elected automaticaly from dsar median and dsar rv correlations (only with FWVZ)

        stas = ['COP']#FWVZ']#,'POS']
        #final_time = '2021 09 30 12 00 00'
        for sta in stas:
            print(sta)
            # get eruptions
            if man_picked:
                path_mp = 'C:\\Users\\aar135\\codes_local_disk\\volc_forecast_tl\\volc_forecast_tl\\features\\lake_data\\'+sta+'\\selection\\'
                path = path_mp
            if auto_picked:
                path_ap = 'C:\\Users\\aar135\\codes_local_disk\\volc_forecast_tl\\volc_forecast_tl\\features\\lake_data\\corr_dsar_ratevar\\'
                path = path_ap
                save_png_path = path_ap + 'plot_auto_pick\\'
            if man_picked and auto_picked:
                path_ap_mp = 'C:\\Users\\aar135\\codes_local_disk\\volc_forecast_tl\\volc_forecast_tl\\features\\lake_data\\corr_dsar_ratevar\\'
                path = path_ap_mp
            dates =[]
            if man_picked:
                path_dates_mp = path_mp+sta+'_temp_eruptive_periods.txt'
            if auto_picked:
                path_dates_ap = path_ap+sta+'_dates_missed_events_from_dsar_median_rv_cc.txt'
                path_dates_ap = path_ap+sta+'_dates_max_CC_missed_events_from_dsar_median_rv_with_filt_peak.txt'
                #path_dates_ap = path_ap+'FWVZ_dates_max_CC_missed_events_from_dsar_median_rv_with_filt_peak.txt'
            #
            if auto_picked:
                with open(path_dates_ap,'r') as fp:
                    dates = [datetimeify(ln.rstrip()) for ln in fp.readlines()]
            if man_picked:
                with open(path_dates_mp,'r') as fp:
                    dates = [datetimeify(ln.rstrip()) for ln in fp.readlines()]
            if auto_picked and man_picked:
                with open(path_dates_ap,'r') as fp:
                    dates1 = [datetimeify(ln.rstrip()) for ln in fp.readlines()]
                with open(path_dates_mp,'r') as fp:
                    dates2 = [datetimeify(ln.rstrip()) for ln in fp.readlines()]
                dates = dates1 + dates2
            #
            for i, date in enumerate(dates):#[0:40]):
                print(str(i+1)+'/'+str(len(dates)))
                print(date)
                #date = datetimeify('2010-10-04 09:30:00')
                look_back = 20 #days
                look_front = 15
                plot_seismic_temp_data(sta = sta, erup_time = date, look_back = look_back, 
                    look_front = look_front, save_png_path = save_png_path, date_line = True, level = True,  rainfall = True)

# lake temperature correlation
sta = 'FWVZ'
roll_mean = True
ite = 100

def temp_erup_ana(sta = sta, ite = ite):
    '''
    Calculate temperature diference and rate before eruptive events read from file. 
    Its a random proccess, where reference dates are 'perturb' (jitter) during several iterations. 
    '''
    # 
    #auto_picked = True # dates elected automaticaly from dsar median and dsar rv correlations (only with FWVZ)
    stas = [sta]#,'POS'] #'POS','FWVZ'
    if stas[0] == 'FWVZ':
        _from = datetimeify('2009-06-01')
    if stas[0] == 'POS':
        _from = datetimeify('2013-01-03')
    #
    for sta in stas:
        jitter = True
        if jitter:
            ite = ite
        else:
            ite = 1
        #
        l_back = 0*day
        l_for = 5*day
        utc_0 = True
        # import event dates
        # get eruptions
        if man_picked:
            path_mp = 'C:\\Users\\aar135\\codes_local_disk\\volc_forecast_tl\\volc_forecast_tl\\features\\lake_data\\'+sta+'\\selection\\'
            path = path_mp
        if auto_picked:
            path_ap = 'C:\\Users\\aar135\\codes_local_disk\\volc_forecast_tl\\volc_forecast_tl\\features\\lake_data\\corr_dsar_ratevar\\'
            path = path_ap
        if man_picked and auto_picked:
            path_ap_mp = 'C:\\Users\\aar135\\codes_local_disk\\volc_forecast_tl\\volc_forecast_tl\\features\\lake_data\\corr_dsar_ratevar\\'
            path = path_ap_mp
        dates =[]
        if man_picked:
            path_dates_mp = path_mp+sta+'_temp_eruptive_periods.txt'
        if auto_picked:
            path_dates_ap = path_ap+sta+'_dates_missed_events_from_dsar_median_rv_cc.txt' # dsar and rsam peak
            if sta == 'FWVZ':
                path_dates_ap = path_ap+sta+'_dates_max_CC_missed_events_from_dsar_median_rv_with_filt_peak_man_refine_dates.txt'
            #
            just_heating = False
            just_cooling = False
            #
            if just_heating:
                path_dates_ap = path_ap+'FWVZ_heating_cycle_date_events.txt'
            if just_cooling:
                path_dates_ap = path_ap+'FWVZ_cooling_cycle_date_events.txt'
        #
        if auto_picked:
            with open(path_dates_ap,'r') as fp:
                dates = [datetimeify(ln.rstrip()) for ln in fp.readlines()]
        if man_picked:
            with open(path_dates_mp,'r') as fp:
                dates = [datetimeify(ln.rstrip()) for ln in fp.readlines()]
        if auto_picked and man_picked:
            with open(path_dates_ap,'r') as fp:
                dates1 = [datetimeify(ln.rstrip()) for ln in fp.readlines()]
            with open(path_dates_mp,'r') as fp:
                dates2 = [datetimeify(ln.rstrip()) for ln in fp.readlines()]
            dates = dates1 + dates2
        #
        dates = [dat for dat in dates if dat > _from]
        # import temp data
        # import temp data
        #ti_e1 = t0
        #tf_e1 = t1
        # import temp data
        if sta == 'FWVZ':
            path2 = '..'+os.sep+'data'+os.sep+"RU001_temp_data.csv"
        if sta == 'POS':
            path2 = '..'+os.sep+'data'+os.sep+"POS_temp_data.csv"
        pd_temp = pd.read_csv(path2, index_col=1)
        if utc_0:
            pd_temp.index = [datetimeify(pd_temp.index[i])-6*hour for i in range(len(pd_temp.index))]
        else:
            pd_temp.index = [datetimeify(pd_temp.index[i]) for i in range(len(pd_temp.index))]
        # lists to save
        dif_l = []
        rate_l = []
        rate_days_l = []
        date_dT_l = []
        temp_ev_l = []
        # loop over dates 
        for k in range(ite):
            for i, dat in enumerate(dates):
                dat_ori = dat
                if jitter:
                    #dat = dat - 7*day + random.randint(0, 14)*day
                    #dat = dat - 4*day + random.randint(0, 7)*day
                    dat = dat - 1*day + random.randint(0, 1)*day
                # (1) explor back and for for min and max
                test = pd_temp[dat-l_back : dat+ l_for]
                if sta == 'POS':
                    pass
                    test[" t (C)"] = test[" t (C)"].rolling(window=10).mean()
                if roll_mean:
                    if sta == 'FWVZ':
                        pass
                        test[" t (C)"] = test[" t (C)"].rolling(window=10).mean()
                #
                if dat_ori == datetimeify('2009-07-13 06:30:00'):
                    a = 1
                    #pass
                    # plt.plot(test.index, test[" t (C)"].values)
                    # test[" t (C)"] = test[" t (C)"].rolling(window=15).mean()
                    # plt.plot(test.index, test[" t (C)"].values)
                    # plt.show()
                    # asdf
                #print(test.head())
                #print(test.tail())
                #_max = max(test[" t (C)"])
                _max_idx = test[" t (C)"].idxmax(axis = 1)
                _max =  test[" t (C)"].max()
                #_min = min(test[" t (C)"])
                _min_idx = test[" t (C)"].idxmin(axis = 1)
                _min =  test[" t (C)"].min()
                #
                #temp_ev_l.append(test[" t (C)"].mean())
                temp_ev_l.append(test[" t (C)"].values[-1])
                #
                try:
                    dif = (_max - _min) #abs(_max - _min) 
                    #
                    sign = (_max_idx - _min_idx).days / abs((_max_idx - _min_idx).days)
                    dif = dif / sign
                    #
                    rate = sign*dif/(_max_idx - _min_idx).days
                    #
                    if not math.isinf(rate):
                        dif_l.append(dif)
                        rate_l.append(rate)
                        rate_days_l.append((_max_idx - _min_idx).days)
                        date_dT_l.append([dat_ori, dif])
                except:
                    pass
                #
        # write output
        #path = '..'+os.sep+'features'+os.sep+'correlations'+os.sep+'stat_sign_per_hig_corr_feat'+os.sep
        with open(path+sta+'_temp_dif_rate_ite'+str(ite)+'.txt', 'w') as f:
            for k in range(len(dif_l)):
                f.write(str(round(dif_l[k],2))+'\t'+str(round(rate_l[k],2))+'\t'+str(round(rate_days_l[k],2))+'\n')
        if auto_picked and not man_picked:
            with open(path+sta+'_date_dT_ite'+str(ite)+'.txt', 'w') as f:
                for k in range(len(date_dT_l)):
                    f.write(str(date_dT_l[k][0])+','+str(round(date_dT_l[k][1],2))+'\n')
        with open(path+sta+'_temp_mean_event_'+str(ite)+'.txt', 'w') as f:
            for k in range(len(temp_ev_l)):
                f.write(str(round(temp_ev_l[k],2))+'\n')

def temp_dif_rate_stats(sta = sta, ite = ite):
    '''
    explore for temperature diference and rates over the whole record
    '''
    #auto_picked = True # dates elected automaticaly from dsar median and dsar rv correlations (only with FWVZ)
    stas = [sta]
    #stas = ['FWVZ','POS']#'FWVZ'
    for sta in stas:
        utc_0 = True
        pass
        # import temp record
        # import temp data
        if sta == 'FWVZ':
            path2 = '..'+os.sep+'data'+os.sep+"RU001_temp_data.csv"
        if sta == 'POS':
            path2 = '..'+os.sep+'data'+os.sep+"POS_temp_data.csv"
        #path2 = 'C:\\Users\\aar135\\codes_local_disk\\volc_forecast_tl\\volc_forecast_tl\\features\\lake_data\\FWVZ\\selection\\'
        pd_temp = pd.read_csv(path2, index_col=1)
        if utc_0:
            pd_temp.index = [datetimeify(pd_temp.index[i])-6*hour for i in range(len(pd_temp.index))]
        else:
            pd_temp.index = [datetimeify(pd_temp.index[i]) for i in range(len(pd_temp.index))]
        
        # import dif and rate
        path = 'C:\\Users\\aar135\\codes_local_disk\\volc_forecast_tl\\volc_forecast_tl\\features\\lake_data\\'+sta+'\\selection\\'
        if auto_picked:
            path = 'C:\\Users\\aar135\\codes_local_disk\\volc_forecast_tl\\volc_forecast_tl\\features\\lake_data\\corr_dsar_ratevar\\'
        _fls = glob.glob(path+sta+"_temp_dif_rate_ite100.txt")
        dif_l = []
        rate_l = []
        rate_days_l = []
        for _fl in _fls:
            fl  = np.genfromtxt(_fl, delimiter="\t")
            [dif_l.append(fl[i][0]) for i in range(len(fl))]
            [rate_l.append(fl[i][1]) for i in range(len(fl))]
            [rate_days_l.append(fl[i][2]) for i in range(len(fl))]
        
        # loop over rate
        ite = ite
        if sta == 'FWVZ':
            s_date = datetimeify('2009 05 01 00 00 00')
            f_date = datetimeify('2021 12 31 00 00 00')
        if sta == 'POS':
            s_date = datetimeify('2012 12 26 00 00 00')
            f_date = datetimeify('2013 06 13 00 00 00')
        #
        pd_temp = pd_temp[s_date: f_date]
        delta = f_date - s_date
        #
        dif_list = []
        rate_list = []
        days_list = []

        # remove rates from eruptive periods
        remv_erup_dates = True
        if remv_erup_dates: # locate_missed_events_seismic() need to be run too
            # get eruptions
            path = 'C:\\Users\\aar135\\codes_local_disk\\volc_forecast_tl\\volc_forecast_tl\\features\\lake_data\\'+sta+'\\selection\\'
            if auto_picked:
                path = 'C:\\Users\\aar135\\codes_local_disk\\volc_forecast_tl\\volc_forecast_tl\\features\\lake_data\\corr_dsar_ratevar\\'
            dates =[]
            path_dates = path+sta+'_temp_eruptive_periods.txt'
            if auto_picked:
                path_dates = path+sta+'_dates_missed_events_from_dsar_median_rv_cc.txt'
            #
            with open(path_dates,'r') as fp:
                dates = [datetimeify(ln.rstrip()) for ln in fp.readlines()]
            l_dates_rej = []
            for date in dates:
                [l_dates_rej.append(date-15*day+ i*day) for i in range(30)]
            #
        for i in range(ite):
            try:
                # sample a day diference 
                _idx = random.randint(0, len(rate_days_l))
                d_day = rate_days_l[_idx]
                # resample pd
                _ = str(int(d_day))+'D'
                _test =  pd_temp.resample(_).ffill()
                #
                if roll_mean:
                    if sta == 'FWVZ':
                        pass
                        _test[" t (C)"] = _test[" t (C)"].rolling(window=10).mean()
                    if sta == 'POS':
                        pass
                        _test[" t (C)"] = _test[" t (C)"].rolling(window=10).mean()
                #
                _difs = _test[" t (C)"].diff()[1:]
                _rates = _difs/int(d_day)

                for k in range(len(_difs.values)):
                    if not np.isnan(_rates.values[k]):
                        if remv_erup_dates:
                            if _rates.index[i] not in l_dates_rej:
                                dif_list.append(_difs.values[k])#(abs(_difs.values[k]))
                                rate_list.append(_rates.values[k])
                                days_list.append(d_day)
                        else:
                            dif_list.append(_difs.values[k])
                            rate_list.append(_rates.values[k])
                            days_list.append(d_day)
                # [dif_list.append(abs(_difs.values[k])) for k in range(len(_difs.values))]
                # [rate_list.append(abs(_rates.values[k])) for k in range(len(_rates.values))]
                # [days_list.append(abs(d_day)) for k in range(len(_rates.values))]
            except:
                pass
        # write output
        #path = '..'+os.sep+'features'+os.sep+'correlations'+os.sep+'stat_sign_per_hig_corr_feat'+os.sep
        with open(path+sta+'_temp_dif_rate_ite'+str(ite)+'_out.txt', 'w') as f:
            for k in range(len(dif_list)):
                f.write(str(round(dif_list[k],2))+'\t'+str(round(rate_list[k],2))+'\t'+str(round(days_list[k],2))+'\n')

def plot_temp_erup_ana(sta = sta, ite = ite):
    '''
    '''
    if False: # plot one 
        # read results 
        path = 'C:\\Users\\aar135\\codes_local_disk\\volc_forecast_tl\\volc_forecast_tl\\features\\lake_data\\FWVZ\\selection\\'

        _fls = glob.glob(path+"FWVZ_temp_dif_rate_ite100.txt")
        dif_l = []
        rate_l = []
        rate_days_l = []
        for _fl in _fls:
            fl  = np.genfromtxt(_fl, delimiter="\t")
            [dif_l.append(fl[i][0]) for i in range(len(fl))]
            [rate_l.append(fl[i][1]) for i in range(len(fl))]
            [rate_days_l.append(fl[i][2]) for i in range(len(fl))]

        # plot histograms 
        fig, (ax1, ax2, ax3) = plt.subplots(nrows = 1, ncols = 3, figsize = (12,4))
        xlim = None#[0,0.25]
        ylim = None#[0,70]
        colors = ['b', 'r', 'g', 'm']

        # dif
        n_bins = 20#int(np.sqrt(len(dif_l)))
        ax1.hist(dif_l, n_bins, histtype='bar', color = colors[1], edgecolor='k', label = 'dif_temp')
        ax1.set_xlabel('temperature [°C]', fontsize=textsize)
        ax1.set_ylabel('samples', fontsize=textsize)
        ax1.grid(True, which='both', linewidth=0.1)
        xlim = [0, 12.5]
        ax1.set_xlim(xlim)
        ax1.set_ylim(ylim)
        #ax1.set_title('RSAM', fontsize = textsize)
        #ax1.plot([np.median(dif_l), np.median(dif_l)],[0, count_rsam+ count_rsam*.1],'k--', label = 'median: '+str(round(np.median(dif_l),2)))
        ax1.legend(loc = 1)

        # days
        #n_bins = int(np.sqrt(len(rate_l))/3)
        ax2.hist(rate_days_l, n_bins, histtype='bar', color = colors[2], edgecolor='k', label = 'days')
        ax2.set_xlabel('days', fontsize=textsize)
        ax2.set_ylabel(' ', fontsize=textsize)
        ax2.grid(True, which='both', linewidth=0.1)
        xlim = [0, 10]
        ax2.set_xlim(xlim)
        ax2.set_ylim(ylim)
        #ax1.set_title('RSAM', fontsize = textsize)
        #ax2.plot([np.median(rate_l), np.median(rate_l)],[0, count_rsam+ count_rsam*.1],'k--', label = 'median: '+str(round(np.median(rate_l),2)))
        ax2.legend(loc = 1)

        # rate
        #n_bins = int(np.sqrt(len(rate_l))/3)
        ax3.hist(rate_l, n_bins, histtype='bar', color = colors[0], edgecolor='k', label = 'rate')
        ax3.set_xlabel('rate [°C/day]', fontsize=textsize)
        ax3.set_ylabel(' ', fontsize=textsize)
        ax3.grid(True, which='both', linewidth=0.1)
        xlim = [0, 8]
        ax3.set_xlim(xlim)
        ax3.set_ylim(ylim)
        #ax1.set_title('RSAM', fontsize = textsize)
        #ax2.plot([np.median(rate_l), np.median(rate_l)],[0, count_rsam+ count_rsam*.1],'k--', label = 'median: '+str(round(np.median(rate_l),2)))
        ax3.legend(loc = 1)

        plt.show()

    if True:
        #auto_picked = True # dates elected automaticaly from dsar median and dsar rv correlations (only with FWVZ)
        sta = sta#'FWVZ' 'POS'
        # read results 
        path1 = 'C:\\Users\\aar135\\codes_local_disk\\volc_forecast_tl\\volc_forecast_tl\\features\\lake_data\\'+sta+'\\selection\\'
        path2 = 'C:\\Users\\aar135\\codes_local_disk\\volc_forecast_tl\\volc_forecast_tl\\features\\lake_data\\'+sta+'\\selection\\'
        if auto_picked:
            path1 = 'C:\\Users\\aar135\\codes_local_disk\\volc_forecast_tl\\volc_forecast_tl\\features\\lake_data\\corr_dsar_ratevar\\'
            path2 = 'C:\\Users\\aar135\\codes_local_disk\\volc_forecast_tl\\volc_forecast_tl\\features\\lake_data\\corr_dsar_ratevar\\'

        path1 = path1 +sta+"_temp_dif_rate_ite"+str(ite)+".txt"
        path2 = path2 +sta+"_temp_dif_rate_ite"+str(ite)+"_out.txt"
        # if sta == 'POS':#'FWVZ'
        #     path2 = path2 +sta+"_temp_dif_rate_ite10_out.txt"
        # if sta == 'FWVZ':#'FWVZ'            
        #     path2 = path2 +sta+"_temp_dif_rate_ite100_out.txt"
        #     #path2 = path2 +sta+"_temp_dif_rate_ite10_out.txt"
        #
        _fls = glob.glob(path1)
        dif_l1 = []
        rate_l1 = []
        rate_days_l1 = []
        for _fl in _fls:
            fl  = np.genfromtxt(_fl, delimiter="\t")
            [dif_l1.append(fl[i][0]) for i in range(len(fl))]
            [rate_l1.append(fl[i][1]) for i in range(len(fl))]
            [rate_days_l1.append(fl[i][2]) for i in range(len(fl))]

        _fls = glob.glob(path2)
        dif_l2 = []
        rate_l2 = []
        rate_days_l2 = []
        for _fl in _fls:
            fl  = np.genfromtxt(_fl, delimiter="\t")
            [dif_l2.append(fl[i][0]) for i in range(len(fl))]
            [rate_l2.append(fl[i][1]) for i in range(len(fl))]
            [rate_days_l2.append(fl[i][2]) for i in range(len(fl))]

        #replace rate > 5
        if True:
            for i, r in enumerate(rate_l1):
                if abs(r)>5:
                    rate_l1[i] = rate_l1[i]/3 

        dif_l1 = dif_l1[:int(len(dif_l1)/2)]
        rate_l1 =rate_l1[:int(len(rate_l1)/2)]
        rate_days_l1 =rate_days_l1[:int(len(rate_days_l1)/2)]

        ## (5) construct p-val histogram of both populations
        plot_rate = True
        if plot_rate:
            fig, (ax1, ax2, ax3) = plt.subplots(nrows = 1, ncols = 3, figsize = (12,4))
        else:
            fig, (ax1, ax2) = plt.subplots(nrows = 1, ncols = 2, figsize = (8,4))

        colors = ['b', 'r', 'g', 'm']
        #_heights, a_bins = np.histogram(pv_samp_in_1)
        
        # select lists 
        #ax.axvline(x=0.05, color = 'k', ls='--', linewidth=1, label = '0.05 threshold')
        multi = [dif_l1, dif_l2]
        colors = ['r', 'b']
        labels = ['in eruption', 'out eruption']
        bins = 20#np.linspace(0, 1, 13)
        ax1.hist(multi, bins, color = colors, label=labels, density = True)
        xlim = None#[0, 7] #12.5]

        ax1.set_xlabel('d_temp [°C]')
        ax1.set_ylabel('pdf')
        ax1.set_title('Lake temperature difference')
        #ax1.set_xlim(xlim)
        #ax.set_xscale('log')
        ##
        rate_days_l1 = abs(np.asarray(rate_days_l1))
        multi = [rate_days_l1, []]# rate_days_l2]
        colors = ['r', 'b']
        labels = ['in eruption', 'out eruption']
        bins = 20#np.linspace(0, 1, 13)
        ax2.hist(multi, bins, color = colors, density = True)
        xlim = None#None #[0, 4.2]#5] 8]
        ax2.set_xlabel('dt [days]')
        ax2.set_title('Days period')
        #ax2.set_ylabel('pdf')
        ax2.set_xlim(xlim)
        #ax.set_xscale('log')

        if plot_rate:
            multi = [rate_l1, rate_l2]
            colors = ['r', 'b']
            labels = ['in eruption', 'out eruption']
            bins = 20#np.linspace(0, 1, 13)
            ax3.hist(multi, bins, color = colors, label=labels, density = True)
            xlim = None#[-2, 5]#None#[-3, 3]#None #[0, 4.2]#5] 8]
            ax3.set_xlabel('[°C/days]')
            #ax2.set_ylabel('pdf')
            ax3.set_title('Rate')
            ax3.set_xlim(xlim)
            #ax.set_xscale('log')

        plt.legend(loc='upper right')
        plt.show()

def temp_level_change_corr_ana(sta = sta, ite = ite):
    '''
    out txt with temp and level change during events (for scattet plot)
    Its a random proccess, where reference dates are 'perturb' (jitter) during several iterations. 
    '''
    # 
    #auto_picked = True # dates elected automaticaly from dsar median and dsar rv correlations (only with FWVZ)
    stas = [sta]#,'POS'] #'POS','FWVZ'
    if stas[0] == 'FWVZ':
        _from = datetimeify('2009-06-01')
    #
    for sta in stas:
        jitter = True
        if jitter:
            ite = ite
        else:
            ite = 1
        #
        l_back = 30*day
        l_for = 0*day
        utc_0 = True
        # import event dates
        # get eruptions
        if auto_picked:
            path_ap = 'C:\\Users\\aar135\\codes_local_disk\\volc_forecast_tl\\volc_forecast_tl\\features\\lake_data\\corr_dsar_ratevar\\'
            path = path_ap
        dates =[]
        if auto_picked:
            path_dates_ap = path_ap+sta+'_dates_max_CC_missed_events_from_dsar_median_rv_without_filt_peak.txt' # dsar and rsam peak
            #
            just_heating = False
            just_cooling = False
            #
            if just_heating:
                path_dates_ap = path_ap+'FWVZ_heating_cycle_date_events.txt'
            if just_cooling:
                path_dates_ap = path_ap+'FWVZ_cooling_cycle_date_events.txt'
        #
        if auto_picked:
            with open(path_dates_ap,'r') as fp:
                #for ln in fp.readlines():
                #    a = ln.rstrip().split(' ')[0]
                dates = [datetimeify(ln.rstrip().split(' ')[0]) for ln in fp.readlines()]
        #
        dates = [dat for dat in dates if dat > _from]
        
        ## import temp data
        #ti_e1 = t0
        #tf_e1 = t1
        # import temp data
        if sta == 'FWVZ':
            path1 = '..'+os.sep+'data'+os.sep+"RU001_temp_data.csv"
        pd_temp = pd.read_csv(path1, index_col=1)
        if utc_0:
            pd_temp.index = [datetimeify(pd_temp.index[i])-6*hour for i in range(len(pd_temp.index))]
        else:
            pd_temp.index = [datetimeify(pd_temp.index[i]) for i in range(len(pd_temp.index))]

        ## import level data
        if sta == 'FWVZ':
            path2 = '..'+os.sep+'data'+os.sep+"RU001A_level_data.csv"
            path2 = '..'+os.sep+'data'+os.sep+"RU001A_level_data_full.csv"
            #path2 = '..'+os.sep+'data'+os.sep+"RU001_level_data.csv"
        pd_lev = pd.read_csv(path2, index_col=1)
        #
        #
        if utc_0:
            pd_lev.index = [datetimeify(pd_lev.index[i])-6*hour for i in range(len(pd_lev.index))]
        else:
            pd_lev.index = [datetimeify(pd_lev.index[i]) for i in range(len(pd_lev.index))]
        #
        if True: # rolling median over data
            N = 2
            pd_lev = pd_lev[:].rolling(40).median()#[N*24*6:]

        # lists to save
        date_event = [] 
        temp_dif_l = []
        level_dif_l = []

        # loop over dates 
        for k in range(ite):
            for i, dat in enumerate(dates):
                dat_ori = dat
                if jitter:
                    #dat = dat - 7*day + random.randint(0, 14)*day
                    dat = dat - 4*day + random.randint(0, 7)*day
                    #dat = dat - 1*day + random.randint(0, 1)*day
                # (1) explor back and for for min and max
                test = pd_temp[dat-l_back : dat+ l_for]
                if sta == 'POS':
                    pass
                    test[" t (C)"] = test[" t (C)"].rolling(window=10).mean()
                if roll_mean:
                    if sta == 'FWVZ':
                        pass
                        test[" t (C)"] = test[" t (C)"].rolling(window=10).mean()
                #
                if dat_ori == datetimeify('2009-07-13 06:30:00'):
                    a = 1
                    #pass
                    # plt.plot(test.index, test[" t (C)"].values)
                    # test[" t (C)"] = test[" t (C)"].rolling(window=15).mean()
                    # plt.plot(test.index, test[" t (C)"].values)
                    # plt.show()
                    # asdf
                #print(test.head())
                #print(test.tail())
                #_max = max(test[" t (C)"])
                _max_idx = test[" t (C)"].idxmax(axis = 1)
                _max =  test[" t (C)"].max()
                #_min = min(test[" t (C)"])
                _min_idx = test[" t (C)"].idxmin(axis = 1)
                _min =  test[" t (C)"].min()
                #
                #temp_ev_l.append(test[" t (C)"].mean())
                #temp_ev_l.append(test[" t (C)"].values[-1])
                #
                try:
                    dif_temp = (_max - _min) #abs(_max - _min) 
                    #
                    sign = (_max_idx - _min_idx).days / abs((_max_idx - _min_idx).days)
                    dif_temp = dif_temp / sign
                    #
                    rate2 = sign*dif_temp/(_max_idx - _min_idx).days
                    #

                    #level
                    # (1) explor back and for for min and max
                    test = pd_lev[dat-l_back : dat+ l_for]
                    # rolling median to avoid outliers
                    #test[" z (m)"] = test[" z (m)"].rolling(window=10).mean()
                    #print(test.head())
                    #print(test.tail())
                    #_max = max(test[" t (C)"])
                    _max_idx = test[" z (m)"].idxmax(axis = 1)
                    _max =  test[" z (m)"].max()
                    #_min = min(test[" z (m)"])
                    _min_idx = test[" z (m)"].idxmin(axis = 1)
                    _min =  test[" z (m)"].min()
                    #
                    #try:
                    dif_lev = (_max - _min) #abs(_max - _min) 
                    #
                    sign = (_max_idx - _min_idx).days / abs((_max_idx - _min_idx).days)
                    dif_lev = dif_lev / sign
                    #
                    rate2 = dif_lev/(_max_idx - _min_idx).days
                    #
                    #if not math.isinf(rate1) or not math.isinf(rate2):
                    temp_dif_l.append(dif_temp)
                    level_dif_l.append(dif_lev)
                    date_event.append(dat)
                except:
                    pass
                #
        # write output
        #path = '..'+os.sep+'features'+os.sep+'correlations'+os.sep+'stat_sign_per_hig_corr_feat'+os.sep
        with open(path+sta+'_date_temp_lev_difs_ite'+str(ite)+'.txt', 'w') as f:
            for k in range(len(date_event)):
                f.write(str(date_event[k])+','+str(round(temp_dif_l[k],2))+','+str(round(level_dif_l[k],2))+'\n')
        #
        #plt.scatter([abs(t) for t in temp_dif_l], [abs(l) for l in level_dif_l], c ="blue")
        #plt.figure()
        #ax = plt.axes([0, 0, 1, 1])
        plt.scatter(temp_dif_l, level_dif_l, c ="blue")
        # RIGHT VERTICAL
        plt.axvline(x=0., linewidth=1, color='k')
        plt.axhline(y=0., linewidth=1, color='k')

        # To show the plot
        plt.show()
        ##

def comb_stat_test_temp_dif_rate():
    '''
    Calculate dT and rates on events of multiple volcanoes, 
    and use d_days to explore dT and rates over the whole records.
    Plot histograms of dT and rates comparing observations in events and
    the rest of the record. 
    '''
    # (0) generate pool of eruptive events
    stas = ['POS','FWVZ']#'POS'
    n_evt_sta = 2 # number of events per station per sample
    n_sample = 40
    # extras
    l_back = 20*day
    l_for = 0*day
    utc_0 =True
    # (1) create the sample [[sta, ref_date],...
    samp_in = []
    for sta in stas:
        # get eruptions
        path = 'C:\\Users\\aar135\\codes_local_disk\\volc_forecast_tl\\volc_forecast_tl\\features\\lake_data\\'+sta+'\\selection\\'
        dates =[]
        with open(path+sta+'_temp_eruptive_periods.txt','r') as fp:
            dates = [datetimeify(ln.rstrip()) for ln in fp.readlines()]
        #
        for n in range(n_sample):
            _samp_in = random.sample(dates, int(n_evt_sta/2))
            for s in _samp_in:
                samp_in.append([sta,s])
    
    # (2) For each sample: calculate dT, rate, days; and 
    _samp_in = []
    for i, s in enumerate(samp_in):
        if s[0] == 'FWVZ':
            path2 = '..'+os.sep+'data'+os.sep+"RU001_temp_data.csv"
            #s_date = datetimeify('2009 05 01 00 00 00')
            #f_date = datetimeify('2021 12 31 00 00 00')
        if s[0] == 'POS':
            path2 = '..'+os.sep+'data'+os.sep+"POS_temp_data.csv"
            #s_date = datetimeify('2012 12 26 00 00 00')
            #f_date = datetimeify('2013 06 13 00 00 00')
        pd_temp = pd.read_csv(path2, index_col=1)
        if utc_0:
            pd_temp.index = [datetimeify(pd_temp.index[i])-6*hour for i in range(len(pd_temp.index))]
        else:
            pd_temp.index = [datetimeify(pd_temp.index[i]) for i in range(len(pd_temp.index))]
        # rolling median
        pd_temp[" t (C)"] = pd_temp[" t (C)"].rolling(window=10).mean()
        # normalized
        pd_temp[" t (C)"] =  (pd_temp[" t (C)"] - pd_temp[" t (C)"].mean())/pd_temp[" t (C)"].std()
    
        #
        dat = s[1]
        jitter = True
        if jitter:
            dat = dat - 0*day + random.randint(0, 10)*day
        test = pd_temp[dat-l_back : dat+ l_for]
        #
        _max_idx = test[" t (C)"].idxmax(axis = 1)
        _max =  test[" t (C)"].max()
        #_min = min(test[" t (C)"])
        _min_idx = test[" t (C)"].idxmin(axis = 1)
        _min =  test[" t (C)"].min()
        #
        #try:
        dif = abs(_max - _min)
        rate = dif/abs((_max_idx - _min_idx).days)
        dd = abs((_max_idx - _min_idx).days) #day diference
        if not np.isnan(rate):
            s.append([dif, rate, dd])
            _samp_in.append(s)
        
    # (3) Using d_days: calculate dT and rates over the wholes records
    dif_list = []
    rate_list = []
    days_list = []
    for s in _samp_in:
        for sta in stas: # loop over records 
            #
            if s[0] == 'FWVZ':
                path2 = '..'+os.sep+'data'+os.sep+"RU001_temp_data.csv"
                s_date = datetimeify('2009 05 01 00 00 00')
                f_date = datetimeify('2021 12 31 00 00 00')
            if s[0] == 'POS':
                path2 = '..'+os.sep+'data'+os.sep+"POS_temp_data.csv"
                s_date = datetimeify('2012 12 26 00 00 00')
                f_date = datetimeify('2013 06 13 00 00 00')   
            #
            pd_temp = pd_temp[s_date: f_date]
            #delta = f_date - s_date
            #
            #try:
            # sample a day diference 
            #_idx = random.randint(0, len(rate_days_l))
            d_day = s[2][2] # 
            # resample pd
            _ = str(int(d_day))+'D'
            _test =  pd_temp.resample(_).ffill()
            _difs = _test[" t (C)"].diff()[1:]
            _rates = _difs/int(d_day)
            #
            for k in range(len(_difs.values)):
                if not np.isnan(_rates.values[k]):
                    dif_list.append(abs(_difs.values[k]))
                    rate_list.append(abs(_rates.values[k]))
                    days_list.append(abs(d_day))
            #except:
            #    pass
    
    # (4) 
    fig, (ax1, ax2) = plt.subplots(nrows = 1, ncols = 2, figsize = (8,4))
    colors = ['b', 'r', 'g', 'm']
    
    # select lists 
    dif_l1 = [_samp_in[i][2][0] for i in range(len(_samp_in))]
    dif_l2 = dif_list

    multi = [dif_l1, dif_l2]
    colors = ['r', 'b']
    labels = ['in eruption', 'out eruption']
    bins = 20#np.linspace(0, 1, 13)
    ax1.hist(multi, bins, color = colors, label=labels, density = True)
    xlim = [0, 7] #12.5]

    ax1.set_xlabel('d_temp [°C]')
    ax1.set_ylabel('pdf')
    #ax1.set_xlim(xlim)
    #ax.set_xscale('log')

    ##
    rate_l1 = [_samp_in[i][2][1] for i in range(len(_samp_in))]
    rate_l2 = rate_list
    multi = [rate_l1, rate_l2]
    colors = ['r', 'b']
    labels = ['in eruption', 'out eruption']
    bins = 20#np.linspace(0, 1, 13)
    ax2.hist(multi, bins, color = colors, label=labels, density = True)
    xlim = None #[0, 4.2]#5] 8]

    ax2.set_xlabel('rate [°C/days]')
    #ax2.set_ylabel('pdf')
    ax2.set_xlim(xlim)
    #ax.set_xscale('log')
    
    plt.legend(loc='upper right')
    plt.show()

def pval_comb_stat_temp_dif_rate():
    '''
    Calculate dT and rates for samples taken from volcanic events in multiple volcanoes, 
    and randombly selected from the non-eruptive record. 
    Plot histograms of dT and rates bor both samples in and out eruptions. 
    Plot p-value distributions for dT and rates bor both samples in and out eruptions. 
    '''
    # (0) Pool of samples in and out
    stas = ['FWVZ','POS']#'POS'
    n_evt_sta = 2 # number of events per station per sample
    n_sample = 40
    # extras
    l_back = 20*day
    l_for = 0*day
    utc_0 =True

    # (1) in eruption
    samp_in = []
    for sta in stas:
        # get eruptions
        path = 'C:\\Users\\aar135\\codes_local_disk\\volc_forecast_tl\\volc_forecast_tl\\features\\lake_data\\'+sta+'\\selection\\'
        dates =[]
        with open(path+sta+'_temp_eruptive_periods.txt','r') as fp:
            dates = [datetimeify(ln.rstrip()) for ln in fp.readlines()]
        #
        for n in range(int(n_sample/len(stas))):
            _samp_in = random.sample(dates, int(n_evt_sta))
            for s in _samp_in:
                samp_in.append([sta,s])

    # (2) out eruption
    samp_out = []
    for sta in stas:
        # get eruptions
        path = 'C:\\Users\\aar135\\codes_local_disk\\volc_forecast_tl\\volc_forecast_tl\\features\\lake_data\\'+sta+'\\selection\\'
        dates =[]
        with open(path+sta+'_temp_eruptive_periods.txt','r') as fp:
            dates = [datetimeify(ln.rstrip()) for ln in fp.readlines()]
        #
        if sta == 'FWVZ':
            s_date = datetimeify('2009 05 01 00 00 00')
            f_date = datetimeify('2021 12 31 00 00 00')
        if sta == 'POS':
            s_date = datetimeify('2012 12 26 00 00 00')
            f_date = datetimeify('2013 06 13 00 00 00')   
        #
        # days to avoid (near eruptions)
        d_avoid = []
        for d in dates:
            d_avoid.append([d - 20*day + i*day for i in range(30)])
        d_avoid = [d_avoid[k]+d_avoid[k+1] for k in range(len(d_avoid)-1)]
        # non eruptive day list
        l_days = [s_date+i*day for i in range((f_date - s_date).days) if s_date+i*day not in d_avoid]
        # sample
        _samp_out = random.sample(l_days, int(n_evt_sta/len(stas))*int(n_sample))
        for s in _samp_out:
            samp_out.append([sta,s])
    #
    # 
    def sample_dtemp_rate_sta(stas, s):
        '''
        received a sample and calculate dtemp and rate on eruptions of the station and 
        over the whole record of the station
        Input
        s = [station, date] 
        Returns
        l_dif_erup, l_rate_erup, l_dif_non_erup, l_rec_rates
        '''
        if s[0] == 'FWVZ':
            path2 = '..'+os.sep+'data'+os.sep+"RU001_temp_data.csv"
            s_date = datetimeify('2009 05 01 00 00 00')
            f_date = datetimeify('2021 12 31 00 00 00')
        if s[0] == 'POS':
            path2 = '..'+os.sep+'data'+os.sep+"POS_temp_data.csv"
            s_date = datetimeify('2012 12 26 00 00 00')
            f_date = datetimeify('2013 06 13 00 00 00') 
        pd_temp = pd.read_csv(path2, index_col=1)
        if utc_0:
            pd_temp.index = [datetimeify(pd_temp.index[i])-6*hour for i in range(len(pd_temp.index))]
        else:
            pd_temp.index = [datetimeify(pd_temp.index[i]) for i in range(len(pd_temp.index))]
        # rolling median
        pd_temp[" t (C)"] = pd_temp[" t (C)"].rolling(window=10).mean()
        # normalized
        pd_temp[" t (C)"] =  (pd_temp[" t (C)"] - pd_temp[" t (C)"].mean())/pd_temp[" t (C)"].std()
        #
        dat = s[1]
        jitter = True
        if jitter:
            dat = dat - 0*day + random.randint(0, 10)*day
        test = pd_temp[dat-l_back : dat+ l_for]
        #
        _max_idx = test[" t (C)"].idxmax(axis = 1)
        _max =  test[" t (C)"].max()
        #_min = min(test[" t (C)"])
        _min_idx = test[" t (C)"].idxmin(axis = 1)
        _min =  test[" t (C)"].min()
        #
        #_dif = abs(_max - _min)
        #_rate = dif/abs((_max_idx - _min_idx).days)
        _dd_sample = abs((_max_idx - _min_idx).days) #day diference
        #
        #l_erup = [dif, rate, dd]
        
        # loop over several records
        l_dif_erup = []
        l_rate_erup = []
        l_dif_non_erup = []
        l_rate_non_erup = []

        for sta in stas: 
            # get eruptions
            path = 'C:\\Users\\aar135\\codes_local_disk\\volc_forecast_tl\\volc_forecast_tl\\features\\lake_data\\'+sta+'\\selection\\'
            dates =[]
            with open(path+sta+'_temp_eruptive_periods.txt','r') as fp:
                dates = [datetimeify(ln.rstrip()) for ln in fp.readlines()]
            #
            #
            if sta == 'FWVZ':
                path2 = '..'+os.sep+'data'+os.sep+"RU001_temp_data.csv"
                s_date = datetimeify('2009 05 01 00 00 00')
                f_date = datetimeify('2021 12 31 00 00 00')
            if sta == 'POS':
                path2 = '..'+os.sep+'data'+os.sep+"POS_temp_data.csv"
                s_date = datetimeify('2012 12 26 00 00 00')
                f_date = datetimeify('2013 06 13 00 00 00') 
            #
            pd_temp = pd.read_csv(path2, index_col=1)
            if utc_0:
                pd_temp.index = [datetimeify(pd_temp.index[i])-6*hour for i in range(len(pd_temp.index))]
            else:
                pd_temp.index = [datetimeify(pd_temp.index[i]) for i in range(len(pd_temp.index))]
            # rolling median
            pd_temp[" t (C)"] = pd_temp[" t (C)"].rolling(window=10).mean()
            # normalized
            pd_temp[" t (C)"] =  (pd_temp[" t (C)"] - pd_temp[" t (C)"].mean())/pd_temp[" t (C)"].std()
            pd_temp = pd_temp[s_date: f_date]
            ##
            for dat in dates:
                #
                di = dat- _dd_sample*day #+ _dd_sample
                df = dat
                test = pd_temp[di : df]
                _dif = abs(test[" t (C)"].max()-test[" t (C)"].min())
                _rates = _dif/int(_dd_sample)
                # 
                l_dif_erup.append(_dif)
                l_rate_erup.append(_rates)

            # over the record
            d_day = _dd_sample#s[2][2] # 
            # resample pd
            _ = str(int(d_day))+'D'
            _test =  pd_temp.resample(_).ffill()
            _difs = _test[" t (C)"].diff()[1:].values
            _rates = _difs/int(d_day)
            #
            _difs = [_difs[k] for k in range(len(_difs))]
            _rates = [_rates[k] for k in range(len(_rates))]
            #
            [l_dif_non_erup.append(_difs[k]) for k in range(len(_difs))]
            [l_rate_non_erup.append(_rates[k]) for k in range(len(_rates))]
        #
        ##
        return l_dif_erup, l_rate_erup, l_dif_non_erup, l_rate_non_erup
    
    # loop over sample (in and out)
    # p values
    pv_in_dtemp = []
    pv_in_rate = []
    pv_out_dtemp = []
    pv_out_rate = []
    # dTs
    dT_in_dtemp = []
    dT_in_dtemp_nonerup = []
    dT_in_rate = []
    dT_in_rate_nonerup = []
    dT_out_dtemp = []
    dT_out_rate = []
    #
    print('samp in\n')
    count = 0
    for s in samp_in:
        try:
            print(str(count+1)+'/'+str(len(samp_in)))
            count = count + 1
            #
            _l_dif_erup, _l_rate_erup, _l_dif_non_erup, _l_rate_non_erup = sample_dtemp_rate_sta(stas, s)
            #
            l_rate_erup = []
            l_dif_erup = []
            l_rate_non_erup = []
            l_dif_non_erup = []
            for i in range(len(_l_rate_erup)):
                if not np.isnan(_l_rate_erup[i]):
                    l_rate_erup.append(_l_rate_erup[i])
                    l_dif_erup.append(_l_dif_erup[i])
            for i in range(len(_l_rate_non_erup)):
                if not np.isnan(_l_rate_non_erup[i]):
                    l_rate_non_erup.append(_l_rate_non_erup[i])
                    l_dif_non_erup.append(_l_dif_non_erup[i])
            # calculate p-value 
            _pv_in_dtemp = kstest(_l_dif_erup, _l_dif_non_erup).pvalue
            _pv_in_rate = kstest(l_rate_erup, l_rate_non_erup).pvalue
            #
            pv_in_dtemp.append(_pv_in_dtemp) 
            pv_in_rate.append(_pv_in_rate) 
            #
            dT_in_dtemp = dT_in_dtemp + l_dif_erup 
            dT_in_dtemp_nonerup = dT_in_dtemp_nonerup + l_dif_non_erup
            # 
            dT_in_rate = dT_in_rate + l_rate_erup 
            dT_in_rate_nonerup = dT_in_rate_nonerup + l_rate_non_erup 
        except:
            pass
    #
    print('samp out\n')
    count = 0
    for s in samp_out:
        try:
            print(str(count+1)+'/'+str(len(samp_out)))
            count = count + 1
            _l_dif_erup, _l_rate_erup, _l_dif_non_erup, _l_rate_non_erup = sample_dtemp_rate_sta(stas, s)
            #
            l_rate_erup = []
            l_dif_erup = []
            l_rate_non_erup = []
            l_dif_non_erup = []
            for i in range(len(_l_rate_erup)):
                if not np.isnan(_l_rate_erup[i]):
                    l_rate_erup.append(_l_rate_erup[i])
                    l_dif_erup.append(_l_dif_erup[i])
            for i in range(len(_l_rate_non_erup)):
                if not np.isnan(_l_rate_non_erup[i]):
                    l_rate_non_erup.append(_l_rate_non_erup[i])
                    l_dif_non_erup.append(_l_dif_non_erup[i])
            # calculate p-value 
            _pv_out_dtemp = kstest(_l_dif_erup, _l_dif_non_erup).pvalue
            _pv_out_rate = kstest(l_rate_erup, l_rate_non_erup).pvalue
            #
            pv_out_dtemp.append(_pv_out_dtemp) 
            pv_out_rate.append(_pv_out_rate) 
        except:
            pass
    #
    if True: # plot dTs and rates
        ## (5) construct p-val histogram of both populations
        fig, (ax1, ax2) = plt.subplots(nrows = 1, ncols = 2, figsize = (8,4))
        colors = ['b', 'r', 'g', 'm']

        #_heights, a_bins = np.histogram(pv_samp_in_1)
        
        # select lists 
        #ax.axvline(x=0.05, color = 'k', ls='--', linewidth=1, label = '0.05 threshold')
        multi = [dT_in_dtemp, dT_in_dtemp_nonerup]
        colors = ['r', 'b']
        labels = ['in eruption', 'out eruption']
        bins = 20#np.linspace(0, 1, 13)
        #bins = np.linspace(0, 1, 20)
        ax1.hist(multi, bins, color = colors, label=labels, density = True)
        xlim = [0, 7] #12.5]

        ax1.set_title(' ')
        ax1.set_xlabel('d_temp [°C]')
        ax1.set_ylabel('pdf')
        #ax1.set_xlim(xlim)
        #ax1.set_xscale('log')

        ##
        multi = [dT_in_rate, dT_in_rate_nonerup]
        colors = ['r', 'b']
        labels = ['in eruption', 'out eruption']
        bins = 20#np.linspace(0, 1, 13)
        bins = np.linspace(0, 1, 20)
        ax2.hist(multi, bins, color = colors, label=labels, density = True)
        xlim = None #[0, 4.2]#5] 8]

        ax2.set_title(' ')
        ax2.set_xlabel('rate [°C/days]')
        ax2.set_xlim(xlim)
        #ax2.set_xscale('log')
        
        plt.legend(loc='upper right')
        plt.show()

    if True: # plot p-values
        ## (5) construct p-val histogram of both populations
        fig, (ax1, ax2) = plt.subplots(nrows = 1, ncols = 2, figsize = (8,4))
        colors = ['b', 'r', 'g', 'm']

        #_heights, a_bins = np.histogram(pv_samp_in_1)
        
        # select lists 
        #ax.axvline(x=0.05, color = 'k', ls='--', linewidth=1, label = '0.05 threshold')
        multi = [pv_in_dtemp, pv_out_dtemp]
        colors = ['r', 'b']
        labels = ['in eruption', 'out eruption']
        bins = 20#np.linspace(0, 1, 13)
        bins = np.linspace(0, 1, 20)
        ax1.hist(multi, bins, color = colors, label=labels, density = False)
        xlim = [0, 7] #12.5]

        ax1.set_title('d_temp [°C]')
        ax1.set_xlabel('p-value')
        ax1.set_ylabel('pdf')
        #ax1.set_xlim(xlim)
        #ax1.set_xscale('log')

        ##
        multi = [pv_in_rate, pv_out_rate]
        colors = ['r', 'b']
        labels = ['in eruption', 'out eruption']
        bins = 20#np.linspace(0, 1, 13)
        bins = np.linspace(0, 1, 20)
        ax2.hist(multi, bins, color = colors, label=labels, density = False)
        xlim = None #[0, 4.2]#5] 8]

        ax2.set_title('rate [°C/days]')
        ax2.set_xlabel('p-value')
        ax2.set_xlim(xlim)
        #ax2.set_xscale('log')
        
        plt.legend(loc='upper right')
        plt.show()

def scat_plot_cc_dT():
    '''
    '''
    # import dates and CCs
    path = 'C:\\Users\\aar135\\codes_local_disk\\volc_forecast_tl\\volc_forecast_tl\\features\\lake_data\\corr_dsar_ratevar\\'
    with open(path+'FWVZ_dates_CC_missed_events_from_dsar_median_rv.txt','r') as fp:
        date_cc = []
        cc = []
        dic_cc = {}
        for ln in fp.readlines():
            _ = ln.split(',')
            date_cc.append(datetimeify(_[0]))
            cc.append(_[1][:-1])
            dic_cc[datetimeify(_[0])] = _[1][:-1]

    # import dT and dates
    path = 'C:\\Users\\aar135\\codes_local_disk\\volc_forecast_tl\\volc_forecast_tl\\features\\lake_data\\corr_dsar_ratevar\\'
    with open(path+'FWVZ_date_dT_ite100.txt','r') as fp:
        date_dT = []
        dT = []
        for ln in fp.readlines():
            _ = ln.split(',')
            date_dT.append(datetimeify(_[0]))
            dT.append(_[1][:-1])
    # build pairs between cc and dT (through dates)
    _cc = []
    _dT = [] 
    for i, _date in enumerate(date_dT):  
        _cc.append(float(dic_cc[_date]))
        _dT.append(float(dT[i]))
    # 
    plt.plot(_cc, _dT, 'b*')
    plt.plot([-1,1], [0,0], 'k--')
    plt.xlabel("CC")
    plt.ylabel("dT")
    #plt.legend(loc='upper left')
    plt.show()
    
# lake level correlation (data from 2016 to 2022)
def level_erup_ana():
    '''
    Calculate level diference and rate before eruptive events read from file. 
    Its a random proccess, where reference dates are 'perturb' (jitter) during several iterations. 
    '''
    # 
    #auto_picked = True # dates elected automaticaly from dsar median and dsar rv correlations (only with FWVZ)
    stas = ['FWVZ']#,'POS'] #'POS','FWVZ'
    for sta in stas:
        jitter = True
        if jitter:
            ite = 100
        else:
            ite = 1
        #
        l_back = 0*day
        l_for = 3*day
        utc_0 = True
        _from = 2009
        # import event dates
        if man_picked:
            path_mp = 'C:\\Users\\aar135\\codes_local_disk\\volc_forecast_tl\\volc_forecast_tl\\features\\lake_data\\'+sta+'\\selection\\'
            path = path_mp
        if auto_picked:
            path_ap = 'C:\\Users\\aar135\\codes_local_disk\\volc_forecast_tl\\volc_forecast_tl\\features\\lake_data\\corr_dsar_ratevar\\'
            path = path_ap
        if man_picked and auto_picked:
            path_ap_mp = 'C:\\Users\\aar135\\codes_local_disk\\volc_forecast_tl\\volc_forecast_tl\\features\\lake_data\\corr_dsar_ratevar\\'
            path = path_ap_mp
        dates =[]
        if man_picked:
            path_dates_mp = path_mp+sta+'_temp_eruptive_periods.txt'
        if auto_picked:
            path_dates_ap = path_ap+'FWVZ_dates_missed_events_from_dsar_median_rv_cc.txt'
        #
        if auto_picked:
            with open(path_dates_ap,'r') as fp:
                _dates = [datetimeify(ln.rstrip()) for ln in fp.readlines()]
                dates = [dat for dat in _dates if dat.year > _from]
        if man_picked:
            with open(path_dates_mp,'r') as fp:
                dates = [datetimeify(ln.rstrip()) for ln in fp.readlines()]
                dates = [dat for dat in dates]# if dat.year > 2016]
        if auto_picked and man_picked:
            with open(path_dates_ap,'r') as fp:
                _dates1 = [datetimeify(ln.rstrip()) for ln in fp.readlines()]
                dates1 = [dat for dat in _dates1 if dat.year >= 2016]
            with open(path_dates_mp,'r') as fp:
                _dates2 = [datetimeify(ln.rstrip()) for ln in fp.readlines()]
                dates2 = [dat for dat in _dates2 if dat.year >= 2016]
            dates = dates1 + dates2
        
        # import temp data
        # import temp data
        #ti_e1 = t0
        #tf_e1 = t1
        # import temp data
        if sta == 'FWVZ':
            path2 = '..'+os.sep+'data'+os.sep+"RU001A_level_data.csv"
            path2 = '..'+os.sep+'data'+os.sep+"RU001A_level_data_full.csv"
            #path2 = '..'+os.sep+'data'+os.sep+"RU001_level_data.csv"
        pd_lev = pd.read_csv(path2, index_col=1)
        #
        #
        if utc_0:
            pd_lev.index = [datetimeify(pd_lev.index[i])-6*hour for i in range(len(pd_lev.index))]
        else:
            pd_lev.index = [datetimeify(pd_lev.index[i]) for i in range(len(pd_lev.index))]
        #
        if True: # rolling median over data
            N = 2
            pd_lev = pd_lev[:].rolling(40).median()#[N*24*6:]
        # lists to save
        dif_l = []
        rate_l = []
        rate_days_l = []
        # loop over dates 
        for k in range(ite):
            for i, dat in enumerate(dates):
                try:
                    if jitter:
                        #dat = dat - 7*day + random.randint(0, 14)*day
                        dat = dat - 4*day + random.randint(0, 7)*day
                    # (1) explor back and for for min and max
                    test = pd_lev[dat-l_back : dat+ l_for]
                    # rolling median to avoid outliers
                    #test[" z (m)"] = test[" z (m)"].rolling(window=10).mean()
                    #print(test.head())
                    #print(test.tail())
                    #_max = max(test[" t (C)"])
                    _max_idx = test[" z (m)"].idxmax(axis = 1)
                    _max =  test[" z (m)"].max()
                    #_min = min(test[" z (m)"])
                    _min_idx = test[" z (m)"].idxmin(axis = 1)
                    _min =  test[" z (m)"].min()
                    #
                    #try:
                    dif = (_max - _min) #abs(_max - _min) 
                    #
                    sign = (_max_idx - _min_idx).days / abs((_max_idx - _min_idx).days)
                    dif = dif / sign
                    #
                    rate = dif/(_max_idx - _min_idx).days
                    #
                    if not math.isinf(rate):
                        dif_l.append(dif)
                        rate_l.append(rate)
                        rate_days_l.append((_max_idx - _min_idx).days)
                except:
                    pass
                #
        # write output
        #path = '..'+os.sep+'features'+os.sep+'correlations'+os.sep+'stat_sign_per_hig_corr_feat'+os.sep
        with open(path+sta+'_level_dif_rate_ite'+str(ite)+'.txt', 'w') as f:
            for k in range(len(dif_l)):
                f.write(str(round(dif_l[k],2))+'\t'+str(round(rate_l[k],2))+'\t'+str(round(rate_days_l[k],2))+'\n')

def level_dif_rate_stats():
    '''
    explore for tamperature diference and rates over the whole record
    '''
    #auto_picked = True # dates elected automaticaly from dsar median and dsar rv correlations (only with FWVZ)
    stas = ['FWVZ']
    #stas = ['FWVZ','POS']#'FWVZ'
    for sta in stas:
        utc_0 = True
        pass
        # import temp record
        # import temp data
        if sta == 'FWVZ':
            path2 = '..'+os.sep+'data'+os.sep+"RU001A_level_data.csv"

        #path2 = 'C:\\Users\\aar135\\codes_local_disk\\volc_forecast_tl\\volc_forecast_tl\\features\\lake_data\\FWVZ\\selection\\'
        pd_lev = pd.read_csv(path2, index_col=1)
        if utc_0:
            pd_lev.index = [datetimeify(pd_lev.index[i])-6*hour for i in range(len(pd_lev.index))]
        else:
            pd_lev.index = [datetimeify(pd_lev.index[i]) for i in range(len(pd_lev.index))]
        
        # import dif and rate
        path = 'C:\\Users\\aar135\\codes_local_disk\\volc_forecast_tl\\volc_forecast_tl\\features\\lake_data\\'+sta+'\\selection\\'
        if auto_picked:
            path = 'C:\\Users\\aar135\\codes_local_disk\\volc_forecast_tl\\volc_forecast_tl\\features\\lake_data\\corr_dsar_ratevar\\'
        _fls = glob.glob(path+sta+"_level_dif_rate_ite100.txt")
        dif_l = []
        rate_l = []
        rate_days_l = []
        for _fl in _fls:
            fl  = np.genfromtxt(_fl, delimiter="\t")
            [dif_l.append(fl[i][0]) for i in range(len(fl))]
            [rate_l.append(fl[i][1]) for i in range(len(fl))]
            [rate_days_l.append(fl[i][2]) for i in range(len(fl))]
        
        # loop over rate
        ite = 100
        if sta == 'FWVZ':
            s_date = datetimeify('2016 03 05 00 00 00')
            f_date = datetimeify('2022 01 31 00 00 00')
        #
        pd_lev = pd_lev[s_date: f_date]
        delta = f_date - s_date
        #
        dif_list = []
        rate_list = []
        days_list = []

        # remove rates from eruptive periods
        remv_erup_dates = True
        if remv_erup_dates: # locate_missed_events_seismic() need to be run too
            # get eruptions
            path = 'C:\\Users\\aar135\\codes_local_disk\\volc_forecast_tl\\volc_forecast_tl\\features\\lake_data\\'+sta+'\\selection\\'
            if auto_picked:
                path = 'C:\\Users\\aar135\\codes_local_disk\\volc_forecast_tl\\volc_forecast_tl\\features\\lake_data\\corr_dsar_ratevar\\'
            dates =[]
            path_dates = path+sta+'_temp_eruptive_periods.txt'
            if auto_picked:
                path_dates = path+'FWVZ_dates_missed_events_from_dsar_median_rv_cc.txt'
            #
            with open(path_dates,'r') as fp:
                dates = [datetimeify(ln.rstrip()) for ln in fp.readlines()]
            l_dates_rej = []
            for date in dates:
                [l_dates_rej.append(date-30*day+ i*day) for i in range(60)]
            #
        for i in range(ite):
            try:
                # sample a day diference 
                _idx = random.randint(0, len(rate_days_l))
                d_day = rate_days_l[_idx]
                # resample pd
                _ = str(int(d_day))+'D'
                _test =  pd_lev.resample(_).ffill()
                _difs = _test[" z (m)"].diff()[1:]
                _rates = _difs/int(d_day)

                for k in range(len(_difs.values)):
                    if not np.isnan(_rates.values[k]):
                        if remv_erup_dates:
                            if _rates.index[i] not in l_dates_rej:
                                dif_list.append(_difs.values[k])
                                rate_list.append(_rates.values[k])
                                days_list.append(abs(d_day))
                        else:
                            dif_list.append(_difs.values[k])
                            rate_list.append(_rates.values[k])
                            days_list.append(abs(d_day))
                # [dif_list.append(abs(_difs.values[k])) for k in range(len(_difs.values))]
                # [rate_list.append(abs(_rates.values[k])) for k in range(len(_rates.values))]
                # [days_list.append(abs(d_day)) for k in range(len(_rates.values))]
            except:
                pass
        # write output
        #path = '..'+os.sep+'features'+os.sep+'correlations'+os.sep+'stat_sign_per_hig_corr_feat'+os.sep
        with open(path+sta+'_level_dif_rate_ite'+str(ite)+'_out.txt', 'w') as f:
            for k in range(len(dif_list)):
                f.write(str(round(dif_list[k],2))+'\t'+str(round(rate_list[k],2))+'\t'+str(round(days_list[k],2))+'\n')

def plot_level_erup_ana():
    '''
    '''
    if False: # plot one 
        # read results 
        path = 'C:\\Users\\aar135\\codes_local_disk\\volc_forecast_tl\\volc_forecast_tl\\features\\lake_data\\FWVZ\\selection\\'

        _fls = glob.glob(path+"FWVZ_temp_dif_rate_ite100.txt")
        dif_l = []
        rate_l = []
        rate_days_l = []
        for _fl in _fls:
            fl  = np.genfromtxt(_fl, delimiter="\t")
            [dif_l.append(fl[i][0]) for i in range(len(fl))]
            [rate_l.append(fl[i][1]) for i in range(len(fl))]
            [rate_days_l.append(fl[i][2]) for i in range(len(fl))]

        # plot histograms 
        fig, (ax1, ax2, ax3) = plt.subplots(nrows = 1, ncols = 3, figsize = (12,4))
        xlim = None#[0,0.25]
        ylim = None#[0,70]
        colors = ['b', 'r', 'g', 'm']

        # dif
        n_bins = 20#int(np.sqrt(len(dif_l)))
        ax1.hist(dif_l, n_bins, histtype='bar', color = colors[1], edgecolor='k', label = 'dif_temp')
        ax1.set_xlabel('temperature [°C]', fontsize=textsize)
        ax1.set_ylabel('samples', fontsize=textsize)
        ax1.grid(True, which='both', linewidth=0.1)
        xlim = [0, 12.5]
        ax1.set_xlim(xlim)
        ax1.set_ylim(ylim)
        #ax1.set_title('RSAM', fontsize = textsize)
        #ax1.plot([np.median(dif_l), np.median(dif_l)],[0, count_rsam+ count_rsam*.1],'k--', label = 'median: '+str(round(np.median(dif_l),2)))
        ax1.legend(loc = 1)

        # days
        #n_bins = int(np.sqrt(len(rate_l))/3)
        ax2.hist(rate_days_l, n_bins, histtype='bar', color = colors[2], edgecolor='k', label = 'days')
        ax2.set_xlabel('days', fontsize=textsize)
        ax2.set_ylabel(' ', fontsize=textsize)
        ax2.grid(True, which='both', linewidth=0.1)
        xlim = [0, 10]
        ax2.set_xlim(xlim)
        ax2.set_ylim(ylim)
        #ax1.set_title('RSAM', fontsize = textsize)
        #ax2.plot([np.median(rate_l), np.median(rate_l)],[0, count_rsam+ count_rsam*.1],'k--', label = 'median: '+str(round(np.median(rate_l),2)))
        ax2.legend(loc = 1)

        # rate
        #n_bins = int(np.sqrt(len(rate_l))/3)
        ax3.hist(rate_l, n_bins, histtype='bar', color = colors[0], edgecolor='k', label = 'rate')
        ax3.set_xlabel('rate [°C/day]', fontsize=textsize)
        ax3.set_ylabel(' ', fontsize=textsize)
        ax3.grid(True, which='both', linewidth=0.1)
        xlim = [0, 8]
        ax3.set_xlim(xlim)
        ax3.set_ylim(ylim)
        #ax1.set_title('RSAM', fontsize = textsize)
        #ax2.plot([np.median(rate_l), np.median(rate_l)],[0, count_rsam+ count_rsam*.1],'k--', label = 'median: '+str(round(np.median(rate_l),2)))
        ax3.legend(loc = 1)

        plt.show()

    if True:
        #auto_picked = True # dates elected automaticaly from dsar median and dsar rv correlations (only with FWVZ)
        sta = 'FWVZ'#'FWVZ' 'POS'
        # read results 
        path1 = 'C:\\Users\\aar135\\codes_local_disk\\volc_forecast_tl\\volc_forecast_tl\\features\\lake_data\\'+sta+'\\selection\\'
        path2 = 'C:\\Users\\aar135\\codes_local_disk\\volc_forecast_tl\\volc_forecast_tl\\features\\lake_data\\'+sta+'\\selection\\'
        if auto_picked:
            path1 = 'C:\\Users\\aar135\\codes_local_disk\\volc_forecast_tl\\volc_forecast_tl\\features\\lake_data\\corr_dsar_ratevar\\'
            path2 = 'C:\\Users\\aar135\\codes_local_disk\\volc_forecast_tl\\volc_forecast_tl\\features\\lake_data\\corr_dsar_ratevar\\'

        path1 = path1 +sta+"_level_dif_rate_ite100.txt"
        path2 = path2 +sta+"_level_dif_rate_ite100_out.txt"
        # if sta == 'POS':#'FWVZ'
        #     path2 = path2 +sta+"_temp_dif_rate_ite10_out.txt"
        # if sta == 'FWVZ':#'FWVZ'            
        #     path2 = path2 +sta+"_temp_dif_rate_ite100_out.txt"
        #     #path2 = path2 +sta+"_temp_dif_rate_ite10_out.txt"
        #
        _fls = glob.glob(path1)
        dif_l1 = []
        rate_l1 = []
        rate_days_l1 = []
        for _fl in _fls:
            fl  = np.genfromtxt(_fl, delimiter="\t")
            [dif_l1.append(fl[i][0]) for i in range(len(fl))]
            [rate_l1.append(fl[i][1]) for i in range(len(fl))]
            [rate_days_l1.append(fl[i][2]) for i in range(len(fl))]

        _fls = glob.glob(path2)
        dif_l2 = []
        rate_l2 = []
        rate_days_l2 = []
        for _fl in _fls:
            fl  = np.genfromtxt(_fl, delimiter="\t")
            [dif_l2.append(fl[i][0]) for i in range(len(fl))]
            [rate_l2.append(fl[i][1]) for i in range(len(fl))]
            [rate_days_l2.append(fl[i][2]) for i in range(len(fl))]
        #replace rate > 5
        if True:
            for i, r in enumerate(rate_l1): 
                if r>.2:
                    rate_l1[i] = rate_l1[i]/4

        ## (5) construct p-val histogram of both populations
        plot_rate = True
        if plot_rate:
            fig, (ax1, ax2, ax3) = plt.subplots(nrows = 1, ncols = 3, figsize = (12,4))
        else:
            fig, (ax1, ax2) = plt.subplots(nrows = 1, ncols = 2, figsize = (8,4))

        colors = ['b', 'r', 'g', 'm']
        #_heights, a_bins = np.histogram(pv_samp_in_1)
        
        # select lists 
        #ax.axvline(x=0.05, color = 'k', ls='--', linewidth=1, label = '0.05 threshold')
        multi = [dif_l1, dif_l2]
        colors = ['r', 'b']
        labels = ['in eruption', 'out eruption']
        bins = 20#np.linspace(0, 1, 13)
        ax1.hist(multi, bins, color = colors, label=labels, density = True)
        xlim = None#[0, 7] #12.5]

        ax1.set_xlabel('d_level [m]')
        ax1.set_ylabel('pdf')
        ax1.set_title('Lake level difference')
        #ax1.set_xlim(xlim)
        #ax.set_xscale('log')

        ##
        rate_days_l1 = abs(np.asarray(rate_days_l1))
        multi = [rate_days_l1, []]#rate_days_l2]
        colors = ['r', 'b']
        labels = ['in eruption', 'out eruption']
        bins = 20#np.linspace(0, 1, 13)
        ax2.hist(multi, bins, color = colors, label=labels, density = True)
        xlim = None #[0, 4.2]#5] 8]
        ax2.set_xlabel('dt [days]')
        #ax2.set_ylabel('pdf')
        ax2.set_xlim(xlim)
        ax2.set_title('Days period')
        #ax.set_xscale('log')

        if plot_rate:
            multi = [rate_l1, rate_l2]
            colors = ['r', 'b']
            labels = ['in eruption', 'out eruption']
            bins = 20#np.linspace(0, 1, 13)
            ax3.hist(multi, bins, color = colors, label=labels, density = True)
            xlim = None #[0, 4.2]#5] 8]
            ax3.set_xlabel('[m/days]')
            #ax2.set_ylabel('pdf')
            ax3.set_xlim(xlim)
            ax3.set_title('Rate')
            #ax.set_xscale('log')
        
        plt.legend(loc='upper right')
        plt.show()
        asdf

# 
def download_acoustic():
    '''
    Download and plot pressure data from geonet
    '''
    #
    from obspy import UTCDateTime
    #
    from obspy.clients.fdsn import Client
    client = Client("GEONET")
    from obspy import UTCDateTime
    #t = UTCDateTime("2012-02-27T00:00:00.000")
    starttime = UTCDateTime("2009-07-12") 
    endtime = UTCDateTime("2009-07-14")
    inventory = client.get_stations(network="NZ", station="COVZ", starttime=starttime, endtime=endtime)
    #st = client.get_waveforms(network = "AV", station = "FWVZ", location = None, channel = "EHZ", starttime=starttime, endtime=endtime)
    st = client.get_waveforms(network = "NZ", station = "COVZ", location = 30, channel = None, starttime=starttime, endtime=endtime)
    st.plot()  
    asdf

# temperature cycles and events
def map_events_in_temp_cycles():
    '''
    '''
    sta = 'FWVZ'#'POS'#'FWVZ'#'COP'
    ## import events
    path = 'C:\\Users\\aar135\\codes_local_disk\\volc_forecast_tl\\volc_forecast_tl\\features\\lake_data\\'+sta+'\\selection\\'
    if auto_picked:
        path = 'C:\\Users\\aar135\\codes_local_disk\\volc_forecast_tl\\volc_forecast_tl\\features\\lake_data\\corr_dsar_ratevar\\'
    dates =[]
    path_dates = path+sta+'_dates_missed_events_from_dsar_median_rv_cc.txt'
    path_dates = path+sta+'_dates_max_CC_missed_events_from_dsar_median_rv.txt'

    path_dates_filt_off = path+sta+'_dates_max_CC_missed_events_from_dsar_median_rv_without_filt_peak.txt'
    path_dates_filt_on = path+sta+'_dates_max_CC_missed_events_from_dsar_median_rv_with_filt_peak.txt'
    #
    if False: # man picked
        path_mp = 'C:\\Users\\aar135\\codes_local_disk\\volc_forecast_tl\\volc_forecast_tl\\features\\lake_data\\'+sta+'\\selection\\'
        path_dates = path_mp+'FWVZ_temp_eruptive_periods.txt'
    #
    date_events = []
    cc_events = []
    max_events = []
    with open(path_dates_filt_on,'r') as fp:
        for ln in fp.readlines():
            _d, _cc, _mx =ln.rstrip().split(',')
            date_events.append(datetimeify(_d))
            cc_events.append(_cc)
            max_events.append(_mx)
        #date_events = [datetimeify(ln.rstrip()) for ln in fp.readlines()]
    #
    date_events_filt_off = []
    cc_events_filt_off = []
    max_events_filt_off = []
    with open(path_dates_filt_off,'r') as fp:
        for ln in fp.readlines():
            _d, _cc, _mx =ln.rstrip().split(',')
            date_events_filt_off.append(datetimeify(_d))
            cc_events_filt_off.append(_cc)
            max_events_filt_off.append(_mx)
        #date_events = [datetimeify(ln.rstrip()) for ln in fp.readlines()]

    ## import temperature cycle dates
    path = 'C:\\Users\\aar135\\codes_local_disk\\volc_forecast_tl\\volc_forecast_tl\\features\\lake_data\\corr_dsar_ratevar\\'+sta+'_temp_cycles_ini_mid_end.txt'
    cycle_date_ini = []
    cycle_date_mid = []
    cycle_date_fin = []
    cycle_temp_ini = []
    cycle_temp_mid = []
    cycle_temp_fin = []
    
    #
    with open(path,'r') as fp:
        #
        for ln in fp.readlines():
            #
            _cycle_date_ini, _cycle_date_mid, _cycle_date_fin, _cycle_temp_ini, _cycle_temp_mid, _cycle_temp_fin  = ln.rstrip().split(',')
            #
            cycle_date_ini.append(datetimeify(_cycle_date_ini))
            cycle_date_mid.append(datetimeify(_cycle_date_mid))
            cycle_date_fin.append(datetimeify(_cycle_date_fin))
            #
            cycle_temp_ini.append(float(_cycle_temp_ini))
            cycle_temp_mid.append(float(_cycle_temp_mid))
            cycle_temp_fin.append(float(_cycle_temp_fin))
            #
    #

    ## loop over events
    heat_l = []
    mid_l = []
    cool_l = []
    heat_cc_l = []
    heat_max_l = []
    mid_l = []
    cool_l = []
    cool_cc_l = []
    cool_max_l = []
    #
    for j, d_events in enumerate(date_events):
        for i in range(len(cycle_date_ini)):
            if d_events >  cycle_date_ini[i] and d_events <= cycle_date_mid[i]-0*day:
                heat_l.append(d_events)
                heat_cc_l.append(cc_events[j])
                heat_max_l.append(max_events[j])
            if d_events >  cycle_date_mid[i]-0*day and d_events <=  cycle_date_mid[i]+0*day:
                mid_l.append(d_events)
            if d_events >=  cycle_date_mid[i]+0*day and d_events <  cycle_date_fin[i]:
                cool_l.append(d_events)
                cool_cc_l.append(cc_events[j])
                cool_max_l.append(max_events[j])

    # save
    # write output
    # path = 'C:\\Users\\aar135\\codes_local_disk\\volc_forecast_tl\\volc_forecast_tl\\features\\lake_data\\corr_dsar_ratevar\\'
    # with open(path+'FWVZ_heating_cycle_date_events.txt', 'w') as f:
    #     for i in range(len(heat_l)):
    #         f.write(str(heat_l[i])+'\n')
    # with open(path+'FWVZ_cooling_cycle_date_events.txt', 'w') as f:
    #     for i in range(len(cool_l)):
    #         f.write(str(cool_l[i])+'\n')

    ## loop over events
    heat_l_filt_off = []
    mid_l_filt_off = []
    cool_l_filt_off = []
    heat_cc_l_filt_off = []
    heat_max_l_filt_off = []
    mid_l_filt_off = []
    cool_l_filt_off = []
    cool_cc_l_filt_off = []
    cool_max_l_filt_off = []
    #
    for j, d_events in enumerate(date_events_filt_off):
        for i in range(len(cycle_date_ini)):
            if d_events >  cycle_date_ini[i] and d_events <= cycle_date_mid[i]-0*day:
                heat_l_filt_off.append(d_events)
                heat_cc_l_filt_off.append(cc_events_filt_off[j])
                heat_max_l_filt_off.append(max_events_filt_off[j])
            if d_events >  cycle_date_mid[i]-0*day and d_events <=  cycle_date_mid[i]+0*day:
                mid_l.append(d_events_filt_off)
            if d_events >=  cycle_date_mid[i]+0*day and d_events <  cycle_date_fin[i]:
                cool_l_filt_off.append(d_events)
                cool_cc_l_filt_off.append(cc_events_filt_off[j])
                cool_max_l_filt_off.append(max_events_filt_off[j])
    #
    if True:  # statistics from cc's and max's during heating and cooling cycles
        #stats
        try:
            per_events_on_heat = len(heat_l)/(len(heat_l)+len(cool_l))*100 #%
            #
            mean_cc_on_heat = np.mean([float(heat_cc_l[i]) for i in range(len(heat_cc_l))]) #
            mean_cc_on_cool =  np.mean([float(cool_cc_l[i]) for i in range(len(cool_cc_l))]) #
            #
            mean_max_on_heat = np.mean([float(heat_max_l[i]) for i in range(len(heat_max_l))]) #
            mean_max_on_cool = np.mean([float(cool_max_l[i]) for i in range(len(cool_max_l))]) #
            #
            #plt.text(str(len(heat_l)/len(date_events)*100)+' %')
            plt.bar(['Heating', 'Cooling'], [len(heat_l), len(cool_l)])
            plt.show()
        except:
            pass
        ##
        per_events_on_heat_filt_off = len(heat_l_filt_off)/(len(heat_l_filt_off)+len(cool_l_filt_off))*100 #%
        #
        mean_cc_on_heat_filt_off = np.mean([float(heat_cc_l_filt_off[i]) for i in range(len(heat_cc_l_filt_off))]) #
        mean_cc_on_cool_filt_off =  np.mean([float(cool_cc_l_filt_off[i]) for i in range(len(cool_cc_l_filt_off))]) #
        #
        mean_max_on_heat_filt_off = np.mean([float(heat_max_l_filt_off[i]) for i in range(len(heat_max_l_filt_off))]) #
        mean_max_on_cool_filt_off = np.mean([float(cool_max_l_filt_off[i]) for i in range(len(cool_max_l_filt_off))]) #
        #
        #plt.text(str(len(heat_l)/len(date_events)*100)+' %')
        plt.bar(['Heating', 'Cooling'], [len(heat_l_filt_off), len(cool_l_filt_off)])
        plt.show()
        asdf
    
    # plot event locations in temp cycles  
    if True:
        #
        for i in range(len(cycle_date_ini)):
            #
            nrow = 2
            ncol = 1
            fig, (ax,ax2) = plt.subplots(nrows=nrow, ncols=ncol,figsize=(10,4))#(14,4))
            #
            col = ['r','g','b']
            alpha = [.5, 1., 1.]
            thick_line = [1., 3., 3.]

            mov_avg = True # moving average for temp and level data
            utc_0 = True
            # plot temp data
            if True:
                #
                if sta == 'FWVZ':
                    #
                    ti_e1 = cycle_date_ini[i]
                    tf_e1 = cycle_date_fin[i]
                    # import temp data
                    path = '..'+os.sep+'data'+os.sep+"RU001_temp_data.csv"
                    pd_temp = pd.read_csv(path, index_col=1)
                    if utc_0:
                        pd_temp.index = [datetimeify(pd_temp.index[i])-6*hour for i in range(len(pd_temp.index))]
                    else:
                        pd_temp.index = [datetimeify(pd_temp.index[i]) for i in range(len(pd_temp.index))]
                    # plot data in axis twin axis
                    # Trim the data
                    temp_e1_tim = pd_temp[ti_e1: tf_e1].index.values

                    #temp_e1_tim=to_nztimezone(temp_e1_tim)
                    #
                    temp_e1_val = pd_temp[ti_e1: tf_e1].loc[:,' t (C)'].values
                    # ax2
                    #ax2b = ax2.twinx()   
                    if mov_avg: # plot moving average
                        n=50
                        #v_plot = (temp_e1_val-np.min(temp_e1_val))/np.max((temp_e1_val-np.min(temp_e1_val)))
                        v_plot = temp_e1_val
                        ax.plot(temp_e1_tim, v_plot, '-', color='g', label='lake temperature', alpha = 1.)
                        #
                        #ax.plot(temp_e1_tim[:-n+1], moving_average(temp_e1_val, n=n), '--', color='k')#, label='temp. mov. avg.')
                        ax.plot(temp_e1_tim[n-1-20:-20], moving_average(v_plot[::-1], n=n)[::-1], '--', color='k')#, label='lake temperature')
                    else:
                        #ax2b.plot(temp_e1_tim, temp_e1_val, '-', color='g', label='temperature')
                        #ax2.set_ylim([-40,40])
                        #plt.show()
                        v_plot = temp_e1_val
                        ax.plot(temp_e1_tim, v_plot, '-', color='g', label='lake temperature', alpha = 1.)
                    ax.set_ylabel('Temperature °C')
                    ax.set_ylim([10,50])
                    #ax2b.set_ylabel('temperature C')   
                    #ax.legend(loc = 2)   
                    ## plot event

                    for d_events in date_events_filt_off:
                        if d_events > ti_e1 and d_events <= tf_e1:
                            ax.axvline(x=d_events, color='gray', ls='--')#, lw = 14)
                            #ax.axvline(x=cycle_date_mid[i], color='gray', ls='--')#, lw = 14)
                    for d_events in date_events:
                        if d_events > ti_e1 and d_events <= tf_e1:
                            ax.axvline(x=d_events, color='k', ls='-')#, lw = 14)
                            ax.axvline(x=cycle_date_mid[i], color='blue', ls='-')#, lw = 14)
                    ax.plot([],[], color='gray', ls='--', label = 'non-expl events')
                    ax.plot([],[], color='black', ls='-', label = 'expl events')
                    ax.plot([],[], color='blue', ls='-', label = 'mid heat/cool cycle')
                    ax.legend(loc = 2)   
                    #ax.grid()
                    #
                if sta == 'POS':
                    #
                    ti_e1 = cycle_date_ini[i]
                    tf_e1 = cycle_date_fin[i]
                    # import temp data
                    path = '..'+os.sep+'data'+os.sep+"POS_temp_data.csv"
                    pd_temp = pd.read_csv(path, index_col=1)
                    if utc_0:
                        pd_temp.index = [datetimeify(pd_temp.index[i])-6*hour for i in range(len(pd_temp.index))]
                    else:
                        pd_temp.index = [datetimeify(pd_temp.index[i]) for i in range(len(pd_temp.index))]
                    # plot data in axis twin axis
                    # Trim the data
                    temp_e1_tim = pd_temp[ti_e1: tf_e1].index.values

                    #temp_e1_tim=to_nztimezone(temp_e1_tim)
                    #
                    temp_e1_val = pd_temp[ti_e1: tf_e1].loc[:,' t (C)'].values
                    # ax2
                    #ax2b = ax2.twinx()   
                    if mov_avg: # plot moving average
                        n=50
                        #v_plot = (temp_e1_val-np.min(temp_e1_val))/np.max((temp_e1_val-np.min(temp_e1_val)))
                        v_plot = temp_e1_val
                        ax.plot(temp_e1_tim, v_plot, '-', color='g', label='lake temperature', alpha = 1.)
                        #
                        #ax.plot(temp_e1_tim[:-n+1], moving_average(temp_e1_val, n=n), '--', color='k')#, label='temp. mov. avg.')
                        ax.plot(temp_e1_tim[n-1-20:-20], moving_average(v_plot[::-1], n=n)[::-1], '--', color='k')#, label='lake temperature')
                    else:
                        #ax2b.plot(temp_e1_tim, temp_e1_val, '-', color='g', label='temperature')
                        #ax2.set_ylim([-40,40])
                        #plt.show()
                        v_plot = temp_e1_val
                        ax.plot(temp_e1_tim, v_plot, '-', color='g', label='lake temperature', alpha = 1.)
                    ax.set_ylabel('Temperature °C')
                    #ax.set_ylim([10,50])
                    #ax2b.set_ylabel('temperature C')   
                    #ax.legend(loc = 2)   
                    ## plot event

                    for d_events in date_events_filt_off:
                        if d_events > ti_e1 and d_events <= tf_e1:
                            ax.axvline(x=d_events, color='gray', ls='--')#, lw = 14)
                            #ax.axvline(x=cycle_date_mid[i], color='gray', ls='--')#, lw = 14)
                    for d_events in date_events:
                        if d_events > ti_e1 and d_events <= tf_e1:
                            ax.axvline(x=d_events, color='k', ls='-')#, lw = 14)
                            ax.axvline(x=cycle_date_mid[i], color='blue', ls='-')#, lw = 14)
                    ax.plot([],[], color='gray', ls='--', label = 'non-expl events')
                    ax.plot([],[], color='black', ls='-', label = 'expl events')
                    ax.plot([],[], color='blue', ls='-', label = 'mid heat/cool cycle')
                    ax.legend(loc = 2)   
                    #ax.grid()
                if sta == 'COP':
                    #
                    ti_e1 = cycle_date_ini[i]
                    tf_e1 = cycle_date_fin[i]
                    # import temp data
                    path = '..'+os.sep+'data'+os.sep+"COP_temp_data.csv"
                    pd_temp = pd.read_csv(path, index_col=1)
                    if utc_0:
                        pd_temp.index = [datetimeify(pd_temp.index[i])-6*hour for i in range(len(pd_temp.index))]
                    else:
                        pd_temp.index = [datetimeify(pd_temp.index[i]) for i in range(len(pd_temp.index))]
                    # plot data in axis twin axis
                    # Trim the data
                    temp_e1_tim = pd_temp[ti_e1: tf_e1].index.values

                    #temp_e1_tim=to_nztimezone(temp_e1_tim)
                    #
                    temp_e1_val = pd_temp[ti_e1: tf_e1].loc[:,' t (C)'].values
                    # ax2
                    #ax2b = ax2.twinx()   
                    if mov_avg: # plot moving average
                        n=50
                        #v_plot = (temp_e1_val-np.min(temp_e1_val))/np.max((temp_e1_val-np.min(temp_e1_val)))
                        v_plot = temp_e1_val
                        ax.plot(temp_e1_tim, v_plot, '-', color='g', label='lake temperature', alpha = 1.)
                        #
                        #ax.plot(temp_e1_tim[:-n+1], moving_average(temp_e1_val, n=n), '--', color='k')#, label='temp. mov. avg.')
                        ax.plot(temp_e1_tim[n-1-20:-20], moving_average(v_plot[::-1], n=n)[::-1], '--', color='k')#, label='lake temperature')
                    else:
                        #ax2b.plot(temp_e1_tim, temp_e1_val, '-', color='g', label='temperature')
                        #ax2.set_ylim([-40,40])
                        #plt.show()
                        v_plot = temp_e1_val
                        ax.plot(temp_e1_tim, v_plot, '-', color='g', label='lake temperature', alpha = 1.)
                    ax.set_ylabel('Temperature °C')
                    #ax.set_ylim([10,50])
                    #ax2b.set_ylabel('temperature C')   
                    #ax.legend(loc = 2)   
                    ## plot event

                    for d_events in date_events_filt_off:
                        if d_events > ti_e1 and d_events <= tf_e1:
                            ax.axvline(x=d_events, color='gray', ls='--')#, lw = 14)
                            #ax.axvline(x=cycle_date_mid[i], color='gray', ls='--')#, lw = 14)
                    for d_events in date_events:
                        if d_events > ti_e1 and d_events <= tf_e1:
                            ax.axvline(x=d_events, color='k', ls='-')#, lw = 14)
                            ax.axvline(x=cycle_date_mid[i], color='blue', ls='-')#, lw = 14)
                    ax.plot([],[], color='gray', ls='--', label = 'non-expl events')
                    ax.plot([],[], color='black', ls='-', label = 'expl events')
                    ax.plot([],[], color='blue', ls='-', label = 'mid heat/cool cycle')
                    ax.legend(loc = 2)   
                    #ax.grid()
            # plot so2
            if False:
                #
                #par2 = host.twinx()
                #ax2 = ax.twinx()
                if sta == 'FWVZ':
                    ti_e1 = cycle_date_ini[i]
                    tf_e1 = cycle_date_fin[i]
                    # import temp data
                    path = '..'+os.sep+'data'+os.sep+"RU000_SO2-flux-a_data.csv"
                    pd_so = pd.read_csv(path, index_col=1)
                    if utc_0:
                        pd_so.index = [datetimeify(pd_so.index[i])-6*hour for i in range(len(pd_so.index))]
                    else:
                        pd_so.index = [datetimeify(pd_so.index[i]) for i in range(len(pd_so.index))]
                    # plot data in axis twin axis
                    # Trim the data
                    so_e1_tim = pd_so[ti_e1: tf_e1].index.values
                    #temp_e1_tim=to_nztimezone(temp_e1_tim)
                    so_e1_val = pd_so[ti_e1: tf_e1].loc[:,' SO2-flux-a (kg/s)'].values
                    so_e1_val_err = pd_so[ti_e1: tf_e1].loc[:,' error (kg/s)'].values
                    # ax2
                    #ax2b = ax2.twinx()   
                    if mov_avg: # plot moving average
                        n=50
                        #v_plot = (temp_e1_val-np.min(temp_e1_val))/np.max((temp_e1_val-np.min(temp_e1_val)))
                        v_plot = so_e1_val
                        ax2.errorbar(so_e1_tim, v_plot, yerr = so_e1_val_err, fmt='o', ecolor = 'b',color='b')#'-', color='g', label='lake temperature', alpha = 1.)
                        #matplotlib.pyplot.errorbar(x, y, yerr=None
                        ax2.plot([], [], 'ob', label = 'SO2')
                    else:
                        #ax2b.plot(temp_e1_tim, temp_e1_val, '-', color='g', label='temperature')
                        #ax2.set_ylim([-40,40])
                        #plt.show()
                        v_plot = so_e1_val
                        #par1.plot(so_e1_tim, v_plot, '-', color='g', label='lake temperature', alpha = 1.)
                        ax2.errorbar(so_e1_tim, v_plot, yerr = so_e1_val_err)
                    #
                    ax2.set_ylabel('SO2 [kg/s]')
                    if max(v_plot) > 2:
                        ax2.set_ylim([0,max(v_plot)])
                    else:
                        ax2.set_ylim([0,2])
                    #ax.grid()
            # plot co2
            if False:
                #
                #par2 = host.twinx()
                ax2b = ax2.twinx()
                if sta == 'FWVZ':
                    ti_e1 = cycle_date_ini[i]
                    tf_e1 = cycle_date_fin[i]
                    # import temp data
                    path = '..'+os.sep+'data'+os.sep+"RU000_CO2-flux-a_data.csv"
                    pd_so = pd.read_csv(path, index_col=1)
                    if utc_0:
                        pd_so.index = [datetimeify(pd_so.index[i])-6*hour for i in range(len(pd_so.index))]
                    else:
                        pd_so.index = [datetimeify(pd_so.index[i]) for i in range(len(pd_so.index))]
                    # plot data in axis twin axis
                    # Trim the data
                    so_e1_tim = pd_so[ti_e1: tf_e1].index.values
                    #temp_e1_tim=to_nztimezone(temp_e1_tim)
                    so_e1_val = pd_so[ti_e1: tf_e1].loc[:,' CO2-flux-a (kg/s)'].values
                    so_e1_val_err = pd_so[ti_e1: tf_e1].loc[:,' error (kg/s)'].values
                    # ax2
                    #ax2b = ax2.twinx()   
                    if mov_avg: # plot moving average
                        n=50
                        #v_plot = (temp_e1_val-np.min(temp_e1_val))/np.max((temp_e1_val-np.min(temp_e1_val)))
                        v_plot = so_e1_val
                        ax2b.errorbar(so_e1_tim, v_plot, yerr = so_e1_val_err, fmt='o', ecolor = 'r',color='r')#'-', color='g', label='lake temperature', alpha = 1.)
                        ax2.plot([], [], 'or', label = 'CO2')#'-', color='g', label='lake temperature', alpha = 1.)
                        #matplotlib.pyplot.errorbar(x, y, yerr=None
                    else:
                        #ax2b.plot(temp_e1_tim, temp_e1_val, '-', color='g', label='temperature')
                        #ax2.set_ylim([-40,40])
                        #plt.show()
                        v_plot = so_e1_val
                        #par1.plot(so_e1_tim, v_plot, '-', color='g', label='lake temperature', alpha = 1.)
                        ax2b.errorbar(so_e1_tim, v_plot, yerr = so_e1_val_err)
                    #
                    ax2b.set_ylabel('CO2 [kg/s]')
                    if max(v_plot) > 20:
                        ax2b.set_ylim([0,max(v_plot)])
                    else:
                        ax2b.set_ylim([0,20])
                    #ax2b.legend(loc = 4)   
                    #ax2b.grid()
            #            
            # plot co2
            #ax2.legend(loc = 3)
            #ax.set_xlim([ti_e1-60*day, tf_e1+1*day])
            #ax2.set_xlim([ti_e1-60*day, tf_e1+1*day])
            #plt.show()

            save_png_path = 'C:\\Users\\aar135\\codes_local_disk\\volc_forecast_tl\\volc_forecast_tl\\features\\lake_data\\corr_dsar_ratevar\\temp_cycles\\'
            if save_png_path:
                plt.savefig(save_png_path+sta+'_temp_cycle_'+str(i+1)+'.png')
            #plt.show()
            plt.close('all')

# figures paper
def figure_1():
    '''
    plot precursor and four eruptions: Ruapehu 06 and 07, Kawa 13, Copahue 20
    '''
    def conv(at, x):
        y = ((x-np.mean(x))/np.std(x)*at.values).mean()
        return y
    def chqv(y):
        '''
        quick calc of change quantile variance .4-.6
        for calling:
            df[dt].rolling(N*24*6).apply(chqv)[N*24*6:]
        '''
        y0,y1 = np.percentile(y, [40,60])
        # return y1-y0
        inds = np.where((y>y0)&(y<y1))
        return np.var(np.diff(y, prepend=0)[inds])
    nrow = 5
    ncol = 1
    fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(nrows=nrow, ncols=ncol,figsize=(12,12))#(14,4))
    # subplot one: normalize features
    col = ['b','g','r']
    alpha = [1., .5, .5]
    thick_line = [2., 1., 1.]
    N, M = [2,15]
    # plot 1: WIZ (precursor, reference)
    if True:
        ## DSAR median 
        day = timedelta(days=1)
        sta_arch = 'WIZ'
        dt = 'zsc2_dsarF'
        fm = ForecastModel(window=2., overlap=1., station=sta_arch,
            look_forward=2., data_streams=[dt], 
            data_dir=r'C:\Users\aar135\codes_local_disk\volc_forecast_tl\volc_forecast_tl\data'
            )
        #
        te = fm.data.tes[-1]
        # rolling median and signature length window
        #N, M = [2,15]
        l_forw = 2
        # time
        j = fm.data.df.index
        # median 
        df = fm.data.df[(j>(te-(M+N)*day))&(j<te+l_forw*day)]
        archtype = df[dt].rolling(N*24*6).median()[N*24*6:]
        #
        _times = archtype.index
        _val = archtype.values
        _val_max = max(_val)
        #
        ft = 'DSAR median'
        ax1.plot(_times, _val, '-', color=col[0], alpha = alpha[0],linewidth=thick_line[0], label=' '+ ft,zorder=1)
        
        ## rsam raw background
        dt = 'rsam'
        fm = ForecastModel(window=2., overlap=1., station=sta_arch,
            look_forward=2., data_streams=[dt], 
            data_dir=r'C:\Users\aar135\codes_local_disk\volc_forecast_tl\volc_forecast_tl\data'
            )
        #
        te = fm.data.tes[-1]
        # rolling median and signature length window
        #N, M = [2,30]
        # time
        j = fm.data.df.index
        # median 
        df = fm.data.df[(j>(te-(M+N)*day))&(j<te+l_forw*day)]
        archtype = df[dt]#.rolling(N*24*6).median()[N*24*6:]
        #
        _times = archtype.index
        _val = archtype.values
        #
        ft = 'RSAM'
        _val = _val/max(_val) * _val_max*.7
        ax1.plot(_times, _val, '-', color='c', alpha = 1., linewidth=1., zorder=0)#, label=' '+ ft)
        ax1.plot([], [], '-', color='c', alpha = 1., linewidth=1., label=' '+ ft)
        
        ##
        ax1b = ax1.twinx()
        ## HF median 
        dt = 'zsc2_hfF'
        fm = ForecastModel(window=2., overlap=1., station=sta_arch,
            look_forward=2., data_streams=[dt], 
            data_dir=r'C:\Users\aar135\codes_local_disk\volc_forecast_tl\volc_forecast_tl\data'
            )
        #
        te = fm.data.tes[-1]
        # rolling median and signature length window
        #N, M = [2,15]
        # time
        j = fm.data.df.index
        # median 
        df = fm.data.df[(j>(te-(M+N)*day))&(j<te+l_forw*day)]
        archtype = df[dt].rolling(N*24*6).median()[N*24*6:]
        #
        _times = archtype.index
        _val = archtype.values
        #
        ft = 'HF median'
        ax1b.plot(_times, _val, '-', color=col[1], alpha = alpha[1],linewidth=thick_line[1])#, label=' '+ ft)
        ax1.plot([], [], '-', color=col[1], alpha = alpha[1],linewidth=thick_line[1], label=' '+ ft)
        ##
        ## MF median 
        dt = 'zsc2_mfF'
        fm = ForecastModel(window=2., overlap=1., station=sta_arch,
            look_forward=2., data_streams=[dt], 
            data_dir=r'C:\Users\aar135\codes_local_disk\volc_forecast_tl\volc_forecast_tl\data'
            )
        #
        te = fm.data.tes[-1]
        # rolling median and signature length window
        #N, M = [2,30]
        # time
        j = fm.data.df.index
        # median 
        df = fm.data.df[(j>(te-(M+N)*day))&(j<te+l_forw*day)]
        archtype = df[dt].rolling(N*24*6).median()[N*24*6:]
        #
        _times = archtype.index
        _val = archtype.values
        #
        ft = 'MF median'
        ax1b.plot(_times, _val, '-', color=col[2], alpha = alpha[2],linewidth=thick_line[2])#, label=' '+ ft)
        ax1.plot([], [], '-', color=col[2], alpha = alpha[2],linewidth=thick_line[2], label=' '+ ft)

        # plot eruption 
        if True: # plot vertical lines
            #te = datetimeify(erup_times[j])#fm_e1.data.tes[int(erup[-1:])-1]
            if True:
                ax1.axvline(te, color='red', alpha = .25, linestyle='-', linewidth=12, zorder = 0)
                ax1.plot([], color='red', alpha = .25, linestyle='-', linewidth=12, label = 'eruption')
            else:
                ax1.axvline(te, color='gray', alpha = .25, linestyle='-', linewidth=12, zorder = 0)
                ax1.plot([], color='gray', alpha = .25, linestyle='-', linewidth=12, label = 'fluid release event')
        ## 
        ax1.legend(loc=2)
        ax1.set_ylabel('normalized data')
        ax1.set_ylabel('nDSAR median')
        ax1b.set_ylabel('nHF and nMF median')
        #
        ax1.grid(color='gray', linestyle='-', linewidth=.3, alpha = 0.5)

    # plot 2: FWVZ 2006
    if True:
        ## DSAR median 
        day = timedelta(days=1)
        sta_arch = 'FWVZ'
        dt = 'zsc2_dsarF'
        fm = ForecastModel(window=2., overlap=1., station=sta_arch,
            look_forward=2., data_streams=[dt], 
            data_dir=r'C:\Users\aar135\codes_local_disk\volc_forecast_tl\volc_forecast_tl\data'
            )
        #
        te = fm.data.tes[0]
        te = datetimeify('2006-10-05') # pre filter
        te = datetimeify('2006-10-04 09:20:00')
        # rolling median and signature length window
        #N, M = [2,15]
        l_forw = 1
        # time
        j = fm.data.df.index
        # median 
        df = fm.data.df[(j>(te-(M+N)*day))&(j<te+l_forw*day)]
        archtype = df[dt].rolling(N*24*6).median()[N*24*6:]
        #
        _times = archtype.index
        _val = archtype.values
        _val_max = max(_val)
        #
        ft = 'DSAR median'
        ax2.plot(_times, _val, '-', color=col[0], alpha = alpha[0],linewidth=thick_line[0], label=' '+ ft,zorder=1)
        
        ## rsam raw background
        dt = 'rsam'
        fm = ForecastModel(window=2., overlap=1., station=sta_arch,
            look_forward=2., data_streams=[dt], 
            data_dir=r'C:\Users\aar135\codes_local_disk\volc_forecast_tl\volc_forecast_tl\data'
            )
        # rolling median and signature length window
        #N, M = [2,30]
        # time
        j = fm.data.df.index
        # median 
        df = fm.data.df[(j>(te-(M+N)*day))&(j<te+l_forw*day)]
        archtype = df[dt]#.rolling(N*24*6).median()[N*24*6:]
        #
        _times = archtype.index
        _val = archtype.values
        #
        ft = 'RSAM'
        _val = _val/max(_val) * _val_max*.7
        ax2.plot(_times, _val, '-', color='c', alpha = 1., linewidth=2., zorder=2)#, label=' '+ ft)
        ax2.plot([], [], '-', color='c', alpha = 1., linewidth=2, label=' '+ ft)
        

        ##
        ax2b = ax2.twinx()
        ## HF median 
        dt = 'zsc2_hfF'
        fm = ForecastModel(window=2., overlap=1., station=sta_arch,
            look_forward=2., data_streams=[dt], 
            data_dir=r'C:\Users\aar135\codes_local_disk\volc_forecast_tl\volc_forecast_tl\data'
            )
        #
        # rolling median and signature length window
        #N, M = [2,15]
        # time
        j = fm.data.df.index
        # median 
        df = fm.data.df[(j>(te-(M+N)*day))&(j<te+l_forw*day)]
        archtype = df[dt].rolling(N*24*6).median()[N*24*6:]
        #
        _times = archtype.index
        _val = archtype.values
        #
        ft = 'HF median'
        ax2b.plot(_times, _val, '-', color=col[1], alpha = alpha[1],linewidth=thick_line[1])#, label=' '+ ft)
        ax2.plot([], [], '-', color=col[1], alpha = alpha[1],linewidth=thick_line[1], label=' '+ ft)
        ##
        ## MF median 
        dt = 'zsc2_mfF'
        fm = ForecastModel(window=2., overlap=1., station=sta_arch,
            look_forward=2., data_streams=[dt], 
            data_dir=r'C:\Users\aar135\codes_local_disk\volc_forecast_tl\volc_forecast_tl\data'
            )
        #
        # rolling median and signature length window
        #N, M = [2,30]
        # time
        j = fm.data.df.index
        # median 
        df = fm.data.df[(j>(te-(M+N)*day))&(j<te+l_forw*day)]
        archtype = df[dt].rolling(N*24*6).median()[N*24*6:]
        #
        _times = archtype.index
        _val = archtype.values
        #
        ft = 'MF median'
        ax2b.plot(_times, _val, '-', color=col[2], alpha = alpha[2],linewidth=thick_line[2])#, label=' '+ ft)
        ax2.plot([], [], '-', color=col[2], alpha = alpha[2],linewidth=thick_line[2], label=' '+ ft)

        # plot event 
        ax2.axvline(te+0.07*day, color='k',linestyle='--', linewidth=3, zorder = 1)
        ax2.plot([], color='k', linestyle='--', linewidth=3, label = 'auto picker')
        # plot eruption 
        if True: # plot vertical lines
            #te = datetimeify(erup_times[j])#fm_e1.data.tes[int(erup[-1:])-1]
            if True:
                ax2.axvline(te, color='red', alpha = .25, linestyle='-', linewidth=12, zorder = 0)
                ax2.plot([], color='red', alpha = .25, linestyle='-', linewidth=12, label = 'eruption')
            else:
                ax2.axvline(te, color='gray', alpha = .25, linestyle='-', linewidth=12, zorder = 0)
                ax2.plot([], color='gray', alpha = .25, linestyle='-', linewidth=12, label = 'fluid release event')
        ## 
        ax2.legend(loc=2)
        ax2.set_ylabel('normalized data')
        ax2.set_ylabel('nDSAR median')
        ax2b.set_ylabel('nHF and nMF median')
        #
        ax2.grid(color='gray', linestyle='-', linewidth=.3, alpha = 0.5)

    # plot 2: FWVZ 2007
    if True:
        ## DSAR median 
        day = timedelta(days=1)
        sta_arch = 'FWVZ'
        dt = 'zsc2_dsarF'
        fm = ForecastModel(window=2., overlap=1., station=sta_arch,
            look_forward=2., data_streams=[dt], 
            data_dir=r'C:\Users\aar135\codes_local_disk\volc_forecast_tl\volc_forecast_tl\data'
            )
        #
        te = fm.data.tes[1]
        te = datetimeify('2007-09-25')
        te = datetimeify('2007-09-25 08:20:00')
        # rolling median and signature length window
        #N, M = [2,15]
        _M = 22
        l_forw = .5
        # time
        j = fm.data.df.index
        # median 
        df = fm.data.df[(j>(te-(_M+N)*day))&(j<te+l_forw*day)]
        archtype = df[dt].rolling(N*24*6).median()[N*24*6:]
        #
        _times = archtype.index
        _val = archtype.values
        _val_max = max(_val)
        #
        ft = 'DSAR median'
        ax3.plot(_times, _val, '-', color=col[0], alpha = alpha[0],linewidth=thick_line[0], label=' '+ ft,zorder=1)
        
        ## rsam raw background
        dt = 'rsam'
        fm = ForecastModel(window=2., overlap=1., station=sta_arch,
            look_forward=2., data_streams=[dt], 
            data_dir=r'C:\Users\aar135\codes_local_disk\volc_forecast_tl\volc_forecast_tl\data'
            )
        # rolling median and signature length window
        #N, M = [2,30]
        # time
        j = fm.data.df.index
        # median 
        df = fm.data.df[(j>(te-(_M+N+2)*day))&(j<te+l_forw*day)]
        archtype = df[dt]#.rolling(N*24*6).median()[N*24*6:]
        #
        _times = archtype.index
        _val = archtype.values
        #
        ft = 'RSAM'
        _val = _val/max(_val) * _val_max*.7
        ax3.plot(_times, _val, '-', color='c', alpha = 1.0, linewidth=2., zorder=0)#, label=' '+ ft)
        ax3.plot([], [], '-', color='c', alpha = 1., linewidth=2., label=' '+ ft)
        
        ##
        ax3b = ax3.twinx()
        ## HF median 
        dt = 'zsc2_hfF'
        fm = ForecastModel(window=2., overlap=1., station=sta_arch,
            look_forward=2., data_streams=[dt], 
            data_dir=r'C:\Users\aar135\codes_local_disk\volc_forecast_tl\volc_forecast_tl\data'
            )
        #
        # rolling median and signature length window
        #N, M = [2,15]
        # time
        j = fm.data.df.index
        # median 
        df = fm.data.df[(j>(te-(_M+N)*day))&(j<te+l_forw*day)]
        archtype = df[dt].rolling(N*24*6).median()[N*24*6:]
        #
        _times = archtype.index
        _val = archtype.values
        #
        ft = 'HF median'
        ax3b.plot(_times, _val, '-', color=col[1], alpha = alpha[1],linewidth=thick_line[1])#, label=' '+ ft)
        ax3.plot([], [], '-', color=col[1], alpha = alpha[1],linewidth=thick_line[1], label=' '+ ft)
        ##
        ## MF median 
        dt = 'zsc2_mfF'
        fm = ForecastModel(window=2., overlap=1., station=sta_arch,
            look_forward=2., data_streams=[dt], 
            data_dir=r'C:\Users\aar135\codes_local_disk\volc_forecast_tl\volc_forecast_tl\data'
            )
        #
        # rolling median and signature length window
        #N, M = [2,30]
        # time
        j = fm.data.df.index
        # median 
        df = fm.data.df[(j>(te-(_M+N)*day))&(j<te+l_forw*day)]
        archtype = df[dt].rolling(N*24*6).median()[N*24*6:]
        #
        _times = archtype.index
        _val = archtype.values
        #
        ft = 'MF median'
        ax3b.plot(_times, _val, '-', color=col[2], alpha = alpha[2],linewidth=thick_line[2])#, label=' '+ ft)
        ax3.plot([], [], '-', color=col[2], alpha = alpha[2],linewidth=thick_line[2], label=' '+ ft)

        #ax3b.set_ylim([0,1.5])

        # plot event 
        ax3.axvline(te+0.09*day, color='k',linestyle='--', linewidth=3, zorder = 4)
        ax3.plot([], color='k', linestyle='--', linewidth=3, label = 'auto picker')
        # plot eruption 
        if True: # plot vertical lines
            #te = datetimeify(erup_times[j])#fm_e1.data.tes[int(erup[-1:])-1]
            if True:
                ax3.axvline(te, color='red', alpha = .25, linestyle='-', linewidth=12, zorder = 0)
                ax3.plot([], color='red', alpha = .25, linestyle='-', linewidth=12, label = 'eruption')
            else:
                ax3.axvline(te, color='gray', alpha = .25, linestyle='-', linewidth=12, zorder = 0)
                ax3.plot([], color='gray', alpha = .25, linestyle='-', linewidth=12, label = 'fluid release event')

        ## 
        ax3.legend(loc=2)
        ax3.set_ylabel('normalized data')
        ax3.set_ylabel('nDSAR median')
        ax3b.set_ylabel('nHF and nMF median')
        #
        ax3.grid(color='gray', linestyle='-', linewidth=.3, alpha = 0.5)

    # plot 3: Kawa 2007
    if True:
        ## DSAR median 
        day = timedelta(days=1)
        sta_arch = 'POS'
        dt = 'zsc2_dsarF'
        fm = ForecastModel(window=2., overlap=1., station=sta_arch,
            look_forward=2., data_streams=[dt], 
            data_dir=r'C:\Users\aar135\codes_local_disk\volc_forecast_tl\volc_forecast_tl\data'
            )
        #
        #te = fm.data.tes[0] # 2013 03 20 
        te = datetimeify('2013-03-29')#('2011-04-22')
        te = datetimeify('2013-03-27 04:30:00')#('2011-04-22')
        te = datetimeify('2013-04-01 00:00:00')#('2011-04-22')
        # rolling median and signature length window
        #N, M = [2,7]
        l_forw = 3
        # time
        j = fm.data.df.index
        # median 
        _N, _M = [2,30]
        df = fm.data.df[(j>(te-(_M+N)*day))&(j<te+l_forw*day)]
        archtype = df[dt].rolling(N*24*6).median()[N*24*6:]
        #
        _times = archtype.index
        _val = archtype.values
        _val_max = max(_val)
        #
        ft = 'DSAR median'
        ax4.plot(_times, _val, '-', color=col[0], alpha = alpha[0],linewidth=thick_line[0], label=' '+ ft,zorder=1)
        
        ## rsam raw background
        dt = 'rsam'
        fm = ForecastModel(window=2., overlap=1., station=sta_arch,
            look_forward=2., data_streams=[dt], 
            data_dir=r'C:\Users\aar135\codes_local_disk\volc_forecast_tl\volc_forecast_tl\data'
            )
        # rolling median and signature length window
        #N, M = [2,30]
        # time
        j = fm.data.df.index
        # median 
        df = fm.data.df[(j>(te-(_M+N)*day))&(j<te+l_forw*day)]
        archtype = df[dt]#.rolling(N*24*6).median()[N*24*6:]
        #
        _times = archtype.index
        _val = archtype.values
        #
        ft = 'RSAM'
        _val = _val/max(_val) * _val_max*.7
        ax4.plot(_times, _val, '-', color='c', alpha = 1., linewidth=2, zorder=0)#, label=' '+ ft)
        ax4.plot([], [], '-', color='c', alpha = 1., linewidth=2, label=' '+ ft)
        
        ##
        ax4b = ax4.twinx()
        ## HF median 
        dt = 'zsc2_hfF'
        fm = ForecastModel(window=2., overlap=1., station=sta_arch,
            look_forward=2., data_streams=[dt], 
            data_dir=r'C:\Users\aar135\codes_local_disk\volc_forecast_tl\volc_forecast_tl\data'
            )
        #
        # rolling median and signature length window
        #N, M = [2,15]
        # time
        j = fm.data.df.index
        # median 
        df = fm.data.df[(j>(te-(_M+N)*day))&(j<te+l_forw*day)]
        archtype = df[dt].rolling(N*24*6).median()[N*24*6:]
        #
        _times = archtype.index
        _val = archtype.values
        #
        ft = 'HF median'
        ax4b.plot(_times, _val, '-', color=col[1], alpha = alpha[1],linewidth=thick_line[1])#, label=' '+ ft)
        ax4.plot([], [], '-', color=col[1], alpha = alpha[1],linewidth=thick_line[1], label=' '+ ft)
        ##
        ## MF median 
        dt = 'zsc2_mfF'
        fm = ForecastModel(window=2., overlap=1., station=sta_arch,
            look_forward=2., data_streams=[dt], 
            data_dir=r'C:\Users\aar135\codes_local_disk\volc_forecast_tl\volc_forecast_tl\data'
            )
        #
        # rolling median and signature length window
        #N, M = [2,30]
        # time
        j = fm.data.df.index
        # median 
        df = fm.data.df[(j>(te-(_M+N)*day))&(j<te+l_forw*day)]
        archtype = df[dt].rolling(N*24*6).median()[N*24*6:]
        #
        _times = archtype.index
        _val = archtype.values
        #
        ft = 'MF median'
        ax4b.plot(_times, _val, '-', color=col[2], alpha = alpha[2],linewidth=thick_line[2])#, label=' '+ ft)
        ax4.plot([], [], '-', color=col[2], alpha = alpha[2],linewidth=thick_line[2], label=' '+ ft)

        # plot event  
        ax4.axvline(te+0.1*day, color='k',linestyle='--', linewidth=3, zorder = 4)
        ax4.plot([], color='k', linestyle='--', linewidth=3, label = 'auto picker')
        # plot period of high activity
        te = fm.data.tes[0]  
        te = datetimeify('2013 03 08 00 00 00')
        ax4.axvline(te+0*day, color='m',linestyle='--', linewidth=3, zorder = 4)
        te = datetimeify('2013 04 02 00 00 00')
        ax4.axvline(te+0*day, color='m',linestyle='--', linewidth=3, zorder = 4)
        #
        ax4.plot([], color='m', linestyle='--', linewidth=3, label = 'high activity period')
        # plot eruption 
        if True: # plot vertical lines
            #te = datetimeify(erup_times[j])#fm_e1.data.tes[int(erup[-1:])-1]
            te = datetimeify('2013-03-20 00:00:00')#
            if True:
                ax4.axvline(te, color='red', alpha = .25, linestyle='-', linewidth=12, zorder = 0)
                ax4.plot([], color='red', alpha = .25, linestyle='-', linewidth=12, label = 'eruption')
            else:
                ax4.axvline(te, color='gray', alpha = .25, linestyle='-', linewidth=12, zorder = 0)
                ax4.plot([], color='gray', alpha = .25, linestyle='-', linewidth=12, label = 'fluid release event')
        ## 
        ax4.legend(loc=2)
        ax4.set_ylabel('normalized data')
        ax4.set_ylabel('nDSAR median')
        ax4b.set_ylabel('nHF and nMF median')
        #
        ax4.grid(color='gray', linestyle='-', linewidth=.3, alpha = 0.5)
        ax4.set_yscale('log')
    
    # plot 3: Copahue 2020
    if True:
        ## DSAR median 
        day = timedelta(days=1)
        sta_arch = 'COP'
        dt = 'zsc2_dsarF'
        fm = ForecastModel(window=2., overlap=1., station=sta_arch,
            look_forward=2., data_streams=[dt], 
            data_dir=r'C:\Users\aar135\codes_local_disk\volc_forecast_tl\volc_forecast_tl\data'
            )
        #
        #te = datetimeify('2020-08-30')#fm.data.tes[0] '2020-10-20')#
        #te = datetimeify('2020-09-01 04:10:00')
        te = datetimeify('2020-08-05 18:50:00')
        # rolling median and signature length window
        #N, M = [2,10]
        # time
        j = fm.data.df.index
        l_forw = 1.2
        # median 
        df = fm.data.df[(j>(te-(M+N)*day))&(j<te+l_forw*day)]
        archtype = df[dt].rolling(N*24*6).median()[N*24*6:]
        #
        _times = archtype.index
        _val = archtype.values
        _val_max = max(_val)
        #
        ft = 'DSAR median'
        ax5.plot(_times, _val, '-', color=col[0], alpha = alpha[0],linewidth=thick_line[0], label=' '+ ft,zorder=1)
        
        ## rsam raw background
        dt = 'rsam'
        fm = ForecastModel(window=2., overlap=1., station=sta_arch,
            look_forward=2., data_streams=[dt], 
            data_dir=r'C:\Users\aar135\codes_local_disk\volc_forecast_tl\volc_forecast_tl\data'
            )
        # rolling median and signature length window
        #N, M = [2,30]
        # time
        j = fm.data.df.index
        # median 
        df = fm.data.df[(j>(te-(M+N)*day))&(j<te+l_forw*day)]
        archtype = df[dt]#.rolling(N*24*6).median()[N*24*6:]
        #
        _times = archtype.index
        _val = archtype.values
        #
        ft = 'RSAM'
        _val = _val/max(_val) * _val_max*.7
        ax5.plot(_times, _val, '-', color='c', alpha = 1.0, linewidth=2.0, zorder=0)#, label=' '+ ft)
        ax5.plot([], [], '-', color='c', alpha = 1.0, linewidth=2.0, label=' '+ ft)
        ##
        ax5b = ax5.twinx()
        ## HF median 
        dt = 'zsc2_hfF'
        fm = ForecastModel(window=2., overlap=1., station=sta_arch,
            look_forward=2., data_streams=[dt], 
            data_dir=r'C:\Users\aar135\codes_local_disk\volc_forecast_tl\volc_forecast_tl\data'
            )
        #
        # rolling median and signature length window
        #N, M = [2,15]
        # time
        j = fm.data.df.index
        # median 
        df = fm.data.df[(j>(te-(M+N)*day))&(j<te+l_forw*day)]
        archtype = df[dt].rolling(N*24*6).median()[N*24*6:]
        #
        _times = archtype.index
        _val = archtype.values
        #
        ft = 'HF median'
        ax5b.plot(_times, _val, '-', color=col[1], alpha = alpha[1],linewidth=thick_line[1])#, label=' '+ ft)
        ax5.plot([], [], '-', color=col[1], alpha = alpha[1],linewidth=thick_line[1], label=' '+ ft)
        ##
        ## MF median 
        dt = 'zsc2_mfF'
        fm = ForecastModel(window=2., overlap=1., station=sta_arch,
            look_forward=2., data_streams=[dt], 
            data_dir=r'C:\Users\aar135\codes_local_disk\volc_forecast_tl\volc_forecast_tl\data'
            )
        #
        # rolling median and signature length window
        #N, M = [2,30]
        # time
        j = fm.data.df.index
        # median 
        df = fm.data.df[(j>(te-(M+N)*day))&(j<te+l_forw*day)]
        archtype = df[dt].rolling(N*24*6).median()[N*24*6:]
        #
        _times = archtype.index
        _val = archtype.values
        #
        ft = 'MF median'
        ax5b.plot(_times, _val, '-', color=col[2], alpha = alpha[2],linewidth=thick_line[2])#, label=' '+ ft)
        ax5.plot([], [], '-', color=col[2], alpha = alpha[2],linewidth=thick_line[2], label=' '+ ft)

        # plot eruption 
        ax5.axvline(te+0.07*day, color='k',linestyle='--', linewidth=3, zorder = 4)
        ax5.plot([], color='k', linestyle='--', linewidth=3, label = 'auto picker')
        # plot eruption registered 
        te = datetimeify('2020-08-06 08:25:00')
        #ax5.axvline(te+0.15*day, color='r',linestyle='-', linewidth=3, zorder = 4)
        #ax5.plot([], color='r', linestyle='-', linewidth=3, label = 'eruption')
        #
        if True: # plot vertical lines
            #te = datetimeify(erup_times[j])#fm_e1.data.tes[int(erup[-1:])-1]
            if True:
                ax5.axvline(te, color='red', alpha = .25, linestyle='-', linewidth=12, zorder = 0)
                ax5.plot([], color='red', alpha = .25, linestyle='-', linewidth=12, label = 'eruption')
            else:
                ax5.axvline(te, color='gray', alpha = .25, linestyle='-', linewidth=12, zorder = 0)
                ax5.plot([], color='gray', alpha = .25, linestyle='-', linewidth=12, label = 'fluid release event')
        ## 
        ax5.legend(loc=2)
        ax5.set_ylabel('normalized data')
        ax5.set_ylabel('nDSAR median')
        ax5b.set_ylabel('nHF and nMF median')
        #
        ax5.grid(color='gray', linestyle='-', linewidth=.3, alpha = 0.5)

    ax1.set_title('Whakaari 2019 eruption: sealing, pressurization and eruption')
    ax2.set_title('Ruapehu 2006 eruption')
    ax3.set_title('Ruapehu 2007 eruption')
    ax4.set_title('Kawa Ijen 2013 eruption')
    ax5.set_title('Copahue 2020 small eruption')
    #####################################################
    plt.tight_layout()
    plt.show()
    plt.close()

def figure_2(): # temperature cycles 
    '''
    temperature cycle for Ruapehu ~2016
    temperature cycle for Kawa Ijen ~2016

    '''
    pass
    nrow = 7
    ncol = 1
    fig, (ax0, ax1, ax2, ax5, ax6, ax3, ax4) = plt.subplots(nrows=nrow, ncols=ncol,figsize=(10,10))#(14,4))
    # subplot one: normalize features
    col = ['b','g','r']
    alpha = [1., .5, .5]
    thick_line = [2., 1., 1.]
    utc_0 = True
    mov_avg = True

    # plot 0: temp cycles Ruapehu 
    if True:
        if True: # temp
            #
            #ti_e1 = datetimeify('2012-06-01') 
            ti_e1 = datetimeify('2009-01-01') 
            tf_e1 = datetimeify('2021-12-31') 
            sta = 'FWVZ'
            ## import events
            path = 'C:\\Users\\aar135\\codes_local_disk\\volc_forecast_tl\\volc_forecast_tl\\features\\lake_data\\'+sta+'\\selection\\'
            if auto_picked:
                path = 'C:\\Users\\aar135\\codes_local_disk\\volc_forecast_tl\\volc_forecast_tl\\features\\lake_data\\corr_dsar_ratevar\\'
            dates =[]
            path_dates = path+sta+'_dates_missed_events_from_dsar_median_rv_cc.txt'
            path_dates = path+sta+'_dates_max_CC_missed_events_from_dsar_median_rv.txt'

            path_dates_filt_off = path+sta+'_dates_max_CC_missed_events_from_dsar_median_rv_without_filt_peak.txt'
            path_dates_filt_on = path+sta+'_dates_max_CC_missed_events_from_dsar_median_rv_with_filt_peak - Copy.txt'
            path_dates_filt_on = path+sta+'_dates_max_CC_missed_events_from_dsar_median_rv_with_filt_peak_man_refine.txt'
            #
            date_events = []
            cc_events = []
            max_events = []
            with open(path_dates_filt_on,'r') as fp:
                for ln in fp.readlines():
                    _d, _cc, _mx =ln.rstrip().split(',')
                    #_d =ln.rstrip().split(',')
                    date_events.append(datetimeify(_d))
                    cc_events.append(_cc)
                    max_events.append(_mx)
                #date_events = [datetimeify(ln.rstrip()) for ln in fp.readlines()]
            #
            date_events_filt_off = []
            cc_events_filt_off = []
            max_events_filt_off = []
            with open(path_dates_filt_off,'r') as fp:
                for ln in fp.readlines():
                    _d, _cc, _mx =ln.rstrip().split(',')
                    date_events_filt_off.append(datetimeify(_d))
                    cc_events_filt_off.append(_cc)
                    max_events_filt_off.append(_mx)
                #date_events = [datetimeify(ln.rstrip()) for ln in fp.readlines()]

            # import temp data
            path = '..'+os.sep+'data'+os.sep+"RU001_temp_data.csv"
            pd_temp = pd.read_csv(path, index_col=1)
            if utc_0:
                pd_temp.index = [datetimeify(pd_temp.index[i])-6*hour for i in range(len(pd_temp.index))]
            else:
                pd_temp.index = [datetimeify(pd_temp.index[i]) for i in range(len(pd_temp.index))]
            # plot data in axis twin axis
            # Trim the data
            temp_e1_tim = pd_temp[ti_e1: tf_e1].index.values

            #temp_e1_tim=to_nztimezone(temp_e1_tim)
            #
            temp_e1_val = pd_temp[ti_e1: tf_e1].loc[:,' t (C)'].values
            # ax2
            #ax2b = ax2.twinx()   
            if mov_avg: # plot moving average
                n=50
                #v_plot = (temp_e1_val-np.min(temp_e1_val))/np.max((temp_e1_val-np.min(temp_e1_val)))
                v_plot = temp_e1_val
                ax0.plot(temp_e1_tim, v_plot, '-', color='g', label='lake temperature', alpha = 1. , zorder = 4)
                #
                #ax.plot(temp_e1_tim[:-n+1], moving_average(temp_e1_val, n=n), '--', color='k')#, label='temp. mov. avg.')
                #ax0.plot(temp_e1_tim[n-1-20:-20], moving_average(v_plot[::-1], n=n)[::-1], '--', color='k')#, label='lake temperature')
            else:
                #ax2b.plot(temp_e1_tim, temp_e1_val, '-', color='g', label='temperature')
                #ax2.set_ylim([-40,40])
                #plt.show()
                v_plot = temp_e1_val
                ax0.plot(temp_e1_tim, v_plot, '-', color='g', label='lake temperature', alpha = 1. , zorder = 4)
            ax0.set_ylabel('Temperature °C')
            ax0.set_ylim([10,50])
            #ax2b.set_ylabel('temperature C')   
            #ax.legend(loc = 2)   
            ## plot event

            # for d_events in date_events_filt_off:
            #     if d_events > ti_e1 and d_events <= tf_e1:
            #         ax0.axvline(x=d_events, color='gray', ls='--', lw = 1. , alpha = 0.7, zorder = 4)
            #         #ax.axvline(x=cycle_date_mid[i], color='gray', ls='--')#, lw = 14)
            for d_events in date_events:
                if d_events > ti_e1 and d_events <= tf_e1:
                    ax0.axvline(x=d_events, color='k', ls='--', lw = 2. , alpha = 1.0, zorder = 4)
                    #ax0.axvline(x=cycle_date_mid[i], color='blue', ls='-')#, lw = 14)
            #ax0.plot([],[], color='gray', ls='--', label = 'non-expl events')
            ax0.plot([],[], color='k', ls='--', lw = 3, label = 'sealing + fluid release events')
            #ax1.plot([],[], color='blue', ls='-', label = 'mid heat/cool cycle')  
            #
        if False: # so2
            ax1b = ax1.twinx()
            # import temp data
            path = '..'+os.sep+'data'+os.sep+"RU000_SO2-flux-a_data.csv"
            pd_so = pd.read_csv(path, index_col=1)
            if utc_0:
                pd_so.index = [datetimeify(pd_so.index[i])-6*hour for i in range(len(pd_so.index))]
            else:
                pd_so.index = [datetimeify(pd_so.index[i]) for i in range(len(pd_so.index))]
            # plot data in axis twin axis
            # Trim the data
            so_e1_tim = pd_so[ti_e1: tf_e1].index.values
            #temp_e1_tim=to_nztimezone(temp_e1_tim)
            so_e1_val = pd_so[ti_e1: tf_e1].loc[:,' SO2-flux-a (kg/s)'].values
            so_e1_val_err = pd_so[ti_e1: tf_e1].loc[:,' error (kg/s)'].values
            # ax2
            ax0b = ax0.twinx()   
            if mov_avg: # plot moving average
                n=50
                #v_plot = (temp_e1_val-np.min(temp_e1_val))/np.max((temp_e1_val-np.min(temp_e1_val)))
                v_plot = so_e1_val
                ax0b.errorbar(so_e1_tim, v_plot, yerr = so_e1_val_err, fmt='.', ecolor = 'b',color='b', alpha = 1.)#'-', color='g', label='lake temperature', alpha = 1.)
                #matplotlib.pyplot.errorbar(x, y, yerr=None
                ax0.plot([], [], '.b', label = 'SO2')
            else:
                #ax2b.plot(temp_e1_tim, temp_e1_val, '-', color='g', label='temperature')
                #ax2.set_ylim([-40,40])
                #plt.show()
                v_plot = so_e1_val
                #par1.plot(so_e1_tim, v_plot, '-', color='g', label='lake temperature', alpha = 1.)
                ax0b.errorbar(so_e1_tim, v_plot, yerr = so_e1_val_err)
            #
            ax0b.set_ylabel('SO2 [kg/s]')
            # if max(v_plot) > 2:
            #     ax1b.set_ylim([0,max(v_plot)])
            # else:
            #     ax1b.set_ylim([0,2])
            ax0b.set_ylim([0,5])
            #ax.grid()
        if False: # co2
            ax1c = ax1.twinx()
            # Offset the right spine of twin2.  The ticks and label have already been
            # placed on the right by twinx above.
            #ax1c.spines.right.set_position(("axes", 1.2))
            # import temp data
            path = '..'+os.sep+'data'+os.sep+"RU000_CO2-flux-a_data.csv"
            pd_so = pd.read_csv(path, index_col=1)
            if utc_0:
                pd_so.index = [datetimeify(pd_so.index[i])-6*hour for i in range(len(pd_so.index))]
            else:
                pd_so.index = [datetimeify(pd_so.index[i]) for i in range(len(pd_so.index))]
            # plot data in axis twin axis
            # Trim the data
            so_e1_tim = pd_so[ti_e1: tf_e1].index.values
            #temp_e1_tim=to_nztimezone(temp_e1_tim)
            so_e1_val = pd_so[ti_e1: tf_e1].loc[:,' CO2-flux-a (kg/s)'].values
            so_e1_val_err = pd_so[ti_e1: tf_e1].loc[:,' error (kg/s)'].values
            # ax2
            #ax2b = ax2.twinx()   
            if mov_avg: # plot moving average
                n=50
                #v_plot = (temp_e1_val-np.min(temp_e1_val))/np.max((temp_e1_val-np.min(temp_e1_val)))
                v_plot = so_e1_val
                ax1c.errorbar(so_e1_tim, v_plot, yerr = so_e1_val_err, fmt='o', ecolor = 'r',color='r')#'-', color='g', label='lake temperature', alpha = 1.)
                ax1c.plot([], [], 'or', label = 'CO2')#'-', color='g', label='lake temperature', alpha = 1.)
                #matplotlib.pyplot.errorbar(x, y, yerr=None
            else:
                #ax2b.plot(temp_e1_tim, temp_e1_val, '-', color='g', label='temperature')
                #ax2.set_ylim([-40,40])
                #plt.show()
                v_plot = so_e1_val
                #par1.plot(so_e1_tim, v_plot, '-', color='g', label='lake temperature', alpha = 1.)
                ax1c.errorbar(so_e1_tim, v_plot, yerr = so_e1_val_err)
            #
            ax1c.set_ylabel('CO2 [kg/s]')
            # if max(v_plot) > 20:
            #     ax1c.set_ylim([0,max(v_plot)])
            # else:
            #     ax1c.set_ylim([0,20])
        ax0.legend(loc = 1) 
        
        ax0.set_xlim([ti_e1-month*3,tf_e1])
        dates = [tf_e1-i*year for i in range(14)]
        #ax0.set_xticks(dates)
        #look_back = 365.25+14
        #look_front = 0
        #ax0.set_xticks([tf_e1 - 365.25*day*i for i in range(int((look_back+look_front)/365.25)+1)])
        #ax0b.set_xlim([ti_e1-month*3,tf_e1])

    # plot 1: temp cycle Ruapehu 2012
    if True:
        if True: # temp
            #
            #ti_e1 = datetimeify('2012-06-01') 
            ti_e1 = datetimeify('2012-04-15') 
            tf_e1 = datetimeify('2013-08-01') 
            sta = 'FWVZ'
            ## import events
            path = 'C:\\Users\\aar135\\codes_local_disk\\volc_forecast_tl\\volc_forecast_tl\\features\\lake_data\\'+sta+'\\selection\\'
            if auto_picked:
                path = 'C:\\Users\\aar135\\codes_local_disk\\volc_forecast_tl\\volc_forecast_tl\\features\\lake_data\\corr_dsar_ratevar\\'
            dates =[]
            path_dates = path+sta+'_dates_missed_events_from_dsar_median_rv_cc.txt'
            path_dates = path+sta+'_dates_max_CC_missed_events_from_dsar_median_rv.txt'

            path_dates_filt_off = path+sta+'_dates_max_CC_missed_events_from_dsar_median_rv_without_filt_peak.txt'
            path_dates_filt_on = path+sta+'_dates_max_CC_missed_events_from_dsar_median_rv_with_filt_peak - Copy.txt'
            path_dates_filt_on = path+sta+'_dates_max_CC_missed_events_from_dsar_median_rv_with_filt_peak_man_refine.txt'
            #
            date_events = []
            cc_events = []
            max_events = []
            with open(path_dates_filt_on,'r') as fp:
                for ln in fp.readlines():
                    _d, _cc, _mx =ln.rstrip().split(',')
                    date_events.append(datetimeify(_d))
                    cc_events.append(_cc)
                    max_events.append(_mx)
                #date_events = [datetimeify(ln.rstrip()) for ln in fp.readlines()]
            #
            date_events_filt_off = []
            cc_events_filt_off = []
            max_events_filt_off = []
            with open(path_dates_filt_off,'r') as fp:
                for ln in fp.readlines():
                    _d, _cc, _mx =ln.rstrip().split(',')
                    date_events_filt_off.append(datetimeify(_d))
                    cc_events_filt_off.append(_cc)
                    max_events_filt_off.append(_mx)
                #date_events = [datetimeify(ln.rstrip()) for ln in fp.readlines()]

            # import temp data
            path = '..'+os.sep+'data'+os.sep+"RU001_temp_data.csv"
            pd_temp = pd.read_csv(path, index_col=1)
            if utc_0:
                pd_temp.index = [datetimeify(pd_temp.index[i])-6*hour for i in range(len(pd_temp.index))]
            else:
                pd_temp.index = [datetimeify(pd_temp.index[i]) for i in range(len(pd_temp.index))]
            # plot data in axis twin axis
            # Trim the data
            temp_e1_tim = pd_temp[ti_e1: tf_e1].index.values

            #temp_e1_tim=to_nztimezone(temp_e1_tim)
            #
            temp_e1_val = pd_temp[ti_e1: tf_e1].loc[:,' t (C)'].values
            # ax2
            #ax2b = ax2.twinx()   
            if mov_avg: # plot moving average
                n=50
                #v_plot = (temp_e1_val-np.min(temp_e1_val))/np.max((temp_e1_val-np.min(temp_e1_val)))
                v_plot = temp_e1_val
                ax1.plot(temp_e1_tim, v_plot, '-', color='g', label='lake temperature', alpha = 1.)
                #
                #ax.plot(temp_e1_tim[:-n+1], moving_average(temp_e1_val, n=n), '--', color='k')#, label='temp. mov. avg.')
                ax1.plot(temp_e1_tim[n-1-20:-20], moving_average(v_plot[::-1], n=n)[::-1], '--', color='k')#, label='lake temperature')
            else:
                #ax2b.plot(temp_e1_tim, temp_e1_val, '-', color='g', label='temperature')
                #ax2.set_ylim([-40,40])
                #plt.show()
                v_plot = temp_e1_val
                ax1.plot(temp_e1_tim, v_plot, '-', color='g', label='lake temperature', alpha = 1.)
            ax1.set_ylabel('Temperature °C')
            ax1.set_ylim([10,50])
            #ax2b.set_ylabel('temperature C')   
            #ax.legend(loc = 2)   
            ## plot event

            for d_events in date_events_filt_off:
                if d_events > ti_e1 and d_events <= tf_e1:
                    ax1.axvline(x=d_events, color='gray', ls='--')#, lw = 14)
                    #ax.axvline(x=cycle_date_mid[i], color='gray', ls='--')#, lw = 14)
            for d_events in date_events:
                if d_events > ti_e1 and d_events <= tf_e1:
                    ax1.axvline(x=d_events, color='k', ls='--', lw = 3)
                    #ax1.axvline(x=cycle_date_mid[i], color='blue', ls='-')#, lw = 14)
            ax1.plot([],[], color='gray', ls='--', label = 'sealing')
            ax1.plot([],[], color='k', ls='--', lw = 3, label = 'sealing + fluid release')
            #ax1.plot([],[], color='blue', ls='-', label = 'mid heat/cool cycle')
            ax1.legend(loc = 7)   
        if True: # so2
            ax1b = ax1.twinx()
            # import temp data
            path = '..'+os.sep+'data'+os.sep+"RU000_SO2-flux-a_data.csv"
            pd_so = pd.read_csv(path, index_col=1)
            if utc_0:
                pd_so.index = [datetimeify(pd_so.index[i])-6*hour for i in range(len(pd_so.index))]
            else:
                pd_so.index = [datetimeify(pd_so.index[i]) for i in range(len(pd_so.index))]
            # plot data in axis twin axis
            # Trim the data
            so_e1_tim = pd_so[ti_e1: tf_e1].index.values
            #temp_e1_tim=to_nztimezone(temp_e1_tim)
            so_e1_val = pd_so[ti_e1: tf_e1].loc[:,' SO2-flux-a (kg/s)'].values
            so_e1_val_err = pd_so[ti_e1: tf_e1].loc[:,' error (kg/s)'].values
            # ax2
            #ax2b = ax2.twinx()   
            if mov_avg: # plot moving average
                n=50
                #v_plot = (temp_e1_val-np.min(temp_e1_val))/np.max((temp_e1_val-np.min(temp_e1_val)))
                v_plot = so_e1_val
                ax1b.errorbar(so_e1_tim, v_plot, yerr = so_e1_val_err, fmt='o', ecolor = 'b',color='b')#'-', color='g', label='lake temperature', alpha = 1.)
                #matplotlib.pyplot.errorbar(x, y, yerr=None
                ax1.plot([], [], 'ob', label = 'SO2')
            else:
                #ax2b.plot(temp_e1_tim, temp_e1_val, '-', color='g', label='temperature')
                #ax2.set_ylim([-40,40])
                #plt.show()
                v_plot = so_e1_val
                #par1.plot(so_e1_tim, v_plot, '-', color='g', label='lake temperature', alpha = 1.)
                ax1b.errorbar(so_e1_tim, v_plot, yerr = so_e1_val_err)
            #
            ax1b.set_ylabel('SO2 [kg/s]')
            # if max(v_plot) > 2:
            #     ax2b.set_ylim([0,max(v_plot)])
            # else:
            #     ax2b.set_ylim([0,2])
            ax1b.set_ylim([0,5])
            #ax.grid()
        if False: # co2
            ax1c = ax1.twinx()
            # Offset the right spine of twin2.  The ticks and label have already been
            # placed on the right by twinx above.
            #ax1c.spines.right.set_position(("axes", 1.2))
            # import temp data
            path = '..'+os.sep+'data'+os.sep+"RU000_CO2-flux-a_data.csv"
            pd_so = pd.read_csv(path, index_col=1)
            if utc_0:
                pd_so.index = [datetimeify(pd_so.index[i])-6*hour for i in range(len(pd_so.index))]
            else:
                pd_so.index = [datetimeify(pd_so.index[i]) for i in range(len(pd_so.index))]
            # plot data in axis twin axis
            # Trim the data
            so_e1_tim = pd_so[ti_e1: tf_e1].index.values
            #temp_e1_tim=to_nztimezone(temp_e1_tim)
            so_e1_val = pd_so[ti_e1: tf_e1].loc[:,' CO2-flux-a (kg/s)'].values
            so_e1_val_err = pd_so[ti_e1: tf_e1].loc[:,' error (kg/s)'].values
            # ax2
            #ax2b = ax2.twinx()   
            if mov_avg: # plot moving average
                n=50
                #v_plot = (temp_e1_val-np.min(temp_e1_val))/np.max((temp_e1_val-np.min(temp_e1_val)))
                v_plot = so_e1_val
                ax1c.errorbar(so_e1_tim, v_plot, yerr = so_e1_val_err, fmt='o', ecolor = 'r',color='r')#'-', color='g', label='lake temperature', alpha = 1.)
                ax1c.plot([], [], 'or', label = 'CO2')#'-', color='g', label='lake temperature', alpha = 1.)
                #matplotlib.pyplot.errorbar(x, y, yerr=None
            else:
                #ax2b.plot(temp_e1_tim, temp_e1_val, '-', color='g', label='temperature')
                #ax2.set_ylim([-40,40])
                #plt.show()
                v_plot = so_e1_val
                #par1.plot(so_e1_tim, v_plot, '-', color='g', label='lake temperature', alpha = 1.)
                ax1c.errorbar(so_e1_tim, v_plot, yerr = so_e1_val_err)
            #
            ax1c.set_ylabel('CO2 [kg/s]')
            # if max(v_plot) > 20:
            #     ax1c.set_ylim([0,max(v_plot)])
            # else:
            #     ax1c.set_ylim([0,20])
        ax1.legend(loc = 2) 
        
    # plot 2: temp cycle Ruapehu 2016
    if True:
        if True: # temp
            #
            #ti_e1 = datetimeify('2015-09-01') 
            ti_e1 = datetimeify('2015-08-01') 
            tf_e1 = datetimeify('2016-08-01') 
            sta = 'FWVZ'
            ## import events
            path = 'C:\\Users\\aar135\\codes_local_disk\\volc_forecast_tl\\volc_forecast_tl\\features\\lake_data\\'+sta+'\\selection\\'
            if auto_picked:
                path = 'C:\\Users\\aar135\\codes_local_disk\\volc_forecast_tl\\volc_forecast_tl\\features\\lake_data\\corr_dsar_ratevar\\'
            dates =[]
            path_dates = path+sta+'_dates_missed_events_from_dsar_median_rv_cc.txt'
            path_dates = path+sta+'_dates_max_CC_missed_events_from_dsar_median_rv.txt'

            path_dates_filt_off = path+sta+'_dates_max_CC_missed_events_from_dsar_median_rv_without_filt_peak.txt'
            path_dates_filt_on = path+sta+'_dates_max_CC_missed_events_from_dsar_median_rv_with_filt_peak - Copy.txt'
            path_dates_filt_on = path+sta+'_dates_max_CC_missed_events_from_dsar_median_rv_with_filt_peak_man_refine.txt'
            #
            date_events = []
            cc_events = []
            max_events = []
            with open(path_dates_filt_on,'r') as fp:
                for ln in fp.readlines():
                    _d, _cc, _mx =ln.rstrip().split(',')
                    date_events.append(datetimeify(_d))
                    cc_events.append(_cc)
                    max_events.append(_mx)
                #date_events = [datetimeify(ln.rstrip()) for ln in fp.readlines()]
            #
            date_events_filt_off = []
            cc_events_filt_off = []
            max_events_filt_off = []
            with open(path_dates_filt_off,'r') as fp:
                for ln in fp.readlines():
                    _d, _cc, _mx =ln.rstrip().split(',')
                    date_events_filt_off.append(datetimeify(_d))
                    cc_events_filt_off.append(_cc)
                    max_events_filt_off.append(_mx)
                #date_events = [datetimeify(ln.rstrip()) for ln in fp.readlines()]
    

            # import temp data
            path = '..'+os.sep+'data'+os.sep+"RU001_temp_data.csv"
            pd_temp = pd.read_csv(path, index_col=1)
            if utc_0:
                pd_temp.index = [datetimeify(pd_temp.index[i])-6*hour for i in range(len(pd_temp.index))]
            else:
                pd_temp.index = [datetimeify(pd_temp.index[i]) for i in range(len(pd_temp.index))]
            # plot data in axis twin axis
            # Trim the data
            temp_e1_tim = pd_temp[ti_e1: tf_e1].index.values

            #temp_e1_tim=to_nztimezone(temp_e1_tim)
            #
            temp_e1_val = pd_temp[ti_e1: tf_e1].loc[:,' t (C)'].values
            # ax2
            #ax2b = ax2.twinx()   
            if mov_avg: # plot moving average
                n=50
                #v_plot = (temp_e1_val-np.min(temp_e1_val))/np.max((temp_e1_val-np.min(temp_e1_val)))
                v_plot = temp_e1_val
                ax2.plot(temp_e1_tim, v_plot, '-', color='g', label='lake temperature', alpha = 1.)
                #
                #ax.plot(temp_e1_tim[:-n+1], moving_average(temp_e1_val, n=n), '--', color='k')#, label='temp. mov. avg.')
                ax2.plot(temp_e1_tim[n-1-20:-20], moving_average(v_plot[::-1], n=n)[::-1], '--', color='k')#, label='lake temperature')
            else:
                #ax2b.plot(temp_e1_tim, temp_e1_val, '-', color='g', label='temperature')
                #ax2.set_ylim([-40,40])
                #plt.show()
                v_plot = temp_e1_val
                ax2.plot(temp_e1_tim, v_plot, '-', color='g', label='lake temperature', alpha = 1.)
            ax2.set_ylabel('Temperature °C')
            ax2.set_ylim([10,50])
            #ax2b.set_ylabel('temperature C')   
            #ax.legend(loc = 2)   
            ## plot event

            for d_events in date_events_filt_off:
                if d_events > ti_e1 and d_events <= tf_e1:
                    ax2.axvline(x=d_events, color='gray', ls='--')#, lw = 14)
                    #ax.axvline(x=cycle_date_mid[i], color='gray', ls='--')#, lw = 14)
            for d_events in date_events:
                if d_events > ti_e1 and d_events <= tf_e1:
                    ax2.axvline(x=d_events, color='k', ls='--', lw = 3)
                    #ax1.axvline(x=cycle_date_mid[i], color='blue', ls='-')#, lw = 14)
            ax2.plot([],[], color='gray', ls='--', label = 'sealing')
            ax2.plot([],[], color='black', ls='--', lw = 3, label = 'sealing + fluid release')
            #ax1.plot([],[], color='blue', ls='-', label = 'mid heat/cool cycle')
            ax2.legend(loc = 2)   
        if True: # so2
            ax2b = ax2.twinx()
            # import temp data
            path = '..'+os.sep+'data'+os.sep+"RU000_SO2-flux-a_data.csv"
            pd_so = pd.read_csv(path, index_col=1)
            if utc_0:
                pd_so.index = [datetimeify(pd_so.index[i])-6*hour for i in range(len(pd_so.index))]
            else:
                pd_so.index = [datetimeify(pd_so.index[i]) for i in range(len(pd_so.index))]
            # plot data in axis twin axis
            # Trim the data
            so_e1_tim = pd_so[ti_e1: tf_e1].index.values
            #temp_e1_tim=to_nztimezone(temp_e1_tim)
            so_e1_val = pd_so[ti_e1: tf_e1].loc[:,' SO2-flux-a (kg/s)'].values
            so_e1_val_err = pd_so[ti_e1: tf_e1].loc[:,' error (kg/s)'].values
            # ax2
            #ax2b = ax2.twinx()   
            if mov_avg: # plot moving average
                n=50
                #v_plot = (temp_e1_val-np.min(temp_e1_val))/np.max((temp_e1_val-np.min(temp_e1_val)))
                v_plot = so_e1_val
                ax2b.errorbar(so_e1_tim, v_plot, yerr = so_e1_val_err, fmt='o', ecolor = 'b',color='b')#'-', color='g', label='lake temperature', alpha = 1.)
                #matplotlib.pyplot.errorbar(x, y, yerr=None
                ax2.plot([], [], 'ob', label = 'SO2')
            else:
                #ax2b.plot(temp_e1_tim, temp_e1_val, '-', color='g', label='temperature')
                #ax2.set_ylim([-40,40])
                #plt.show()
                v_plot = so_e1_val
                #par1.plot(so_e1_tim, v_plot, '-', color='g', label='lake temperature', alpha = 1.)
                ax2b.errorbar(so_e1_tim, v_plot, yerr = so_e1_val_err)
            #
            ax2b.set_ylabel('SO2 [kg/s]')
            # if max(v_plot) > 2:
            #     ax2b.set_ylim([0,max(v_plot)])
            # else:
            #     ax2b.set_ylim([0,2])
            ax2b.set_ylim([0,5])
            #ax.grid()
        if False: # co2
            ax2c = ax2.twinx()
            # Offset the right spine of twin2.  The ticks and label have already been
            # placed on the right by twinx above.
            #ax1c.spines.right.set_position(("axes", 1.2))
            # import temp data
            path = '..'+os.sep+'data'+os.sep+"RU000_CO2-flux-a_data.csv"
            pd_so = pd.read_csv(path, index_col=1)
            if utc_0:
                pd_so.index = [datetimeify(pd_so.index[i])-6*hour for i in range(len(pd_so.index))]
            else:
                pd_so.index = [datetimeify(pd_so.index[i]) for i in range(len(pd_so.index))]
            # plot data in axis twin axis
            # Trim the data
            so_e1_tim = pd_so[ti_e1: tf_e1].index.values
            #temp_e1_tim=to_nztimezone(temp_e1_tim)
            so_e1_val = pd_so[ti_e1: tf_e1].loc[:,' CO2-flux-a (kg/s)'].values
            so_e1_val_err = pd_so[ti_e1: tf_e1].loc[:,' error (kg/s)'].values
            # ax2
            #ax2b = ax2.twinx()   
            if mov_avg: # plot moving average
                n=50
                #v_plot = (temp_e1_val-np.min(temp_e1_val))/np.max((temp_e1_val-np.min(temp_e1_val)))
                v_plot = so_e1_val
                ax2c.errorbar(so_e1_tim, v_plot, yerr = so_e1_val_err, fmt='o', ecolor = 'r',color='r')#'-', color='g', label='lake temperature', alpha = 1.)
                ax2c.plot([], [], 'or', label = 'CO2')#'-', color='g', label='lake temperature', alpha = 1.)
                #matplotlib.pyplot.errorbar(x, y, yerr=None
            else:
                #ax2b.plot(temp_e1_tim, temp_e1_val, '-', color='g', label='temperature')
                #ax2.set_ylim([-40,40])
                #plt.show()
                v_plot = so_e1_val
                #par1.plot(so_e1_tim, v_plot, '-', color='g', label='lake temperature', alpha = 1.)
                ax2c.errorbar(so_e1_tim, v_plot, yerr = so_e1_val_err)
            #
            ax2c.set_ylabel('CO2 [kg/s]')
            # if max(v_plot) > 20:
            #     ax1c.set_ylim([0,max(v_plot)])
            # else:
            #     ax1c.set_ylim([0,20])
        ax2.legend(loc = 2) 
        
    # plot 5: temp cycle Ruapehu 2014
    if True:
        if True: # temp
            #
            #ti_e1 = datetimeify('2012-06-01') 
            ti_e1 = datetimeify('2014-12-01') 
            tf_e1 = datetimeify('2015-09-01') 
            sta = 'FWVZ'
            ## import events
            path = 'C:\\Users\\aar135\\codes_local_disk\\volc_forecast_tl\\volc_forecast_tl\\features\\lake_data\\'+sta+'\\selection\\'
            if auto_picked:
                path = 'C:\\Users\\aar135\\codes_local_disk\\volc_forecast_tl\\volc_forecast_tl\\features\\lake_data\\corr_dsar_ratevar\\'
            dates =[]
            path_dates = path+sta+'_dates_missed_events_from_dsar_median_rv_cc.txt'
            path_dates = path+sta+'_dates_max_CC_missed_events_from_dsar_median_rv.txt'

            path_dates_filt_off = path+sta+'_dates_max_CC_missed_events_from_dsar_median_rv_without_filt_peak.txt'
            path_dates_filt_on = path+sta+'_dates_max_CC_missed_events_from_dsar_median_rv_with_filt_peak - Copy.txt'
            path_dates_filt_on = path+sta+'_dates_max_CC_missed_events_from_dsar_median_rv_with_filt_peak_man_refine.txt'
            #
            date_events = []
            cc_events = []
            max_events = []
            with open(path_dates_filt_on,'r') as fp:
                for ln in fp.readlines():
                    _d, _cc, _mx =ln.rstrip().split(',')
                    date_events.append(datetimeify(_d))
                    cc_events.append(_cc)
                    max_events.append(_mx)
                #date_events = [datetimeify(ln.rstrip()) for ln in fp.readlines()]
            #
            date_events_filt_off = []
            cc_events_filt_off = []
            max_events_filt_off = []
            with open(path_dates_filt_off,'r') as fp:
                for ln in fp.readlines():
                    _d, _cc, _mx =ln.rstrip().split(',')
                    date_events_filt_off.append(datetimeify(_d))
                    cc_events_filt_off.append(_cc)
                    max_events_filt_off.append(_mx)
                #date_events = [datetimeify(ln.rstrip()) for ln in fp.readlines()]

            # import temp data
            path = '..'+os.sep+'data'+os.sep+"RU001_temp_data.csv"
            pd_temp = pd.read_csv(path, index_col=1)
            if utc_0:
                pd_temp.index = [datetimeify(pd_temp.index[i])-6*hour for i in range(len(pd_temp.index))]
            else:
                pd_temp.index = [datetimeify(pd_temp.index[i]) for i in range(len(pd_temp.index))]
            # plot data in axis twin axis
            # Trim the data
            temp_e1_tim = pd_temp[ti_e1: tf_e1].index.values

            #temp_e1_tim=to_nztimezone(temp_e1_tim)
            #
            temp_e1_val = pd_temp[ti_e1: tf_e1].loc[:,' t (C)'].values
            # ax2
            #ax2b = ax2.twinx()   
            if mov_avg: # plot moving average
                n=50
                #v_plot = (temp_e1_val-np.min(temp_e1_val))/np.max((temp_e1_val-np.min(temp_e1_val)))
                v_plot = temp_e1_val
                ax5.plot(temp_e1_tim, v_plot, '-', color='g', label='lake temperature', alpha = 1.)
                #
                #ax.plot(temp_e1_tim[:-n+1], moving_average(temp_e1_val, n=n), '--', color='k')#, label='temp. mov. avg.')
                ax5.plot(temp_e1_tim[n-1-20:-20], moving_average(v_plot[::-1], n=n)[::-1], '--', color='k')#, label='lake temperature')
            else:
                #ax2b.plot(temp_e1_tim, temp_e1_val, '-', color='g', label='temperature')
                #ax2.set_ylim([-40,40])
                #plt.show()
                v_plot = temp_e1_val
                ax5.plot(temp_e1_tim, v_plot, '-', color='g', label='lake temperature', alpha = 1.)
            ax5.set_ylabel('Temperature °C')
            ax5.set_ylim([10,50])
            #ax2b.set_ylabel('temperature C')   
            #ax.legend(loc = 2)   
            ## plot event

            for d_events in date_events_filt_off:
                if d_events > ti_e1 and d_events <= tf_e1:
                    ax5.axvline(x=d_events, color='gray', ls='--')#, lw = 14)
                    #ax.axvline(x=cycle_date_mid[i], color='gray', ls='--')#, lw = 14)
            for d_events in date_events:
                if d_events > ti_e1 and d_events <= tf_e1:
                    ax5.axvline(x=d_events, color='k', ls='--', lw = 3)
                    #ax1.axvline(x=cycle_date_mid[i], color='blue', ls='-')#, lw = 14)
            ax5.plot([],[], color='gray', ls='--', label = 'sealing')
            ax5.plot([],[], color='k', ls='--', lw = 3, label = 'sealing + fluid release')
            #ax1.plot([],[], color='blue', ls='-', label = 'mid heat/cool cycle')
            ax5.legend(loc = 2)   
        if True: # so2
            ax5b = ax5.twinx()
            # import temp data
            path = '..'+os.sep+'data'+os.sep+"RU000_SO2-flux-a_data.csv"
            pd_so = pd.read_csv(path, index_col=1)
            if utc_0:
                pd_so.index = [datetimeify(pd_so.index[i])-6*hour for i in range(len(pd_so.index))]
            else:
                pd_so.index = [datetimeify(pd_so.index[i]) for i in range(len(pd_so.index))]
            # plot data in axis twin axis
            # Trim the data
            so_e1_tim = pd_so[ti_e1: tf_e1].index.values
            #temp_e1_tim=to_nztimezone(temp_e1_tim)
            so_e1_val = pd_so[ti_e1: tf_e1].loc[:,' SO2-flux-a (kg/s)'].values
            so_e1_val_err = pd_so[ti_e1: tf_e1].loc[:,' error (kg/s)'].values
            # ax2
            #ax2b = ax2.twinx()   
            if mov_avg: # plot moving average
                n=50
                #v_plot = (temp_e1_val-np.min(temp_e1_val))/np.max((temp_e1_val-np.min(temp_e1_val)))
                v_plot = so_e1_val
                ax5b.errorbar(so_e1_tim, v_plot, yerr = so_e1_val_err, fmt='o', ecolor = 'b',color='b')#'-', color='g', label='lake temperature', alpha = 1.)
                #matplotlib.pyplot.errorbar(x, y, yerr=None
                ax5.plot([], [], 'ob', label = 'SO2')
            else:
                #ax2b.plot(temp_e1_tim, temp_e1_val, '-', color='g', label='temperature')
                #ax2.set_ylim([-40,40])
                #plt.show()
                v_plot = so_e1_val
                #par1.plot(so_e1_tim, v_plot, '-', color='g', label='lake temperature', alpha = 1.)
                ax5b.errorbar(so_e1_tim, v_plot, yerr = so_e1_val_err)
            #
            ax5b.set_ylabel('SO2 [kg/s]')
            # if max(v_plot) > 2:
            #     ax1b.set_ylim([0,max(v_plot)])
            # else:
            #     ax1b.set_ylim([0,2])
            ax5b.set_ylim([0,5])
            #ax.grid()
        if False: # co2
            ax1c = ax1.twinx()
            # Offset the right spine of twin2.  The ticks and label have already been
            # placed on the right by twinx above.
            #ax1c.spines.right.set_position(("axes", 1.2))
            # import temp data
            path = '..'+os.sep+'data'+os.sep+"RU000_CO2-flux-a_data.csv"
            pd_so = pd.read_csv(path, index_col=1)
            if utc_0:
                pd_so.index = [datetimeify(pd_so.index[i])-6*hour for i in range(len(pd_so.index))]
            else:
                pd_so.index = [datetimeify(pd_so.index[i]) for i in range(len(pd_so.index))]
            # plot data in axis twin axis
            # Trim the data
            so_e1_tim = pd_so[ti_e1: tf_e1].index.values
            #temp_e1_tim=to_nztimezone(temp_e1_tim)
            so_e1_val = pd_so[ti_e1: tf_e1].loc[:,' CO2-flux-a (kg/s)'].values
            so_e1_val_err = pd_so[ti_e1: tf_e1].loc[:,' error (kg/s)'].values
            # ax2
            #ax2b = ax2.twinx()   
            if mov_avg: # plot moving average
                n=50
                #v_plot = (temp_e1_val-np.min(temp_e1_val))/np.max((temp_e1_val-np.min(temp_e1_val)))
                v_plot = so_e1_val
                ax1c.errorbar(so_e1_tim, v_plot, yerr = so_e1_val_err, fmt='o', ecolor = 'r',color='r')#'-', color='g', label='lake temperature', alpha = 1.)
                ax1c.plot([], [], 'or', label = 'CO2')#'-', color='g', label='lake temperature', alpha = 1.)
                #matplotlib.pyplot.errorbar(x, y, yerr=None
            else:
                #ax2b.plot(temp_e1_tim, temp_e1_val, '-', color='g', label='temperature')
                #ax2.set_ylim([-40,40])
                #plt.show()
                v_plot = so_e1_val
                #par1.plot(so_e1_tim, v_plot, '-', color='g', label='lake temperature', alpha = 1.)
                ax1c.errorbar(so_e1_tim, v_plot, yerr = so_e1_val_err)
            #
            ax1c.set_ylabel('CO2 [kg/s]')
            # if max(v_plot) > 20:
            #     ax1c.set_ylim([0,max(v_plot)])
            # else:
            #     ax1c.set_ylim([0,20])
        ax5.legend(loc = 2) 
    
    # plot 6: temp cycle Ruapehu 2016
    if True:
        if True: # temp
            #
            #ti_e1 = datetimeify('2015-09-01') 
            ti_e1 = datetimeify('2019-07-15') 
            tf_e1 = datetimeify('2020-10-01') 
            sta = 'FWVZ'
            ## import events
            path = 'C:\\Users\\aar135\\codes_local_disk\\volc_forecast_tl\\volc_forecast_tl\\features\\lake_data\\'+sta+'\\selection\\'
            if auto_picked:
                path = 'C:\\Users\\aar135\\codes_local_disk\\volc_forecast_tl\\volc_forecast_tl\\features\\lake_data\\corr_dsar_ratevar\\'
            dates =[]
            path_dates = path+sta+'_dates_missed_events_from_dsar_median_rv_cc.txt'
            path_dates = path+sta+'_dates_max_CC_missed_events_from_dsar_median_rv.txt'

            path_dates_filt_off = path+sta+'_dates_max_CC_missed_events_from_dsar_median_rv_without_filt_peak.txt'
            path_dates_filt_on = path+sta+'_dates_max_CC_missed_events_from_dsar_median_rv_with_filt_peak - Copy.txt'
            path_dates_filt_on = path+sta+'_dates_max_CC_missed_events_from_dsar_median_rv_with_filt_peak_man_refine.txt'
            #
            date_events = []
            cc_events = []
            max_events = []
            with open(path_dates_filt_on,'r') as fp:
                for ln in fp.readlines():
                    _d, _cc, _mx =ln.rstrip().split(',')
                    date_events.append(datetimeify(_d))
                    cc_events.append(_cc)
                    max_events.append(_mx)
                #date_events = [datetimeify(ln.rstrip()) for ln in fp.readlines()]
            #
            date_events_filt_off = []
            cc_events_filt_off = []
            max_events_filt_off = []
            with open(path_dates_filt_off,'r') as fp:
                for ln in fp.readlines():
                    _d, _cc, _mx =ln.rstrip().split(',')
                    date_events_filt_off.append(datetimeify(_d))
                    cc_events_filt_off.append(_cc)
                    max_events_filt_off.append(_mx)
                #date_events = [datetimeify(ln.rstrip()) for ln in fp.readlines()]
    

            # import temp data
            path = '..'+os.sep+'data'+os.sep+"RU001_temp_data.csv"
            pd_temp = pd.read_csv(path, index_col=1)
            if utc_0:
                pd_temp.index = [datetimeify(pd_temp.index[i])-6*hour for i in range(len(pd_temp.index))]
            else:
                pd_temp.index = [datetimeify(pd_temp.index[i]) for i in range(len(pd_temp.index))]
            # plot data in axis twin axis
            # Trim the data
            temp_e1_tim = pd_temp[ti_e1: tf_e1].index.values

            #temp_e1_tim=to_nztimezone(temp_e1_tim)
            #
            temp_e1_val = pd_temp[ti_e1: tf_e1].loc[:,' t (C)'].values
            # ax2
            #ax2b = ax2.twinx()   
            if mov_avg: # plot moving average
                n=50
                #v_plot = (temp_e1_val-np.min(temp_e1_val))/np.max((temp_e1_val-np.min(temp_e1_val)))
                v_plot = temp_e1_val
                ax6.plot(temp_e1_tim, v_plot, '-', color='g', label='lake temperature', alpha = 1.)
                #
                #ax.plot(temp_e1_tim[:-n+1], moving_average(temp_e1_val, n=n), '--', color='k')#, label='temp. mov. avg.')
                ax6.plot(temp_e1_tim[n-1-20:-20], moving_average(v_plot[::-1], n=n)[::-1], '--', color='k')#, label='lake temperature')
            else:
                #ax2b.plot(temp_e1_tim, temp_e1_val, '-', color='g', label='temperature')
                #ax2.set_ylim([-40,40])
                #plt.show()
                v_plot = temp_e1_val
                ax6.plot(temp_e1_tim, v_plot, '-', color='g', label='lake temperature', alpha = 1.)
            ax6.set_ylabel('Temperature °C')
            ax6.set_ylim([10,50])
            #ax2b.set_ylabel('temperature C')   
            #ax.legend(loc = 2)   
            ## plot event

            for d_events in date_events_filt_off:
                if d_events > ti_e1 and d_events <= tf_e1:
                    ax6.axvline(x=d_events, color='gray', ls='--')#, lw = 14)
                    #ax.axvline(x=cycle_date_mid[i], color='gray', ls='--')#, lw = 14)
            for d_events in date_events:
                if d_events > ti_e1 and d_events <= tf_e1:
                    ax6.axvline(x=d_events, color='k', ls='--', lw = 3)
                    #ax1.axvline(x=cycle_date_mid[i], color='blue', ls='-')#, lw = 14)
            ax6.plot([],[], color='gray', ls='--', label = 'sealing')
            ax6.plot([],[], color='black', ls='--', lw = 3, label = 'sealing + fluid release')
            #ax1.plot([],[], color='blue', ls='-', label = 'mid heat/cool cycle')
            ax6.legend(loc = 2)   
        if True: # so2
            ax6b = ax6.twinx()
            # import temp data
            path = '..'+os.sep+'data'+os.sep+"RU000_SO2-flux-a_data.csv"
            pd_so = pd.read_csv(path, index_col=1)
            if utc_0:
                pd_so.index = [datetimeify(pd_so.index[i])-6*hour for i in range(len(pd_so.index))]
            else:
                pd_so.index = [datetimeify(pd_so.index[i]) for i in range(len(pd_so.index))]
            # plot data in axis twin axis
            # Trim the data
            so_e1_tim = pd_so[ti_e1: tf_e1].index.values
            #temp_e1_tim=to_nztimezone(temp_e1_tim)
            so_e1_val = pd_so[ti_e1: tf_e1].loc[:,' SO2-flux-a (kg/s)'].values
            so_e1_val_err = pd_so[ti_e1: tf_e1].loc[:,' error (kg/s)'].values
            # ax2
            #ax2b = ax2.twinx()   
            if mov_avg: # plot moving average
                n=50
                #v_plot = (temp_e1_val-np.min(temp_e1_val))/np.max((temp_e1_val-np.min(temp_e1_val)))
                v_plot = so_e1_val
                ax6b.errorbar(so_e1_tim, v_plot, yerr = so_e1_val_err, fmt='o', ecolor = 'b',color='b')#'-', color='g', label='lake temperature', alpha = 1.)
                #matplotlib.pyplot.errorbar(x, y, yerr=None
                ax6.plot([], [], 'ob', label = 'SO2')
            else:
                #ax2b.plot(temp_e1_tim, temp_e1_val, '-', color='g', label='temperature')
                #ax2.set_ylim([-40,40])
                #plt.show()
                v_plot = so_e1_val
                #par1.plot(so_e1_tim, v_plot, '-', color='g', label='lake temperature', alpha = 1.)
                ax6b.errorbar(so_e1_tim, v_plot, yerr = so_e1_val_err)
            #
            ax6b.set_ylabel('SO2 [kg/s]')
            # if max(v_plot) > 2:
            #     ax2b.set_ylim([0,max(v_plot)])
            # else:
            #     ax2b.set_ylim([0,2])
            ax6b.set_ylim([0,5])
            #ax.grid()
        if False: # co2
            ax2c = ax2.twinx()
            # Offset the right spine of twin2.  The ticks and label have already been
            # placed on the right by twinx above.
            #ax1c.spines.right.set_position(("axes", 1.2))
            # import temp data
            path = '..'+os.sep+'data'+os.sep+"RU000_CO2-flux-a_data.csv"
            pd_so = pd.read_csv(path, index_col=1)
            if utc_0:
                pd_so.index = [datetimeify(pd_so.index[i])-6*hour for i in range(len(pd_so.index))]
            else:
                pd_so.index = [datetimeify(pd_so.index[i]) for i in range(len(pd_so.index))]
            # plot data in axis twin axis
            # Trim the data
            so_e1_tim = pd_so[ti_e1: tf_e1].index.values
            #temp_e1_tim=to_nztimezone(temp_e1_tim)
            so_e1_val = pd_so[ti_e1: tf_e1].loc[:,' CO2-flux-a (kg/s)'].values
            so_e1_val_err = pd_so[ti_e1: tf_e1].loc[:,' error (kg/s)'].values
            # ax2
            #ax2b = ax2.twinx()   
            if mov_avg: # plot moving average
                n=50
                #v_plot = (temp_e1_val-np.min(temp_e1_val))/np.max((temp_e1_val-np.min(temp_e1_val)))
                v_plot = so_e1_val
                ax2c.errorbar(so_e1_tim, v_plot, yerr = so_e1_val_err, fmt='o', ecolor = 'r',color='r')#'-', color='g', label='lake temperature', alpha = 1.)
                ax2c.plot([], [], 'or', label = 'CO2')#'-', color='g', label='lake temperature', alpha = 1.)
                #matplotlib.pyplot.errorbar(x, y, yerr=None
            else:
                #ax2b.plot(temp_e1_tim, temp_e1_val, '-', color='g', label='temperature')
                #ax2.set_ylim([-40,40])
                #plt.show()
                v_plot = so_e1_val
                #par1.plot(so_e1_tim, v_plot, '-', color='g', label='lake temperature', alpha = 1.)
                ax2c.errorbar(so_e1_tim, v_plot, yerr = so_e1_val_err)
            #
            ax2c.set_ylabel('CO2 [kg/s]')
            # if max(v_plot) > 20:
            #     ax1c.set_ylim([0,max(v_plot)])
            # else:
            #     ax1c.set_ylim([0,20])
        ax6.legend(loc = 2) 
        ax6.set_title('Ruapehu Temperature cycle 2019-20') 

    # plot 3: POS 2013
    if True: # temp
        #
        ti_e1 = datetimeify('2012-12-25') 
        tf_e1 = datetimeify('2013-06-14') 
        sta = 'POS'
        ## import events
        path = 'C:\\Users\\aar135\\codes_local_disk\\volc_forecast_tl\\volc_forecast_tl\\features\\lake_data\\'+sta+'\\selection\\'
        if auto_picked:
            path = 'C:\\Users\\aar135\\codes_local_disk\\volc_forecast_tl\\volc_forecast_tl\\features\\lake_data\\corr_dsar_ratevar\\'
        dates =[]
        path_dates = path+sta+'_dates_missed_events_from_dsar_median_rv_cc.txt'
        path_dates = path+sta+'_dates_max_CC_missed_events_from_dsar_median_rv.txt'

        path_dates_filt_off = path+sta+'_dates_max_CC_missed_events_from_dsar_median_rv_without_filt_peak.txt'
        path_dates_filt_on = path+sta+'_dates_max_CC_missed_events_from_dsar_median_rv_with_filt_peak - Copy.txt'
        #
        date_events = []
        cc_events = []
        max_events = []
        with open(path_dates_filt_on,'r') as fp:
            for ln in fp.readlines():
                _d, _cc, _mx =ln.rstrip().split(',')
                date_events.append(datetimeify(_d))
                cc_events.append(_cc)
                max_events.append(_mx)
            #date_events = [datetimeify(ln.rstrip()) for ln in fp.readlines()]
        #
        date_events_filt_off = []
        cc_events_filt_off = []
        max_events_filt_off = []
        with open(path_dates_filt_off,'r') as fp:
            for ln in fp.readlines():
                _d, _cc, _mx =ln.rstrip().split(',')
                date_events_filt_off.append(datetimeify(_d))
                cc_events_filt_off.append(_cc)
                max_events_filt_off.append(_mx)
            #date_events = [datetimeify(ln.rstrip()) for ln in fp.readlines()]


        # import temp data
        path = '..'+os.sep+'data'+os.sep+"RU001_temp_data.csv"
        pd_temp = pd.read_csv(path, index_col=1)
        if utc_0:
            pd_temp.index = [datetimeify(pd_temp.index[i])-6*hour for i in range(len(pd_temp.index))]
        else:
            pd_temp.index = [datetimeify(pd_temp.index[i]) for i in range(len(pd_temp.index))]
        # plot data in axis twin axis
        # Trim the data
        temp_e1_tim = pd_temp[ti_e1: tf_e1].index.values

        #temp_e1_tim=to_nztimezone(temp_e1_tim)
        #
        temp_e1_val = pd_temp[ti_e1: tf_e1].loc[:,' t (C)'].values
        # ax2
        #ax2b = ax2.twinx()   
        if mov_avg: # plot moving average
            n=50
            #v_plot = (temp_e1_val-np.min(temp_e1_val))/np.max((temp_e1_val-np.min(temp_e1_val)))
            v_plot = temp_e1_val
            ax3.plot(temp_e1_tim, v_plot, '-', color='g', label='lake temperature', alpha = 1.)
            #
            #ax.plot(temp_e1_tim[:-n+1], moving_average(temp_e1_val, n=n), '--', color='k')#, label='temp. mov. avg.')
            ax3.plot(temp_e1_tim[n-1-20:-20], moving_average(v_plot[::-1], n=n)[::-1], '--', color='k')#, label='lake temperature')
        else:
            #ax2b.plot(temp_e1_tim, temp_e1_val, '-', color='g', label='temperature')
            #ax2.set_ylim([-40,40])
            #plt.show()
            v_plot = temp_e1_val
            ax3.plot(temp_e1_tim, v_plot, '-', color='g', label='lake temperature', alpha = 1.)
        ax3.set_ylabel('Temperature °C')
        ax3.set_ylim([20,40])
        #ax2b.set_ylabel('temperature C')   
        #ax.legend(loc = 2)   
        ## plot event

        for d_events in date_events_filt_off:
            if d_events > ti_e1 and d_events <= tf_e1:
                ax3.axvline(x=d_events, color='gray', ls='--')#, lw = 14)
                #ax.axvline(x=cycle_date_mid[i], color='gray', ls='--')#, lw = 14)
        for d_events in date_events:
            if d_events > ti_e1 and d_events <= tf_e1:
                ax3.axvline(x=d_events, color='k', ls='--', lw = 3)
                #ax1.axvline(x=cycle_date_mid[i], color='blue', ls='-')#, lw = 14)
        ax3.plot([],[], color='gray', ls='--', label = 'sealing')
        ax3.plot([],[], color='black', ls='--', lw = 3, label = 'sealing + fluid release')
        #ax1.plot([],[], color='blue', ls='-', label = 'mid heat/cool cycle')
        ax3.legend(loc = 2)   
        ax3.legend(loc = 2)  

    # plot 4: temp cycle Copahue 2016
    if True:
        if True: # temp
            #
            ti_e1 = datetimeify('2020-03-06') 
            tf_e1 = datetimeify('2020-11-25') 
            sta = 'COP'
            ## import events
            path = 'C:\\Users\\aar135\\codes_local_disk\\volc_forecast_tl\\volc_forecast_tl\\features\\lake_data\\'+sta+'\\selection\\'
            if auto_picked:
                path = 'C:\\Users\\aar135\\codes_local_disk\\volc_forecast_tl\\volc_forecast_tl\\features\\lake_data\\corr_dsar_ratevar\\'
            dates =[]
            path_dates = path+sta+'_dates_missed_events_from_dsar_median_rv_cc.txt'
            path_dates = path+sta+'_dates_max_CC_missed_events_from_dsar_median_rv.txt'

            path_dates_filt_off = path+sta+'_dates_max_CC_missed_events_from_dsar_median_rv_without_filt_peak.txt'
            path_dates_filt_on = path+sta+'_dates_max_CC_missed_events_from_dsar_median_rv_with_filt_peak - Copy.txt'
            #
            date_events = []
            cc_events = []
            max_events = []
            with open(path_dates_filt_on,'r') as fp:
                for ln in fp.readlines():
                    _d, _cc, _mx =ln.rstrip().split(',')
                    date_events.append(datetimeify(_d))
                    cc_events.append(_cc)
                    max_events.append(_mx)
                #date_events = [datetimeify(ln.rstrip()) for ln in fp.readlines()]
            #
            date_events_filt_off = []
            cc_events_filt_off = []
            max_events_filt_off = []
            with open(path_dates_filt_off,'r') as fp:
                for ln in fp.readlines():
                    _d, _cc, _mx =ln.rstrip().split(',')
                    date_events_filt_off.append(datetimeify(_d))
                    cc_events_filt_off.append(_cc)
                    max_events_filt_off.append(_mx)
                #date_events = [datetimeify(ln.rstrip()) for ln in fp.readlines()]
    
            # import temp data
            path = '..'+os.sep+'data'+os.sep+"COP_temp_data.csv"
            pd_temp = pd.read_csv(path, index_col=1)
            if utc_0:
                pd_temp.index = [datetimeify(pd_temp.index[i])-6*hour for i in range(len(pd_temp.index))]
            else:
                pd_temp.index = [datetimeify(pd_temp.index[i]) for i in range(len(pd_temp.index))]
            # plot data in axis twin axis
            # Trim the data
            temp_e1_tim = pd_temp[ti_e1: tf_e1].index.values

            #temp_e1_tim=to_nztimezone(temp_e1_tim)
            #
            temp_e1_val = pd_temp[ti_e1: tf_e1].loc[:,' t (C)'].values
            # ax2
            #ax2b = ax2.twinx()   
            if mov_avg: # plot moving average
                n=50
                #v_plot = (temp_e1_val-np.min(temp_e1_val))/np.max((temp_e1_val-np.min(temp_e1_val)))
                v_plot = temp_e1_val
                ax4.plot(temp_e1_tim, v_plot, '.', color='g', label='lake temperature', alpha = 1.)
                ax4.plot(temp_e1_tim, v_plot, '-', color='g', alpha = .6)
                #
                #ax.plot(temp_e1_tim[:-n+1], moving_average(temp_e1_val, n=n), '--', color='k')#, label='temp. mov. avg.')
                ax4.plot(temp_e1_tim[n-1-20:-20], moving_average(v_plot[::-1], n=n)[::-1], '--', color='k')#, label='lake temperature')
            else:
                #ax2b.plot(temp_e1_tim, temp_e1_val, '-', color='g', label='temperature')
                #ax2.set_ylim([-40,40])
                #plt.show()
                v_plot = temp_e1_val
                ax4.plot(temp_e1_tim, v_plot, '.', color='g', label='lake temperature', alpha = 1.)
                ax4.plot(temp_e1_tim, v_plot, '-', color='g', label='lake temperature', alpha = .7)

            ax4.set_ylabel('Temperature °C')
            ax4.set_ylim([20,60])
            #ax2b.set_ylabel('temperature C')   
            #ax.legend(loc = 2)   
            ## plot event

            for d_events in date_events_filt_off:
                if d_events > ti_e1 and d_events <= tf_e1:
                    ax4.axvline(x=d_events, color='gray', ls='--')#, lw = 14)
                    #ax.axvline(x=cycle_date_mid[i], color='gray', ls='--')#, lw = 14)
            for d_events in date_events:
                if d_events > ti_e1 and d_events <= tf_e1:
                    ax4.axvline(x=d_events, color='k', ls='--', lw = 3)
                    #ax1.axvline(x=cycle_date_mid[i], color='blue', ls='-')#, lw = 14)
            ax4.plot([],[], color='gray', ls='--', label = 'sealing')
            ax4.plot([],[], color='black', ls='--', lw = 3, label = 'sealing + fluid release')
            #ax1.plot([],[], color='blue', ls='-', label = 'mid heat/cool cycle')   
        if True: # so2
            ax4b = ax4.twinx()
            # import temp data
            path = '..'+os.sep+'data'+os.sep+"COP_so2_data.csv"
            pd_temp = pd.read_csv(path, index_col=0)
            if utc_0:
                pd_temp.index = [datetimeify(pd_temp.index[i])-6*hour for i in range(len(pd_temp.index))]
            else:
                pd_temp.index = [datetimeify(pd_temp.index[i]) for i in range(len(pd_temp.index))]
            # plot data in axis twin axis
            # Trim the data
            temp_e1_tim = pd_temp[ti_e1: tf_e1].index.values

            #temp_e1_tim=to_nztimezone(temp_e1_tim)
            #
            temp_e1_val = pd_temp[ti_e1: tf_e1].loc[:,'SO2_column_number_density_15km'].values
            # ax2
            #ax2b = ax2.twinx()   
            mov_avg = False
            if mov_avg: # plot moving average
                n=50
                #v_plot = (temp_e1_val-np.min(temp_e1_val))/np.max((temp_e1_val-np.min(temp_e1_val)))
                v_plot = temp_e1_val*1000
                #
                #ax.plot(temp_e1_tim[:-n+1], moving_average(temp_e1_val, n=n), '--', color='k')#, label='temp. mov. avg.')
                ax4b.plot(temp_e1_tim[n-1-20:-20], moving_average(v_plot[::-1], n=n)[::-1], 'ob')#, label='lake temperature')
                ax4.plot([], [], 'ob', label = 'SO2')
            else:
                #ax2b.plot(temp_e1_tim, temp_e1_val, '-', color='g', label='temperature')
                #ax2.set_ylim([-40,40])
                #plt.show()
                v_plot = temp_e1_val
                ax4b.plot(temp_e1_tim, v_plot, 'ob')
                ax4.plot([], [], 'ob', label = 'SO2')
            ax4b.set_ylabel(r'SO2 [mol/m$^2$]')
            ax4b.set_ylim([0,0.005])
        ax4.legend(loc = 2) 

    #####################################################
    ax0.set_title('(a) Ruapehu Temperature cycles') 
    ax1.set_title('(b) Ruapehu Temperature cycle 2012-13') 
    ax2.set_title('(c) Ruapehu Temperature cycle 2015-16') 
    ax5.set_title('(d) Ruapehu Temperature cycle 2014-15') 
    ax6.set_title('(e) Ruapehu Temperature cycle 2019-20')
    ax3.set_title('(f) Kawah Ijen Temperature cycle 2013')
    ax4.set_title('(g) Copahue Temperature cycle 2020') 

    plt.tight_layout()
    plt.show()
    plt.close()

def figure_3():
    '''
    '''
    #fig, (ax1, ax2, ax3, ax4) = plt.subplots(nrows = 2, ncols = 2, figsize = (12,8))
    nrow = 2
    ncol = 2
    fig, ((ax4, ax1), (ax2, ax3)) = plt.subplots(nrows=nrow, ncols=ncol,figsize=(8,8))#(14,4))
    #
    # nrow = 3
    # ncol = 2
    # fig, ((ax01, ax02), (ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=nrow, ncols=ncol,figsize=(8,12))#(14,4))
    #
    roll_mean = False # results with rolling median 
    #
    if False: # plot 01: # events per month
        #auto_picked = True # dates elected automaticaly from dsar median and dsar rv correlations (only with FWVZ)
        '''
        '''
        sta = 'FWVZ'#'POS'#'FWVZ'#'COP'
        ## import events
        path = 'C:\\Users\\aar135\\codes_local_disk\\volc_forecast_tl\\volc_forecast_tl\\features\\lake_data\\'+sta+'\\selection\\'
        if auto_picked:
            path = 'C:\\Users\\aar135\\codes_local_disk\\volc_forecast_tl\\volc_forecast_tl\\features\\lake_data\\corr_dsar_ratevar\\'
        dates =[]
        path_dates = path+sta+'_dates_missed_events_from_dsar_median_rv_cc.txt'
        path_dates = path+sta+'_dates_max_CC_missed_events_from_dsar_median_rv.txt'

        path_dates_filt_off = path+sta+'_dates_max_CC_missed_events_from_dsar_median_rv_without_filt_peak.txt'
        path_dates_filt_on = path+sta+'_dates_max_CC_missed_events_from_dsar_median_rv_with_filt_peak.txt'
        #
        if False: # man picked
            path_mp = 'C:\\Users\\aar135\\codes_local_disk\\volc_forecast_tl\\volc_forecast_tl\\features\\lake_data\\'+sta+'\\selection\\'
            path_dates = path_mp+'FWVZ_temp_eruptive_periods.txt'
        #
        date_events = []
        cc_events = []
        max_events = []
        with open(path_dates_filt_on,'r') as fp:
            for ln in fp.readlines():
                _d, _cc, _mx =ln.rstrip().split(',')
                date_events.append(datetimeify(_d))
                cc_events.append(_cc)
                max_events.append(_mx)
            #date_events = [datetimeify(ln.rstrip()) for ln in fp.readlines()]
        #
        date_events_filt_off = []
        cc_events_filt_off = []
        max_events_filt_off = []
        with open(path_dates_filt_off,'r') as fp:
            for ln in fp.readlines():
                _d, _cc, _mx =ln.rstrip().split(',')
                date_events_filt_off.append(datetimeify(_d))
                cc_events_filt_off.append(_cc)
                max_events_filt_off.append(_mx)
            #date_events = [datetimeify(ln.rstrip()) for ln in fp.readlines()]

        # explosive 
        _jan, _feb , _mar, _apr, _may, _jun, _jul, _agu, _sep, _oct, _nov, _dec = 0,0,0,0,0,0,0,0,0,0,0,0
        for dat in date_events: 
            if dat.month == 1:
                _jan += 1
            if dat.month == 2:
                _feb += 1
            if dat.month == 3:
                _mar += 1
            if dat.month == 4:
                _apr += 1
            if dat.month == 5:
                _may += 1
            if dat.month == 6:
                _jun += 1
            if dat.month == 7:
                _jul += 1
            if dat.month == 8:
                _agu += 1
            if dat.month == 9:
                _sep += 1
            if dat.month == 10:
                _oct += 1
            if dat.month == 11:
                _nov += 1
            if dat.month == 12:
                _dec += 1
        # plot
        months = ['jan', 'feb' , 'mar', 'apr', 'may', 'jun', 'jul', 'agu', 'sep', 'oct', 'nov', 'dec']
        months = [1,2,3,4,5,6,7,8,9,10,11,12]
        n_events_expl = [_jan, _feb , _mar, _apr, _may, _jun, _jul, _agu, _sep, _oct, _nov, _dec]

        # non explosive 
        _jan, _feb , _mar, _apr, _may, _jun, _jul, _agu, _sep, _oct, _nov, _dec = 0,0,0,0,0,0,0,0,0,0,0,0
        for dat in date_events_filt_off: 
            if dat.month == 1:
                _jan += 1
            if dat.month == 2:
                _feb += 1
            if dat.month == 3:
                _mar += 1
            if dat.month == 4:
                _apr += 1
            if dat.month == 5:
                _may += 1
            if dat.month == 6:
                _jun += 1
            if dat.month == 7:
                _jul += 1
            if dat.month == 8:
                _agu += 1
            if dat.month == 9:
                _sep += 1
            if dat.month == 10:
                _oct += 1
            if dat.month == 11:
                _nov += 1
            if dat.month == 12:
                _dec += 1
        # 
        months = ['jan', 'feb' , 'mar', 'apr', 'may', 'jun', 'jul', 'agu', 'sep', 'oct', 'nov', 'dec']
        #months = [1,2,3,4,5,6,7,8,9,10,11,12]
        n_events_non_expl = [_jan, _feb , _mar, _apr, _may, _jun, _jul, _agu, _sep, _oct, _nov, _dec]

        # plot 
        x = np.arange(len(months))
        width = 0.35  # the width of the bars
        ax01.bar(x - width/2, n_events_non_expl, width, label='non-explosive')
        ax01.bar(x + width/2, n_events_expl, width, label='explosive')
        ax01.set_xticks(x, months)
        ax01.legend()
        ax01.set_xticks(x)
        ax01.set_xticklabels(months, rotation=90, ha='right')
        #ax01.set_xlabel('month')
        ax01.set_ylabel('# events')
        ax01.set_title('#Number events per month in Ruapehu')

    if False: # plot 02: # Events per year 
        #auto_picked = True # dates elected automaticaly from dsar median and dsar rv correlations (only with FWVZ)
        '''
        '''
        sta = 'FWVZ'#'POS'#'FWVZ'#'COP'
        ## import events
        path = 'C:\\Users\\aar135\\codes_local_disk\\volc_forecast_tl\\volc_forecast_tl\\features\\lake_data\\'+sta+'\\selection\\'
        if auto_picked:
            path = 'C:\\Users\\aar135\\codes_local_disk\\volc_forecast_tl\\volc_forecast_tl\\features\\lake_data\\corr_dsar_ratevar\\'
        dates =[]
        path_dates = path+sta+'_dates_missed_events_from_dsar_median_rv_cc.txt'
        path_dates = path+sta+'_dates_max_CC_missed_events_from_dsar_median_rv.txt'

        path_dates_filt_off = path+sta+'_dates_max_CC_missed_events_from_dsar_median_rv_without_filt_peak.txt'
        path_dates_filt_on = path+sta+'_dates_max_CC_missed_events_from_dsar_median_rv_with_filt_peak.txt'
        #
        if False: # man picked
            path_mp = 'C:\\Users\\aar135\\codes_local_disk\\volc_forecast_tl\\volc_forecast_tl\\features\\lake_data\\'+sta+'\\selection\\'
            path_dates = path_mp+'FWVZ_temp_eruptive_periods.txt'
        #
        date_events = []
        cc_events = []
        max_events = []
        with open(path_dates_filt_on,'r') as fp:
            for ln in fp.readlines():
                _d, _cc, _mx =ln.rstrip().split(',')
                date_events.append(datetimeify(_d))
                cc_events.append(_cc)
                max_events.append(_mx)
            #date_events = [datetimeify(ln.rstrip()) for ln in fp.readlines()]
        #
        date_events_filt_off = []
        cc_events_filt_off = []
        max_events_filt_off = []
        with open(path_dates_filt_off,'r') as fp:
            for ln in fp.readlines():
                _d, _cc, _mx =ln.rstrip().split(',')
                date_events_filt_off.append(datetimeify(_d))
                cc_events_filt_off.append(_cc)
                max_events_filt_off.append(_mx)
            #date_events = [datetimeify(ln.rstrip()) for ln in fp.readlines()]

        _09, _10 , _11, _12, _13, _14, _15, _16, _17, _18, _19, _20, _21 = 0,0,0,0,0,0,0,0,0,0,0,0,0
        for dat in date_events: 
            if dat.year == 2009:
                _09 += 1
            if dat.year == 2010:
                _10 += 1
            if dat.year == 2011:
                _11 += 1
            if dat.year == 2012:
                _12 += 1
            if dat.year == 2013:
                _13 += 1
            if dat.year == 2014:
                _14 += 1
            if dat.year == 2015:
                _15 += 1
            if dat.year == 2016:
                _16 += 1
            if dat.year == 2017:
                _17 += 1
            if dat.year == 2018:
                _18 += 1
            if dat.year == 2019:
                _19 += 1
            if dat.year == 2020:
                _20 += 1
            if dat.year == 2021:
                _21 += 1
        # 
        year = [2009,2010,2011,2012,2013,2014,2015,2016,2017,2018,2019,2020,2021]
        n_events_expl = [_09, _10 , _11, _12, _13, _14, _15, _16, _17, _18, _19, _20, _21]
        
        # non explosive 
        _09, _10 , _11, _12, _13, _14, _15, _16, _17, _18, _19, _20, _21 = 0,0,0,0,0,0,0,0,0,0,0,0,0
        for dat in date_events_filt_off: 
            if dat.year == 2009:
                _09 += 1
            if dat.year == 2010:
                _10 += 1
            if dat.year == 2011:
                _11 += 1
            if dat.year == 2012:
                _12 += 1
            if dat.year == 2013:
                _13 += 1
            if dat.year == 2014:
                _14 += 1
            if dat.year == 2015:
                _15 += 1
            if dat.year == 2016:
                _16 += 1
            if dat.year == 2017:
                _17 += 1
            if dat.year == 2018:
                _18 += 1
            if dat.year == 2019:
                _19 += 1
            if dat.year == 2020:
                _20 += 1
            if dat.year == 2021:
                _21 += 1
        # 
        year = ['2009','2010','2011','2012','2013','2014','2015','2016','2017','2018','2019','2020','2021']
        n_events_non_expl = [_09, _10 , _11, _12, _13, _14, _15, _16, _17, _18, _19, _20, _21]
        
        # plot 
        x = np.arange(len(year))
        width = 0.35  # the width of the bars
        ax02.bar(x - width/2, n_events_non_expl, width, label='non-explosive')
        ax02.bar(x + width/2, n_events_expl, width, label='explosive')
        #ax02.set_xtickslabels(year)
        ax02.set_xticks(x)
        ax02.set_xticklabels(year, rotation=90, ha='right')
        ax02.legend()
        #ax02.set_xlabel('Year')
        ax02.set_ylabel('# events')
        ax02.set_title('Number events per year in Ruapehu')

    if True: # plot 1
        #auto_picked = True # dates elected automaticaly from dsar median and dsar rv correlations (only with FWVZ)
        sta = 'FWVZ'#'FWVZ' 'POS'
        # read results 
        _path = 'C:\\Users\\aar135\\codes_local_disk\\volc_forecast_tl\\volc_forecast_tl\\features\\lake_data\\corr_dsar_ratevar\\hist_fig\\'
        if roll_mean:
            _path = 'C:\\Users\\aar135\\codes_local_disk\\volc_forecast_tl\\volc_forecast_tl\\features\\lake_data\\corr_dsar_ratevar\\hist_fig\\roll_mean\\'

        #
        plot_4days = False
        plot_non_explosive = True
        #
        if plot_4days:
            path1 = _path +sta+"_temp_dif_rate_ite100_4days.txt"
            path3 = _path +sta+"_temp_dif_rate_ite100_out_4days.txt"
        #
        if plot_non_explosive:
            path1 = _path +sta+"_temp_dif_rate_ite100_20days_nofiltpeak.txt"
            path3 = _path +sta+"_temp_dif_rate_ite100_out_20days_nofiltpeak.txt"
        #    
        path2 = _path +sta+"_temp_dif_rate_ite100_20days.txt"
        path4 = _path +sta+"_temp_dif_rate_ite100_out_20days.txt"
        #
        if True:
            _fls = glob.glob(path1)
            dif_l1 = []
            rate_l1 = []
            rate_days_l1 = []
            for _fl in _fls:
                fl  = np.genfromtxt(_fl, delimiter="\t")
                [dif_l1.append(fl[i][0]) for i in range(len(fl))]
                [rate_l1.append(fl[i][1]) for i in range(len(fl))]
                [rate_days_l1.append(fl[i][2]) for i in range(len(fl))]

            _fls = glob.glob(path2)
            dif_l2 = []
            rate_l2 = []
            rate_days_l2 = []
            for _fl in _fls:
                fl  = np.genfromtxt(_fl, delimiter="\t")
                [dif_l2.append(fl[i][0]) for i in range(len(fl))]
                [rate_l2.append(fl[i][1]) for i in range(len(fl))]
                [rate_days_l2.append(fl[i][2]) for i in range(len(fl))]

            _fls = glob.glob(path3)
            dif_l3 = []
            rate_l3 = []
            rate_days_l3 = []
            for _fl in _fls:
                fl  = np.genfromtxt(_fl, delimiter="\t")
                [dif_l3.append(fl[i][0]) for i in range(len(fl))]
                [rate_l3.append(fl[i][1]) for i in range(len(fl))]
                [rate_days_l3.append(fl[i][2]) for i in range(len(fl))]

            _fls = glob.glob(path4)
            dif_l4 = []
            rate_l4 = []
            rate_days_l4 = []
            for _fl in _fls:
                fl  = np.genfromtxt(_fl, delimiter="\t")
                [dif_l4.append(fl[i][0]) for i in range(len(fl))]
                [rate_l4.append(fl[i][1]) for i in range(len(fl))]
                [rate_days_l4.append(fl[i][2]) for i in range(len(fl))]

        #replace rate > 5
        if False:
            for i, r in enumerate(rate_l1):
                if abs(r)>5:
                    rate_l1[i] = rate_l1[i]/3 

        #dif_l1 = dif_l1[:int(len(dif_l1)/2)]
        #rate_l1 =rate_l1[:int(len(rate_l1)/2)]
        #rate_days_l1 =rate_days_l1[:int(len(rate_days_l1)/2)]
        #
        colors = ['b', 'r', 'gray']#, 'm']
        #_heights, a_bins = np.histogram(pv_samp_in_1)
        
        # select lists 
        #ax.axvline(x=0.05, color = 'k', ls='--', linewidth=1, label = '0.05 threshold')
        if True:
            multi = [dif_l1, dif_l2, dif_l3]# + dif_l4]
            #colors = ['lightgrey', 'r', 'b']
            if plot_non_explosive:
                labels = ['non-explosive events', 'explosive events', 'out eruption']
                colors = ['lightgrey', 'r', 'b']
            else:
                labels = ['4 days back', '20 days back', 'out eruption']
                colors = ['lightgrey', 'r', 'b']
            alpha = [1,1,.5]
            bins = 20#np.linspace(0, 1, 13)
        else:
            multi = [dif_l1, dif_l2]# + dif_l4]
            colors = ['r', 'b']
            labels = ['4 days back', 'out eruption']
            bins = 20#np.linspace(0, 1, 13)
        ax1.hist(multi, bins, color = colors, label=labels, alpha= None, density = True)
        xlim = [-11, 11] #12.5]

        ax1.set_xlim(xlim)
        ax1.set_xlabel('d_temp [°C]')
        ax1.set_ylabel('pdf')
        #ax1.set_title('Ruapehu lake temperature')
        ax1.legend(loc = 1)

    if True: # plot 2
        #auto_picked = True # dates elected automaticaly from dsar median and dsar rv correlations (only with FWVZ)
        sta = 'FWVZ'#'FWVZ' 'POS'
        # read results 
        _path = 'C:\\Users\\aar135\\codes_local_disk\\volc_forecast_tl\\volc_forecast_tl\\features\\lake_data\\corr_dsar_ratevar\\hist_fig\\'
        #
        # path1 = _path +sta+"_level_dif_rate_ite100_4days.txt" 
        # path2 = _path +sta+"_level_dif_rate_ite100_20days.txt"
        # path3 = _path +sta+"_level_dif_rate_ite100_out_4days.txt"
        # path4 = _path +sta+"_level_dif_rate_ite100_out_20days.txt"
        #
        if plot_4days:
            path1 = _path +sta+"_level_dif_rate_ite100_4days.txt"
            path3 = _path +sta+"_level_dif_rate_ite100_out_4days.txt"
        #
        if plot_non_explosive:
            path1 = _path +sta+"_level_dif_rate_ite100_20days_nofiltpeak.txt"
            path3 = _path +sta+"_level_dif_rate_ite100_out_20days_nofiltpeak.txt"
        #    
        path2 = _path +sta+"_level_dif_rate_ite100_20days.txt"
        path4 = _path +sta+"_level_dif_rate_ite100_out_20days.txt"
        #
        if True:
            _fls = glob.glob(path1)
            dif_l1 = []
            rate_l1 = []
            rate_days_l1 = []
            for _fl in _fls:
                fl  = np.genfromtxt(_fl, delimiter="\t")
                [dif_l1.append(fl[i][0]) for i in range(len(fl))]
                [rate_l1.append(fl[i][1]) for i in range(len(fl))]
                [rate_days_l1.append(fl[i][2]) for i in range(len(fl))]

            _fls = glob.glob(path2)
            dif_l2 = []
            rate_l2 = []
            rate_days_l2 = []
            for _fl in _fls:
                fl  = np.genfromtxt(_fl, delimiter="\t")
                [dif_l2.append(fl[i][0]) for i in range(len(fl))]
                [rate_l2.append(fl[i][1]) for i in range(len(fl))]
                [rate_days_l2.append(fl[i][2]) for i in range(len(fl))]

            _fls = glob.glob(path3)
            dif_l3 = []
            rate_l3 = []
            rate_days_l3 = []
            for _fl in _fls:
                fl  = np.genfromtxt(_fl, delimiter="\t")
                [dif_l3.append(fl[i][0]) for i in range(len(fl))]
                [rate_l3.append(fl[i][1]) for i in range(len(fl))]
                [rate_days_l3.append(fl[i][2]) for i in range(len(fl))]

            _fls = glob.glob(path4)
            dif_l4 = []
            rate_l4 = []
            rate_days_l4 = []
            for _fl in _fls:
                fl  = np.genfromtxt(_fl, delimiter="\t")
                [dif_l4.append(fl[i][0]) for i in range(len(fl))]
                [rate_l4.append(fl[i][1]) for i in range(len(fl))]
                [rate_days_l4.append(fl[i][2]) for i in range(len(fl))]

        #replace rate > 5
        if False:
            for i, r in enumerate(rate_l1):
                if abs(r)>5:
                    rate_l1[i] = rate_l1[i]/3 

        #dif_l1 = dif_l1[:int(len(dif_l1)/2)]
        #rate_l1 =rate_l1[:int(len(rate_l1)/2)]
        #rate_days_l1 =rate_days_l1[:int(len(rate_days_l1)/2)]
        #
        #_heights, a_bins = np.histogram(pv_samp_in_1)
        
        # select lists 
        #ax.axvline(x=0.05, color = 'k', ls='--', linewidth=1, label = '0.05 threshold')
        multi = [dif_l1, dif_l2, dif_l3]# + dif_l4]
        #colors = ['lightgrey', 'r', 'b']
        #labels = ['4 days back', '20 days back', 'out eruption']
        #
        if plot_non_explosive:
            labels = ['non-explosive events', 'explosive events', 'out eruption']
            colors = ['lightgrey', 'r', 'b']
        else:
            labels = ['4 days back', '20 days back', 'out eruption']
            colors = ['lightgrey', 'r', 'b']
        #
        bins = 20#np.linspace(0, 1, 13)
        ax2.hist(multi, bins, color = colors, label=labels, density = True)
        xlim = None#[0, 7] #12.5]

        ax2.set_xlabel('d_z [m]')
        ax2.set_xlim([-.35,.35])
        #ax2.set_title('Ruapehu lake level')
        ax2.legend(loc = 1)

    if True: # plot 3
        #auto_picked = True # dates elected automaticaly from dsar median and dsar rv correlations (only with FWVZ)
        sta = 'POS'#'FWVZ' 'POS'
        # read results 
        _path = 'C:\\Users\\aar135\\codes_local_disk\\volc_forecast_tl\\volc_forecast_tl\\features\\lake_data\\corr_dsar_ratevar\\hist_fig\\'
        #
        # path1 = _path +sta+"_temp_dif_rate_ite100_4days.txt"
        # path2 = _path +sta+"_temp_dif_rate_ite100_20days.txt"
        # path3 = _path +sta+"_temp_dif_rate_ite100_out_4days.txt"
        # path4 = _path +sta+"_temp_dif_rate_ite100_out_20days.txt"
        #
        if plot_4days:
            path1 = _path +sta+"_temp_dif_rate_ite100_4days.txt"
            path3 = _path +sta+"_temp_dif_rate_ite100_out_4days.txt"
        #
        if plot_non_explosive:
            path1 = _path +sta+"_temp_dif_rate_ite100_20days_nofiltpeak.txt"
            path3 = _path +sta+"_temp_dif_rate_ite100_out_20days_nofiltpeak.txt"
        #    
        path2 = _path +sta+"_temp_dif_rate_ite100_20days.txt"
        path4 = _path +sta+"_temp_dif_rate_ite100_out_20days.txt"
        #
        if True:
            _fls = glob.glob(path1)
            dif_l1 = []
            rate_l1 = []
            rate_days_l1 = []
            for _fl in _fls:
                fl  = np.genfromtxt(_fl, delimiter="\t")
                [dif_l1.append(fl[i][0]) for i in range(len(fl))]
                [rate_l1.append(fl[i][1]) for i in range(len(fl))]
                [rate_days_l1.append(fl[i][2]) for i in range(len(fl))]

            _fls = glob.glob(path2)
            dif_l2 = []
            rate_l2 = []
            rate_days_l2 = []
            for _fl in _fls:
                fl  = np.genfromtxt(_fl, delimiter="\t")
                [dif_l2.append(fl[i][0]) for i in range(len(fl))]
                [rate_l2.append(fl[i][1]) for i in range(len(fl))]
                [rate_days_l2.append(fl[i][2]) for i in range(len(fl))]

            _fls = glob.glob(path3)
            dif_l3 = []
            rate_l3 = []
            rate_days_l3 = []
            for _fl in _fls:
                fl  = np.genfromtxt(_fl, delimiter="\t")
                [dif_l3.append(fl[i][0]) for i in range(len(fl))]
                [rate_l3.append(fl[i][1]) for i in range(len(fl))]
                [rate_days_l3.append(fl[i][2]) for i in range(len(fl))]

            _fls = glob.glob(path4)
            dif_l4 = []
            rate_l4 = []
            rate_days_l4 = []
            for _fl in _fls:
                fl  = np.genfromtxt(_fl, delimiter="\t")
                [dif_l4.append(fl[i][0]) for i in range(len(fl))]
                [rate_l4.append(fl[i][1]) for i in range(len(fl))]
                [rate_days_l4.append(fl[i][2]) for i in range(len(fl))]

        #replace rate > 5
        if False:
            for i, r in enumerate(rate_l1):
                if abs(r)>5:
                    rate_l1[i] = rate_l1[i]/3 

        #dif_l1 = dif_l1[:int(len(dif_l1)/2)]
        #rate_l1 =rate_l1[:int(len(rate_l1)/2)]
        #rate_days_l1 =rate_days_l1[:int(len(rate_days_l1)/2)]
        #
        colors = ['b', 'r', 'gray']#, 'm']
        #_heights, a_bins = np.histogram(pv_samp_in_1)
        
        # select lists 
        #ax.axvline(x=0.05, color = 'k', ls='--', linewidth=1, label = '0.05 threshold')
        multi = [dif_l1, dif_l2, dif_l3]# + dif_l4]
        #
        if plot_non_explosive:
            labels = ['non-explosive events', 'explosive events', 'out eruption']
            colors = ['lightgrey', 'r', 'b']
        else:
            labels = ['4 days back', '20 days back', 'out eruption']
            colors = ['lightgrey', 'r', 'b']
        #
        alpha = [1,1,.5]
        bins = 12#np.linspace(0, 1, 13)
        ax3.hist(multi, bins, color = colors, label=labels, density = True)
        xlim = None#[0, 7] #12.5]
        #
        ax3.set_xlabel('d_temp [°C]')
        ax3.set_ylabel('pdf')
        #ax3.set_title('Kawa Ijen lake temperature')
        ax3.legend(loc = 1)

    if True: # plot 4
        path =  'C:\\Users\\aar135\\codes_local_disk\\volc_forecast_tl\\volc_forecast_tl\\features\\lake_data\\corr_dsar_ratevar\\hist_fig\\'
        _temp_filt = np.genfromtxt(path+'FWVZ_temp_mean_event_100.txt')
        _temp_nofilt = np.genfromtxt(path+'FWVZ_temp_mean_event_100_nofiltpeak.txt')
        #
        colors = ['r', 'lightgrey']#, 'gray']#
        multi = [_temp_filt, _temp_nofilt]
        labels = ['explosive events','non-explosive events']
        bins = 20
        #
        ax4.hist(multi, bins, color = colors, label=labels, alpha= None, density = True)
        ax4.set_xlabel('temp [°C]')
        #ax4.set_ylabel('freq')
        #ax4.set_title('Ruapehu events temperature')
        ax4.legend(loc = 1)
    #
    ax1.set_title('\nRuapehu lake temperature\ndifference before events')
    ax2.set_title('\nRuapehu lake level\ndifference before events')
    ax3.set_title('Kawa Ijen lake temperature\ndifference before events')
    ax4.set_title('Ruapehu lake temperature\n during events')
    #####################################################
    plt.tight_layout()
    plt.show()
    plt.close()

def figure_4():
    '''
    plot: temperature cycle (1), rsam and dsar before event (2), and lake levels (3)
    '''
    sta = 'FWVZ' 
    if sta == 'FWVZ':
        ffm = False
        server = False # files imported from server to local pc 
        server2 = True # server at uni 
        #plot_erup = False
    #
    look_back = 28
    look_front = 2
    #
    erup_time = datetimeify('2009 07 13 06 30 00')
    #erup_time = datetimeify('2010 09 03 00 00 00')
    #erup_time = datetimeify('2021 03 04 12 00 00')
    #erup_time = datetimeify('2016 11 13 12 00 00')
    #
    #erup_time = datetimeify('2021 09 09 00 00 00')
    #
    day = timedelta(days=1)
    t0 = erup_time - look_back*day#30*day
    t1 = erup_time + look_front*day#hour
    #
    ## plot other data
    temp = True
    level = True
    rainfall = True
    ## 
    plot_erup = True
    # figure
    nrow = 3
    ncol = 1
    #fig, (ax0, ax1, ax2, ax4) = plt.subplots(nrows=nrow, ncols=ncol, figsize=(12,12))#(14,4)) #, ax4)
    fig, (ax0, ax1, ax2) = plt.subplots(nrows=nrow, ncols=ncol, figsize=(12,8))#(14,4)) #, ax4)
    #
    for ax in [ax1,ax2]:#,ax4]: # plot eruption times 
        # plot event 
        #te = datetimeify('2009-07-07') 
        te = datetimeify('2009-07-13 06:30:00')
        ax.axvline(te+0.*day, color='r',linestyle='-', linewidth=7, alpha=.3, zorder = 0)
        ax.plot([], color='r', linestyle='-', linewidth=7, alpha=.3, label = 'fluid release event')
        # plot eruption 
        #te = datetimeify('2009 07 13 06 30 00') 
        #ax.axvline(te+0.22*day, color='r',linestyle='-', linewidth=3, zorder = 4)
        #ax.plot([], color='r', linestyle='-', linewidth=3, label = 'eruption')  
  
    #####################################################
    # subplot cero
    if True:
        #
        sta = 'FWVZ'#'POS'#'FWVZ'#'COP'
        ## import events
        path = 'C:\\Users\\aar135\\codes_local_disk\\volc_forecast_tl\\volc_forecast_tl\\features\\lake_data\\'+sta+'\\selection\\'
        if auto_picked:
            path = 'C:\\Users\\aar135\\codes_local_disk\\volc_forecast_tl\\volc_forecast_tl\\features\\lake_data\\corr_dsar_ratevar\\'
        dates =[]
        path_dates = path+sta+'_dates_missed_events_from_dsar_median_rv_cc.txt'
        path_dates = path+sta+'_dates_max_CC_missed_events_from_dsar_median_rv.txt'

        path_dates_filt_off = path+sta+'_dates_max_CC_missed_events_from_dsar_median_rv_without_filt_peak.txt'
        path_dates_filt_on = path+sta+'_dates_max_CC_missed_events_from_dsar_median_rv_with_filt_peak.txt'
        #
        #
        date_events = []
        cc_events = []
        max_events = []
        # with open(path_dates_filt_on,'r') as fp:
        #     for ln in fp.readlines():
        #         _d, _cc, _mx =ln.rstrip().split(',')
        #         date_events.append(datetimeify(_d))
        #         cc_events.append(_cc)
        #         max_events.append(_mx)
        #     #date_events = [datetimeify(ln.rstrip()) for ln in fp.readlines()]
        # #
        # date_events_filt_off = []
        # cc_events_filt_off = []
        # max_events_filt_off = []
        # with open(path_dates_filt_off,'r') as fp:
        #     for ln in fp.readlines():
        #         _d, _cc, _mx =ln.rstrip().split(',')
        #         date_events_filt_off.append(datetimeify(_d))
        #         cc_events_filt_off.append(_cc)
        #         max_events_filt_off.append(_mx)
        #

        col = ['r','g','b']
        alpha = [.5, 1., 1.]
        thick_line = [1., 3., 3.]
        #
        mov_avg = True # moving average for temp and level data
        utc_0 = True
        # plot temp data
        if True:
            #
            if sta == 'FWVZ':
                #
                ti_e1 = datetimeify('2009 05 10 00 00 00')
                tf_e1 = datetimeify('2010 02 01 00 00 00')
                # import temp data
                path = '..'+os.sep+'data'+os.sep+"RU001_temp_data.csv"
                pd_temp = pd.read_csv(path, index_col=1)
                if utc_0:
                    pd_temp.index = [datetimeify(pd_temp.index[i])-6*hour for i in range(len(pd_temp.index))]
                else:
                    pd_temp.index = [datetimeify(pd_temp.index[i]) for i in range(len(pd_temp.index))]
                # plot data in axis twin axis
                # Trim the data
                temp_e1_tim = pd_temp[ti_e1: tf_e1].index.values

                #temp_e1_tim=to_nztimezone(temp_e1_tim)
                #
                temp_e1_val = pd_temp[ti_e1: tf_e1].loc[:,' t (C)'].values
                # ax2
                #ax2b = ax2.twinx()   
                if mov_avg: # plot moving average
                    n=50
                    #v_plot = (temp_e1_val-np.min(temp_e1_val))/np.max((temp_e1_val-np.min(temp_e1_val)))
                    v_plot = temp_e1_val
                    ax0.plot(temp_e1_tim, v_plot, '-', color='g', label='lake temperature', alpha = 1.)
                    #
                    #ax.plot(temp_e1_tim[:-n+1], moving_average(temp_e1_val, n=n), '--', color='k')#, label='temp. mov. avg.')
                    ax0.plot(temp_e1_tim[n-1-20:-20], moving_average(v_plot[::-1], n=n)[::-1], '--', color='k')#, label='lake temperature')
                else:
                    #ax2b.plot(temp_e1_tim, temp_e1_val, '-', color='g', label='temperature')
                    #ax2.set_ylim([-40,40])
                    #plt.show()
                    v_plot = temp_e1_val
                    ax0.plot(temp_e1_tim, v_plot, '-', color='g', label='lake temperature', alpha = 1.)
                ax0.set_ylabel('Temperature °C')
                ax0.set_ylim([10,50])
                #ax2b.set_ylabel('temperature C')   
                #ax.legend(loc = 2)   
                ## plot event
                # for d_events in date_events_filt_off:
                #     if d_events > ti_e1 and d_events <= tf_e1:
                #         ax0.axvline(x=d_events, color='gray', ls='--', lw = 3)#, lw = 14)
                #         #ax.axvline(x=cycle_date_mid[i], color='gray', ls='--')#, lw = 14)
                # for d_events in date_events:
                #     if d_events > ti_e1 and d_events <= tf_e1:
                #         ax0.axvline(x=d_events, color='k', ls='--', lw = 3)#, lw = 14)
                #         #ax0.axvline(x=cycle_date_mid[i], color='blue', ls='-')#, lw = 14)
                #
                te = datetimeify('2009 07 13 06 30 00') 
                ax0.axvline(te+0*day, color='r',linestyle='-', linewidth=3, alpha = .5, zorder = 4)
                ax0.plot([], color='r', linestyle='-', linewidth=3, label = 'fluid release event') 
                #
                #ax0.plot([],[], color='gray', ls='--', lw = 3, label = 'non-expl events')
                #ax0.plot([],[], color='black', ls='--', lw = 3, label = 'expl events')
                #ax0.plot([],[], color='blue', ls='-', label = 'mid heat/cool cycle')
                ax0.legend(loc = 1)   
                ax0.grid()
                ax0.set_ylim([12,36])
                #plt.show()
    # subplot one: MF, HF, DSAR medians (DSAR yaxis left; MF, HF yaxis rigth). 1/RSAM (normalized)
    if True:
        # features
        fts_yleft = ['zsc2_dsarF__median']
        fts_yrigth = []#['zsc2_dsarF__rate_variance']#['zsc2_dsarF__change_quantiles__f_agg_"var"__isabs_False__qh_0.6__ql_0.4']#['zsc2_mfF__median','zsc2_hfF__median']
        data_yrigth = ['rsam']
        
        #
        col = ['b','b','r']
        alpha = [1., 1., .5]
        thick_line = [2., 6., 1.]
        ax1b = ax1.twinx() 
        for i, ft in enumerate(fts_yleft):
            if True: # load feature (else: cal feature. median or rv)
                
                if 'zsc2_dsarF' in ft:
                    ds = ['zsc2_dsarF'] 
                if server:
                    path_feat_serv = 'C:\\Users\\aar135\\codes_local_disk\\volc_forecast_tl\\features_bkp\\features_server\\'
                    fm_e1 = ForecastModel(window=2., overlap=1., station =  sta,
                        look_forward=2., data_streams=ds, 
                        feature_dir=path_feat_serv, 
                        savefile_type='pkl') 
                elif server2:
                    path_feat_serv = 'U:\\Research\\EruptionForecasting\\eruptions\\features\\'
                    fm_e1 = ForecastModel(window=2., overlap=1., station =  sta,
                        look_forward=2., data_streams=ds, 
                        feature_dir=path_feat_serv, 
                        savefile_type='pkl') 
                else:
                    try:
                        fm_e1 = ForecastModel(window=2., overlap=1., station = sta,
                            look_forward=2., data_streams=ds, savefile_type='csv')
                    except:
                        fm_e1 = ForecastModel(window=2., overlap=1., station = sta,
                            look_forward=2., data_streams=ds, savefile_type='pkl')                    
                ##  
                ft = ft.replace("-",'"')
                # adding multiple Axes objects
                ft_e1 = fm_e1.get_features(ti=t0, tf=t1, n_jobs=1, compute_only_features=[ft])
                # extract values to plot 
                ft_e1_t = ft_e1[0].index.values
                ft_e1_v = ft_e1[0].loc[:,ft]
                #
                v_plot = ft_e1_v

                #v_plot = ft_e1_v-np.min(ft_e1_v)/np.max((ft_e1_v-np.min(ft_e1_v)))
                #v_plot = ft_e1_v/np.max(ft_e1_v)
            else: 
                #
                if 'zsc2_dsarF' in ft:
                    ds = 'zsc2_dsarF'
                if 'zsc2_mfF' in ft:
                    ds = 'zsc2_mfF' 
                if 'zsc2_hfF' in ft:
                    ds = 'zsc2_hfF' 
                # 
                #
                day = timedelta(days=1)
                fm = ForecastModel(window=2., overlap=1., station=sta,
                    look_forward=2., data_streams=[ds], 
                    data_dir=r'C:\Users\aar135\codes_local_disk\volc_forecast_tl\volc_forecast_tl\data'
                    )
                #
                N, M = [2,30]
                df = fm.data.df[t0:t1]
                if 'median' in ft:
                    test = df[ds].rolling(N*24*6).median()[N*24*6:]
                if 'rate_variance' in ft:
                    test = df[ds].rolling(N*24*6).apply(chqv)[N*24*6:]
                #
                #out = test.rolling(archtype.shape[0]).apply(partial(conv, (archtype-archtype.mean())/archtype.std()))       
                #out = out.resample('1D').ffill()
                #
                ft_e1_t = test.index
                v_plot = test.values
            #
            if ft == 'zsc2_dsarF__median':
                ft = 'DSAR median'
            #
            ax1b.plot(ft_e1_t, v_plot, '-', color=col[i], alpha = alpha[i], linewidth = thick_line[i], label=' '+ ft, zorder = 2)
            #
            #
            if ffm: # ffm 
                #ax1b = ax1.twinx() 
                #v_plot = data[data_stream].loc[inds]
                inv_rsam = fm_e1.data.get_data(ti=t0, tf=t1)['rsamF']#.loc[ft_e1_t]
                inv_rsam = 1./inv_rsam
                # normalized it to yaxis rigth 
                inv_rsam = inv_rsam/max(inv_rsam)
                inv_rsam = inv_rsam*0.5*max(v_plot)
                #
                ax1.plot(ft_e1_t, inv_rsam, '-', color= 'gray', linewidth=0.5, markersize=0.5, alpha = 1.)
                ax1.plot([], [], '-', color= 'gray', markersize=1, label='1/RSAM', alpha = 1.0)
                #ax1.set_ylim([0,1])
                #ax1.set_yticks([])
            #
            if False:#plot_erup: # plot vertical lines
                te = datetimeify(erup_time)#fm_e1.data.tes[int(erup[-1:])-1]
                ax1b.axvline(te, color='r',linestyle='-', linewidth=5, alpha=0.5,zorder = 4)
                ax1b.plot([], color='r', linestyle='-', linewidth=5, alpha=0.5, label = 'fluid release event')
            #
            #ax1.legend(loc = 2)
            #
            te = datetimeify(erup_time)#fm_e1.data.tes[int(erup[-1:])-1]
            #ax1.set_xticks([ft_e1[0].index[-1] - 7*day*i for i in range(int(30/7)+1)])#[dat.index.values[0],dat.index.values[-1]])#, ]np.arange(0, len(x)+1, 5))
            
            #ax1b.set_yticks([])
            ax1b.grid()
            ax1b.set_ylabel('DSAR value')        #ax.set_xticks([ft_e1[0].index[-1]-7*day*i +day for i in range(5)])
            #ax1.set_yscale('log') #ax.set_yscale('log')
            #ax.set_xticks([ft_e1[0].index[-1] - 7*day*i for i in range(int(30/7)+1)])#[dat.index.values[0],dat.index.values[-1]])#, ]np.arange(0, len(x)+1, 5))
            #ax1.set_xticks([t1 - 5*day*i for i in range(int(look_back/5)+1)])#[dat.index.values[0],dat.index.values[-1]])#, ]np.arange(0, len(x)+1, 5))
        #     #
        # except:
        #     pass
        if fts_yrigth:
            #ax1b = ax1.twinx() 
            col = ['r','g']
            alpha = [1., .5]
            thick_line = [2.,1.]
            #try: 
            for i, ft in enumerate(fts_yrigth):
                if 'zsc2_dsarF' in ft:
                    ds = 'zsc2_dsarF'
                if 'zsc2_mfF' in ft:
                    ds = 'zsc2_mfF' 
                if 'zsc2_hfF' in ft:
                    ds = 'zsc2_hfF' 
                # 
                if False: # look feature in the prev cacl features (else: calculate feat from data; only for median and rv)
                    if server:
                        path_feat_serv = 'C:\\Users\\aar135\\codes_local_disk\\volc_forecast_tl\\features_bkp\\features_server\\'
                        fm_e1 = ForecastModel(window=2., overlap=1., station =  sta,
                            look_forward=2., data_streams=ds, 
                            feature_dir=path_feat_serv, 
                            savefile_type='pkl') 
                    elif server2:
                        path_feat_serv = 'U:\\Research\\EruptionForecasting\\eruptions\\features\\'
                        fm_e1 = ForecastModel(window=2., overlap=1., station =  sta,
                            look_forward=2., data_streams=ds, 
                            feature_dir=path_feat_serv, 
                            savefile_type='pkl') 
                    else:
                        try:
                            fm_e1 = ForecastModel(window=2., overlap=1., station = sta,
                                look_forward=2., data_streams=ds, savefile_type='csv')
                        except:
                            fm_e1 = ForecastModel(window=2., overlap=1., station = sta,
                                look_forward=2., data_streams=ds, savefile_type='pkl')                    
                    ##  
                    ft = ft.replace("-",'"')
                    
                    ft_e1 = fm_e1.get_features(ti=t0, tf=t1, n_jobs=1, compute_only_features=[ft])
                    # adding multiple Axes objects

                    # extract values to plot 
                    ft_e1_t = ft_e1[0].index.values
                    ft_e1_v = ft_e1[0].loc[:,ft]
                    #
                    v_plot = ft_e1_v

                else:
                    day = timedelta(days=1)
                    fm = ForecastModel(window=2., overlap=1., station=sta,
                        look_forward=2., data_streams=[ds], 
                        data_dir=r'C:\Users\aar135\codes_local_disk\volc_forecast_tl\volc_forecast_tl\data'
                        )
                    #
                    N, M = [2,30]
                    df = fm.data.df[t0:t1]
                    if 'median' in ft:
                        test = df[ds].rolling(N*24*6).median()[N*24*6:]
                    if 'rate_variance' in ft:
                        test = df[ds].rolling(N*24*6).apply(chqv)[N*24*6:]
                    #
                    #out = test.rolling(archtype.shape[0]).apply(partial(conv, (archtype-archtype.mean())/archtype.std()))       
                    #out = out.resample('1D').ffill()
                    #
                    ft_e1_t = test.index
                    v_plot = test.values

                #v_plot = ft_e1_v-np.min(ft_e1_v)/np.max((ft_e1_v-np.min(ft_e1_v)))
                #v_plot = ft_e1_v/np.max(ft_e1_v)
                #
                if ft == 'zsc2_mfF__median':
                    ft = 'nMF median'
                if ft == 'zsc2_hfF__median':
                    ft = 'nHF median'
                #
                ax1.plot(ft_e1_t, v_plot, '-', color=col[i], alpha = alpha[i],label=' '+ ft, zorder = 4)
                #
                ax1.legend(loc = 3)
                #
                te = datetimeify(erup_time)#fm_e1.data.tes[int(erup[-1:])-1]
                #ax1.set_xticks([ft_e1[0].index[-1] - 7*day*i for i in range(int(30/7)+1)])#[dat.index.values[0],dat.index.values[-1]])#, ]np.arange(0, len(x)+1, 5))
                
                #ax1b.set_yticks([])
                ax1.grid()
                ax1.set_ylabel('Feature value')        #ax.set_xticks([ft_e1[0].index[-1]-7*day*i +day for i in range(5)])
                #ax.set_xticks([ft_e1[0].index[-1] - 7*day*i for i in range(int(30/7)+1)])#[dat.index.values[0],dat.index.values[-1]])#, ]np.arange(0, len(x)+1, 5))
                #ax1.set_xticks([t1 - 5*day*i for i in range(int(look_back/5)+1)])#[dat.index.values[0],dat.index.values[-1]])#, ]np.arange(0, len(x)+1, 5))
                #
            #except:
            #    pass

        else:
            pass
            if data_yrigth:
                #
                #ax1b = ax1.twinx() 
                #
                td = TremorData(station = sta)
                #td.update(ti=t0, tf=t1)
                data_streams = data_yrigth#['hf','mf', 'rsam']#, 'dsarF']
                label = ['RSAM','MF','HF','DSAR']
                #label = ['1/RSAM']
                inv = False
                if False:
                    data_streams = ['rsam']
                    label = ['RSAM']

                if type(data_streams) is str:
                    data_streams = [data_streams,]
                if any(['_' in ds for ds in data_streams]):
                    td._compute_transforms()
                #ax.set_xlim(*range)
                # plot data for each year
                norm= False
                _range = [t0,t1]
                log =False
                col_def = None
                data = td.get_data(*_range)
                xi = datetime(year=1,month=1,day=1,hour=0,minute=0,second=0)
                cols = ['gray','r','g','m',[0.5,0.5,0.5],[0.75,0.75,0.75]]
                if inv:
                    cols = ['gray','g','r','m',[0.5,0.5,0.5],[0.75,0.75,0.75]]
                inds = (data.index>=datetimeify(_range[0]))&(data.index<=datetimeify(_range[1]))
                i=0
                for data_stream, col in zip(data_streams,cols):
                    v_plot = data[data_stream].loc[inds]
                    if inv:
                        v_plot = 1/v_plot
                    if log:
                        v_plot = np.log10(v_plot)
                    if norm:
                        v_plot = (v_plot-np.min(v_plot))/np.max((v_plot-np.min(v_plot)))
                    if label:
                        if col_def:
                            ax1.plot(data.index[inds], v_plot, '-', color=col_def, label=label[i], linewidth=1., alpha = 1., zorder = 0)
                        else:
                            #ax1b.plot(data.index[inds], v_plot, '-', color=col, label=label[i], linewidth=1., alpha = 1.0, zorder = 0)
                            ax1.plot(data.index[inds], v_plot, '-', color=col, linewidth=1., alpha = 1., zorder = 1)
                            ax1b.plot([], [], '-', color=col, label=label[i], linewidth=1., alpha = 1., zorder = 1)
                    else:
                        ax1.plot(data.index[inds], v_plot, '-', color=col, label=data_stream, linewidth=2., alpha = .9, zorder = 1)
                    i+=1
                for te in td.tes:
                    if [te>=datetimeify(_range[0]) and te<=datetimeify(_range[1])]:
                        pass
                        #ax.axvline(te, color='k', linestyle='--', linewidth=2, zorder = 0)
                #
                #ax1.plot([], color='k', linestyle='--', linewidth=2, label = 'eruption')
                #ax4.set_xlim(_range)
                #ax1b.legend(loc = 2)
                #ax1b.grid()
                if log:
                    ax1.set_ylabel(' ')
                else:
                    ax1.set_ylabel('RSAM')# \u03BC m/s')
                #ax4.set_xlabel('Time [month-day hour]')
                #ax4.title.set_text('Station '+td.station+' ('+sta_code[td.station]+'): Tremor data')
                #
                #ax4.set_xticks([te - 5*day*i for i in range(int(look_back/5)+1)])#[dat.index.values[0],dat.index.values[-1]])#, ]np.arange(0, len(x)+1, 5))
                #ax4.set_ylim([1e9,1e13])
                ax1.set_yscale('log')
        ax1b.legend(loc = 2)      
        #
    # subplot two: temp data (if any: level and rainfall)
    if True:  
        mov_avg = True # moving average for temp and level data
        # convert to UTC 0
        utc_0 = True
        if utc_0:
            _utc_0 = 0#-13 # hours
        # plot temp data
        if sta == 'FWVZ':
            # plot temperature data 
            if temp:
                try:
                    ti_e1 = t0
                    tf_e1 = t1
                    # import temp data
                    path = '..'+os.sep+'data'+os.sep+"RU001_temp_data.csv"
                    pd_temp = pd.read_csv(path, index_col=1)
                    if utc_0:
                        #pd_temp.index = [datetimeify(pd_temp.index[i])-6*hour for i in range(len(pd_temp.index))]
                        pd_temp.index = [datetimeify(pd_temp.index[i])+_utc_0*hour for i in range(len(pd_temp.index))]
                    else:
                        pd_temp.index = [datetimeify(pd_temp.index[i]) for i in range(len(pd_temp.index))]
                    
                    # plot data in axis twin axis
                    # Trim the data
                    temp_e1_tim = pd_temp[ti_e1: tf_e1].index.values
                    #temp_e1_tim=to_nztimezone(temp_e1_tim)
                    #
                    temp_e1_val = pd_temp[ti_e1: tf_e1].loc[:,' t (C)'].values
                    # ax2
                    #ax2b = ax2.twinx()   
                    if mov_avg: # plot moving average
                        n=30
                        #v_plot = (temp_e1_val-np.min(temp_e1_val))/np.max((temp_e1_val-np.min(temp_e1_val)))
                        v_plot = temp_e1_val
                        ax2.plot(temp_e1_tim, v_plot, '-', color='g', label='lake temperature', alpha = 1.)
                        
                        #
                        #ax.plot(temp_e1_tim[:-n+1], moving_average(temp_e1_val, n=n), '--', color='k')#, label='temp. mov. avg.')
                        ax2.plot(temp_e1_tim[n-1-20:-20], moving_average(v_plot[::-1], n=n)[::-1], '--', color='g')#, label='lake temperature')
                    else:
                        v_plot = temp_e1_val
                        ax2.plot(temp_e1_tim, v_plot, '-', color='g', label='lake temperature', alpha = 1.)
                    ax2.set_ylabel('Temperature °C')
                    #
                    temp_min = min(temp_e1_val)
                    temp_max = max(temp_e1_val)
                    temp_mu = np.mean(temp_e1_tim)
                    temp_sigma = np.std(temp_e1_tim)
                    ax2.set_ylim([temp_mu-3*temp_sigma,temp_mu+3*temp_sigma])
                    #ax2.set_ylabel('temperature C')   
                except:
                    pass

            # plot lake level data
            if level:
                #try:
                if True:
                    ax2b = ax2.twinx()
                    #
                    # import temp data
                    if t0.year >= 2016:# and t0.month >= 3:
                        path = '..'+os.sep+'data'+os.sep+"RU001A_level_data.csv"

                    elif t0.year < 2012:# and t0.month < 7:
                        path = '..'+os.sep+'data'+os.sep+"RU001_level_data.csv"      
                    else:
                        path = '..'+os.sep+'data'+os.sep+"RU001A_level_data_full.csv"
                    #    
                    pd_lev = pd.read_csv(path, index_col=1)
                    if utc_0:
                        #pd_temp.index = [datetimeify(pd_temp.index[i])-6*hour for i in range(len(pd_temp.index))]
                        #pd_temp.index = [datetimeify(pd_temp.index[i])-0*hour for i in range(len(pd_temp.index))]
                        pd_lev.index = [datetimeify(pd_lev.index[i])+_utc_0*hour for i in range(len(pd_lev.index))]
                    else:
                        pd_lev.index = [datetimeify(pd_lev.index[i]) for i in range(len(pd_lev.index))]

                    if t0.year>2010 and t1.year<2016: # rolling median over data
                        N = 2
                        pd_lev = pd_lev[:].rolling(40).median()#[N*24*6:]

                    # plot data in axis twin axis
                    # Trim the data
                    lev_e1_tim = pd_lev[ti_e1: tf_e1].index.values
                    #temp_e1_tim=to_nztimezone(temp_e1_tim)
                    lev_e1_val = pd_lev[ti_e1: tf_e1].loc[:,' z (m)'].values
                    # ax2
                    #ax2b = ax2.twinx()
                    if False:#mov_avg: # plot moving average
                        n=10
                        v_plot = temp_e1_val
                        ax2b.plot(temp_e1_tim, v_plot, '-', color='royalblue', alpha = 1.)
                        ax2.plot([], [], '-', color='royalblue', label='lake level')
                        #ax2b.plot(temp_e1_tim[:-n+1], moving_average(temp_e1_val, n=n), '--', color='k', label='temp. mov. avg.')
                        #v_plot = (temp_e1_val-np.min(temp_e1_val))/np.max((temp_e1_val-np.min(temp_e1_val)))
                        ax2b.plot(temp_e1_tim[n-1-10:-10], moving_average(v_plot[::-1], n=n)[::-1], '--', color='royalblue', label='lake level')
                        #ax2b.plot(temp_e1_tim, v_plot, '-', color='b', alpha = 0.3)
                    else:
                        v_plot = lev_e1_val
                        ax2b.plot(lev_e1_tim, v_plot, '-', color='royalblue', label='lake level')
                    #
                    ax2b.set_ylabel('Lake level cm') 
                    ax2.plot([], [], '-', color='royalblue', label='lake level')

                    if False:#plot_erup: # plot vertical lines
                        te = datetimeify(erup_time)#fm_e1.data.tes[int(erup[-1:])-1]
                        ax2.axvline(te, color='r',linestyle='-', linewidth=3, alpha = 0.5, zorder = 4)
                        #ax2.plot([], color='k', linestyle='--', linewidth=2, label = 'eruption')
                #except:
                #    pass
            
            # plot rainfall data
            if rainfall:
                try:
                    ti_e1 = t0#datetimeify(t0)
                    tf_e1 = t1#datetimeify(t1)
                    #
                    # import temp data
                    path = '..'+os.sep+'data'+os.sep+"_chateau_rain.csv"
                    pd_rf = pd.read_csv(path, index_col=1)
                    pd_rf.index = pd.to_datetime(pd_rf.index, format='%Y%m%d:%H%M')
                    if utc_0:
                        pd_rf.index = [pd_rf.index[i]+_utc_0*hour for i in range(len(pd_rf.index))]

                    # Trim the data
                    rf_e2_tim = pd_rf[ti_e1: tf_e1].index#.values
                    rf_e2_val = pd_rf[ti_e1: tf_e1].loc[:,'Amount(mm)'].values /3
                    # ax2
                    #ax2b = ax2.twinx()
                    v_plot = (rf_e2_val-np.min(rf_e2_val))/np.max((rf_e2_val-np.min(rf_e2_val)))
                    #v_plot = v_plot*5 + 14
                    if temp_max:
                        v_plot = v_plot*(temp_max-temp_min)*0.6 + temp_min
                    ax2.plot(rf_e2_tim, v_plot, '-', color='c', label='rain fall', alpha = 0.6)
                    #ax2b.set_ylabel('temperature C')
                    #ax2b.legend(loc = 1)
                except:
                    pass

        if sta == 'DAM' or sta == 'POS':
            lake = False # no data
            rainfall = False # no data
            try:
                if temp:
                    ti_e1 = t0
                    tf_e1 = t1
                    # import temp data
                    path = '..'+os.sep+'data'+os.sep+"DAM_temp_data.csv"
                    pd_temp = pd.read_csv(path, index_col=1)

                    if utc_0:
                        pd_temp.index = [datetimeify(pd_temp.index[i])-6*hour for i in range(len(pd_temp.index))]
                    else:
                        pd_temp.index = [datetimeify(pd_temp.index[i]) for i in range(len(pd_temp.index))]
                    # plot data in axis twin axis
                    # Trim the data
                    temp_e1_tim = pd_temp[ti_e1: tf_e1].index.values
                    #temp_e1_tim=to_nztimezone(temp_e1_tim)
                    #
                    temp_e1_val = pd_temp[ti_e1: tf_e1].loc[:,' t (C)'].values
                    # ax2
                    #ax2b = ax2.twinx()   
                    if mov_avg: # plot moving average
                        n=50
                        #v_plot = (temp_e1_val-np.min(temp_e1_val))/np.max((temp_e1_val-np.min(temp_e1_val)))
                        v_plot = temp_e1_val
                        ax2.plot(temp_e1_tim, v_plot, '-', color='g', label='lake temperature', alpha = 1.)
                        #
                        #ax.plot(temp_e1_tim[:-n+1], moving_average(temp_e1_val, n=n), '--', color='k')#, label='temp. mov. avg.')
                        _x = temp_e1_tim[n-1-20:-20]
                        _y = moving_average(v_plot[::-1], n=n)[::-1]
                        ax2.plot(_x, _y, '--', color='k')#, label='lake temperature')
                    else:
                        v_plot = temp_e1_val
                        ax2.plot(temp_e1_tim, v_plot, '-', color='g', label='lake temperature', alpha = 1.)
                    ax2.set_ylabel('Temperature °C')
                    
                    _ylim = [min(_y)-1,max(_y)+1] 
                    ax2.set_ylim(_ylim)
                    #ax2.set_ylabel('temperature C')   
            except:
                pass
            if plot_erup: # plot vertical lines
                te = datetimeify(erup_time)#fm_e1.data.tes[int(erup[-1:])-1]
                ax2.axvline(te, color='k',linestyle='--', linewidth=2, zorder = 4)
                #ax2.plot([], color='k', linestyle='--', linewidth=2, label = 'eruption')

        ax2.legend(loc = 2)   
        ax2.grid()
        #ax2.set_xticks([t1 - 5*day*i for i in range(int(look_back/5)+1)])#[dat.index.values[0],dat.index.values[-1]])#, ]np.arange(0, len(x)+1, 5))
    # subplot three: filtered  RSAM, MF, HF datastreams
    if False:
        #
        td = TremorData(station = sta)
        #td.update(ti=t0, tf=t1)
        data_streams = ['rsam','hf', 'mf']#, 'dsarF']
        label = ['nRSAM','nHF','nMF','nDSAR']
        #label = ['1/RSAM']
        inv = False
        if False:
            data_streams = ['hf', 'mf', 'rsam', 'lf', 'vlf']
            label = ['HF','MF','RSAM','LF', 'VLF']

        if type(data_streams) is str:
            data_streams = [data_streams,]
        if any(['_' in ds for ds in data_streams]):
            td._compute_transforms()
        #ax.set_xlim(*range)
        # plot data for each year
        norm= False
        _range = [t0,t1]
        log =False
        col_def = None
        data = td.get_data(*_range)
        xi = datetime(year=1,month=1,day=1,hour=0,minute=0,second=0)
        cols = ['r','g','gray','m',[0.5,0.5,0.5],[0.75,0.75,0.75]]
        if inv:
            cols = ['gray','g','r','m',[0.5,0.5,0.5],[0.75,0.75,0.75]]
        inds = (data.index>=datetimeify(_range[0]))&(data.index<=datetimeify(_range[1]))
        i=0
        for data_stream, col in zip(data_streams,cols):
            v_plot = data[data_stream].loc[inds]
            if inv:
                v_plot = 1/v_plot
            if log:
                v_plot = np.log10(v_plot)
            if norm:
                v_plot = (v_plot-np.min(v_plot))/np.max((v_plot-np.min(v_plot)))
            if label:
                if col_def:
                    ax3.plot(data.index[inds], v_plot, '-', color=col_def, label=label[i], alpha = 1.0)
                else:
                    ax3.plot(data.index[inds], v_plot, '-', color=col, label=label[i], alpha = 1.0)
            else:
                ax3.plot(data.index[inds], v_plot, '-', color=col, label=data_stream, alpha = 1.0)
            i+=1
        for te in td.tes:
            if [te>=datetimeify(_range[0]) and te<=datetimeify(_range[1])]:
                pass
                #ax.axvline(te, color='k', linestyle='--', linewidth=2, zorder = 0)
        #
        #ax.plot([], color='k', linestyle='--', linewidth=2, label = 'eruption')
        #ax3.set_xlim(_range)
        ax3.legend(loc = 2)
        ax3.grid()
        if log:
            ax3.set_ylabel(' ')
        else:
            ax3.set_ylabel('\u03BC m/s')
        #ax3.set_xlabel('Time [month-day hour]')
        #ax3.title.set_text('Station '+td.station+' ('+sta_code[td.station]+'): Tremor data')
        #
        if plot_erup: # plot vertical lines
            te = datetimeify(erup_time)#fm_e1.data.tes[int(erup[-1:])-1]
            ax3.axvline(te, color='k',linestyle='--', linewidth=2, zorder = 4)
            ax3.plot([], color='k', linestyle='--', linewidth=2, label = 'event')
        
        #
        te = datetimeify(erup_time)#fm_e1.data.tes[int(erup[-1:])-1]
        #ax3.set_xticks([te - 5*day*i for i in range(int(look_back/5)+1)])#[dat.index.values[0],dat.index.values[-1]])#, ]np.arange(0, len(x)+1, 5))

        ax3.set_yscale('log')
        lims = [10e-2,10e2]#[np.mean(np.log(v_plot))-3*np.std(np.log(v_plot)), np.mean(np.log(v_plot))+6*np.std(np.log(v_plot))]
        if sta == 'COP':
            lims = None
        ax3.set_ylim(lims)
    # subplot four: non filtered  RSAM, MF, HF datastreams
    if False:
        #
        td = TremorData(station = sta)
        #td.update(ti=t0, tf=t1)
        data_streams = ['hf','mf', 'rsam']#, 'dsarF']
        label = ['HF','MF','RSAM','DSAR']
        #label = ['1/RSAM']
        inv = False
        if False:
            data_streams = ['hf', 'mf', 'rsam', 'lf', 'vlf']
            label = ['HF','MF','RSAM','LF', 'VLF']

        if type(data_streams) is str:
            data_streams = [data_streams,]
        if any(['_' in ds for ds in data_streams]):
            td._compute_transforms()
        #ax.set_xlim(*range)
        # plot data for each year
        norm= False
        _range = [t0,t1]
        log =False
        col_def = None
        data = td.get_data(*_range)
        xi = datetime(year=1,month=1,day=1,hour=0,minute=0,second=0)
        cols = ['g','r','gray','m',[0.5,0.5,0.5],[0.75,0.75,0.75]]
        if inv:
            cols = ['gray','g','r','m',[0.5,0.5,0.5],[0.75,0.75,0.75]]
        inds = (data.index>=datetimeify(_range[0]))&(data.index<=datetimeify(_range[1]))
        i=0
        for data_stream, col in zip(data_streams,cols):
            v_plot = data[data_stream].loc[inds]
            if inv:
                v_plot = 1/v_plot
            if log:
                v_plot = np.log10(v_plot)
            if norm:
                v_plot = (v_plot-np.min(v_plot))/np.max((v_plot-np.min(v_plot)))
            if label:
                if col_def:
                    ax4.plot(data.index[inds], v_plot, '-', color=col_def, label=label[i], linewidth=1., alpha = 1.0)
                else:
                    ax4.plot(data.index[inds], v_plot, '-', color=col, label=label[i], linewidth=1., alpha = 1.0)
            else:
                ax4.plot(data.index[inds], v_plot, '-', color=col, label=data_stream, linewidth=1., alpha = 1.0)
            i+=1
        for te in td.tes:
            if [te>=datetimeify(_range[0]) and te<=datetimeify(_range[1])]:
                pass
                #ax.axvline(te, color='k', linestyle='--', linewidth=2, zorder = 0)
        #
        #ax.plot([], color='k', linestyle='--', linewidth=2, label = 'eruption')
        #ax4.set_xlim(_range)
        ax4.legend(loc = 2)
        ax4.grid()
        if log:
            ax4.set_ylabel(' ')
        else:
            ax4.set_ylabel('\u03BC m/s')
        #ax4.set_xlabel('Time [month-day hour]')
        #ax4.title.set_text('Station '+td.station+' ('+sta_code[td.station]+'): Tremor data')
        #
        te = datetimeify(erup_time)#fm_e1.data.tes[int(erup[-1:])-1]
        #ax4.set_xticks([te - 5*day*i for i in range(int(look_back/5)+1)])#[dat.index.values[0],dat.index.values[-1]])#, ]np.arange(0, len(x)+1, 5))
        #ax4.set_ylim([1e9,1e13])
        ax4.set_yscale('log')
    #
    if False: # plot vertical lines
        te = datetimeify(erup_time)#fm_e1.data.tes[int(erup[-1:])-1]
        ax1.axvline(te, color='k',linestyle='--', linewidth=2, zorder = 4)
        ax2.axvline(te, color='k',linestyle='--', linewidth=2, zorder = 4)
        ax3.axvline(te, color='k',linestyle='--', linewidth=2, zorder = 4)
        ax1.plot([], color='k', linestyle='--', linewidth=2, label = 'event')
    #
    if False:#save_png_path:
        dat = erup_time.strftime('%Y-%m-%d')
        title =  sta+'_'+dat+'_'+'look_back'+str(look_back)
        ax1.set_title(title)
        plt.savefig(save_png_path+sta+'_'+dat+'_'+'look_back'+str(look_back)+'.png')
    #
    ax1.set_xticks([t1 - 4*day*i for i in range(int((look_back+look_front)/4)+1)])
    ax2.set_xticks([t1 - 4*day*i for i in range(int((look_back+look_front)/4)+1)])
    #ax3.set_xticks([t1 - 2*day*i for i in range(int((look_back+look_front)/2)+1)])
    #ax4.set_xticks([t1 - 4*day*i for i in range(int((look_back+look_front)/4)+1)])
    #    
    ax1.set_xlim([t0+2*day,t1])
    ax2.set_xlim([t0+2*day,t1])
    #ax2b.set_ylim([0.2,0.5])
    #ax2b.set_ylim([0.1,0.5])

    #ax3.set_xlim([t0+2*day,t1])
    #ax4.set_xlim([t0+2*day,t1])
    #
    ax0.set_title('(a) Ruapehu 2009 temperature cycle and fluid release event on 07/13')
    ax1.set_title('(b) DSAR median and RSAM before fluid release event on 07/13')
    ax2.set_title('(c) Lake temperature and level before fluid release event on 07/13')
    #ax4.set_title('Seismic datastreams before hydrothermal event on 07/13')
    #
    plt.tight_layout()
    plt.show()
    plt.close('all')

def figure_sup_ruapehu_events(): # events in seismic and lake levels 
    '''
    plot: rsam and dsar before event (1), and lake levels (2), for multiple events in Ruapehu
    '''
    sta = 'FWVZ' 
    if sta == 'FWVZ':
        ffm = False
        server = False # files imported from server to local pc 
        server2 = False # server at uni 
        plot_erup = True
    #
    look_back = 10
    look_front = 5
    #
    #erup_time = datetimeify('2009 07 13 06 30 00')
    #erup_time = datetimeify('2010 09 03 00 00 00')
    #erup_time = datetimeify('2021 03 04 12 00 00')
    erup_times = [datetimeify('2009 07 13 06 30 00'), 
                    datetimeify('2010 09 03 16 00 00'),
                    datetimeify('2016 11 13 12 00 00'),
                    datetimeify('2021 03 04 12 00 00')]
    #
    #erup_time = datetimeify('2021 09 09 00 00 00')
    #
    day = timedelta(days=1)
    #t0 = erup_times - look_back*day#30*day
    #t1 = erup_times + look_front*day#hour
    #
    ## plot other data
    temp = True
    level = True
    rainfall = True
    ## 
    plot_erup = True
    # figure
    nrow = 4
    ncol = 2
    #fig, (ax0, ax1, ax2, ax4) = plt.subplots(nrows=nrow, ncols=ncol, figsize=(12,12))#(14,4)) #, ax4)
    fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6), (ax7, ax8)) = plt.subplots(nrows=nrow, ncols=ncol, figsize=(12,12))#(14,4)) #, ax4)
    #
    # for ax in [ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8]:#,ax4]: # plot eruption times 
    #     # plot event 
    #     #te = datetimeify('2009-07-07') 
    #     te = datetimeify('2009-07-13 06:30:00')
    #     ax.axvline(te+0.12*day, color='k',linestyle='--', linewidth=3, zorder = 4)
    #     ax.plot([], color='k', linestyle='--', linewidth=3, label = 'located event')
    #     # plot eruption 
    #     te = datetimeify('2009 07 13 06 30 00') 
    #     #ax.axvline(te+0.22*day, color='r',linestyle='-', linewidth=3, zorder = 4)
    #     #ax.plot([], color='r', linestyle='-', linewidth=3, label = 'eruption')  
  
    #####################################################
    for j,ax in enumerate([ax1, ax3, ax5, ax7]):
        #
        t0 = erup_times[j] - look_back*day#30*day
        t1 = erup_times[j] + look_front*day#hour
        # subplot one: MF, HF, DSAR medians (DSAR yaxis left; MF, HF yaxis rigth). 1/RSAM (normalized)
        if True:
            # features
            fts_yleft = ['zsc2_dsarF__median']
            fts_yrigth = []#['zsc2_dsarF__rate_variance']#['zsc2_dsarF__change_quantiles__f_agg_"var"__isabs_False__qh_0.6__ql_0.4']#['zsc2_mfF__median','zsc2_hfF__median']
            data_yrigth = ['rsam']
            
            #
            col = ['b','b','r']
            alpha = [1., 1., .5]
            thick_line = [2., 6., 1.]
            axb = ax.twinx() 
            for i, ft in enumerate(fts_yleft):
                if False: # load feature (else: cal feature. median or rv)
                    if 'zsc2_dsarF' in ft:
                        ds = ['zsc2_dsarF'] 
                    if server:
                        path_feat_serv = 'C:\\Users\\aar135\\codes_local_disk\\volc_forecast_tl\\features_bkp\\features_server\\'
                        fm_e1 = ForecastModel(window=2., overlap=1., station =  sta,
                            look_forward=2., data_streams=ds, 
                            feature_dir=path_feat_serv, 
                            savefile_type='pkl') 
                    elif server2:
                        path_feat_serv = 'U:\\Research\\EruptionForecasting\\eruptions\\features\\'
                        fm_e1 = ForecastModel(window=2., overlap=1., station =  sta,
                            look_forward=2., data_streams=ds, 
                            feature_dir=path_feat_serv, 
                            savefile_type='pkl') 
                    else:
                        try:
                            fm_e1 = ForecastModel(window=2., overlap=1., station = sta,
                                look_forward=2., data_streams=ds, savefile_type='csv')
                        except:
                            fm_e1 = ForecastModel(window=2., overlap=1., station = sta,
                                look_forward=2., data_streams=ds, savefile_type='pkl')                    
                    ##  
                    ft = ft.replace("-",'"')
                    # adding multiple Axes objects
                    ft_e1 = fm_e1.get_features(ti=t0, tf=t1, n_jobs=1, compute_only_features=[ft])
                    # extract values to plot 
                    ft_e1_t = ft_e1[0].index.values
                    ft_e1_v = ft_e1[0].loc[:,ft]
                    #
                    v_plot = ft_e1_v

                    #v_plot = ft_e1_v-np.min(ft_e1_v)/np.max((ft_e1_v-np.min(ft_e1_v)))
                    #v_plot = ft_e1_v/np.max(ft_e1_v)
                else: 
                    #
                    if 'zsc2_dsarF' in ft:
                        ds = 'zsc2_dsarF'
                    if 'zsc2_mfF' in ft:
                        ds = 'zsc2_mfF' 
                    if 'zsc2_hfF' in ft:
                        ds = 'zsc2_hfF' 
                    # 
                    #
                    day = timedelta(days=1)
                    fm = ForecastModel(window=2., overlap=1., station=sta,
                        look_forward=2., data_streams=[ds], 
                        data_dir=r'C:\Users\aar135\codes_local_disk\volc_forecast_tl\volc_forecast_tl\data'
                        )
                    #
                    N, M = [2,30]
                    df = fm.data.df[t0:t1]
                    if 'median' in ft:
                        test = df[ds].rolling(N*24*6).median()[N*24*6:]
                    if 'rate_variance' in ft:
                        test = df[ds].rolling(N*24*6).apply(chqv)[N*24*6:]
                    #
                    #out = test.rolling(archtype.shape[0]).apply(partial(conv, (archtype-archtype.mean())/archtype.std()))       
                    #out = out.resample('1D').ffill()
                    #
                    ft_e1_t = test.index
                    v_plot = test.values
                #
                if ft == 'zsc2_dsarF__median':
                    ft = 'DSAR median'
                #
                axb.plot(ft_e1_t, v_plot, '-', color=col[i], alpha = alpha[i], linewidth = thick_line[i], label=' '+ ft, zorder = 2)
                #
                #
                if ffm: # ffm 
                    #ax1b = ax1.twinx() 
                    #v_plot = data[data_stream].loc[inds]
                    inv_rsam = fm_e1.data.get_data(ti=t0, tf=t1)['rsamF']#.loc[ft_e1_t]
                    inv_rsam = 1./inv_rsam
                    # normalized it to yaxis rigth 
                    inv_rsam = inv_rsam/max(inv_rsam)
                    inv_rsam = inv_rsam*0.5*max(v_plot)
                    #
                    ax.plot(ft_e1_t, inv_rsam, '-', color= 'gray', linewidth=0.5, markersize=0.5, alpha = 1.)
                    ax.plot([], [], '-', color= 'gray', markersize=1, label='1/RSAM', alpha = 1.0)
                    #ax1.set_ylim([0,1])
                    #ax1.set_yticks([])
                #
                if plot_erup: # plot vertical lines
                    te = datetimeify(erup_times[j])#fm_e1.data.tes[int(erup[-1:])-1]
                    ax.axvline(te, color='gray', alpha = .25, linestyle='-', linewidth=12, zorder = 0)
                    axb.plot([], color='gray', alpha = .25, linestyle='-', linewidth=12, label = 'eruption')
                #
                #ax1.legend(loc = 2)
                #
                te = datetimeify(erup_times[j])#fm_e1.data.tes[int(erup[-1:])-1]
                #ax1.set_xticks([ft_e1[0].index[-1] - 7*day*i for i in range(int(30/7)+1)])#[dat.index.values[0],dat.index.values[-1]])#, ]np.arange(0, len(x)+1, 5))
                
                #ax1b.set_yticks([])
                axb.grid()
                axb.set_ylabel('nDSAR value')        #ax.set_xticks([ft_e1[0].index[-1]-7*day*i +day for i in range(5)])
                #ax1.set_yscale('log') #ax.set_yscale('log')
                #ax.set_xticks([ft_e1[0].index[-1] - 7*day*i for i in range(int(30/7)+1)])#[dat.index.values[0],dat.index.values[-1]])#, ]np.arange(0, len(x)+1, 5))
                #ax1.set_xticks([t1 - 5*day*i for i in range(int(look_back/5)+1)])#[dat.index.values[0],dat.index.values[-1]])#, ]np.arange(0, len(x)+1, 5))
            #     #
            # except:
            #     pass
            if fts_yrigth:
                #ax1b = ax1.twinx() 
                col = ['r','g']
                alpha = [1., .5]
                thick_line = [2.,1.]
                #try: 
                for i, ft in enumerate(fts_yrigth):
                    if 'zsc2_dsarF' in ft:
                        ds = 'zsc2_dsarF'
                    if 'zsc2_mfF' in ft:
                        ds = 'zsc2_mfF' 
                    if 'zsc2_hfF' in ft:
                        ds = 'zsc2_hfF' 
                    # 
                    if False: # look feature in the prev cacl features (else: calculate feat from data; only for median and rv)
                        if server:
                            path_feat_serv = 'C:\\Users\\aar135\\codes_local_disk\\volc_forecast_tl\\features_bkp\\features_server\\'
                            fm_e1 = ForecastModel(window=2., overlap=1., station =  sta,
                                look_forward=2., data_streams=ds, 
                                feature_dir=path_feat_serv, 
                                savefile_type='pkl') 
                        elif server2:
                            path_feat_serv = 'U:\\Research\\EruptionForecasting\\eruptions\\features\\'
                            fm_e1 = ForecastModel(window=2., overlap=1., station =  sta,
                                look_forward=2., data_streams=ds, 
                                feature_dir=path_feat_serv, 
                                savefile_type='pkl') 
                        else:
                            try:
                                fm_e1 = ForecastModel(window=2., overlap=1., station = sta,
                                    look_forward=2., data_streams=ds, savefile_type='csv')
                            except:
                                fm_e1 = ForecastModel(window=2., overlap=1., station = sta,
                                    look_forward=2., data_streams=ds, savefile_type='pkl')                    
                        ##  
                        ft = ft.replace("-",'"')
                        
                        ft_e1 = fm_e1.get_features(ti=t0, tf=t1, n_jobs=1, compute_only_features=[ft])
                        # adding multiple Axes objects

                        # extract values to plot 
                        ft_e1_t = ft_e1[0].index.values
                        ft_e1_v = ft_e1[0].loc[:,ft]
                        #
                        v_plot = ft_e1_v

                    else:
                        day = timedelta(days=1)
                        fm = ForecastModel(window=2., overlap=1., station=sta,
                            look_forward=2., data_streams=[ds], 
                            data_dir=r'C:\Users\aar135\codes_local_disk\volc_forecast_tl\volc_forecast_tl\data'
                            )
                        #
                        N, M = [2,30]
                        df = fm.data.df[t0:t1]
                        if 'median' in ft:
                            test = df[ds].rolling(N*24*6).median()[N*24*6:]
                        if 'rate_variance' in ft:
                            test = df[ds].rolling(N*24*6).apply(chqv)[N*24*6:]
                        #
                        #out = test.rolling(archtype.shape[0]).apply(partial(conv, (archtype-archtype.mean())/archtype.std()))       
                        #out = out.resample('1D').ffill()
                        #
                        ft_e1_t = test.index
                        v_plot = test.values

                    #v_plot = ft_e1_v-np.min(ft_e1_v)/np.max((ft_e1_v-np.min(ft_e1_v)))
                    #v_plot = ft_e1_v/np.max(ft_e1_v)
                    #
                    if ft == 'zsc2_mfF__median':
                        ft = 'nMF median'
                    if ft == 'zsc2_hfF__median':
                        ft = 'nHF median'
                    #
                    ax.plot(ft_e1_t, v_plot, '-', color=col[i], alpha = alpha[i],label=' '+ ft, zorder = 4)
                    #
                    ax.legend(loc = 3)
                    #
                    te = datetimeify(erup_time)#fm_e1.data.tes[int(erup[-1:])-1]
                    #ax1.set_xticks([ft_e1[0].index[-1] - 7*day*i for i in range(int(30/7)+1)])#[dat.index.values[0],dat.index.values[-1]])#, ]np.arange(0, len(x)+1, 5))
                    
                    #ax1b.set_yticks([])
                    #ax.grid()
                    ax.set_ylabel('Feature value')        #ax.set_xticks([ft_e1[0].index[-1]-7*day*i +day for i in range(5)])
                    #ax.set_xticks([ft_e1[0].index[-1] - 7*day*i for i in range(int(30/7)+1)])#[dat.index.values[0],dat.index.values[-1]])#, ]np.arange(0, len(x)+1, 5))
                    #ax1.set_xticks([t1 - 5*day*i for i in range(int(look_back/5)+1)])#[dat.index.values[0],dat.index.values[-1]])#, ]np.arange(0, len(x)+1, 5))
                    #
                #except:
                #    pass

            else:
                pass
                if data_yrigth:
                    #
                    #ax1b = ax1.twinx() 
                    #
                    td = TremorData(station = sta)
                    #td.update(ti=t0, tf=t1)
                    data_streams = data_yrigth#['hf','mf', 'rsam']#, 'dsarF']
                    label = ['RSAM','MF','HF','DSAR']
                    #label = ['1/RSAM']
                    inv = False
                    if False:
                        data_streams = ['rsam']
                        label = ['RSAM']

                    if type(data_streams) is str:
                        data_streams = [data_streams,]
                    if any(['_' in ds for ds in data_streams]):
                        td._compute_transforms()
                    #ax.set_xlim(*range)
                    # plot data for each year
                    norm= False
                    _range = [t0,t1]
                    log =False
                    col_def = None
                    data = td.get_data(*_range)
                    xi = datetime(year=1,month=1,day=1,hour=0,minute=0,second=0)
                    cols = ['k','r','g','m',[0.5,0.5,0.5],[0.75,0.75,0.75]]
                    if inv:
                        cols = ['k','g','r','m',[0.5,0.5,0.5],[0.75,0.75,0.75]]
                    inds = (data.index>=datetimeify(_range[0]))&(data.index<=datetimeify(_range[1]))
                    i=0
                    for data_stream, col in zip(data_streams,cols):
                        v_plot = data[data_stream].loc[inds]
                        if inv:
                            v_plot = 1/v_plot
                        if log:
                            v_plot = np.log10(v_plot)
                        if norm:
                            v_plot = (v_plot-np.min(v_plot))/np.max((v_plot-np.min(v_plot)))
                        if label:
                            if col_def:
                                ax.plot(data.index[inds], v_plot, '-', color=col_def, label=label[i], linewidth=1., alpha = 1., zorder = 0)
                            else:
                                #ax1b.plot(data.index[inds], v_plot, '-', color=col, label=label[i], linewidth=1., alpha = 1.0, zorder = 0)
                                ax.plot(data.index[inds], v_plot, '-', color=col, linewidth=1., alpha = .7, zorder = 3)
                                axb.plot([], [], '-', color=col, label=label[i], linewidth=1., alpha = .7, zorder = 3)
                        else:
                            ax.plot(data.index[inds], v_plot, '-', color=col, label=data_stream, linewidth=1., alpha = .7, zorder = 3)
                        i+=1
                    for te in td.tes:
                        if [te>=datetimeify(_range[0]) and te<=datetimeify(_range[1])]:
                            pass
                            #ax.axvline(te, color='k', linestyle='--', linewidth=2, zorder = 0)
                    #
                    ax.plot([], color='k', linestyle='--', linewidth=1, label = 'eruption')
                    #ax4.set_xlim(_range)
                    #ax1b.legend(loc = 2)
                    #ax1b.grid()
                    if log:
                        ax.set_ylabel(' ')
                    else:
                        ax.set_ylabel('RSAM \u03BC m/s')
                    #ax4.set_xlabel('Time [month-day hour]')
                    #ax4.title.set_text('Station '+td.station+' ('+sta_code[td.station]+'): Tremor data')
                    #
                    #ax4.set_xticks([te - 5*day*i for i in range(int(look_back/5)+1)])#[dat.index.values[0],dat.index.values[-1]])#, ]np.arange(0, len(x)+1, 5))
                    #ax4.set_ylim([1e9,1e13])
                    ax.set_yscale('log')
            axb.legend(loc = 2)      
            axb.grid(False)
            ax.grid(color='gray', linestyle='-', linewidth=.5, alpha = 0.5)
            #

    for j, ax in enumerate([ax2, ax4, ax6, ax8]):
        t0 = erup_times[j] - look_back*day#30*day
        t1 = erup_times[j] + look_front*day#hour
        # subplot two: temp data (if any: level and rainfall)
        if True:  
            mov_avg = True # moving average for temp and level data
            # convert to UTC 0
            utc_0 = True
            if utc_0:
                _utc_0 = 0#-13 # hours
            # plot temp data
            if sta == 'FWVZ':
                # plot temperature data 
                if temp:
                    try:
                        ti_e1 = t0
                        tf_e1 = t1
                        # import temp data
                        path = '..'+os.sep+'data'+os.sep+"RU001_temp_data.csv"
                        pd_temp = pd.read_csv(path, index_col=1)
                        if utc_0:
                            #pd_temp.index = [datetimeify(pd_temp.index[i])-6*hour for i in range(len(pd_temp.index))]
                            pd_temp.index = [datetimeify(pd_temp.index[i])+_utc_0*hour for i in range(len(pd_temp.index))]
                        else:
                            pd_temp.index = [datetimeify(pd_temp.index[i]) for i in range(len(pd_temp.index))]
                        
                        # plot data in axis twin axis
                        # Trim the data
                        temp_e1_tim = pd_temp[ti_e1: tf_e1].index.values
                        #temp_e1_tim=to_nztimezone(temp_e1_tim)
                        #
                        temp_e1_val = pd_temp[ti_e1: tf_e1].loc[:,' t (C)'].values
                        # ax2
                        #ax2b = ax2.twinx()   
                        if mov_avg: # plot moving average
                            n=30
                            #v_plot = (temp_e1_val-np.min(temp_e1_val))/np.max((temp_e1_val-np.min(temp_e1_val)))
                            v_plot = temp_e1_val
                            ax.plot(temp_e1_tim, v_plot, '-', color='g', label='lake temperature', alpha = 1.)
                            
                            #
                            #ax.plot(temp_e1_tim[:-n+1], moving_average(temp_e1_val, n=n), '--', color='k')#, label='temp. mov. avg.')
                            ax.plot(temp_e1_tim[n-1-20:-20], moving_average(v_plot[::-1], n=n)[::-1], '--', color='g')#, label='lake temperature')
                        else:
                            v_plot = temp_e1_val
                            ax.plot(temp_e1_tim, v_plot, '-', color='g', label='lake temperature', alpha = 1.)
                        ax.set_ylabel('Temperature °C')
                        #
                        temp_min = min(temp_e1_val)
                        temp_max = max(temp_e1_val)
                        temp_mu = np.mean(temp_e1_tim)
                        temp_sigma = np.std(temp_e1_tim)
                        ax.set_ylim([temp_mu-3*temp_sigma,temp_mu+3*temp_sigma])
                        #ax2.set_ylabel('temperature C')   
                    except:
                        pass

                # plot lake level data
                if level:
                    #try:
                    if True:
                        axb = ax.twinx()
                        #
                        # import temp data
                        if t0.year >= 2016:# and t0.month >= 3:
                            path = '..'+os.sep+'data'+os.sep+"RU001A_level_data.csv"

                        elif t0.year < 2012:# and t0.month < 7:
                            path = '..'+os.sep+'data'+os.sep+"RU001_level_data.csv"      
                        else:
                            path = '..'+os.sep+'data'+os.sep+"RU001A_level_data_full.csv"
                        #    
                        pd_lev = pd.read_csv(path, index_col=1)
                        if utc_0:
                            #pd_temp.index = [datetimeify(pd_temp.index[i])-6*hour for i in range(len(pd_temp.index))]
                            #pd_temp.index = [datetimeify(pd_temp.index[i])-0*hour for i in range(len(pd_temp.index))]
                            pd_lev.index = [datetimeify(pd_lev.index[i])+_utc_0*hour for i in range(len(pd_lev.index))]
                        else:
                            pd_lev.index = [datetimeify(pd_lev.index[i]) for i in range(len(pd_lev.index))]

                        if t0.year>2010 and t1.year<2016: # rolling median over data
                            N = 2
                            pd_lev = pd_lev[:].rolling(40).median()#[N*24*6:]

                        # plot data in axis twin axis
                        # Trim the data
                        lev_e1_tim = pd_lev[ti_e1: tf_e1].index.values
                        #temp_e1_tim=to_nztimezone(temp_e1_tim)
                        lev_e1_val = pd_lev[ti_e1: tf_e1].loc[:,' z (m)'].values
                        # ax2
                        #ax2b = ax2.twinx()
                        if False:#mov_avg: # plot moving average
                            n=10
                            v_plot = temp_e1_val
                            axb.plot(temp_e1_tim, v_plot, '-', color='royalblue', alpha = 1.)
                            ax.plot([], [], '-', color='royalblue', label='lake level')
                            #ax2b.plot(temp_e1_tim[:-n+1], moving_average(temp_e1_val, n=n), '--', color='k', label='temp. mov. avg.')
                            #v_plot = (temp_e1_val-np.min(temp_e1_val))/np.max((temp_e1_val-np.min(temp_e1_val)))
                            axb.plot(temp_e1_tim[n-1-10:-10], moving_average(v_plot[::-1], n=n)[::-1], '--', color='royalblue', label='lake level')
                            #ax2b.plot(temp_e1_tim, v_plot, '-', color='b', alpha = 0.3)
                        else:
                            v_plot = lev_e1_val
                            axb.plot(lev_e1_tim, v_plot, '-', color='royalblue', label='lake level')
                        #
                        axb.set_ylabel('Lake level cm') 
                        ax.plot([], [], '-', color='royalblue', label='lake level')

                        if plot_erup: # plot vertical lines
                            te = datetimeify(erup_times[j])#fm_e1.data.tes[int(erup[-1:])-1]
                            ax.axvline(te, color='gray', alpha = .25, linestyle='-', linewidth=12, zorder = 0)
                            axb.plot([], color='gray', alpha = .25, linestyle='-', linewidth=12, label = 'eruption')
                    #except:
                    #    pass
                
                # plot rainfall data
                if rainfall:
                    try:
                        ti_e1 = t0#datetimeify(t0)
                        tf_e1 = t1#datetimeify(t1)
                        #
                        # import temp data
                        path = '..'+os.sep+'data'+os.sep+"_chateau_rain.csv"
                        pd_rf = pd.read_csv(path, index_col=1)
                        pd_rf.index = pd.to_datetime(pd_rf.index, format='%Y%m%d:%H%M')
                        if utc_0:
                            pd_rf.index = [pd_rf.index[i]+_utc_0*hour for i in range(len(pd_rf.index))]

                        # Trim the data
                        rf_e2_tim = pd_rf[ti_e1: tf_e1].index#.values
                        rf_e2_val = pd_rf[ti_e1: tf_e1].loc[:,'Amount(mm)'].values /3
                        # ax2
                        #ax2b = ax2.twinx()
                        v_plot = (rf_e2_val-np.min(rf_e2_val))/np.max((rf_e2_val-np.min(rf_e2_val)))
                        #v_plot = v_plot*5 + 14
                        if temp_max:
                            v_plot = v_plot*(temp_max-temp_min)*0.6 + temp_min
                        ax.plot(rf_e2_tim, v_plot, '-', color='c', label='rain fall', alpha = 0.6)
                        #ax2b.set_ylabel('temperature C')
                        #ax2b.legend(loc = 1)
                    except:
                        pass

            if sta == 'DAM' or sta == 'POS':
                lake = False # no data
                rainfall = False # no data
                try:
                    if temp:
                        ti_e1 = t0
                        tf_e1 = t1
                        # import temp data
                        path = '..'+os.sep+'data'+os.sep+"DAM_temp_data.csv"
                        pd_temp = pd.read_csv(path, index_col=1)

                        if utc_0:
                            pd_temp.index = [datetimeify(pd_temp.index[i])-6*hour for i in range(len(pd_temp.index))]
                        else:
                            pd_temp.index = [datetimeify(pd_temp.index[i]) for i in range(len(pd_temp.index))]
                        # plot data in axis twin axis
                        # Trim the data
                        temp_e1_tim = pd_temp[ti_e1: tf_e1].index.values
                        #temp_e1_tim=to_nztimezone(temp_e1_tim)
                        #
                        temp_e1_val = pd_temp[ti_e1: tf_e1].loc[:,' t (C)'].values
                        # ax2
                        #ax2b = ax2.twinx()   
                        if mov_avg: # plot moving average
                            n=50
                            #v_plot = (temp_e1_val-np.min(temp_e1_val))/np.max((temp_e1_val-np.min(temp_e1_val)))
                            v_plot = temp_e1_val
                            ax2.plot(temp_e1_tim, v_plot, '-', color='g', label='lake temperature', alpha = 1.)
                            #
                            #ax.plot(temp_e1_tim[:-n+1], moving_average(temp_e1_val, n=n), '--', color='k')#, label='temp. mov. avg.')
                            _x = temp_e1_tim[n-1-20:-20]
                            _y = moving_average(v_plot[::-1], n=n)[::-1]
                            ax2.plot(_x, _y, '--', color='k')#, label='lake temperature')
                        else:
                            v_plot = temp_e1_val
                            ax2.plot(temp_e1_tim, v_plot, '-', color='g', label='lake temperature', alpha = 1.)
                        ax2.set_ylabel('Temperature °C')
                        
                        _ylim = [min(_y)-1,max(_y)+1] 
                        ax2.set_ylim(_ylim)
                        #ax2.set_ylabel('temperature C')   
                except:
                    pass
                if plot_erup: # plot vertical lines
                    te = datetimeify(erup_time)#fm_e1.data.tes[int(erup[-1:])-1]
                    ax2.axvline(te, color='k',linestyle='--', linewidth=2, zorder = 4)
                    #ax2.plot([], color='k', linestyle='--', linewidth=2, label = 'eruption')

            ax.legend(loc = 2)   
            ax.grid(color='gray', linestyle='-', linewidth=.5, alpha = 0.5)
            #ax2.set_xticks([t1 - 5*day*i for i in range(int(look_back/5)+1)])#[dat.index.values[0],dat.index.values[-1]])#, ]np.arange(0, len(x)+1, 5))
        #
        
    if False: # plot vertical lines
        te = datetimeify(erup_time)#fm_e1.data.tes[int(erup[-1:])-1]
        ax1.axvline(te, color='k',linestyle='--', linewidth=2, zorder = 4)
        ax2.axvline(te, color='k',linestyle='--', linewidth=2, zorder = 4)
        ax3.axvline(te, color='k',linestyle='--', linewidth=2, zorder = 4)
        ax1.plot([], color='k', linestyle='--', linewidth=2, label = 'event')
    #
    if False:#save_png_path:
        dat = erup_time.strftime('%Y-%m-%d')
        title =  sta+'_'+dat+'_'+'look_back'+str(look_back)
        ax1.set_title(title)
        plt.savefig(save_png_path+sta+'_'+dat+'_'+'look_back'+str(look_back)+'.png')
    #
    _d = 5 
    t1 = erup_times[0] + look_front*day#hour
    ax1.set_xticks([t1 - _d*day*i for i in range(int((look_back+look_front)/_d)+1)])
    ax2.set_xticks([t1 - _d*day*i for i in range(int((look_back+look_front)/_d)+1)])
    #
    t1 = erup_times[1] + look_front*day#hour
    ax3.set_xticks([t1 - _d*day*i for i in range(int((look_back+look_front)/_d)+1)])
    ax4.set_xticks([t1 - _d*day*i for i in range(int((look_back+look_front)/_d)+1)])
    #
    t1 = erup_times[2] + look_front*day#hour
    ax5.set_xticks([t1 - _d*day*i for i in range(int((look_back+look_front)/_d)+1)])
    ax6.set_xticks([t1 - _d*day*i for i in range(int((look_back+look_front)/_d)+1)])
    #
    t1 = erup_times[3] + look_front*day#hour
    ax7.set_xticks([t1 - _d*day*i for i in range(int((look_back+look_front)/_d)+1)])
    ax8.set_xticks([t1 - _d*day*i for i in range(int((look_back+look_front)/_d)+1)])
    #
    #ax3.set_xticks([t1 - 2*day*i for i in range(int((look_back+look_front)/2)+1)])
    #ax4.set_xticks([t1 - 4*day*i for i in range(int((look_back+look_front)/4)+1)])
    #    
    #ax.set_xlim([t0+2*day,t1])
    #ax.set_xlim([t0+2*day,t1])
    #ax1.set_ylim([10**.4,10**2.1])
    ax3.set_ylim([10**0,10**3.5])
    #ax4.set_ylim([1,100])
    #ax7.set_ylim([1,100])
    #ax2b.set_ylim([0.1,0.5])

    #ax3.set_xlim([t0+2*day,t1])
    #ax4.set_xlim([t0+2*day,t1])
    #
    # erup_times = [datetimeify('2009 07 13 06 30 00'), 
    #                 datetimeify('2010 09 03 12 00 00'),
    #                 datetimeify('2016 11 13 12 00 00'),
    #                 datetimeify('2021 03 04 12 00 00')]
    ax1.set_title('(a) Ruapehu 2009/07/13 seismic RSAM and DSAR median')
    ax3.set_title('(c) Ruapehu 2010/09/03 seismic RSAM and DSAR median')
    ax5.set_title('(e) Ruapehu 2016/11/13 seismic RSAM and DSAR median')
    ax7.set_title('(g) Ruapehu 2021/03/04 seismic RSAM and DSAR median')
    #
    ax2.set_title('(b) Ruapehu 2009/07/13 lake levels data')
    ax4.set_title('(d) Ruapehu 2010/09/03 lake levels data')
    ax6.set_title('(f) Ruapehu 2016/11/13 lake levels data')
    ax8.set_title('(h) Ruapehu 2021/03/04 lake levels data')
    #ax1.set_title('(b) DSAR median and RSAM before hydrothermal event on 07/13')
    #ax2.set_title('(c) Lake temperature and level before hydrothermal event on 07/13')
    #ax4.set_title('Seismic datastreams before hydrothermal event on 07/13')
    #
    plt.tight_layout()
    plt.show()
    plt.close('all')

def figure_sup_kawahijen_events(): # events in seismic and lake levels
    '''
    plot: rsam and dsar before event (1), and lake levels (2), for multiple events in Ruapehu
    '''
    sta = 'POS' 
    if sta == 'POS':
        ffm = False
        server = False # files imported from server to local pc 
        server2 = False # server at uni 
        #plot_erup = False
    #
    look_back = 14
    look_front = 4
    #
    #erup_time = datetimeify('2009 07 13 06 30 00')
    #erup_time = datetimeify('2010 09 03 00 00 00')
    #erup_time = datetimeify('2021 03 04 12 00 00')
    erup_times = [datetimeify('2013 03 31 20 00 00'), 
                    datetimeify('2012 11 23 12 00 00'),
                    datetimeify('2013 01 24 00 00 00')]
                    #datetimeify('2011 02 09 00 00 00')]
    #
    #erup_time = datetimeify('2021 09 09 00 00 00')
    #
    day = timedelta(days=1)
    #t0 = erup_times - look_back*day#30*day
    #t1 = erup_times + look_front*day#hour
    #
    ## plot other data
    temp = True
    level = False
    rainfall = False
    ## 
    plot_erup = True
    # figure
    nrow = 3
    ncol = 2
    #fig, (ax0, ax1, ax2, ax4) = plt.subplots(nrows=nrow, ncols=ncol, figsize=(12,12))#(14,4)) #, ax4)
    #fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6), (ax7, ax8)) = plt.subplots(nrows=nrow, ncols=ncol, figsize=(12,12))#(14,4)) #, ax4)
    fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(nrows=nrow, ncols=ncol, figsize=(12,12))#(14,4)) #, ax4)
    #
    # for ax in [ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8]:#,ax4]: # plot eruption times 
    #     # plot event 
    #     #te = datetimeify('2009-07-07') 
    #     te = datetimeify('2009-07-13 06:30:00')
    #     ax.axvline(te+0.12*day, color='k',linestyle='--', linewidth=3, zorder = 4)
    #     ax.plot([], color='k', linestyle='--', linewidth=3, label = 'located event')
    #     # plot eruption 
    #     te = datetimeify('2009 07 13 06 30 00') 
    #     #ax.axvline(te+0.22*day, color='r',linestyle='-', linewidth=3, zorder = 4)
    #     #ax.plot([], color='r', linestyle='-', linewidth=3, label = 'eruption')  
    #
    #####################################################
    for j,ax in enumerate([ax1, ax3, ax5]):
        #
        t0 = erup_times[j] - look_back*day#30*day
        t1 = erup_times[j] + look_front*day#hour
        # subplot one: MF, HF, DSAR medians (DSAR yaxis left; MF, HF yaxis rigth). 1/RSAM (normalized)
        if True:
            # features
            fts_yleft = ['zsc2_dsarF__median']
            fts_yrigth = []#['zsc2_dsarF__rate_variance']#['zsc2_dsarF__change_quantiles__f_agg_"var"__isabs_False__qh_0.6__ql_0.4']#['zsc2_mfF__median','zsc2_hfF__median']
            data_yrigth = ['rsamF']
            
            #
            col = ['b','b','r']
            alpha = [1., 1., .5]
            thick_line = [2., 6., 1.]
            axb = ax.twinx() 
            for i, ft in enumerate(fts_yleft):
                if True: # load feature (else: cal feature. median or rv)
                    if 'zsc2_dsarF' in ft:
                        ds = ['zsc2_dsarF'] 
                    if server:
                        path_feat_serv = 'C:\\Users\\aar135\\codes_local_disk\\volc_forecast_tl\\features_bkp\\features_server\\'
                        fm_e1 = ForecastModel(window=2., overlap=1., station =  sta,
                            look_forward=2., data_streams=ds, 
                            feature_dir=path_feat_serv, 
                            savefile_type='pkl') 
                    elif server2:
                        path_feat_serv = 'U:\\Research\\EruptionForecasting\\eruptions\\features\\'
                        fm_e1 = ForecastModel(window=2., overlap=1., station =  sta,
                            look_forward=2., data_streams=ds, 
                            feature_dir=path_feat_serv, 
                            savefile_type='pkl') 
                    else:
                        try:
                            fm_e1 = ForecastModel(window=2., overlap=1., station = sta,
                                look_forward=2., data_streams=ds, savefile_type='csv')
                        except:
                            fm_e1 = ForecastModel(window=2., overlap=1., station = sta,
                                look_forward=2., data_streams=ds, savefile_type='pkl')                    
                    ##  
                    ft = ft.replace("-",'"')
                    # adding multiple Axes objects
                    ft_e1 = fm_e1.get_features(ti=t0, tf=t1, n_jobs=1, compute_only_features=[ft])
                    # extract values to plot 
                    ft_e1_t = ft_e1[0].index.values
                    ft_e1_v = ft_e1[0].loc[:,ft]
                    #
                    v_plot = ft_e1_v

                    #v_plot = ft_e1_v-np.min(ft_e1_v)/np.max((ft_e1_v-np.min(ft_e1_v)))
                    #v_plot = ft_e1_v/np.max(ft_e1_v)
                else: 
                    #
                    if 'zsc2_dsarF' in ft:
                        ds = 'zsc2_dsarF'
                    if 'zsc2_mfF' in ft:
                        ds = 'zsc2_mfF' 
                    if 'zsc2_hfF' in ft:
                        ds = 'zsc2_hfF' 
                    # 
                    #
                    day = timedelta(days=1)
                    fm = ForecastModel(window=2., overlap=1., station=sta,
                        look_forward=2., data_streams=[ds], 
                        data_dir=r'C:\Users\aar135\codes_local_disk\volc_forecast_tl\volc_forecast_tl\data'
                        )
                    #
                    N, M = [2,30]
                    df = fm.data.df[t0:t1]
                    if 'median' in ft:
                        test = df[ds].rolling(N*24*6).median()[N*24*6:]
                    if 'rate_variance' in ft:
                        test = df[ds].rolling(N*24*6).apply(chqv)[N*24*6:]
                    #
                    #out = test.rolling(archtype.shape[0]).apply(partial(conv, (archtype-archtype.mean())/archtype.std()))       
                    #out = out.resample('1D').ffill()
                    #
                    ft_e1_t = test.index
                    v_plot = test.values
                #
                if ft == 'zsc2_dsarF__median':
                    ft = 'nDSAR median'
                #
                axb.plot(ft_e1_t, v_plot, '-', color=col[i], alpha = alpha[i], linewidth = thick_line[i], label=' '+ ft, zorder = 2)
                #
                #
                if ffm: # ffm 
                    #ax1b = ax1.twinx() 
                    #v_plot = data[data_stream].loc[inds]
                    inv_rsam = fm_e1.data.get_data(ti=t0, tf=t1)['rsamF']#.loc[ft_e1_t]
                    inv_rsam = 1./inv_rsam
                    # normalized it to yaxis rigth 
                    inv_rsam = inv_rsam/max(inv_rsam)
                    inv_rsam = inv_rsam*0.5*max(v_plot)
                    #
                    ax.plot(ft_e1_t, inv_rsam, '-', color= 'gray', linewidth=0.5, markersize=0.5, alpha = 1.)
                    ax.plot([], [], '-', color= 'gray', markersize=1, label='1/RSAM', alpha = 1.0)
                    #ax1.set_ylim([0,1])
                    #ax1.set_yticks([])
                #
                if plot_erup: # plot vertical lines
                    te = datetimeify(erup_times[j])#fm_e1.data.tes[int(erup[-1:])-1]
                    ax.axvline(te, color='gray', alpha = .25, linestyle='-', linewidth=12, zorder = 0)
                    axb.plot([], color='gray', alpha = .25, linestyle='-', linewidth=12, label = 'eruption')
                #
                #ax1.legend(loc = 2)
                #
                te = datetimeify(erup_times[j])#fm_e1.data.tes[int(erup[-1:])-1]
                #ax1.set_xticks([ft_e1[0].index[-1] - 7*day*i for i in range(int(30/7)+1)])#[dat.index.values[0],dat.index.values[-1]])#, ]np.arange(0, len(x)+1, 5))
                
                #ax1b.set_yticks([])
                axb.grid()
                axb.set_ylabel('DSAR value')        #ax.set_xticks([ft_e1[0].index[-1]-7*day*i +day for i in range(5)])
                #ax1.set_yscale('log') #ax.set_yscale('log')
                #ax.set_xticks([ft_e1[0].index[-1] - 7*day*i for i in range(int(30/7)+1)])#[dat.index.values[0],dat.index.values[-1]])#, ]np.arange(0, len(x)+1, 5))
                #ax1.set_xticks([t1 - 5*day*i for i in range(int(look_back/5)+1)])#[dat.index.values[0],dat.index.values[-1]])#, ]np.arange(0, len(x)+1, 5))
            #     #
            # except:
            #     pass
            if fts_yrigth:
                #ax1b = ax1.twinx() 
                col = ['r','g']
                alpha = [1., .5]
                thick_line = [2.,1.]
                #try: 
                for i, ft in enumerate(fts_yrigth):
                    if 'zsc2_dsarF' in ft:
                        ds = 'zsc2_dsarF'
                    if 'zsc2_mfF' in ft:
                        ds = 'zsc2_mfF' 
                    if 'zsc2_hfF' in ft:
                        ds = 'zsc2_hfF' 
                    # 
                    if False: # look feature in the prev cacl features (else: calculate feat from data; only for median and rv)
                        if server:
                            path_feat_serv = 'C:\\Users\\aar135\\codes_local_disk\\volc_forecast_tl\\features_bkp\\features_server\\'
                            fm_e1 = ForecastModel(window=2., overlap=1., station =  sta,
                                look_forward=2., data_streams=ds, 
                                feature_dir=path_feat_serv, 
                                savefile_type='pkl') 
                        elif server2:
                            path_feat_serv = 'U:\\Research\\EruptionForecasting\\eruptions\\features\\'
                            fm_e1 = ForecastModel(window=2., overlap=1., station =  sta,
                                look_forward=2., data_streams=ds, 
                                feature_dir=path_feat_serv, 
                                savefile_type='pkl') 
                        else:
                            try:
                                fm_e1 = ForecastModel(window=2., overlap=1., station = sta,
                                    look_forward=2., data_streams=ds, savefile_type='csv')
                            except:
                                fm_e1 = ForecastModel(window=2., overlap=1., station = sta,
                                    look_forward=2., data_streams=ds, savefile_type='pkl')                    
                        ##  
                        ft = ft.replace("-",'"')
                        
                        ft_e1 = fm_e1.get_features(ti=t0, tf=t1, n_jobs=1, compute_only_features=[ft])
                        # adding multiple Axes objects

                        # extract values to plot 
                        ft_e1_t = ft_e1[0].index.values
                        ft_e1_v = ft_e1[0].loc[:,ft]
                        #
                        v_plot = ft_e1_v

                    else:
                        day = timedelta(days=1)
                        fm = ForecastModel(window=2., overlap=1., station=sta,
                            look_forward=2., data_streams=[ds], 
                            data_dir=r'C:\Users\aar135\codes_local_disk\volc_forecast_tl\volc_forecast_tl\data'
                            )
                        #
                        N, M = [2,30]
                        df = fm.data.df[t0:t1]
                        if 'median' in ft:
                            test = df[ds].rolling(N*24*6).median()[N*24*6:]
                        if 'rate_variance' in ft:
                            test = df[ds].rolling(N*24*6).apply(chqv)[N*24*6:]
                        #
                        #out = test.rolling(archtype.shape[0]).apply(partial(conv, (archtype-archtype.mean())/archtype.std()))       
                        #out = out.resample('1D').ffill()
                        #
                        ft_e1_t = test.index
                        v_plot = test.values

                    #v_plot = ft_e1_v-np.min(ft_e1_v)/np.max((ft_e1_v-np.min(ft_e1_v)))
                    #v_plot = ft_e1_v/np.max(ft_e1_v)
                    #
                    if ft == 'zsc2_mfF__median':
                        ft = 'nMF median'
                    if ft == 'zsc2_hfF__median':
                        ft = 'nHF median'
                    #
                    ax.plot(ft_e1_t, v_plot/1e9, '-', color=col[i], alpha = alpha[i],label=' '+ ft, zorder = 4)
                    #
                    ax.legend(loc = 3)
                    #
                    te = datetimeify(erup_time)#fm_e1.data.tes[int(erup[-1:])-1]
                    #ax1.set_xticks([ft_e1[0].index[-1] - 7*day*i for i in range(int(30/7)+1)])#[dat.index.values[0],dat.index.values[-1]])#, ]np.arange(0, len(x)+1, 5))
                    
                    #ax1b.set_yticks([])
                    #ax.grid()
                    ax.set_ylabel('Feature value')        #ax.set_xticks([ft_e1[0].index[-1]-7*day*i +day for i in range(5)])
                    #ax.set_xticks([ft_e1[0].index[-1] - 7*day*i for i in range(int(30/7)+1)])#[dat.index.values[0],dat.index.values[-1]])#, ]np.arange(0, len(x)+1, 5))
                    #ax1.set_xticks([t1 - 5*day*i for i in range(int(look_back/5)+1)])#[dat.index.values[0],dat.index.values[-1]])#, ]np.arange(0, len(x)+1, 5))
                    #
                #except:
                #    pass

            else:
                pass
                if data_yrigth:
                    #
                    #ax1b = ax1.twinx() 
                    #
                    td = TremorData(station = sta)
                    #td.update(ti=t0, tf=t1)
                    data_streams = data_yrigth#['hf','mf', 'rsam']#, 'dsarF']
                    label = ['RSAM','MF','HF','DSAR']
                    #label = ['1/RSAM']
                    inv = False
                    if False:
                        data_streams = ['rsam']
                        label = ['RSAM']

                    if type(data_streams) is str:
                        data_streams = [data_streams,]
                    if any(['_' in ds for ds in data_streams]):
                        td._compute_transforms()
                    #ax.set_xlim(*range)
                    # plot data for each year
                    norm= False
                    _range = [t0,t1]
                    log =False
                    col_def = None
                    data = td.get_data(*_range)
                    xi = datetime(year=1,month=1,day=1,hour=0,minute=0,second=0)
                    cols = ['k','r','g','m',[0.5,0.5,0.5],[0.75,0.75,0.75]]
                    if inv:
                        cols = ['k','g','r','m',[0.5,0.5,0.5],[0.75,0.75,0.75]]
                    inds = (data.index>=datetimeify(_range[0]))&(data.index<=datetimeify(_range[1]))
                    i=0
                    for data_stream, col in zip(data_streams,cols):
                        v_plot = data[data_stream].loc[inds]
                        if inv:
                            v_plot = 1/v_plot
                        if log:
                            v_plot = np.log10(v_plot)
                        if norm:
                            v_plot = (v_plot-np.min(v_plot))/np.max((v_plot-np.min(v_plot)))
                        if label:
                            if col_def:
                                ax.plot(data.index[inds], v_plot/1e9, '-', color=col_def, label=label[i], linewidth=1., alpha = 1., zorder = 0)
                            else:
                                #ax1b.plot(data.index[inds], v_plot, '-', color=col, label=label[i], linewidth=1., alpha = 1.0, zorder = 0)
                                ax.plot(data.index[inds], v_plot/1e9, '-', color=col, linewidth=1., alpha = .7, zorder = 3)
                                axb.plot([], [], '-', color=col, label=label[i], linewidth=1., alpha = .7, zorder = 3)
                        else:
                            ax.plot(data.index[inds], v_plot, '-', color=col, label=data_stream, linewidth=1., alpha = .7, zorder = 3)
                        i+=1
                    for te in td.tes:
                        if [te>=datetimeify(_range[0]) and te<=datetimeify(_range[1])]:
                            pass
                            #ax.axvline(te, color='k', linestyle='--', linewidth=2, zorder = 0)
                    #
                    ax.plot([], color='k', linestyle='--', linewidth=1, label = 'eruption')
                    #ax4.set_xlim(_range)
                    #ax1b.legend(loc = 2)
                    #ax1b.grid()
                    if log:
                        ax.set_ylabel(' ')
                    else:
                        ax.set_ylabel('RSAM \u03BC m/s')
                    #ax4.set_xlabel('Time [month-day hour]')
                    #ax4.title.set_text('Station '+td.station+' ('+sta_code[td.station]+'): Tremor data')
                    #
                    #ax4.set_xticks([te - 5*day*i for i in range(int(look_back/5)+1)])#[dat.index.values[0],dat.index.values[-1]])#, ]np.arange(0, len(x)+1, 5))
                    #ax4.set_ylim([1e9,1e13])
                    ax.set_yscale('log')
            axb.legend(loc = 2)      
            axb.grid(False)
            ax.grid(color='gray', linestyle='-', linewidth=.5, alpha = 0.5)
            #

    for j, ax in enumerate([ax2, ax4, ax6]):
        t0 = erup_times[j] - look_back*day#30*day
        t1 = erup_times[j] + look_front*day#hour
        # subplot two: temp data (if any: level and rainfall)
        if True:  
            mov_avg = True # moving average for temp and level data
            # convert to UTC 0
            utc_0 = True
            if utc_0:
                _utc_0 = 0#-13 # hours
            # plot temp data
            if sta == 'FWVZ':
                # plot temperature data 
                if temp:
                    try:
                        ti_e1 = t0
                        tf_e1 = t1
                        # import temp data
                        path = '..'+os.sep+'data'+os.sep+"FWVZ_temp_data.csv"
                        pd_temp = pd.read_csv(path, index_col=1)
                        if utc_0:
                            #pd_temp.index = [datetimeify(pd_temp.index[i])-6*hour for i in range(len(pd_temp.index))]
                            pd_temp.index = [datetimeify(pd_temp.index[i])+_utc_0*hour for i in range(len(pd_temp.index))]
                        else:
                            pd_temp.index = [datetimeify(pd_temp.index[i]) for i in range(len(pd_temp.index))]
                        
                        # plot data in axis twin axis
                        # Trim the data
                        temp_e1_tim = pd_temp[ti_e1: tf_e1].index.values
                        #temp_e1_tim=to_nztimezone(temp_e1_tim)
                        #
                        temp_e1_val = pd_temp[ti_e1: tf_e1].loc[:,' t (C)'].values
                        # ax2
                        #ax2b = ax2.twinx()   
                        if mov_avg: # plot moving average
                            n=30
                            #v_plot = (temp_e1_val-np.min(temp_e1_val))/np.max((temp_e1_val-np.min(temp_e1_val)))
                            v_plot = temp_e1_val
                            ax.plot(temp_e1_tim, v_plot, '-', color='g', label='lake temperature', alpha = 1.)
                            #
                            #ax.plot(temp_e1_tim[:-n+1], moving_average(temp_e1_val, n=n), '--', color='k')#, label='temp. mov. avg.')
                            ax.plot(temp_e1_tim[n-1-20:-20], moving_average(v_plot[::-1], n=n)[::-1], '--', color='g')#, label='lake temperature')
                        else:
                            v_plot = temp_e1_val
                            ax.plot(temp_e1_tim, v_plot, '-', color='g', label='lake temperature', alpha = 1.)
                        ax.set_ylabel('Temperature °C')
                        #
                        temp_min = min(temp_e1_val)
                        temp_max = max(temp_e1_val)
                        temp_mu = np.mean(temp_e1_tim)
                        temp_sigma = np.std(temp_e1_tim)
                        ax.set_ylim([temp_mu-3*temp_sigma,temp_mu+3*temp_sigma])
                        #ax2.set_ylabel('temperature C')   
                    except:
                        pass

            if sta == 'DAM' or sta == 'POS':
                lake = False # no data
                rainfall = False # no data
                try:
                    if temp:
                        ti_e1 = t0
                        tf_e1 = t1
                        # import temp data
                        path = '..'+os.sep+'data'+os.sep+"DAM_temp_data.csv"
                        pd_temp = pd.read_csv(path, index_col=1)

                        if utc_0:
                            pd_temp.index = [datetimeify(pd_temp.index[i])-6*hour for i in range(len(pd_temp.index))]
                        else:
                            pd_temp.index = [datetimeify(pd_temp.index[i]) for i in range(len(pd_temp.index))]
                        # plot data in axis twin axis
                        # Trim the data
                        temp_e1_tim = pd_temp[ti_e1: tf_e1].index.values
                        #temp_e1_tim=to_nztimezone(temp_e1_tim)
                        #
                        temp_e1_val = pd_temp[ti_e1: tf_e1].loc[:,' t (C)'].values
                        # ax2
                        #ax2b = ax2.twinx()   
                        if mov_avg: # plot moving average
                            n=50
                            #v_plot = (temp_e1_val-np.min(temp_e1_val))/np.max((temp_e1_val-np.min(temp_e1_val)))
                            v_plot = temp_e1_val
                            ax.plot(temp_e1_tim, v_plot, '-', color='g', label='lake temperature', alpha = 1.)
                            #
                            #ax.plot(temp_e1_tim[:-n+1], moving_average(temp_e1_val, n=n), '--', color='k')#, label='temp. mov. avg.')
                            _x = temp_e1_tim[n-1-20:-20]
                            _y = moving_average(v_plot[::-1], n=n)[::-1]
                            ax.plot(_x, _y, '--', color='k')#, label='lake temperature')
                        else:
                            v_plot = temp_e1_val
                            ax.plot(temp_e1_tim, v_plot, '-', color='g', label='lake temperature', alpha = 1.)
                        ax.set_ylabel('Temperature °C')
                        
                        _ylim = [min(_y)-1,max(_y)+1] 
                        ax.set_ylim(_ylim)
                        #ax2.set_ylabel('temperature C')   
                except:
                    pass
                if True: # plot vertical lines
                    te = datetimeify(erup_times[j])#fm_e1.data.tes[int(erup[-1:])-1]
                    #ax.axvline(te, color='gray',linestyle='-', linewidth=2, zorder = 4)
                    ax.axvline(te, color='gray', alpha = .25, linestyle='-', linewidth=12, zorder = 0)
                    ax.plot([], color='gray', alpha = .25, linestyle='-', linewidth=12, label = 'eruption')
                    #ax2.plot([], color='k', linestyle='--', linewidth=2, label = 'eruption')

            ax.legend(loc = 2)   
            ax.grid(color='gray', linestyle='-', linewidth=.5, alpha = 0.5)
            #ax2.set_xticks([t1 - 5*day*i for i in range(int(look_back/5)+1)])#[dat.index.values[0],dat.index.values[-1]])#, ]np.arange(0, len(x)+1, 5))
        #    
    #
    if False: # plot vertical lines
        te = datetimeify(erup_time)#fm_e1.data.tes[int(erup[-1:])-1]
        ax1.axvline(te, color='k',linestyle='--', linewidth=2, zorder = 4)
        ax2.axvline(te, color='k',linestyle='--', linewidth=2, zorder = 4)
        ax3.axvline(te, color='k',linestyle='--', linewidth=2, zorder = 4)
        ax1.plot([], color='k', linestyle='--', linewidth=2, label = 'event')
    #
    if False:#save_png_path:
        dat = erup_time.strftime('%Y-%m-%d')
        title =  sta+'_'+dat+'_'+'look_back'+str(look_back)
        ax1.set_title(title)
        plt.savefig(save_png_path+sta+'_'+dat+'_'+'look_back'+str(look_back)+'.png')
    #
    _d = 5 
    t1 = erup_times[0] + look_front*day#hour
    ax1.set_xticks([t1 - _d*day*i for i in range(int((look_back+look_front)/_d)+1)])
    ax2.set_xticks([t1 - _d*day*i for i in range(int((look_back+look_front)/_d)+1)])
    #
    t1 = erup_times[1] + look_front*day#hour
    ax3.set_xticks([t1 - _d*day*i for i in range(int((look_back+look_front)/_d)+1)])
    ax4.set_xticks([t1 - _d*day*i for i in range(int((look_back+look_front)/_d)+1)])
    #
    t1 = erup_times[2] + look_front*day#hour
    ax5.set_xticks([t1 - _d*day*i for i in range(int((look_back+look_front)/_d)+1)])
    ax6.set_xticks([t1 - _d*day*i for i in range(int((look_back+look_front)/_d)+1)])
    #
    #t1 = erup_times[3] + look_front*day#hour
    #ax7.set_xticks([t1 - _d*day*i for i in range(int((look_back+look_front)/_d)+1)])
    #ax8.set_xticks([t1 - _d*day*i for i in range(int((look_back+look_front)/_d)+1)])
    #
    #ax3.set_xticks([t1 - 2*day*i for i in range(int((look_back+look_front)/2)+1)])
    #ax4.set_xticks([t1 - 4*day*i for i in range(int((look_back+look_front)/4)+1)])
    #    
    #ax.set_xlim([t0+2*day,t1])
    #ax.set_xlim([t0+2*day,t1])
    #ax1.set_ylim([10**.4,10**2.1])
    #ax3.set_ylim([10**.5,10**2])
    #ax4.set_ylim([1,100])
    #ax7.set_ylim([1,100])
    #ax2b.set_ylim([0.1,0.5])

    #ax3.set_xlim([t0+2*day,t1])
    #ax4.set_xlim([t0+2*day,t1])
    #
    # erup_times = [datetimeify('2009 07 13 06 30 00'), 
    #                 datetimeify('2010 09 03 12 00 00'),
    #                 datetimeify('2016 11 13 12 00 00'),
    #                 datetimeify('2021 03 04 12 00 00')]
    #ax1.set_title('(a) Kawah Ijen 2013/03/31 seismic RSAM and DSAR median')
    #ax3.set_title('(c) Kawah Ijen 2012/11/23 seismic RSAM and DSAR median')
    #ax5.set_title('(e) Kawah Ijen 2013/01/24 seismic RSAM and DSAR median')
    #ax7.set_title(' Ruapehu 2021/03/04 seismic RSAM and DSAR median')
    #
    ax2.set_title('(b) Kawah Ijen 2013/03/31 lake levels data')
    ax4.set_title('(d) Kawah Ijen 2012/11/23 lake levels data')
    ax6.set_title('(f) Kawah Ijen 2013/01/24 lake levels data')
    #ax8.set_title(' Ruapehu 2021/03/04 lake levels data')
    #ax1.set_title('(b) DSAR median and RSAM before hydrothermal event on 07/13')
    #ax2.set_title('(c) Lake temperature and level before hydrothermal event on 07/13')
    #ax4.set_title('Seismic datastreams before hydrothermal event on 07/13')
    #
    plt.tight_layout()
    plt.show()
    plt.close('all')

def figure_sup_copahue_events(): # events in seismic and lake levels
    '''
    plot: rsam and dsar before event (1), and lake levels (2), for multiple events in Ruapehu
    '''
    sta = 'COP' 
    if sta == 'COP':
        ffm = False
        server = False # files imported from server to local pc 
        server2 = False # server at uni 
        #plot_erup = False
    #
    look_back = 14
    look_front = 4
    #
    #erup_time = datetimeify('2009 07 13 06 30 00')
    #erup_time = datetimeify('2010 09 03 00 00 00')
    #erup_time = datetimeify('2021 03 04 12 00 00')
    erup_times = [datetimeify('2020 08 23 12 00 00'), 
                    datetimeify('2020 07 15 18 00 00'), #'2020 07 16 00 00 00'
                    datetimeify('2020 06 16 12 00 00'),
                    datetimeify('2020 08 08 12 00 00')]
    #
    #erup_time = datetimeify('2021 09 09 00 00 00')
    #
    day = timedelta(days=1)
    #t0 = erup_times - look_back*day#30*day
    #t1 = erup_times + look_front*day#hour
    #
    ## plot other data
    temp = True
    level = True
    rainfall = True
    ## 
    plot_erup = True
    # figure
    nrow = 4
    ncol = 1
    #fig, (ax0, ax1, ax2, ax4) = plt.subplots(nrows=nrow, ncols=ncol, figsize=(12,12))#(14,4)) #, ax4)
    fig, (ax1, ax3, ax5, ax7) = plt.subplots(nrows=nrow, ncols=ncol, figsize=(12,6))#(14,4)) #, ax4)
    #
    # for ax in [ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8]:#,ax4]: # plot eruption times 
    #     # plot event 
    #     #te = datetimeify('2009-07-07') 
    #     te = datetimeify('2009-07-13 06:30:00')
    #     ax.axvline(te+0.12*day, color='k',linestyle='--', linewidth=3, zorder = 4)
    #     ax.plot([], color='k', linestyle='--', linewidth=3, label = 'located event')
    #     # plot eruption 
    #     te = datetimeify('2009 07 13 06 30 00') 
    #     #ax.axvline(te+0.22*day, color='r',linestyle='-', linewidth=3, zorder = 4)
    #     #ax.plot([], color='r', linestyle='-', linewidth=3, label = 'eruption')  
  
    #####################################################
    for j,ax in enumerate([ax1, ax3, ax5, ax7]):
        #
        t0 = erup_times[j] - look_back*day#30*day
        t1 = erup_times[j] + look_front*day#hour
        # subplot one: MF, HF, DSAR medians (DSAR yaxis left; MF, HF yaxis rigth). 1/RSAM (normalized)
        if True:
            # features
            fts_yleft = ['zsc2_dsarF__median']
            fts_yrigth = []#['zsc2_dsarF__rate_variance']#['zsc2_dsarF__change_quantiles__f_agg_"var"__isabs_False__qh_0.6__ql_0.4']#['zsc2_mfF__median','zsc2_hfF__median']
            data_yrigth = ['rsamF']
            
            #
            col = ['b','b','r']
            alpha = [1., 1., .5]
            thick_line = [2., 6., 1.]
            axb = ax.twinx() 
            for i, ft in enumerate(fts_yleft):
                if True: # load feature (else: cal feature. median or rv)
                    if 'zsc2_dsarF' in ft:
                        ds = ['zsc2_dsarF'] 
                    if server:
                        path_feat_serv = 'C:\\Users\\aar135\\codes_local_disk\\volc_forecast_tl\\features_bkp\\features_server\\'
                        fm_e1 = ForecastModel(window=2., overlap=1., station =  sta,
                            look_forward=2., data_streams=ds, 
                            feature_dir=path_feat_serv, 
                            savefile_type='pkl') 
                    elif server2:
                        path_feat_serv = 'U:\\Research\\EruptionForecasting\\eruptions\\features\\'
                        fm_e1 = ForecastModel(window=2., overlap=1., station =  sta,
                            look_forward=2., data_streams=ds, 
                            feature_dir=path_feat_serv, 
                            savefile_type='pkl') 
                    else:
                        try:
                            fm_e1 = ForecastModel(window=2., overlap=1., station = sta,
                                look_forward=2., data_streams=ds, savefile_type='csv')
                        except:
                            fm_e1 = ForecastModel(window=2., overlap=1., station = sta,
                                look_forward=2., data_streams=ds, savefile_type='pkl')                    
                    ##  
                    ft = ft.replace("-",'"')
                    # adding multiple Axes objects
                    ft_e1 = fm_e1.get_features(ti=t0, tf=t1, n_jobs=1, compute_only_features=[ft])
                    # extract values to plot 
                    ft_e1_t = ft_e1[0].index.values
                    ft_e1_v = ft_e1[0].loc[:,ft]
                    #
                    v_plot = ft_e1_v

                    #v_plot = ft_e1_v-np.min(ft_e1_v)/np.max((ft_e1_v-np.min(ft_e1_v)))
                    #v_plot = ft_e1_v/np.max(ft_e1_v)
                else: 
                    #
                    if 'zsc2_dsarF' in ft:
                        ds = 'zsc2_dsarF'
                    if 'zsc2_mfF' in ft:
                        ds = 'zsc2_mfF' 
                    if 'zsc2_hfF' in ft:
                        ds = 'zsc2_hfF' 
                    # 
                    #
                    day = timedelta(days=1)
                    fm = ForecastModel(window=2., overlap=1., station=sta,
                        look_forward=2., data_streams=[ds], 
                        data_dir=r'C:\Users\aar135\codes_local_disk\volc_forecast_tl\volc_forecast_tl\data'
                        )
                    #
                    N, M = [2,30]
                    df = fm.data.df[t0:t1]
                    if 'median' in ft:
                        test = df[ds].rolling(N*24*6).median()[N*24*6:]
                    if 'rate_variance' in ft:
                        test = df[ds].rolling(N*24*6).apply(chqv)[N*24*6:]
                    #
                    #out = test.rolling(archtype.shape[0]).apply(partial(conv, (archtype-archtype.mean())/archtype.std()))       
                    #out = out.resample('1D').ffill()
                    #
                    ft_e1_t = test.index
                    v_plot = test.values
                #
                if ft == 'zsc2_dsarF__median':
                    ft = 'DSAR median'
                #
                axb.plot(ft_e1_t, v_plot, '-', color=col[i], alpha = alpha[i], linewidth = thick_line[i], label=' '+ ft, zorder = 2)
                #
                #
                if ffm: # ffm 
                    #ax1b = ax1.twinx() 
                    #v_plot = data[data_stream].loc[inds]
                    inv_rsam = fm_e1.data.get_data(ti=t0, tf=t1)['rsamF']#.loc[ft_e1_t]
                    inv_rsam = 1./inv_rsam
                    # normalized it to yaxis rigth 
                    inv_rsam = inv_rsam/max(inv_rsam)
                    inv_rsam = inv_rsam*0.5*max(v_plot)
                    #
                    ax.plot(ft_e1_t, inv_rsam, '-', color= 'gray', linewidth=0.5, markersize=0.5, alpha = 1.)
                    ax.plot([], [], '-', color= 'gray', markersize=1, label='1/RSAM', alpha = 1.0)
                    #ax1.set_ylim([0,1])
                    #ax1.set_yticks([])
                #
                if plot_erup: # plot vertical lines
                    te = datetimeify(erup_times[j])#fm_e1.data.tes[int(erup[-1:])-1]
                    ax.axvline(te, color='gray', alpha = .25, linestyle='-', linewidth=12, zorder = 0)
                    axb.plot([], color='gray', alpha = .25, linestyle='-', linewidth=12, label = 'eruption')
                #
                #ax1.legend(loc = 2)
                #
                te = datetimeify(erup_times[j])#fm_e1.data.tes[int(erup[-1:])-1]
                #ax1.set_xticks([ft_e1[0].index[-1] - 7*day*i for i in range(int(30/7)+1)])#[dat.index.values[0],dat.index.values[-1]])#, ]np.arange(0, len(x)+1, 5))
                
                #ax1b.set_yticks([])
                axb.grid()
                axb.set_ylabel('DSAR value')        #ax.set_xticks([ft_e1[0].index[-1]-7*day*i +day for i in range(5)])
                #ax1.set_yscale('log') #ax.set_yscale('log')
                #ax.set_xticks([ft_e1[0].index[-1] - 7*day*i for i in range(int(30/7)+1)])#[dat.index.values[0],dat.index.values[-1]])#, ]np.arange(0, len(x)+1, 5))
                #ax1.set_xticks([t1 - 5*day*i for i in range(int(look_back/5)+1)])#[dat.index.values[0],dat.index.values[-1]])#, ]np.arange(0, len(x)+1, 5))
            #     #
            # except:
            #     pass
            if fts_yrigth:
                #ax1b = ax1.twinx() 
                col = ['r','g']
                alpha = [1., .5]
                thick_line = [2.,1.]
                #try: 
                for i, ft in enumerate(fts_yrigth):
                    if 'zsc2_dsarF' in ft:
                        ds = 'zsc2_dsarF'
                    if 'zsc2_mfF' in ft:
                        ds = 'zsc2_mfF' 
                    if 'zsc2_hfF' in ft:
                        ds = 'zsc2_hfF' 
                    # 
                    if False: # look feature in the prev cacl features (else: calculate feat from data; only for median and rv)
                        if server:
                            path_feat_serv = 'C:\\Users\\aar135\\codes_local_disk\\volc_forecast_tl\\features_bkp\\features_server\\'
                            fm_e1 = ForecastModel(window=2., overlap=1., station =  sta,
                                look_forward=2., data_streams=ds, 
                                feature_dir=path_feat_serv, 
                                savefile_type='pkl') 
                        elif server2:
                            path_feat_serv = 'U:\\Research\\EruptionForecasting\\eruptions\\features\\'
                            fm_e1 = ForecastModel(window=2., overlap=1., station =  sta,
                                look_forward=2., data_streams=ds, 
                                feature_dir=path_feat_serv, 
                                savefile_type='pkl') 
                        else:
                            try:
                                fm_e1 = ForecastModel(window=2., overlap=1., station = sta,
                                    look_forward=2., data_streams=ds, savefile_type='csv')
                            except:
                                fm_e1 = ForecastModel(window=2., overlap=1., station = sta,
                                    look_forward=2., data_streams=ds, savefile_type='pkl')                    
                        ##  
                        ft = ft.replace("-",'"')
                        
                        ft_e1 = fm_e1.get_features(ti=t0, tf=t1, n_jobs=1, compute_only_features=[ft])
                        # adding multiple Axes objects

                        # extract values to plot 
                        ft_e1_t = ft_e1[0].index.values
                        ft_e1_v = ft_e1[0].loc[:,ft]
                        #
                        v_plot = ft_e1_v

                    else:
                        day = timedelta(days=1)
                        fm = ForecastModel(window=2., overlap=1., station=sta,
                            look_forward=2., data_streams=[ds], 
                            data_dir=r'C:\Users\aar135\codes_local_disk\volc_forecast_tl\volc_forecast_tl\data'
                            )
                        #
                        N, M = [2,30]
                        df = fm.data.df[t0:t1]
                        if 'median' in ft:
                            test = df[ds].rolling(N*24*6).median()[N*24*6:]
                        if 'rate_variance' in ft:
                            test = df[ds].rolling(N*24*6).apply(chqv)[N*24*6:]
                        #
                        #out = test.rolling(archtype.shape[0]).apply(partial(conv, (archtype-archtype.mean())/archtype.std()))       
                        #out = out.resample('1D').ffill()
                        #
                        ft_e1_t = test.index
                        v_plot = test.values

                    #v_plot = ft_e1_v-np.min(ft_e1_v)/np.max((ft_e1_v-np.min(ft_e1_v)))
                    #v_plot = ft_e1_v/np.max(ft_e1_v)
                    #
                    if ft == 'zsc2_mfF__median':
                        ft = 'nMF median'
                    if ft == 'zsc2_hfF__median':
                        ft = 'nHF median'
                    #
                    ax.plot(ft_e1_t, v_plot, '-', color=col[i], alpha = alpha[i],label=' '+ ft, zorder = 4)
                    #
                    ax.legend(loc = 3)
                    #
                    te = datetimeify(erup_time)#fm_e1.data.tes[int(erup[-1:])-1]
                    #ax1.set_xticks([ft_e1[0].index[-1] - 7*day*i for i in range(int(30/7)+1)])#[dat.index.values[0],dat.index.values[-1]])#, ]np.arange(0, len(x)+1, 5))
                    
                    #ax1b.set_yticks([])
                    #ax.grid()
                    ax.set_ylabel('Feature value')        #ax.set_xticks([ft_e1[0].index[-1]-7*day*i +day for i in range(5)])
                    #ax.set_xticks([ft_e1[0].index[-1] - 7*day*i for i in range(int(30/7)+1)])#[dat.index.values[0],dat.index.values[-1]])#, ]np.arange(0, len(x)+1, 5))
                    #ax1.set_xticks([t1 - 5*day*i for i in range(int(look_back/5)+1)])#[dat.index.values[0],dat.index.values[-1]])#, ]np.arange(0, len(x)+1, 5))
                    #
                #except:
                #    pass

            else:
                pass
                if data_yrigth:
                    #
                    #ax1b = ax1.twinx() 
                    #
                    td = TremorData(station = sta)
                    #td.update(ti=t0, tf=t1)
                    data_streams = data_yrigth#['hf','mf', 'rsam']#, 'dsarF']
                    label = ['RSAM','MF','HF','DSAR']
                    #label = ['1/RSAM']
                    inv = False
                    if False:
                        data_streams = ['rsam']
                        label = ['RSAM']

                    if type(data_streams) is str:
                        data_streams = [data_streams,]
                    if any(['_' in ds for ds in data_streams]):
                        td._compute_transforms()
                    #ax.set_xlim(*range)
                    # plot data for each year
                    norm= False
                    _range = [t0,t1]
                    log =False
                    col_def = None
                    data = td.get_data(*_range)
                    xi = datetime(year=1,month=1,day=1,hour=0,minute=0,second=0)
                    cols = ['k','r','g','m',[0.5,0.5,0.5],[0.75,0.75,0.75]]
                    if inv:
                        cols = ['k','g','r','m',[0.5,0.5,0.5],[0.75,0.75,0.75]]
                    inds = (data.index>=datetimeify(_range[0]))&(data.index<=datetimeify(_range[1]))
                    i=0
                    for data_stream, col in zip(data_streams,cols):
                        v_plot = data[data_stream].loc[inds]
                        if inv:
                            v_plot = 1/v_plot
                        if log:
                            v_plot = np.log10(v_plot)
                        if norm:
                            v_plot = (v_plot-np.min(v_plot))/np.max((v_plot-np.min(v_plot)))
                        if label:
                            if col_def:
                                ax.plot(data.index[inds], v_plot, '-', color=col_def, label=label[i], linewidth=1., alpha = 1., zorder = 0)
                            else:
                                #ax1b.plot(data.index[inds], v_plot, '-', color=col, label=label[i], linewidth=1., alpha = 1.0, zorder = 0)
                                ax.plot(data.index[inds], v_plot*1e9, '-', color=col, linewidth=1., alpha = .7, zorder = 3)
                                axb.plot([], [], '-', color=col, label=label[i], linewidth=1., alpha = .7, zorder = 3)
                        else:
                            ax.plot(data.index[inds], v_plot*1e9, '-', color=col, label=data_stream, linewidth=1., alpha = .7, zorder = 3)
                        i+=1
                    for te in td.tes:
                        if [te>=datetimeify(_range[0]) and te<=datetimeify(_range[1])]:
                            pass
                            #ax.axvline(te, color='k', linestyle='--', linewidth=2, zorder = 0)
                    #
                    ax.plot([], color='k', linestyle='--', linewidth=1, label = 'eruption')
                    #ax4.set_xlim(_range)
                    #ax1b.legend(loc = 2)
                    #ax1b.grid()
                    if log:
                        ax.set_ylabel(' ')
                    else:
                        ax.set_ylabel('RSAM \u03BC m/s')
                    #ax4.set_xlabel('Time [month-day hour]')
                    #ax4.title.set_text('Station '+td.station+' ('+sta_code[td.station]+'): Tremor data')
                    #
                    #ax4.set_xticks([te - 5*day*i for i in range(int(look_back/5)+1)])#[dat.index.values[0],dat.index.values[-1]])#, ]np.arange(0, len(x)+1, 5))
                    #ax4.set_ylim([1e9,1e13])
                    #ax.set_yscale('log')
            axb.legend(loc = 2)      
            axb.grid(False)
            ax.grid(color='gray', linestyle='-', linewidth=.5, alpha = 0.5)
            #



            ax.legend(loc = 2)   
            ax.grid(color='gray', linestyle='-', linewidth=.5, alpha = 0.5)
            #ax2.set_xticks([t1 - 5*day*i for i in range(int(look_back/5)+1)])#[dat.index.values[0],dat.index.values[-1]])#, ]np.arange(0, len(x)+1, 5))
        #
        
    if False: # plot vertical lines
        te = datetimeify(erup_time)#fm_e1.data.tes[int(erup[-1:])-1]
        ax1.axvline(te, color='k',linestyle='--', linewidth=2, zorder = 4)
        ax2.axvline(te, color='k',linestyle='--', linewidth=2, zorder = 4)
        ax3.axvline(te, color='k',linestyle='--', linewidth=2, zorder = 4)
        ax1.plot([], color='k', linestyle='--', linewidth=2, label = 'event')
    #
    if False:#save_png_path:
        dat = erup_time.strftime('%Y-%m-%d')
        title =  sta+'_'+dat+'_'+'look_back'+str(look_back)
        ax1.set_title(title)
        plt.savefig(save_png_path+sta+'_'+dat+'_'+'look_back'+str(look_back)+'.png')
    #
    _d = 5 
    t1 = erup_times[0] + look_front*day#hour
    ax1.set_xticks([t1 - _d*day*i for i in range(int((look_back+look_front)/_d)+1)])
    #ax2.set_xticks([t1 - _d*day*i for i in range(int((look_back+look_front)/_d)+1)])
    #
    t1 = erup_times[1] + look_front*day#hour
    ax3.set_xticks([t1 - _d*day*i for i in range(int((look_back+look_front)/_d)+1)])
    #ax4.set_xticks([t1 - _d*day*i for i in range(int((look_back+look_front)/_d)+1)])
    #
    t1 = erup_times[2] + look_front*day#hour
    ax5.set_xticks([t1 - _d*day*i for i in range(int((look_back+look_front)/_d)+1)])
    #ax6.set_xticks([t1 - _d*day*i for i in range(int((look_back+look_front)/_d)+1)])
    #
    t1 = erup_times[3] + look_front*day#hour
    ax7.set_xticks([t1 - _d*day*i for i in range(int((look_back+look_front)/_d)+1)])
    #ax8.set_xticks([t1 - _d*day*i for i in range(int((look_back+look_front)/_d)+1)])
    #
    #ax3.set_xticks([t1 - 2*day*i for i in range(int((look_back+look_front)/2)+1)])
    #ax4.set_xticks([t1 - 4*day*i for i in range(int((look_back+look_front)/4)+1)])
    #    
    #ax.set_xlim([t0+2*day,t1])
    #ax.set_xlim([t0+2*day,t1])
    #ax1.set_ylim([10**.4,10**2.1])
    #ax3.set_ylim([10**.5,10**2])
    #ax4.set_ylim([1,100])
    #ax7.set_ylim([1,100])
    #ax2b.set_ylim([0.1,0.5])

    #ax3.set_xlim([t0+2*day,t1])
    #ax4.set_xlim([t0+2*day,t1])
    #
    # erup_times = [datetimeify('2009 07 13 06 30 00'), 
    #                 datetimeify('2010 09 03 12 00 00'),
    #                 datetimeify('2016 11 13 12 00 00'),
    #                 datetimeify('2021 03 04 12 00 00')]
    ax1.set_title('(a) Copahue 2020/08/23 seismic RSAM and DSAR median')
    ax3.set_title('(b) Copahue 2020/07/16 seismic RSAM and DSAR median')
    ax5.set_title('(c) Copahue 2020/06/16 seismic RSAM and DSAR median')
    ax7.set_title('(d) Copahue 2020/08/08 seismic RSAM and DSAR median')
    #
    #ax2.set_title(' Ruapehu 2009/07/13 lake levels data')
    #ax4.set_title(' Ruapehu 2010/09/03 lake levels data')
    #ax6.set_title(' Ruapehu 2016/11/13 lake levels data')
    #ax8.set_title(' Ruapehu 2021/03/04 lake levels data')
    #ax1.set_title('(b) DSAR median and RSAM before hydrothermal event on 07/13')
    #ax2.set_title('(c) Lake temperature and level before hydrothermal event on 07/13')
    #ax4.set_title('Seismic datastreams before hydrothermal event on 07/13')
    #
    plt.tight_layout()
    plt.show()
    plt.close('all')

def figure_5_ruapehu():
    '''
    plot events: seismic from multiple events (rsam and dsar)
    '''
    sta = 'FWVZ' 
    if sta == 'FWVZ':
        ffm = False
        server = False # files imported from server to local pc 
        server2 = True # server at uni 
        #plot_erup = False
    #
    look_back = 14
    look_front = 7
    #
    day = timedelta(days=1)
    #
    ## plot other data
    temp = True
    level = True
    rainfall = True
    ## 
    plot_erup = True
    # figure
    nrow = 3
    ncol = 2
    #fig, (ax0, ax1, ax2, ax4) = plt.subplots(nrows=nrow, ncols=ncol, figsize=(12,12))#(14,4)) #, ax4)
    fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(nrows=nrow, ncols=ncol, figsize=(18,12))#(14,4)) #, ax4)

    #####################################################
    # First column
    erup_times = [datetimeify('2006 10 04 09 30 00'), 
                    datetimeify('2009 07 13 06 30 00'), 
                    datetimeify('2010 09 03 16 00 00')]
    for j,ax in enumerate([ax1, ax3, ax5]):
        #
        t0 = erup_times[j] - look_back*day#30*day
        t1 = erup_times[j] + look_front*day#hour
        # subplot one: MF, HF, DSAR medians (DSAR yaxis left; MF, HF yaxis rigth). 1/RSAM (normalized)
        if True:
            # features
            fts_yleft = ['zsc2_dsarF__median']
            fts_yrigth = []#['zsc2_dsarF__rate_variance']#['zsc2_dsarF__change_quantiles__f_agg_"var"__isabs_False__qh_0.6__ql_0.4']#['zsc2_mfF__median','zsc2_hfF__median']
            data_yrigth = ['rsam']
            
            #
            col = ['b','b','r']
            alpha = [1., 1., .5]
            thick_line = [2., 6., 1.]
            axb = ax.twinx() 
            for i, ft in enumerate(fts_yleft):
                if True: # load feature (else: cal feature. median or rv)
                    if 'zsc2_dsarF' in ft:
                        ds = ['zsc2_dsarF'] 
                    if server:
                        path_feat_serv = 'C:\\Users\\aar135\\codes_local_disk\\volc_forecast_tl\\features_bkp\\features_server\\'
                        fm_e1 = ForecastModel(window=2., overlap=1., station =  sta,
                            look_forward=2., data_streams=ds, 
                            feature_dir=path_feat_serv, 
                            savefile_type='pkl') 
                    elif server2:
                        path_feat_serv = 'U:\\Research\\EruptionForecasting\\eruptions\\features\\'
                        fm_e1 = ForecastModel(window=2., overlap=1., station =  sta,
                            look_forward=2., data_streams=ds, 
                            feature_dir=path_feat_serv, 
                            savefile_type='pkl') 
                    else:
                        try:
                            fm_e1 = ForecastModel(window=2., overlap=1., station = sta,
                                look_forward=2., data_streams=ds, savefile_type='csv')
                        except:
                            fm_e1 = ForecastModel(window=2., overlap=1., station = sta,
                                look_forward=2., data_streams=ds, savefile_type='pkl')                    
                    ##  
                    ft = ft.replace("-",'"')
                    # adding multiple Axes objects
                    ft_e1 = fm_e1.get_features(ti=t0, tf=t1, n_jobs=1, compute_only_features=[ft])
                    # extract values to plot 
                    ft_e1_t = ft_e1[0].index.values
                    ft_e1_v = ft_e1[0].loc[:,ft]
                    #
                    v_plot = ft_e1_v

                    #v_plot = ft_e1_v-np.min(ft_e1_v)/np.max((ft_e1_v-np.min(ft_e1_v)))
                    #v_plot = ft_e1_v/np.max(ft_e1_v)
                else: 
                    #
                    if 'zsc2_dsarF' in ft:
                        ds = 'zsc2_dsarF'
                    if 'zsc2_mfF' in ft:
                        ds = 'zsc2_mfF' 
                    if 'zsc2_hfF' in ft:
                        ds = 'zsc2_hfF' 
                    # 
                    #
                    day = timedelta(days=1)
                    fm = ForecastModel(window=2., overlap=1., station=sta,
                        look_forward=2., data_streams=[ds], 
                        data_dir=r'C:\Users\aar135\codes_local_disk\volc_forecast_tl\volc_forecast_tl\data'
                        )
                    #
                    N, M = [2,30]
                    df = fm.data.df[t0:t1]
                    if 'median' in ft:
                        test = df[ds].rolling(N*24*6).median()[N*24*6:]
                    if 'rate_variance' in ft:
                        test = df[ds].rolling(N*24*6).apply(chqv)[N*24*6:]
                    #
                    #out = test.rolling(archtype.shape[0]).apply(partial(conv, (archtype-archtype.mean())/archtype.std()))       
                    #out = out.resample('1D').ffill()
                    #
                    ft_e1_t = test.index
                    v_plot = test.values
                #
                if ft == 'zsc2_dsarF__median':
                    ft = 'nDSAR median'
                #
                axb.plot(ft_e1_t, v_plot, '-', color=col[i], alpha = alpha[i], linewidth = thick_line[i], label=' '+ ft, zorder = 2)
                #
                #
                if ffm: # ffm 
                    #ax1b = ax1.twinx() 
                    #v_plot = data[data_stream].loc[inds]
                    inv_rsam = fm_e1.data.get_data(ti=t0, tf=t1)['rsamF']#.loc[ft_e1_t]
                    inv_rsam = 1./inv_rsam
                    # normalized it to yaxis rigth 
                    inv_rsam = inv_rsam/max(inv_rsam)
                    inv_rsam = inv_rsam*0.5*max(v_plot)
                    #
                    ax.plot(ft_e1_t, inv_rsam, '-', color= 'gray', linewidth=0.5, markersize=0.5, alpha = 1.)
                    ax.plot([], [], '-', color= 'gray', markersize=1, label='1/RSAM', alpha = 1.0)
                    #ax1.set_ylim([0,1])
                    #ax1.set_yticks([])
                #
                if plot_erup: # plot vertical lines
                    te = datetimeify(erup_times[j])#fm_e1.data.tes[int(erup[-1:])-1]
                    if j == 0:
                        ax.axvline(te, color='red', alpha = .25, linestyle='-', linewidth=12, zorder = 0)
                        axb.plot([], color='red', alpha = .25, linestyle='-', linewidth=12, label = 'eruption')
                    else:
                        ax.axvline(te, color='gray', alpha = .25, linestyle='-', linewidth=12, zorder = 0)
                        axb.plot([], color='gray', alpha = .25, linestyle='-', linewidth=12, label = 'fluid release event')
                #
                #ax1.legend(loc = 2)
                #
                te = datetimeify(erup_times[j])#fm_e1.data.tes[int(erup[-1:])-1]
                #ax1.set_xticks([ft_e1[0].index[-1] - 7*day*i for i in range(int(30/7)+1)])#[dat.index.values[0],dat.index.values[-1]])#, ]np.arange(0, len(x)+1, 5))
                
                #ax1b.set_yticks([])
                axb.grid()
                axb.set_ylabel('nDSAR value')        #ax.set_xticks([ft_e1[0].index[-1]-7*day*i +day for i in range(5)])
                #ax1.set_yscale('log') #ax.set_yscale('log')
                #ax.set_xticks([ft_e1[0].index[-1] - 7*day*i for i in range(int(30/7)+1)])#[dat.index.values[0],dat.index.values[-1]])#, ]np.arange(0, len(x)+1, 5))
                #ax1.set_xticks([t1 - 5*day*i for i in range(int(look_back/5)+1)])#[dat.index.values[0],dat.index.values[-1]])#, ]np.arange(0, len(x)+1, 5))
            #     #
            # except:
            #     pass
            if fts_yrigth:
                #ax1b = ax1.twinx() 
                col = ['r','g']
                alpha = [1., .5]
                thick_line = [2.,1.]
                #try: 
                for i, ft in enumerate(fts_yrigth):
                    if 'zsc2_dsarF' in ft:
                        ds = 'zsc2_dsarF'
                    if 'zsc2_mfF' in ft:
                        ds = 'zsc2_mfF' 
                    if 'zsc2_hfF' in ft:
                        ds = 'zsc2_hfF' 
                    # 
                    if False: # look feature in the prev cacl features (else: calculate feat from data; only for median and rv)
                        if server:
                            path_feat_serv = 'C:\\Users\\aar135\\codes_local_disk\\volc_forecast_tl\\features_bkp\\features_server\\'
                            fm_e1 = ForecastModel(window=2., overlap=1., station =  sta,
                                look_forward=2., data_streams=ds, 
                                feature_dir=path_feat_serv, 
                                savefile_type='pkl') 
                        elif server2:
                            path_feat_serv = 'U:\\Research\\EruptionForecasting\\eruptions\\features\\'
                            fm_e1 = ForecastModel(window=2., overlap=1., station =  sta,
                                look_forward=2., data_streams=ds, 
                                feature_dir=path_feat_serv, 
                                savefile_type='pkl') 
                        else:
                            try:
                                fm_e1 = ForecastModel(window=2., overlap=1., station = sta,
                                    look_forward=2., data_streams=ds, savefile_type='csv')
                            except:
                                fm_e1 = ForecastModel(window=2., overlap=1., station = sta,
                                    look_forward=2., data_streams=ds, savefile_type='pkl')                    
                        ##  
                        ft = ft.replace("-",'"')
                        
                        ft_e1 = fm_e1.get_features(ti=t0, tf=t1, n_jobs=1, compute_only_features=[ft])
                        # adding multiple Axes objects

                        # extract values to plot 
                        ft_e1_t = ft_e1[0].index.values
                        ft_e1_v = ft_e1[0].loc[:,ft]
                        #
                        v_plot = ft_e1_v

                    else:
                        day = timedelta(days=1)
                        fm = ForecastModel(window=2., overlap=1., station=sta,
                            look_forward=2., data_streams=[ds], 
                            data_dir=r'C:\Users\aar135\codes_local_disk\volc_forecast_tl\volc_forecast_tl\data'
                            )
                        #
                        N, M = [2,30]
                        df = fm.data.df[t0:t1]
                        if 'median' in ft:
                            test = df[ds].rolling(N*24*6).median()[N*24*6:]
                        if 'rate_variance' in ft:
                            test = df[ds].rolling(N*24*6).apply(chqv)[N*24*6:]
                        #
                        #out = test.rolling(archtype.shape[0]).apply(partial(conv, (archtype-archtype.mean())/archtype.std()))       
                        #out = out.resample('1D').ffill()
                        #
                        ft_e1_t = test.index
                        v_plot = test.values

                    #v_plot = ft_e1_v-np.min(ft_e1_v)/np.max((ft_e1_v-np.min(ft_e1_v)))
                    #v_plot = ft_e1_v/np.max(ft_e1_v)
                    #
                    if ft == 'zsc2_mfF__median':
                        ft = 'nMF median'
                    if ft == 'zsc2_hfF__median':
                        ft = 'nHF median'
                    #
                    ax.plot(ft_e1_t, v_plot, '-', color=col[i], alpha = alpha[i],label=' '+ ft, zorder = 4)
                    #
                    ax.legend(loc = 3)
                    #
                    te = datetimeify(erup_time)#fm_e1.data.tes[int(erup[-1:])-1]
                    #ax1.set_xticks([ft_e1[0].index[-1] - 7*day*i for i in range(int(30/7)+1)])#[dat.index.values[0],dat.index.values[-1]])#, ]np.arange(0, len(x)+1, 5))
                    
                    #ax1b.set_yticks([])
                    #ax.grid()
                    ax.set_ylabel('Feature value')        #ax.set_xticks([ft_e1[0].index[-1]-7*day*i +day for i in range(5)])
                    #ax.set_xticks([ft_e1[0].index[-1] - 7*day*i for i in range(int(30/7)+1)])#[dat.index.values[0],dat.index.values[-1]])#, ]np.arange(0, len(x)+1, 5))
                    #ax1.set_xticks([t1 - 5*day*i for i in range(int(look_back/5)+1)])#[dat.index.values[0],dat.index.values[-1]])#, ]np.arange(0, len(x)+1, 5))
                    #
                #except:
                #    pass

            else:
                pass
                if data_yrigth:
                    #
                    #ax1b = ax1.twinx() 
                    #
                    td = TremorData(station = sta)
                    #td.update(ti=t0, tf=t1)
                    data_streams = data_yrigth#['hf','mf', 'rsam']#, 'dsarF']
                    label = ['RSAM','MF','HF','DSAR']
                    #label = ['1/RSAM']
                    inv = False
                    if False:
                        data_streams = ['rsam']
                        label = ['RSAM']

                    if type(data_streams) is str:
                        data_streams = [data_streams,]
                    if any(['_' in ds for ds in data_streams]):
                        td._compute_transforms()
                    #ax.set_xlim(*range)
                    # plot data for each year
                    norm= False
                    _range = [t0,t1]
                    log =False
                    col_def = None
                    data = td.get_data(*_range)
                    xi = datetime(year=1,month=1,day=1,hour=0,minute=0,second=0)
                    cols = ['k','r','g','m',[0.5,0.5,0.5],[0.75,0.75,0.75]]
                    if inv:
                        cols = ['k','g','r','m',[0.5,0.5,0.5],[0.75,0.75,0.75]]
                    inds = (data.index>=datetimeify(_range[0]))&(data.index<=datetimeify(_range[1]))
                    i=0
                    for data_stream, col in zip(data_streams,cols):
                        v_plot = data[data_stream].loc[inds]
                        if inv:
                            v_plot = 1/v_plot
                        if log:
                            v_plot = np.log10(v_plot)
                        if norm:
                            v_plot = (v_plot-np.min(v_plot))/np.max((v_plot-np.min(v_plot)))
                        if label:
                            if col_def:
                                ax.plot(data.index[inds], v_plot, '-', color=col_def, label=label[i], linewidth=1., alpha = 1., zorder = 0)
                            else:
                                #ax1b.plot(data.index[inds], v_plot, '-', color=col, label=label[i], linewidth=1., alpha = 1.0, zorder = 0)
                                ax.plot(data.index[inds], v_plot, '-', color=col, linewidth=1., alpha = .7, zorder = 3)
                                axb.plot([], [], '-', color=col, label=label[i], linewidth=1., alpha = .7, zorder = 3)
                        else:
                            ax.plot(data.index[inds], v_plot, '-', color=col, label=data_stream, linewidth=1., alpha = .7, zorder = 3)
                        i+=1
                    for te in td.tes:
                        if [te>=datetimeify(_range[0]) and te<=datetimeify(_range[1])]:
                            pass
                            #ax.axvline(te, color='k', linestyle='--', linewidth=2, zorder = 0)
                    #
                    ax.plot([], color='k', linestyle='--', linewidth=1, label = 'eruption')
                    #ax4.set_xlim(_range)
                    #ax1b.legend(loc = 2)
                    #ax1b.grid()
                    if log:
                        ax.set_ylabel(' ')
                    else:
                        ax.set_ylabel('RSAM \u03BC m/s')
                    #ax4.set_xlabel('Time [month-day hour]')
                    #ax4.title.set_text('Station '+td.station+' ('+sta_code[td.station]+'): Tremor data')
                    #
                    #ax4.set_xticks([te - 5*day*i for i in range(int(look_back/5)+1)])#[dat.index.values[0],dat.index.values[-1]])#, ]np.arange(0, len(x)+1, 5))
                    #ax4.set_ylim([1e9,1e13])
                    ax.set_yscale('log')
            axb.legend(loc = 2)      
            axb.grid(False)
            ax.grid(color='gray', linestyle='-', linewidth=.5, alpha = 0.5)
            #

    _d = 5 
    t1 = erup_times[0] + look_front*day#hour
    ax1.set_xticks([t1 - _d*day*i for i in range(int((look_back+look_front)/_d)+1)])
    t3 = erup_times[1] + look_front*day#hour
    ax3.set_xticks([t3 - _d*day*i for i in range(int((look_back+look_front)/_d)+1)])
    t5 = erup_times[2] + look_front*day#hour
    ax5.set_xticks([t5 - _d*day*i for i in range(int((look_back+look_front)/_d)+1)])
    
    #####################################################
    # Second column
    erup_times = [datetimeify('2007 09 25 08 20 00'), 
                    datetimeify('2016 11 13 12 00 00'),
                    datetimeify('2021 03 04 12 00 00')]
    for j,ax in enumerate([ax2, ax4, ax6]):
        #
        t0 = erup_times[j] - look_back*day#30*day
        t1 = erup_times[j] + look_front*day#hour
        # subplot one: MF, HF, DSAR medians (DSAR yaxis left; MF, HF yaxis rigth). 1/RSAM (normalized)
        if True:
            # features
            fts_yleft = ['zsc2_dsarF__median']
            fts_yrigth = []#['zsc2_dsarF__rate_variance']#['zsc2_dsarF__change_quantiles__f_agg_"var"__isabs_False__qh_0.6__ql_0.4']#['zsc2_mfF__median','zsc2_hfF__median']
            data_yrigth = ['rsam']
            #
            col = ['b','b','r']
            alpha = [1., 1., .5]
            thick_line = [2., 6., 1.]
            axb = ax.twinx() 
            for i, ft in enumerate(fts_yleft):
                if True: # load feature (else: cal feature. median or rv)
                    if 'zsc2_dsarF' in ft:
                        ds = ['zsc2_dsarF'] 
                    if server:
                        path_feat_serv = 'C:\\Users\\aar135\\codes_local_disk\\volc_forecast_tl\\features_bkp\\features_server\\'
                        fm_e1 = ForecastModel(window=2., overlap=1., station =  sta,
                            look_forward=2., data_streams=ds, 
                            feature_dir=path_feat_serv, 
                            savefile_type='pkl') 
                    elif server2:
                        path_feat_serv = 'U:\\Research\\EruptionForecasting\\eruptions\\features\\'
                        fm_e1 = ForecastModel(window=2., overlap=1., station =  sta,
                            look_forward=2., data_streams=ds, 
                            feature_dir=path_feat_serv, 
                            savefile_type='pkl') 
                    else:
                        try:
                            fm_e1 = ForecastModel(window=2., overlap=1., station = sta,
                                look_forward=2., data_streams=ds, savefile_type='csv')
                        except:
                            fm_e1 = ForecastModel(window=2., overlap=1., station = sta,
                                look_forward=2., data_streams=ds, savefile_type='pkl')                    
                    ##  
                    ft = ft.replace("-",'"')
                    # adding multiple Axes objects
                    ft_e1 = fm_e1.get_features(ti=t0, tf=t1, n_jobs=1, compute_only_features=[ft])
                    # extract values to plot 
                    ft_e1_t = ft_e1[0].index.values
                    ft_e1_v = ft_e1[0].loc[:,ft]
                    #
                    v_plot = ft_e1_v

                    #v_plot = ft_e1_v-np.min(ft_e1_v)/np.max((ft_e1_v-np.min(ft_e1_v)))
                    #v_plot = ft_e1_v/np.max(ft_e1_v)
                else: 
                    #
                    if 'zsc2_dsarF' in ft:
                        ds = 'zsc2_dsarF'
                    if 'zsc2_mfF' in ft:
                        ds = 'zsc2_mfF' 
                    if 'zsc2_hfF' in ft:
                        ds = 'zsc2_hfF' 
                    # 
                    #
                    day = timedelta(days=1)
                    fm = ForecastModel(window=2., overlap=1., station=sta,
                        look_forward=2., data_streams=[ds], 
                        data_dir=r'C:\Users\aar135\codes_local_disk\volc_forecast_tl\volc_forecast_tl\data'
                        )
                    #
                    N, M = [2,30]
                    df = fm.data.df[t0:t1]
                    if 'median' in ft:
                        test = df[ds].rolling(N*24*6).median()[N*24*6:]
                    if 'rate_variance' in ft:
                        test = df[ds].rolling(N*24*6).apply(chqv)[N*24*6:]
                    #
                    #out = test.rolling(archtype.shape[0]).apply(partial(conv, (archtype-archtype.mean())/archtype.std()))       
                    #out = out.resample('1D').ffill()
                    #
                    ft_e1_t = test.index
                    v_plot = test.values
                #
                if ft == 'zsc2_dsarF__median':
                    ft = 'nDSAR median'
                #
                axb.plot(ft_e1_t, v_plot, '-', color=col[i], alpha = alpha[i], linewidth = thick_line[i], label=' '+ ft, zorder = 2)
                #
                #
                if ffm: # ffm 
                    #ax1b = ax1.twinx() 
                    #v_plot = data[data_stream].loc[inds]
                    inv_rsam = fm_e1.data.get_data(ti=t0, tf=t1)['rsamF']#.loc[ft_e1_t]
                    inv_rsam = 1./inv_rsam
                    # normalized it to yaxis rigth 
                    inv_rsam = inv_rsam/max(inv_rsam)
                    inv_rsam = inv_rsam*0.5*max(v_plot)
                    #
                    ax.plot(ft_e1_t, inv_rsam, '-', color= 'gray', linewidth=0.5, markersize=0.5, alpha = 1.)
                    ax.plot([], [], '-', color= 'gray', markersize=1, label='1/RSAM', alpha = 1.0)
                    #ax1.set_ylim([0,1])
                    #ax1.set_yticks([])
                #
                if plot_erup: # plot vertical lines
                    te = datetimeify(erup_times[j])#fm_e1.data.tes[int(erup[-1:])-1]
                    if j == 0:
                        ax.axvline(te, color='red', alpha = .25, linestyle='-', linewidth=12, zorder = 0)
                        axb.plot([], color='red', alpha = .25, linestyle='-', linewidth=12, label = 'eruption')
                    else:
                        ax.axvline(te, color='gray', alpha = .25, linestyle='-', linewidth=12, zorder = 0)
                        axb.plot([], color='gray', alpha = .25, linestyle='-', linewidth=12, label = 'fluid release event')
                #
                #ax1.legend(loc = 2)
                #
                te = datetimeify(erup_times[j])#fm_e1.data.tes[int(erup[-1:])-1]
                #ax1.set_xticks([ft_e1[0].index[-1] - 7*day*i for i in range(int(30/7)+1)])#[dat.index.values[0],dat.index.values[-1]])#, ]np.arange(0, len(x)+1, 5))
                
                #ax1b.set_yticks([])
                axb.grid()
                axb.set_ylabel('nDSAR value')        #ax.set_xticks([ft_e1[0].index[-1]-7*day*i +day for i in range(5)])
                #ax1.set_yscale('log') #ax.set_yscale('log')
                #ax.set_xticks([ft_e1[0].index[-1] - 7*day*i for i in range(int(30/7)+1)])#[dat.index.values[0],dat.index.values[-1]])#, ]np.arange(0, len(x)+1, 5))
                #ax1.set_xticks([t1 - 5*day*i for i in range(int(look_back/5)+1)])#[dat.index.values[0],dat.index.values[-1]])#, ]np.arange(0, len(x)+1, 5))
            #     #
            # except:
            #     pass
            if fts_yrigth:
                #ax1b = ax1.twinx() 
                col = ['r','g']
                alpha = [1., .5]
                thick_line = [2.,1.]
                #try: 
                for i, ft in enumerate(fts_yrigth):
                    if 'zsc2_dsarF' in ft:
                        ds = 'zsc2_dsarF'
                    if 'zsc2_mfF' in ft:
                        ds = 'zsc2_mfF' 
                    if 'zsc2_hfF' in ft:
                        ds = 'zsc2_hfF' 
                    # 
                    if False: # look feature in the prev cacl features (else: calculate feat from data; only for median and rv)
                        if server:
                            path_feat_serv = 'C:\\Users\\aar135\\codes_local_disk\\volc_forecast_tl\\features_bkp\\features_server\\'
                            fm_e1 = ForecastModel(window=2., overlap=1., station =  sta,
                                look_forward=2., data_streams=ds, 
                                feature_dir=path_feat_serv, 
                                savefile_type='pkl') 
                        elif server2:
                            path_feat_serv = 'U:\\Research\\EruptionForecasting\\eruptions\\features\\'
                            fm_e1 = ForecastModel(window=2., overlap=1., station =  sta,
                                look_forward=2., data_streams=ds, 
                                feature_dir=path_feat_serv, 
                                savefile_type='pkl') 
                        else:
                            try:
                                fm_e1 = ForecastModel(window=2., overlap=1., station = sta,
                                    look_forward=2., data_streams=ds, savefile_type='csv')
                            except:
                                fm_e1 = ForecastModel(window=2., overlap=1., station = sta,
                                    look_forward=2., data_streams=ds, savefile_type='pkl')                    
                        ##  
                        ft = ft.replace("-",'"')
                        
                        ft_e1 = fm_e1.get_features(ti=t0, tf=t1, n_jobs=1, compute_only_features=[ft])
                        # adding multiple Axes objects

                        # extract values to plot 
                        ft_e1_t = ft_e1[0].index.values
                        ft_e1_v = ft_e1[0].loc[:,ft]
                        #
                        v_plot = ft_e1_v

                    else:
                        day = timedelta(days=1)
                        fm = ForecastModel(window=2., overlap=1., station=sta,
                            look_forward=2., data_streams=[ds], 
                            data_dir=r'C:\Users\aar135\codes_local_disk\volc_forecast_tl\volc_forecast_tl\data'
                            )
                        #
                        N, M = [2,30]
                        df = fm.data.df[t0:t1]
                        if 'median' in ft:
                            test = df[ds].rolling(N*24*6).median()[N*24*6:]
                        if 'rate_variance' in ft:
                            test = df[ds].rolling(N*24*6).apply(chqv)[N*24*6:]
                        #
                        #out = test.rolling(archtype.shape[0]).apply(partial(conv, (archtype-archtype.mean())/archtype.std()))       
                        #out = out.resample('1D').ffill()
                        #
                        ft_e1_t = test.index
                        v_plot = test.values

                    #v_plot = ft_e1_v-np.min(ft_e1_v)/np.max((ft_e1_v-np.min(ft_e1_v)))
                    #v_plot = ft_e1_v/np.max(ft_e1_v)
                    #
                    if ft == 'zsc2_mfF__median':
                        ft = 'nMF median'
                    if ft == 'zsc2_hfF__median':
                        ft = 'nHF median'
                    #
                    ax.plot(ft_e1_t, v_plot, '-', color=col[i], alpha = alpha[i],label=' '+ ft, zorder = 4)
                    #
                    ax.legend(loc = 3)
                    #
                    te = datetimeify(erup_time)#fm_e1.data.tes[int(erup[-1:])-1]
                    #ax1.set_xticks([ft_e1[0].index[-1] - 7*day*i for i in range(int(30/7)+1)])#[dat.index.values[0],dat.index.values[-1]])#, ]np.arange(0, len(x)+1, 5))
                    
                    #ax1b.set_yticks([])
                    #ax.grid()
                    ax.set_ylabel('Feature value')        #ax.set_xticks([ft_e1[0].index[-1]-7*day*i +day for i in range(5)])
                    #ax.set_xticks([ft_e1[0].index[-1] - 7*day*i for i in range(int(30/7)+1)])#[dat.index.values[0],dat.index.values[-1]])#, ]np.arange(0, len(x)+1, 5))
                    #ax1.set_xticks([t1 - 5*day*i for i in range(int(look_back/5)+1)])#[dat.index.values[0],dat.index.values[-1]])#, ]np.arange(0, len(x)+1, 5))
                    #
                #except:
                #    pass

            else:
                pass
                if data_yrigth:
                    #
                    #ax1b = ax1.twinx() 
                    #
                    td = TremorData(station = sta)
                    #td.update(ti=t0, tf=t1)
                    data_streams = data_yrigth#['hf','mf', 'rsam']#, 'dsarF']
                    label = ['RSAM','MF','HF','DSAR']
                    #label = ['1/RSAM']
                    inv = False
                    if False:
                        data_streams = ['rsam']
                        label = ['RSAM']

                    if type(data_streams) is str:
                        data_streams = [data_streams,]
                    if any(['_' in ds for ds in data_streams]):
                        td._compute_transforms()
                    #ax.set_xlim(*range)
                    # plot data for each year
                    norm= False
                    _range = [t0,t1]
                    log =False
                    col_def = None
                    data = td.get_data(*_range)
                    xi = datetime(year=1,month=1,day=1,hour=0,minute=0,second=0)
                    cols = ['k','r','g','m',[0.5,0.5,0.5],[0.75,0.75,0.75]]
                    if inv:
                        cols = ['k','g','r','m',[0.5,0.5,0.5],[0.75,0.75,0.75]]
                    inds = (data.index>=datetimeify(_range[0]))&(data.index<=datetimeify(_range[1]))
                    i=0
                    for data_stream, col in zip(data_streams,cols):
                        v_plot = data[data_stream].loc[inds]
                        if inv:
                            v_plot = 1/v_plot
                        if log:
                            v_plot = np.log10(v_plot)
                        if norm:
                            v_plot = (v_plot-np.min(v_plot))/np.max((v_plot-np.min(v_plot)))
                        if label:
                            if col_def:
                                ax.plot(data.index[inds], v_plot, '-', color=col_def, label=label[i], linewidth=1., alpha = 1., zorder = 0)
                            else:
                                #ax1b.plot(data.index[inds], v_plot, '-', color=col, label=label[i], linewidth=1., alpha = 1.0, zorder = 0)
                                ax.plot(data.index[inds], v_plot, '-', color=col, linewidth=1., alpha = .7, zorder = 3)
                                axb.plot([], [], '-', color=col, label=label[i], linewidth=1., alpha = .7, zorder = 3)
                        else:
                            ax.plot(data.index[inds], v_plot, '-', color=col, label=data_stream, linewidth=1., alpha = .7, zorder = 3)
                        i+=1
                    for te in td.tes:
                        if [te>=datetimeify(_range[0]) and te<=datetimeify(_range[1])]:
                            pass
                            #ax.axvline(te, color='k', linestyle='--', linewidth=2, zorder = 0)
                    #
                    ax.plot([], color='k', linestyle='--', linewidth=1, label = 'eruption')
                    #ax4.set_xlim(_range)
                    #ax1b.legend(loc = 2)
                    #ax1b.grid()
                    if log:
                        ax.set_ylabel(' ')
                    else:
                        ax.set_ylabel('RSAM \u03BC m/s')
                    #ax4.set_xlabel('Time [month-day hour]')
                    #ax4.title.set_text('Station '+td.station+' ('+sta_code[td.station]+'): Tremor data')
                    #
                    #ax4.set_xticks([te - 5*day*i for i in range(int(look_back/5)+1)])#[dat.index.values[0],dat.index.values[-1]])#, ]np.arange(0, len(x)+1, 5))
                    #ax4.set_ylim([1e9,1e13])
                    ax.set_yscale('log')
            axb.legend(loc = 2)      
            axb.grid(False)
            ax.grid(color='gray', linestyle='-', linewidth=.5, alpha = 0.5)
            #
    
    _d = 5 
    t2 = erup_times[0] + look_front*day#hour
    ax2.set_xticks([t2 - _d*day*i for i in range(int((look_back+look_front)/_d)+1)])
    t4 = erup_times[1] + look_front*day#hour
    ax4.set_xticks([t4 - _d*day*i for i in range(int((look_back+look_front)/_d)+1)])
    t6 = erup_times[2] + look_front*day#hour
    ax6.set_xticks([t6 - _d*day*i for i in range(int((look_back+look_front)/_d)+1)])

    if False: # plot vertical lines
        te = datetimeify(erup_time)#fm_e1.data.tes[int(erup[-1:])-1]
        ax1.axvline(te, color='k',linestyle='--', linewidth=2, zorder = 4)
        ax2.axvline(te, color='k',linestyle='--', linewidth=2, zorder = 4)
        ax3.axvline(te, color='k',linestyle='--', linewidth=2, zorder = 4)
        ax1.plot([], color='k', linestyle='--', linewidth=2, label = 'event')
    #
    if False:#save_png_path:
        dat = erup_time.strftime('%Y-%m-%d')
        title =  sta+'_'+dat+'_'+'look_back'+str(look_back)
        ax1.set_title(title)
        plt.savefig(save_png_path+sta+'_'+dat+'_'+'look_back'+str(look_back)+'.png')
    #
    for j,ax in enumerate([ax1, ax3, ax5, ax2, ax4, ax6]):
        ax.set_ylim([1e0,1e4])
    #ax.set_xlim([t0+2*day,t1])
    #ax1.set_ylim([10**.4,10**2.1])
    #ax3.set_ylim([10**.5,10**2])
    #ax4.set_ylim([1,100])
    #ax7.set_ylim([1,100])
    #ax2b.set_ylim([0.1,0.5])

    #ax3.set_xlim([t0+2*day,t1])
    #ax4.set_xlim([t0+2*day,t1])
    #
    ax1.set_title('(a) 2006/10/04 Ruapehu eruption: seismic RSAM and DSAR median')
    ax2.set_title('(b) 2007/09/25 Ruapehu eruption: seismic RSAM and DSAR median')
    #
    ax3.set_title('(c) 2009/07/25 Ruapehu possible sealing and fluid release event')
    ax5.set_title('(e) 2010/09/03 Ruapehu possible sealing and fluid release event')
    ax4.set_title('(d) 2016/11/13 Ruapehu possible sealing and fluid release event')
    ax6.set_title('(f) 2021/03/04 Ruapehu possible sealing and fluid release event')
    #
    plt.tight_layout()
    plt.show()
    plt.close('all')
    #

## current

def figure_3_alt_ruap_kawa_copa():  # man selected events RSAM and DSAR
    '''
    plot events: seismic from multiple events (rsam and dsar)
    '''
    sta = 'FWVZ' 
    if sta == 'FWVZ':
        ffm = False
        server = True # files imported from server to local pc 
        server2 = False # server at uni 
        #plot_erup = False
    #
    look_back = 2#14
    look_front = 4#1.5
    #
    day = timedelta(days=1)
    #
    ## plot other data
    temp = True
    level = True
    rainfall = True
    ## 
    plot_erup = True
    # figure
    nrow = 4
    ncol = 2
    #fig, (ax0, ax1, ax2, ax4) = plt.subplots(nrows=nrow, ncols=ncol, figsize=(12,12))#(14,4)) #, ax4)
    fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6), (ax7, ax8)) = plt.subplots(nrows=nrow, ncols=ncol, figsize=(16,8))#(14,4)) #, ax4)

    #####################################################
    # First column (first and second row)
    if True:
        erup_times = [datetimeify('2006 10 04 09 30 00'), 
                        datetimeify('2007 09 25 08 20 00'),#datetimeify('2009 07 13 06 30 00'), 
                        datetimeify('2010 09 03 16 00 00')]
        for j,ax in enumerate([ax1, ax3, ax5]):
            #
            t0 = erup_times[j] - look_back*day#30*day
            t1 = erup_times[j] + look_front*day#hour
            # subplot one: MF, HF, DSAR medians (DSAR yaxis left; MF, HF yaxis rigth). 1/RSAM (normalized)
            if True:
                # features
                fts_yleft = ['zsc2_dsarF__median']
                fts_yrigth = []#['zsc2_dsarF__rate_variance']#['zsc2_dsarF__change_quantiles__f_agg_"var"__isabs_False__qh_0.6__ql_0.4']#['zsc2_mfF__median','zsc2_hfF__median']
                data_yrigth = ['rsam']
                fts_yleft = []#'zsc2_dsarF__median']
                
                #
                col = ['b','b','r']
                alpha = [1., 1., .5]
                thick_line = [1., 1., 1.]
                axb = ax.twinx() 
                for i, ft in enumerate(fts_yleft):
                    if True: # load feature (else: cal feature. median or rv)
                        if 'zsc2_dsarF' in ft:
                            ds = ['zsc2_dsarF'] 
                        if server:
                            path_feat_serv = 'C:\\Users\\aar135\\codes_local_disk\\volc_forecast_tl\\features_bkp\\features_server\\'
                            fm_e1 = ForecastModel(window=2., overlap=1., station =  sta,
                                look_forward=2., data_streams=ds, 
                                feature_dir=path_feat_serv, 
                                savefile_type='pkl') 
                        elif server2:
                            path_feat_serv = 'U:\\Research\\EruptionForecasting\\eruptions\\features\\'
                            fm_e1 = ForecastModel(window=2., overlap=1., station =  sta,
                                look_forward=2., data_streams=ds, 
                                feature_dir=path_feat_serv, 
                                savefile_type='pkl') 
                        else:
                            try:
                                fm_e1 = ForecastModel(window=2., overlap=1., station = sta,
                                    look_forward=2., data_streams=ds, savefile_type='csv')
                            except:
                                fm_e1 = ForecastModel(window=2., overlap=1., station = sta,
                                    look_forward=2., data_streams=ds, savefile_type='pkl')                    
                        ##  
                        ft = ft.replace("-",'"')
                        # adding multiple Axes objects
                        ft_e1 = fm_e1.get_features(ti=t0, tf=t1, n_jobs=1, compute_only_features=[ft])
                        # extract values to plot 
                        ft_e1_t = ft_e1[0].index.values
                        ft_e1_v = ft_e1[0].loc[:,ft]
                        #
                        v_plot = ft_e1_v

                        #v_plot = ft_e1_v-np.min(ft_e1_v)/np.max((ft_e1_v-np.min(ft_e1_v)))
                        #v_plot = ft_e1_v/np.max(ft_e1_v)
                    else: 
                        #
                        if 'zsc2_dsarF' in ft:
                            ds = 'zsc2_dsarF'
                        if 'zsc2_mfF' in ft:
                            ds = 'zsc2_mfF' 
                        if 'zsc2_hfF' in ft:
                            ds = 'zsc2_hfF' 
                        # 
                        #
                        day = timedelta(days=1)
                        fm = ForecastModel(window=2., overlap=1., station=sta,
                            look_forward=2., data_streams=[ds], 
                            data_dir=r'C:\Users\aar135\codes_local_disk\volc_forecast_tl\volc_forecast_tl\data'
                            )
                        #
                        N, M = [2,30]
                        df = fm.data.df[t0:t1]
                        if 'median' in ft:
                            test = df[ds].rolling(N*24*6).median()[N*24*6:]
                        if 'rate_variance' in ft:
                            test = df[ds].rolling(N*24*6).apply(chqv)[N*24*6:]
                        #
                        #out = test.rolling(archtype.shape[0]).apply(partial(conv, (archtype-archtype.mean())/archtype.std()))       
                        #out = out.resample('1D').ffill()
                        #
                        ft_e1_t = test.index
                        v_plot = test.values
                    #
                    if ft == 'zsc2_dsarF__median':
                        ft = 'DSAR median'
                    #
                    axb.plot(ft_e1_t, v_plot, '-', color=col[i], alpha = alpha[i], linewidth = thick_line[i], label=' '+ ft, zorder = 2)
                    #
                    #
                    if ffm: # ffm 
                        #ax1b = ax1.twinx() 
                        #v_plot = data[data_stream].loc[inds]
                        inv_rsam = fm_e1.data.get_data(ti=t0, tf=t1)['rsamF']#.loc[ft_e1_t]
                        inv_rsam = 1./inv_rsam
                        # normalized it to yaxis rigth 
                        inv_rsam = inv_rsam/max(inv_rsam)
                        inv_rsam = inv_rsam*0.5*max(v_plot)
                        #
                        ax.plot(ft_e1_t, inv_rsam, '-', color= 'gray', linewidth=0.5, markersize=0.5, alpha = 1.)
                        ax.plot([], [], '-', color= 'gray', markersize=1, label='1/RSAM', alpha = 1.0)
                        #ax1.set_ylim([0,1])
                        #ax1.set_yticks([])
                    #
                    if plot_erup: # plot vertical lines
                        te = datetimeify(erup_times[j])#fm_e1.data.tes[int(erup[-1:])-1]
                        if j == 0:
                            ax.axvline(te, color='red', alpha = .25, linestyle='-', linewidth=12, zorder = 0)
                            axb.plot([], color='red', alpha = .25, linestyle='-', linewidth=12, label = 'eruption')
                        else:
                            ax.axvline(te, color='gray', alpha = .25, linestyle='-', linewidth=12, zorder = 0)
                            axb.plot([], color='gray', alpha = .25, linestyle='-', linewidth=12, label = 'fluid release event')
                    #
                    #ax1.legend(loc = 2)
                    #
                    te = datetimeify(erup_times[j])#fm_e1.data.tes[int(erup[-1:])-1]
                    #ax1.set_xticks([ft_e1[0].index[-1] - 7*day*i for i in range(int(30/7)+1)])#[dat.index.values[0],dat.index.values[-1]])#, ]np.arange(0, len(x)+1, 5))
                    
                    #ax1b.set_yticks([])
                    axb.grid()
                    axb.set_ylabel('DSAR value')        #ax.set_xticks([ft_e1[0].index[-1]-7*day*i +day for i in range(5)])
                    #ax1.set_yscale('log') #ax.set_yscale('log')
                    #ax.set_xticks([ft_e1[0].index[-1] - 7*day*i for i in range(int(30/7)+1)])#[dat.index.values[0],dat.index.values[-1]])#, ]np.arange(0, len(x)+1, 5))
                    #ax1.set_xticks([t1 - 5*day*i for i in range(int(look_back/5)+1)])#[dat.index.values[0],dat.index.values[-1]])#, ]np.arange(0, len(x)+1, 5))
                #     #

                # except:
                #     pass
                if fts_yrigth:
                    #ax1b = ax1.twinx() 
                    col = ['r','g']
                    alpha = [1., .5]
                    thick_line = [2.,1.]
                    #try: 
                    for i, ft in enumerate(fts_yrigth):
                        if 'zsc2_dsarF' in ft:
                            ds = 'zsc2_dsarF'
                        if 'zsc2_mfF' in ft:
                            ds = 'zsc2_mfF' 
                        if 'zsc2_hfF' in ft:
                            ds = 'zsc2_hfF' 
                        # 
                        if False: # look feature in the prev cacl features (else: calculate feat from data; only for median and rv)
                            if server:
                                path_feat_serv = 'C:\\Users\\aar135\\codes_local_disk\\volc_forecast_tl\\features_bkp\\features_server\\'
                                fm_e1 = ForecastModel(window=2., overlap=1., station =  sta,
                                    look_forward=2., data_streams=ds, 
                                    feature_dir=path_feat_serv, 
                                    savefile_type='pkl') 
                            elif server2:
                                path_feat_serv = 'U:\\Research\\EruptionForecasting\\eruptions\\features\\'
                                fm_e1 = ForecastModel(window=2., overlap=1., station =  sta,
                                    look_forward=2., data_streams=ds, 
                                    feature_dir=path_feat_serv, 
                                    savefile_type='pkl') 
                            else:
                                try:
                                    fm_e1 = ForecastModel(window=2., overlap=1., station = sta,
                                        look_forward=2., data_streams=ds, savefile_type='csv')
                                except:
                                    fm_e1 = ForecastModel(window=2., overlap=1., station = sta,
                                        look_forward=2., data_streams=ds, savefile_type='pkl')                    
                            ##  
                            ft = ft.replace("-",'"')
                            
                            ft_e1 = fm_e1.get_features(ti=t0, tf=t1, n_jobs=1, compute_only_features=[ft])
                            # adding multiple Axes objects

                            # extract values to plot 
                            ft_e1_t = ft_e1[0].index.values
                            ft_e1_v = ft_e1[0].loc[:,ft]
                            #
                            v_plot = ft_e1_v

                        else:
                            day = timedelta(days=1)
                            fm = ForecastModel(window=2., overlap=1., station=sta,
                                look_forward=2., data_streams=[ds], 
                                data_dir=r'C:\Users\aar135\codes_local_disk\volc_forecast_tl\volc_forecast_tl\data'
                                )
                            #
                            N, M = [2,30]
                            df = fm.data.df[t0:t1]
                            if 'median' in ft:
                                test = df[ds].rolling(N*24*6).median()[N*24*6:]
                            if 'rate_variance' in ft:
                                test = df[ds].rolling(N*24*6).apply(chqv)[N*24*6:]
                            #
                            #out = test.rolling(archtype.shape[0]).apply(partial(conv, (archtype-archtype.mean())/archtype.std()))       
                            #out = out.resample('1D').ffill()
                            #
                            ft_e1_t = test.index
                            v_plot = test.values

                        #v_plot = ft_e1_v-np.min(ft_e1_v)/np.max((ft_e1_v-np.min(ft_e1_v)))
                        #v_plot = ft_e1_v/np.max(ft_e1_v)
                        #
                        if ft == 'zsc2_mfF__median':
                            ft = 'nMF median'
                        if ft == 'zsc2_hfF__median':
                            ft = 'nHF median'
                        #
                        ax.plot(ft_e1_t, v_plot, '-', color=col[i], alpha = alpha[i],label=' '+ ft, zorder = 4)
                        #
                        ax.legend(loc = 3)
                        #
                        te = datetimeify(erup_time)#fm_e1.data.tes[int(erup[-1:])-1]
                        #ax1.set_xticks([ft_e1[0].index[-1] - 7*day*i for i in range(int(30/7)+1)])#[dat.index.values[0],dat.index.values[-1]])#, ]np.arange(0, len(x)+1, 5))
                        
                        #ax1b.set_yticks([])
                        #ax.grid()
                        ax.set_ylabel('Feature value')        #ax.set_xticks([ft_e1[0].index[-1]-7*day*i +day for i in range(5)])
                        #ax.set_xticks([ft_e1[0].index[-1] - 7*day*i for i in range(int(30/7)+1)])#[dat.index.values[0],dat.index.values[-1]])#, ]np.arange(0, len(x)+1, 5))
                        #ax1.set_xticks([t1 - 5*day*i for i in range(int(look_back/5)+1)])#[dat.index.values[0],dat.index.values[-1]])#, ]np.arange(0, len(x)+1, 5))
                        #
                        #
                        if plot_erup: # plot vertical lines
                            te = datetimeify(erup_times[j])#fm_e1.data.tes[int(erup[-1:])-1]
                            if True:#j == 0:
                                ax.axvline(te, color='red', alpha = .25, linestyle='-', linewidth=12, zorder = 0)
                                axb.plot([], color='red', alpha = .25, linestyle='-', linewidth=12, label = 'eruption')
                            else:
                                ax.axvline(te, color='gray', alpha = .25, linestyle='-', linewidth=12, zorder = 0)
                                axb.plot([], color='gray', alpha = .25, linestyle='-', linewidth=12, label = 'fluid release event')
                    #except:
                    #    pass

                else:
                    pass
                    if data_yrigth:
                        #
                        #ax1b = ax1.twinx() 
                        #
                        td = TremorData(station = sta)
                        #td.update(ti=t0, tf=t1)
                        data_streams = data_yrigth#['hf','mf', 'rsam']#, 'dsarF']
                        label = ['RSAM','MF','HF','DSAR']
                        #label = ['1/RSAM']
                        inv = False
                        if False:
                            data_streams = ['rsam']
                            label = ['RSAM']

                        if type(data_streams) is str:
                            data_streams = [data_streams,]
                        if any(['_' in ds for ds in data_streams]):
                            td._compute_transforms()
                        #ax.set_xlim(*range)
                        # plot data for each year
                        norm= False
                        _range = [t0,t1]
                        log =False
                        col_def = None
                        data = td.get_data(*_range)
                        xi = datetime(year=1,month=1,day=1,hour=0,minute=0,second=0)
                        cols = ['k','r','g','m',[0.5,0.5,0.5],[0.75,0.75,0.75]]
                        if inv:
                            cols = ['k','g','r','m',[0.5,0.5,0.5],[0.75,0.75,0.75]]
                        inds = (data.index>=datetimeify(_range[0]))&(data.index<=datetimeify(_range[1]))
                        i=0
                        for data_stream, col in zip(data_streams,cols):
                            v_plot = data[data_stream].loc[inds]
                            if inv:
                                v_plot = 1/v_plot
                            if log:
                                v_plot = np.log10(v_plot)
                            if norm:
                                v_plot = (v_plot-np.min(v_plot))/np.max((v_plot-np.min(v_plot)))
                            if label:
                                if col_def:
                                    ax.plot(data.index[inds], v_plot, '-', color=col_def, label=label[i], linewidth=1., alpha = 1., zorder = 0)
                                else:
                                    #ax1b.plot(data.index[inds], v_plot, '-', color=col, label=label[i], linewidth=1., alpha = 1.0, zorder = 0)
                                    ax.plot(data.index[inds], v_plot, '-', color=col, linewidth=1., alpha = .7, zorder = 3)
                                    axb.plot([], [], '-', color=col, label=label[i], linewidth=1., alpha = .7, zorder = 3)
                            else:
                                ax.plot(data.index[inds], v_plot, '-', color=col, label=data_stream, linewidth=1., alpha = .7, zorder = 3)
                            i+=1
                        for te in td.tes:
                            if [te>=datetimeify(_range[0]) and te<=datetimeify(_range[1])]:
                                pass
                                #ax.axvline(te, color='k', linestyle='--', linewidth=2, zorder = 0)
                        #
                        ax.plot([], color='k', linestyle='--', linewidth=1, label = 'eruption')
                        #
                        if plot_erup: # plot vertical lines
                            te = datetimeify(erup_times[j])#fm_e1.data.tes[int(erup[-1:])-1]
                            if True:#j == 0:
                                ax.axvline(te, color='red', alpha = .25, linestyle='-', linewidth=12, zorder = 0)
                                axb.plot([], color='red', alpha = .25, linestyle='-', linewidth=12, label = 'eruption')
                            else:
                                ax.axvline(te, color='gray', alpha = .25, linestyle='-', linewidth=12, zorder = 0)
                                axb.plot([], color='gray', alpha = .25, linestyle='-', linewidth=12, label = 'fluid release event')
                        #ax4.set_xlim(_range)
                        #ax1b.legend(loc = 2)
                        #ax1b.grid()
                        if log:
                            ax.set_ylabel(' ')
                        else:
                            ax.set_ylabel('RSAM')
                        #ax4.set_xlabel('Time [month-day hour]')
                        #ax4.title.set_text('Station '+td.station+' ('+sta_code[td.station]+'): Tremor data')
                        #
                        #ax4.set_xticks([te - 5*day*i for i in range(int(look_back/5)+1)])#[dat.index.values[0],dat.index.values[-1]])#, ]np.arange(0, len(x)+1, 5))
                        #ax4.set_ylim([1e9,1e13])
                        ax.set_yscale('log')
                axb.legend(loc = 2)      
                axb.grid(False)
                ax.grid(color='gray', linestyle='-', linewidth=.5, alpha = 0.5)
                #

        _d = 1 
        t1 = erup_times[0] + look_front*day#hour
        ax1.set_xticks([t1 - _d*day*i for i in range(int((look_back+look_front)/_d)+1)])
        t3 = erup_times[1] + look_front*day#hour
        ax3.set_xticks([t3 - _d*day*i for i in range(int((look_back+look_front)/_d)+1)])
        t5 = erup_times[2] + look_front*day#hour
        ax5.set_xticks([t5 - _d*day*i for i in range(int((look_back+look_front)/_d)+1)])
    
    #####################################################
    # Second column (first and second row)
    if False:
        erup_times = [datetimeify('2007 09 25 08 20 00'), 
                        datetimeify('2016 11 13 12 00 00'),
                        datetimeify('2021 03 04 12 00 00')]
        for j,ax in enumerate([ax2, ax4, ax6]):
            #
            t0 = erup_times[j] - look_back*day#30*day
            t1 = erup_times[j] + look_front*day#hour
            # subplot one: MF, HF, DSAR medians (DSAR yaxis left; MF, HF yaxis rigth). 1/RSAM (normalized)
            if True:
                # features
                fts_yleft = ['zsc2_dsarF__median']
                fts_yrigth = []#['zsc2_dsarF__rate_variance']#['zsc2_dsarF__change_quantiles__f_agg_"var"__isabs_False__qh_0.6__ql_0.4']#['zsc2_mfF__median','zsc2_hfF__median']
                data_yrigth = ['rsam']
                #
                col = ['b','b','r']
                alpha = [1., 1., .5]
                thick_line = [2., 6., 1.]
                axb = ax.twinx() 
                for i, ft in enumerate(fts_yleft):
                    if True: # load feature (else: cal feature. median or rv)
                        if 'zsc2_dsarF' in ft:
                            ds = ['zsc2_dsarF'] 
                        if server:
                            path_feat_serv = 'C:\\Users\\aar135\\codes_local_disk\\volc_forecast_tl\\features_bkp\\features_server\\'
                            fm_e1 = ForecastModel(window=2., overlap=1., station =  sta,
                                look_forward=2., data_streams=ds, 
                                feature_dir=path_feat_serv, 
                                savefile_type='pkl') 
                        elif server2:
                            path_feat_serv = 'U:\\Research\\EruptionForecasting\\eruptions\\features\\'
                            fm_e1 = ForecastModel(window=2., overlap=1., station =  sta,
                                look_forward=2., data_streams=ds, 
                                feature_dir=path_feat_serv, 
                                savefile_type='pkl') 
                        else:
                            try:
                                fm_e1 = ForecastModel(window=2., overlap=1., station = sta,
                                    look_forward=2., data_streams=ds, savefile_type='csv')
                            except:
                                fm_e1 = ForecastModel(window=2., overlap=1., station = sta,
                                    look_forward=2., data_streams=ds, savefile_type='pkl')                    
                        ##  
                        ft = ft.replace("-",'"')
                        # adding multiple Axes objects
                        ft_e1 = fm_e1.get_features(ti=t0, tf=t1, n_jobs=1, compute_only_features=[ft])
                        # extract values to plot 
                        ft_e1_t = ft_e1[0].index.values
                        ft_e1_v = ft_e1[0].loc[:,ft]
                        #
                        v_plot = ft_e1_v

                        #v_plot = ft_e1_v-np.min(ft_e1_v)/np.max((ft_e1_v-np.min(ft_e1_v)))
                        #v_plot = ft_e1_v/np.max(ft_e1_v)
                    else: 
                        #
                        if 'zsc2_dsarF' in ft:
                            ds = 'zsc2_dsarF'
                        if 'zsc2_mfF' in ft:
                            ds = 'zsc2_mfF' 
                        if 'zsc2_hfF' in ft:
                            ds = 'zsc2_hfF' 
                        # 
                        #
                        day = timedelta(days=1)
                        fm = ForecastModel(window=2., overlap=1., station=sta,
                            look_forward=2., data_streams=[ds], 
                            data_dir=r'C:\Users\aar135\codes_local_disk\volc_forecast_tl\volc_forecast_tl\data'
                            )
                        #
                        N, M = [2,30]
                        df = fm.data.df[t0:t1]
                        if 'median' in ft:
                            test = df[ds].rolling(N*24*6).median()[N*24*6:]
                        if 'rate_variance' in ft:
                            test = df[ds].rolling(N*24*6).apply(chqv)[N*24*6:]
                        #
                        #out = test.rolling(archtype.shape[0]).apply(partial(conv, (archtype-archtype.mean())/archtype.std()))       
                        #out = out.resample('1D').ffill()
                        #
                        ft_e1_t = test.index
                        v_plot = test.values
                    #
                    if ft == 'zsc2_dsarF__median':
                        ft = 'DSAR median'
                    #
                    axb.plot(ft_e1_t, v_plot, '-', color=col[i], alpha = alpha[i], linewidth = thick_line[i], label=' '+ ft, zorder = 2)
                    #
                    #
                    if ffm: # ffm 
                        #ax1b = ax1.twinx() 
                        #v_plot = data[data_stream].loc[inds]
                        inv_rsam = fm_e1.data.get_data(ti=t0, tf=t1)['rsamF']#.loc[ft_e1_t]
                        inv_rsam = 1./inv_rsam
                        # normalized it to yaxis rigth 
                        inv_rsam = inv_rsam/max(inv_rsam)
                        inv_rsam = inv_rsam*0.5*max(v_plot)
                        #
                        ax.plot(ft_e1_t, inv_rsam, '-', color= 'gray', linewidth=0.5, markersize=0.5, alpha = 1.)
                        ax.plot([], [], '-', color= 'gray', markersize=1, label='1/RSAM', alpha = 1.0)
                        #ax1.set_ylim([0,1])
                        #ax1.set_yticks([])
                    #
                    if plot_erup: # plot vertical lines
                        te = datetimeify(erup_times[j])#fm_e1.data.tes[int(erup[-1:])-1]
                        if j == 0:
                            ax.axvline(te, color='red', alpha = .25, linestyle='-', linewidth=12, zorder = 0)
                            axb.plot([], color='red', alpha = .25, linestyle='-', linewidth=12, label = 'eruption')
                        else:
                            ax.axvline(te, color='gray', alpha = .25, linestyle='-', linewidth=12, zorder = 0)
                            axb.plot([], color='gray', alpha = .25, linestyle='-', linewidth=12, label = 'fluid release event')
                    #
                    #ax1.legend(loc = 2)
                    #
                    te = datetimeify(erup_times[j])#fm_e1.data.tes[int(erup[-1:])-1]
                    #ax1.set_xticks([ft_e1[0].index[-1] - 7*day*i for i in range(int(30/7)+1)])#[dat.index.values[0],dat.index.values[-1]])#, ]np.arange(0, len(x)+1, 5))
                    
                    #ax1b.set_yticks([])
                    axb.grid()
                    axb.set_ylabel('DSAR value')        #ax.set_xticks([ft_e1[0].index[-1]-7*day*i +day for i in range(5)])
                    #ax1.set_yscale('log') #ax.set_yscale('log')
                    #ax.set_xticks([ft_e1[0].index[-1] - 7*day*i for i in range(int(30/7)+1)])#[dat.index.values[0],dat.index.values[-1]])#, ]np.arange(0, len(x)+1, 5))
                    #ax1.set_xticks([t1 - 5*day*i for i in range(int(look_back/5)+1)])#[dat.index.values[0],dat.index.values[-1]])#, ]np.arange(0, len(x)+1, 5))
                #     #
                # except:
                #     pass
                if fts_yrigth:
                    #ax1b = ax1.twinx() 
                    col = ['r','g']
                    alpha = [1., .5]
                    thick_line = [1.,1.]
                    #try: 
                    for i, ft in enumerate(fts_yrigth):
                        if 'zsc2_dsarF' in ft:
                            ds = 'zsc2_dsarF'
                        if 'zsc2_mfF' in ft:
                            ds = 'zsc2_mfF' 
                        if 'zsc2_hfF' in ft:
                            ds = 'zsc2_hfF' 
                        # 
                        if True: # look feature in the prev cacl features (else: calculate feat from data; only for median and rv)
                            if server:
                                path_feat_serv = 'C:\\Users\\aar135\\codes_local_disk\\volc_forecast_tl\\features_bkp\\features_server\\'
                                fm_e1 = ForecastModel(window=2., overlap=1., station =  sta,
                                    look_forward=2., data_streams=ds, 
                                    feature_dir=path_feat_serv, 
                                    savefile_type='pkl') 
                            elif server2:
                                path_feat_serv = 'U:\\Research\\EruptionForecasting\\eruptions\\features\\'
                                fm_e1 = ForecastModel(window=2., overlap=1., station =  sta,
                                    look_forward=2., data_streams=ds, 
                                    feature_dir=path_feat_serv, 
                                    savefile_type='pkl') 
                            else:
                                try:
                                    fm_e1 = ForecastModel(window=2., overlap=1., station = sta,
                                        look_forward=2., data_streams=ds, savefile_type='csv')
                                except:
                                    fm_e1 = ForecastModel(window=2., overlap=1., station = sta,
                                        look_forward=2., data_streams=ds, savefile_type='pkl')                    
                            ##  
                            ft = ft.replace("-",'"')
                            
                            ft_e1 = fm_e1.get_features(ti=t0, tf=t1-.5*DAY, n_jobs=1, compute_only_features=[ft])
                            # adding multiple Axes objects

                            # extract values to plot 
                            ft_e1_t = ft_e1[0].index.values
                            ft_e1_v = ft_e1[0].loc[:,ft]
                            #
                            v_plot = ft_e1_v

                        else:
                            day = timedelta(days=1)
                            fm = ForecastModel(window=2., overlap=1., station=sta,
                                look_forward=2., data_streams=[ds], 
                                data_dir=r'C:\Users\aar135\codes_local_disk\volc_forecast_tl\volc_forecast_tl\data'
                                )
                            #
                            N, M = [2,30]
                            df = fm.data.df[t0:t1]
                            if 'median' in ft:
                                test = df[ds].rolling(N*24*6).median()[N*24*6:]
                            if 'rate_variance' in ft:
                                test = df[ds].rolling(N*24*6).apply(chqv)[N*24*6:]
                            #
                            #out = test.rolling(archtype.shape[0]).apply(partial(conv, (archtype-archtype.mean())/archtype.std()))       
                            #out = out.resample('1D').ffill()
                            #
                            ft_e1_t = test.index
                            v_plot = test.values

                        #v_plot = ft_e1_v-np.min(ft_e1_v)/np.max((ft_e1_v-np.min(ft_e1_v)))
                        #v_plot = ft_e1_v/np.max(ft_e1_v)
                        #
                        if ft == 'zsc2_mfF__median':
                            ft = 'nMF median'
                        if ft == 'zsc2_hfF__median':
                            ft = 'nHF median'
                        #
                        ax.plot(ft_e1_t, v_plot, '-', color=col[i], alpha = alpha[i],label=' '+ ft, zorder = 4)
                        #
                        ax.legend(loc = 3)
                        #
                        te = datetimeify(erup_time)#fm_e1.data.tes[int(erup[-1:])-1]
                        #ax1.set_xticks([ft_e1[0].index[-1] - 7*day*i for i in range(int(30/7)+1)])#[dat.index.values[0],dat.index.values[-1]])#, ]np.arange(0, len(x)+1, 5))
                        
                        #ax1b.set_yticks([])
                        #ax.grid()
                        ax.set_ylabel('Feature value')        #ax.set_xticks([ft_e1[0].index[-1]-7*day*i +day for i in range(5)])
                        #ax.set_xticks([ft_e1[0].index[-1] - 7*day*i for i in range(int(30/7)+1)])#[dat.index.values[0],dat.index.values[-1]])#, ]np.arange(0, len(x)+1, 5))
                        #ax1.set_xticks([t1 - 5*day*i for i in range(int(look_back/5)+1)])#[dat.index.values[0],dat.index.values[-1]])#, ]np.arange(0, len(x)+1, 5))
                        #
                    #except:
                    #    pass

                else:
                    pass
                    if data_yrigth:
                        #
                        #ax1b = ax1.twinx() 
                        #
                        td = TremorData(station = sta)
                        #td.update(ti=t0, tf=t1)
                        data_streams = data_yrigth#['hf','mf', 'rsam']#, 'dsarF']
                        label = ['RSAM','MF','HF','DSAR']
                        #label = ['1/RSAM']
                        inv = False
                        if False:
                            data_streams = ['rsam']
                            label = ['RSAM']

                        if type(data_streams) is str:
                            data_streams = [data_streams,]
                        if any(['_' in ds for ds in data_streams]):
                            td._compute_transforms()
                        #ax.set_xlim(*range)
                        # plot data for each year
                        norm= False
                        _range = [t0,t1]
                        log =False
                        col_def = None
                        data = td.get_data(*_range)
                        xi = datetime(year=1,month=1,day=1,hour=0,minute=0,second=0)
                        cols = ['k','r','g','m',[0.5,0.5,0.5],[0.75,0.75,0.75]]
                        if inv:
                            cols = ['k','g','r','m',[0.5,0.5,0.5],[0.75,0.75,0.75]]
                        inds = (data.index>=datetimeify(_range[0]))&(data.index<=datetimeify(_range[1]))
                        i=0
                        for data_stream, col in zip(data_streams,cols):
                            v_plot = data[data_stream].loc[inds]
                            if inv:
                                v_plot = 1/v_plot
                            if log:
                                v_plot = np.log10(v_plot)
                            if norm:
                                v_plot = (v_plot-np.min(v_plot))/np.max((v_plot-np.min(v_plot)))
                            if label:
                                if col_def:
                                    ax.plot(data.index[inds], v_plot, '-', color=col_def, label=label[i], linewidth=1., alpha = 1., zorder = 0)
                                else:
                                    #ax1b.plot(data.index[inds], v_plot, '-', color=col, label=label[i], linewidth=1., alpha = 1.0, zorder = 0)
                                    ax.plot(data.index[inds], v_plot, '-', color=col, linewidth=1., alpha = .7, zorder = 3)
                                    axb.plot([], [], '-', color=col, label=label[i], linewidth=1., alpha = .7, zorder = 3)
                            else:
                                ax.plot(data.index[inds], v_plot, '-', color=col, label=data_stream, linewidth=1., alpha = .7, zorder = 3)
                            i+=1
                        for te in td.tes:
                            if [te>=datetimeify(_range[0]) and te<=datetimeify(_range[1])]:
                                pass
                                #ax.axvline(te, color='k', linestyle='--', linewidth=2, zorder = 0)
                        #
                        ax.plot([], color='k', linestyle='--', linewidth=1, label = 'eruption')
                        #ax4.set_xlim(_range)
                        #ax1b.legend(loc = 2)
                        #ax1b.grid()
                        if log:
                            ax.set_ylabel(' ')
                        else:
                            ax.set_ylabel('RSAM \u03BC m/s')
                        #ax4.set_xlabel('Time [month-day hour]')
                        #ax4.title.set_text('Station '+td.station+' ('+sta_code[td.station]+'): Tremor data')
                        #
                        #ax4.set_xticks([te - 5*day*i for i in range(int(look_back/5)+1)])#[dat.index.values[0],dat.index.values[-1]])#, ]np.arange(0, len(x)+1, 5))
                        #ax4.set_ylim([1e9,1e13])
                        ax.set_yscale('log')
                axb.legend(loc = 2)      
                axb.grid(False)
                ax.grid(color='gray', linestyle='-', linewidth=.5, alpha = 0.5)
                #
        
        _d = 5 
        t2 = erup_times[0] + look_front*day#hour
        ax2.set_xticks([t2 - _d*day*i for i in range(int((look_back+look_front)/_d)+1)])
        t4 = erup_times[1] + look_front*day#hour
        ax4.set_xticks([t4 - _d*day*i for i in range(int((look_back+look_front)/_d)+1)])
        t6 = erup_times[2] + look_front*day#hour
        ax6.set_xticks([t6 - _d*day*i for i in range(int((look_back+look_front)/_d)+1)])

        if False: # plot vertical lines
            te = datetimeify(erup_time)#fm_e1.data.tes[int(erup[-1:])-1]
            ax1.axvline(te, color='k',linestyle='--', linewidth=2, zorder = 4)
            ax2.axvline(te, color='k',linestyle='--', linewidth=2, zorder = 4)
            ax3.axvline(te, color='k',linestyle='--', linewidth=2, zorder = 4)
            ax1.plot([], color='k', linestyle='--', linewidth=2, label = 'event')
        #
        if False:#save_png_path:
            dat = erup_time.strftime('%Y-%m-%d')
            title =  sta+'_'+dat+'_'+'look_back'+str(look_back)
            ax1.set_title(title)
            plt.savefig(save_png_path+sta+'_'+dat+'_'+'look_back'+str(look_back)+'.png')
        #
        for j,ax in enumerate([ax1, ax3, ax5, ax2, ax4, ax6]):
            ax.set_ylim([1e0,1e4])

    #####################################################
    # Third row 
    sta = 'POS' 
    if sta == 'POS':
        ffm = False
        server = False # files imported from server to local pc 
        server2 = True # server at uni 

    if False: # Kawah Ijen eruption 
        erup_times = [datetimeify('2013 01 24 00 00 00')]
        #
        # subplot one: MF, HF, DSAR medians (DSAR yaxis left; MF, HF yaxis rigth). 1/RSAM (normalized)
        if True:
            for j,ax in enumerate([ax7]):
                # features
                t0 = erup_times[j] - look_back*day#30*day
                t1 = erup_times[j] + look_front*day#hour
                #
                fts_yleft = ['zsc2_dsarF__median']
                fts_yrigth = []#['zsc2_dsarF__rate_variance']#['zsc2_dsarF__change_quantiles__f_agg_"var"__isabs_False__qh_0.6__ql_0.4']#['zsc2_mfF__median','zsc2_hfF__median']
                data_yrigth = ['rsam']
                #
                col = ['b','b','r']
                alpha = [1., 1., .5]
                thick_line = [2., 6., 1.]
                axb = ax.twinx() 
                for i, ft in enumerate(fts_yleft):
                    if True: # load feature (else: cal feature. median or rv)
                        if 'zsc2_dsarF' in ft:
                            ds = ['zsc2_dsarF'] 
                        if server:
                            path_feat_serv = 'C:\\Users\\aar135\\codes_local_disk\\volc_forecast_tl\\features_bkp\\features_server\\'
                            fm_e1 = ForecastModel(window=2., overlap=1., station =  sta,
                                look_forward=2., data_streams=ds, 
                                feature_dir=path_feat_serv, 
                                savefile_type='pkl') 
                        elif server2:
                            path_feat_serv = 'U:\\Research\\EruptionForecasting\\eruptions\\features\\'
                            fm_e1 = ForecastModel(window=2., overlap=1., station =  sta,
                                look_forward=2., data_streams=ds, 
                                feature_dir=path_feat_serv, 
                                savefile_type='pkl') 
                        else:
                            try:
                                fm_e1 = ForecastModel(window=2., overlap=1., station = sta,
                                    look_forward=2., data_streams=ds, savefile_type='csv')
                            except:
                                fm_e1 = ForecastModel(window=2., overlap=1., station = sta,
                                    look_forward=2., data_streams=ds, savefile_type='pkl')                    
                        ##  
                        ft = ft.replace("-",'"')
                        # adding multiple Axes objects
                        ft_e1 = fm_e1.get_features(ti=t0, tf=t1, n_jobs=1, compute_only_features=[ft])
                        # extract values to plot 
                        ft_e1_t = ft_e1[0].index.values
                        ft_e1_v = ft_e1[0].loc[:,ft]
                        #
                        v_plot = ft_e1_v

                        #v_plot = ft_e1_v-np.min(ft_e1_v)/np.max((ft_e1_v-np.min(ft_e1_v)))
                        #v_plot = ft_e1_v/np.max(ft_e1_v)
                    else: 
                        #
                        if 'zsc2_dsarF' in ft:
                            ds = 'zsc2_dsarF'
                        if 'zsc2_mfF' in ft:
                            ds = 'zsc2_mfF' 
                        if 'zsc2_hfF' in ft:
                            ds = 'zsc2_hfF' 
                        # 
                        #
                        day = timedelta(days=1)
                        fm = ForecastModel(window=2., overlap=1., station=sta,
                            look_forward=2., data_streams=[ds], 
                            data_dir=r'C:\Users\aar135\codes_local_disk\volc_forecast_tl\volc_forecast_tl\data'
                            )
                        #
                        N, M = [2,30]
                        df = fm.data.df[t0:t1]
                        if 'median' in ft:
                            test = df[ds].rolling(N*24*6).median()[N*24*6:]
                        if 'rate_variance' in ft:
                            test = df[ds].rolling(N*24*6).apply(chqv)[N*24*6:]
                        #
                        #out = test.rolling(archtype.shape[0]).apply(partial(conv, (archtype-archtype.mean())/archtype.std()))       
                        #out = out.resample('1D').ffill()
                        #
                        ft_e1_t = test.index
                        v_plot = test.values
                    #
                    if ft == 'zsc2_dsarF__median':
                        ft = 'DSAR median'
                    #
                    axb.plot(ft_e1_t, v_plot, '-', color=col[i], alpha = alpha[i], linewidth = thick_line[i], label=' '+ ft, zorder = 2)
                    #
                    #
                    if ffm: # ffm 
                        #ax1b = ax1.twinx() 
                        #v_plot = data[data_stream].loc[inds]
                        inv_rsam = fm_e1.data.get_data(ti=t0, tf=t1)['rsamF']#.loc[ft_e1_t]
                        inv_rsam = 1./inv_rsam
                        # normalized it to yaxis rigth 
                        inv_rsam = inv_rsam/max(inv_rsam)
                        inv_rsam = inv_rsam*0.5*max(v_plot)
                        #
                        ax.plot(ft_e1_t, inv_rsam, '-', color= 'gray', linewidth=0.5, markersize=0.5, alpha = 1.)
                        ax.plot([], [], '-', color= 'gray', markersize=1, label='1/RSAM', alpha = 1.0)
                        #ax1.set_ylim([0,1])
                        #ax1.set_yticks([])
                    #
                    if plot_erup: # plot vertical lines
                        te = datetimeify(erup_times[j])#fm_e1.data.tes[int(erup[-1:])-1]
                        if j == 1:
                            ax.axvline(te, color='red', alpha = .25, linestyle='-', linewidth=12, zorder = 0)
                            axb.plot([], color='red', alpha = .25, linestyle='-', linewidth=12, label = 'eruption')
                        else:
                            ax.axvline(te, color='gray', alpha = .25, linestyle='-', linewidth=12, zorder = 0)
                            axb.plot([], color='gray', alpha = .25, linestyle='-', linewidth=12, label = 'fluid release event')
                    #
                    #ax1.legend(loc = 2)
                    #
                    te = datetimeify(erup_times[j])#fm_e1.data.tes[int(erup[-1:])-1]
                    #ax1.set_xticks([ft_e1[0].index[-1] - 7*day*i for i in range(int(30/7)+1)])#[dat.index.values[0],dat.index.values[-1]])#, ]np.arange(0, len(x)+1, 5))
                    
                    #ax1b.set_yticks([])
                    axb.grid()
                    axb.set_ylabel('DSAR value')        #ax.set_xticks([ft_e1[0].index[-1]-7*day*i +day for i in range(5)])
                    #ax1.set_yscale('log') #ax.set_yscale('log')
                    #ax.set_xticks([ft_e1[0].index[-1] - 7*day*i for i in range(int(30/7)+1)])#[dat.index.values[0],dat.index.values[-1]])#, ]np.arange(0, len(x)+1, 5))
                    #ax1.set_xticks([t1 - 5*day*i for i in range(int(look_back/5)+1)])#[dat.index.values[0],dat.index.values[-1]])#, ]np.arange(0, len(x)+1, 5))
                #     #
                # except:
                #     pass
                if fts_yrigth:
                    #ax1b = ax1.twinx() 
                    col = ['r','g']
                    alpha = [1., .5]
                    thick_line = [2.,1.]
                    #try: 
                    for i, ft in enumerate(fts_yrigth):
                        if 'zsc2_dsarF' in ft:
                            ds = 'zsc2_dsarF'
                        if 'zsc2_mfF' in ft:
                            ds = 'zsc2_mfF' 
                        if 'zsc2_hfF' in ft:
                            ds = 'zsc2_hfF' 
                        # 
                        if False: # look feature in the prev cacl features (else: calculate feat from data; only for median and rv)
                            if server:
                                path_feat_serv = 'C:\\Users\\aar135\\codes_local_disk\\volc_forecast_tl\\features_bkp\\features_server\\'
                                fm_e1 = ForecastModel(window=2., overlap=1., station =  sta,
                                    look_forward=2., data_streams=ds, 
                                    feature_dir=path_feat_serv, 
                                    savefile_type='pkl') 
                            elif server2:
                                path_feat_serv = 'U:\\Research\\EruptionForecasting\\eruptions\\features\\'
                                fm_e1 = ForecastModel(window=2., overlap=1., station =  sta,
                                    look_forward=2., data_streams=ds, 
                                    feature_dir=path_feat_serv, 
                                    savefile_type='pkl') 
                            else:
                                try:
                                    fm_e1 = ForecastModel(window=2., overlap=1., station = sta,
                                        look_forward=2., data_streams=ds, savefile_type='csv')
                                except:
                                    fm_e1 = ForecastModel(window=2., overlap=1., station = sta,
                                        look_forward=2., data_streams=ds, savefile_type='pkl')                    
                            ##  
                            ft = ft.replace("-",'"')
                            
                            ft_e1 = fm_e1.get_features(ti=t0, tf=t1, n_jobs=1, compute_only_features=[ft])
                            # adding multiple Axes objects

                            # extract values to plot 
                            ft_e1_t = ft_e1[0].index.values
                            ft_e1_v = ft_e1[0].loc[:,ft]
                            #
                            v_plot = ft_e1_v

                        else:
                            day = timedelta(days=1)
                            fm = ForecastModel(window=2., overlap=1., station=sta,
                                look_forward=2., data_streams=[ds], 
                                data_dir=r'C:\Users\aar135\codes_local_disk\volc_forecast_tl\volc_forecast_tl\data'
                                )
                            #
                            N, M = [2,30]
                            df = fm.data.df[t0:t1]
                            if 'median' in ft:
                                test = df[ds].rolling(N*24*6).median()[N*24*6:]
                            if 'rate_variance' in ft:
                                test = df[ds].rolling(N*24*6).apply(chqv)[N*24*6:]
                            #
                            #out = test.rolling(archtype.shape[0]).apply(partial(conv, (archtype-archtype.mean())/archtype.std()))       
                            #out = out.resample('1D').ffill()
                            #
                            ft_e1_t = test.index
                            v_plot = test.values

                        #v_plot = ft_e1_v-np.min(ft_e1_v)/np.max((ft_e1_v-np.min(ft_e1_v)))
                        #v_plot = ft_e1_v/np.max(ft_e1_v)
                        #
                        if ft == 'zsc2_mfF__median':
                            ft = 'nMF median'
                        if ft == 'zsc2_hfF__median':
                            ft = 'nHF median'
                        #
                        ax.plot(ft_e1_t, v_plot, '-', color=col[i], alpha = alpha[i],label=' '+ ft, zorder = 4)
                        #
                        ax.legend(loc = 3)
                        #
                        te = datetimeify(erup_time)#fm_e1.data.tes[int(erup[-1:])-1]
                        #ax1.set_xticks([ft_e1[0].index[-1] - 7*day*i for i in range(int(30/7)+1)])#[dat.index.values[0],dat.index.values[-1]])#, ]np.arange(0, len(x)+1, 5))
                        
                        #ax1b.set_yticks([])
                        #ax.grid()
                        ax.set_ylabel('Feature value')        #ax.set_xticks([ft_e1[0].index[-1]-7*day*i +day for i in range(5)])
                        #ax.set_xticks([ft_e1[0].index[-1] - 7*day*i for i in range(int(30/7)+1)])#[dat.index.values[0],dat.index.values[-1]])#, ]np.arange(0, len(x)+1, 5))
                        #ax1.set_xticks([t1 - 5*day*i for i in range(int(look_back/5)+1)])#[dat.index.values[0],dat.index.values[-1]])#, ]np.arange(0, len(x)+1, 5))
                        #
                    #except:
                    #    pass

                else:
                    pass
                    if data_yrigth:
                        #
                        #ax1b = ax1.twinx() 
                        #
                        td = TremorData(station = sta)
                        #td.update(ti=t0, tf=t1)
                        data_streams = data_yrigth#['hf','mf', 'rsam']#, 'dsarF']
                        label = ['RSAM','MF','HF','DSAR']
                        #label = ['1/RSAM']
                        inv = False
                        if False:
                            data_streams = ['rsam']
                            label = ['RSAM']

                        if type(data_streams) is str:
                            data_streams = [data_streams,]
                        if any(['_' in ds for ds in data_streams]):
                            td._compute_transforms()
                        #ax.set_xlim(*range)
                        # plot data for each year
                        norm= False
                        _range = [t0,t1]
                        log =False
                        col_def = None
                        data = td.get_data(*_range)
                        xi = datetime(year=1,month=1,day=1,hour=0,minute=0,second=0)
                        cols = ['k','r','g','m',[0.5,0.5,0.5],[0.75,0.75,0.75]]
                        if inv:
                            cols = ['k','g','r','m',[0.5,0.5,0.5],[0.75,0.75,0.75]]
                        inds = (data.index>=datetimeify(_range[0]))&(data.index<=datetimeify(_range[1]))
                        i=0
                        for data_stream, col in zip(data_streams,cols):
                            v_plot = data[data_stream].loc[inds]
                            if inv:
                                v_plot = 1/v_plot
                            if log:
                                v_plot = np.log10(v_plot)
                            if norm:
                                v_plot = (v_plot-np.min(v_plot))/np.max((v_plot-np.min(v_plot)))
                            if label:
                                if col_def:
                                    ax.plot(data.index[inds], v_plot, '-', color=col_def, label=label[i], linewidth=1., alpha = 1., zorder = 0)
                                else:
                                    #ax1b.plot(data.index[inds], v_plot, '-', color=col, label=label[i], linewidth=1., alpha = 1.0, zorder = 0)
                                    ax.plot(data.index[inds], v_plot, '-', color=col, linewidth=1., alpha = .7, zorder = 3)
                                    axb.plot([], [], '-', color=col, label=label[i], linewidth=1., alpha = .7, zorder = 3)
                            else:
                                ax.plot(data.index[inds], v_plot, '-', color=col, label=data_stream, linewidth=1., alpha = .7, zorder = 3)
                            i+=1
                        for te in td.tes:
                            if [te>=datetimeify(_range[0]) and te<=datetimeify(_range[1])]:
                                pass
                                #ax.axvline(te, color='k', linestyle='--', linewidth=2, zorder = 0)
                        #
                        ax.plot([], color='k', linestyle='--', linewidth=1, label = 'eruption')
                        #ax4.set_xlim(_range)
                        #ax1b.legend(loc = 2)
                        #ax1b.grid()
                        if log:
                            ax.set_ylabel(' ')
                        else:
                            ax.set_ylabel('RSAM \u03BC m/s')
                        #ax4.set_xlabel('Time [month-day hour]')
                        #ax4.title.set_text('Station '+td.station+' ('+sta_code[td.station]+'): Tremor data')
                        #
                        #ax4.set_xticks([te - 5*day*i for i in range(int(look_back/5)+1)])#[dat.index.values[0],dat.index.values[-1]])#, ]np.arange(0, len(x)+1, 5))
                        #ax4.set_ylim([1e9,1e13])
                        ax.set_yscale('log')
                axb.legend(loc = 2)      
                axb.grid(False)
                ax.grid(color='gray', linestyle='-', linewidth=.5, alpha = 0.5)
                #
                _d = 5 
                t1 = erup_times[0] + look_front*day#hour
                ax.set_xticks([t1 - _d*day*i for i in range(int((look_back+look_front)/_d)+1)])
    
    if False: # Copahue eruption 
        #
        sta = 'COP' 
        if sta == 'COP':
            ffm = False
            server = True # files imported from server to local pc 
            server2 = False # server at uni 
        #
        erup_times = [datetimeify('2020 07 16 00 00 00')]
        #erup_times = [datetimeify('2020 06 16 00 00 00')]
        look_back = 14
        look_front = 7
        #
        col = ['b','b','r']
        alpha = [1., 1., .5]
        thick_line = [2., 6., 1.]
        #
        N, M = [2,15]
        #
        # subplot one: MF, HF, DSAR medians (DSAR yaxis left; MF, HF yaxis rigth). 1/RSAM (normalized)
        if True:
            for j,ax in enumerate([ax8]):
                #
                if True: # RSAM
                    ## DSAR median 
                    day = timedelta(days=1)
                    #sta_arch = 'WIZ'
                    dt = 'zsc2_rsamF'
                    fm = ForecastModel(window=2., overlap=1., station=sta,
                        look_forward=2., data_streams=[dt], 
                        data_dir=r'C:\Users\aar135\codes_local_disk\volc_forecast_tl\volc_forecast_tl\data'
                        )
                    #
                    te = erup_times[j]#fm.data.tes[erup] 
                    # rolling median and signature length window
                    #N, M = [2,15]
                    #l_forw = 0
                    # time
                    k = fm.data.df.index
                    # median 
                    df = fm.data.df[(k>(te-(M+N)*day))&(k<te+look_front*day)]
                    #archtype = df[dt].rolling(N*24*6).median()[N*24*6:]
                    archtype = df[dt]#.rolling(N*24*6).median()[N*24*6:]
                    #
                    _times = archtype.index
                    _val = archtype.values
                    _val_max = max(_val)
                    #
                    ft = 'RSAM'
                    ax.plot(_times, _val, '-', color='k', linewidth=1., alpha = .7,label=' '+ ft, zorder = 4) #'-', color='k', alpha = 0.8, linewidth=thick_line[0], label=' '+ ft,zorder=1)
                    ax.plot([], [], '-', color='k', alpha = .7, linewidth=1., label=' '+ ft,zorder=1)
                    # lim
                    #ax.set_ylim([0,np.mean(_val)+3*np.std(_val)])
                #
                if True: # DSAR
                    axb = ax.twinx()
                    ## DSAR median 
                    day = timedelta(days=1)
                    #sta_arch = 'WIZ'
                    dt = 'zsc2_dsarF'
                    fm = ForecastModel(window=2., overlap=1., station=sta,
                        look_forward=2., data_streams=[dt], 
                        data_dir=r'C:\Users\aar135\codes_local_disk\volc_forecast_tl\volc_forecast_tl\data'
                        )
                    #
                    te = erup_times[j]#fm.data.tes[erup] 
                    # rolling median and signature length window
                    #N, M = [2,15]
                    #l_forw = 0
                    # time
                    k = fm.data.df.index
                    # median 
                    df = fm.data.df[(k>(te-(M+N)*day))&(k<te+look_front*day)]
                    archtype = df[dt].rolling(N*24*6).median()[N*24*6:]
                    #
                    _times = archtype.index
                    _val = archtype.values
                    _val_max = max(_val)
                    #
                    ft = 'DSAR median'
                    axb.plot(_times, _val, '-', color='b', alpha = None, linewidth = 2, label=' '+ ft, zorder = 2) # '-', color='b', alpha = alpha[0], linewidth=thick_line[0], label=' '+ ft,zorder=1)
                    ax.plot([], [], '-', color='b', alpha = None, linewidth = 2, label=' '+ ft, zorder = 2)# color='b', alpha = alpha[0], linewidth=thick_line[0], label=' '+ ft,zorder=1)
                    #ax.plot([], [], '-', color='w', alpha = 0.1, linewidth=thick_line[0], label=str(te.year)+' '+str(te.mo
                    #
                    if plot_erup: # plot vertical lines
                        te = datetimeify(erup_times[j])#fm_e1.data.tes[int(erup[-1:])-1]
                        if j == 1:
                            ax.axvline(te, color='red', alpha = .25, linestyle='-', linewidth=12, zorder = 0)
                            axb.plot([], color='red', alpha = .25, linestyle='-', linewidth=12, label = 'eruption')
                        else:
                            ax.axvline(te, color='gray', alpha = .25, linestyle='-', linewidth=12, zorder = 0)
                            axb.plot([], color='gray', alpha = .25, linestyle='-', linewidth=12, label = 'fluid release event')
                    #
                #
                #ax1.legend(loc = 2)
                #
                te = datetimeify(erup_times[j])#fm_e1.data.tes[int(erup[-1:])-1]
                #ax1.set_xticks([ft_e1[0].index[-1] - 7*day*i for i in range(int(30/7)+1)])#[dat.index.values[0],dat.index.values[-1]])#, ]np.arange(0, len(x)+1, 5))
                #ax1b.set_yticks([])
                axb.grid()
                axb.set_ylabel('DSAR value')        #ax.set_xticks([ft_e1[0].index[-1]-7*day*i +day for i in range(5)])
                ax.set_ylabel('RSAM \u03BC m/s')        #ax.set_xticks([ft_e1[0].index[-1]-7*day*i +day for i in range(5)])
                #ax1.set_yscale('log') #ax.set_yscale('log')
                #ax.set_xticks([ft_e1[0].index[-1] - 7*day*i for i in range(int(30/7)+1)])#[dat.index.values[0],dat.index.values[-1]])#, ]np.arange(0, len(x)+1, 5))
                #ax1.set_xticks([t1 - 5*day*i for i in range(int(look_back/5)+1)])#[dat.index.values[0],dat.index.values[-1]])#, ]np.arange(0, len(x)+1, 5))
                #     #
                # except:
                axb.legend(loc = 2)      
                axb.grid(False)
                ax.grid(color='gray', linestyle='-', linewidth=.5, alpha = 0.5)
                #
                #
                _d = 5 
                t1 = erup_times[0] + look_front*day#hour
                ax.set_xticks([t1 - _d*day*i for i in range(int((look_back+look_front)/_d)+1)])
                #
    
    #ax.set_xlim([t0+2*day,t1])
    #ax1.set_ylim([10**.4,10**2.1])
    #ax3.set_ylim([10**.5,10**2])
    #ax4.set_ylim([1,100])
    #ax7.set_ylim([1,100])
    #ax2b.set_ylim([0.1,0.5])

    #ax3.set_xlim([t0+2*day,t1])
    #ax4.set_xlim([t0+2*day,t1])
    #
    ax1.set_title('(a) 2006/10/04 Ruapehu eruption: RSAM peak and exponential decay')
    ax2.set_title('(b) 2007/09/25 Ruapehu ERUPTION: seismic RSAM and DSAR median')
    ax3.set_title('(b) 2007/09/25 Ruapehu eruption: RSAM peak and exponential decay')

    #
    #ax3.set_title('(c) 2009/07/13 Ruapehu possible sealing and fluid release event')
    ax5.set_title('(e) 2010/09/03 Ruapehu possible sealing and fluid release event')
    ax4.set_title('(d) 2016/11/13 Ruapehu possible sealing and fluid release event')
    ax6.set_title('(f) 2021/03/04 Ruapehu possible sealing and fluid release event')
    #
    ax7.set_title('(g) 2013/01/23 Kawah Ijen possible sealing and fluid release event')
    ax8.set_title('(h) 2020/07/16 Copahue possible sealing and fluid release event')
    #
    plt.tight_layout()
    plt.show()
    plt.close('all')
    #

def figure_2_alt(): # histograms 
    '''
    '''
    #fig, (ax1, ax2, ax3, ax4) = plt.subplots(nrows = 2, ncols = 2, figsize = (12,8))
    nrow = 2
    ncol = 2
    fig, ((ax4, ax1), (ax2, ax3)) = plt.subplots(nrows=nrow, ncols=ncol,figsize=(8,8))#(14,4))
    #
    # nrow = 3
    # ncol = 2
    # fig, ((ax01, ax02), (ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=nrow, ncols=ncol,figsize=(8,12))#(14,4))
    #
    roll_mean = False # results with rolling median 
    #
    if False: # plot 01: # events per month
        #auto_picked = True # dates elected automaticaly from dsar median and dsar rv correlations (only with FWVZ)
        '''
        '''
        sta = 'FWVZ'#'POS'#'FWVZ'#'COP'
        ## import events
        path = 'C:\\Users\\aar135\\codes_local_disk\\volc_forecast_tl\\volc_forecast_tl\\features\\lake_data\\'+sta+'\\selection\\'
        if auto_picked:
            path = 'C:\\Users\\aar135\\codes_local_disk\\volc_forecast_tl\\volc_forecast_tl\\features\\lake_data\\corr_dsar_ratevar\\'
        dates =[]
        path_dates = path+sta+'_dates_missed_events_from_dsar_median_rv_cc.txt'
        path_dates = path+sta+'_dates_max_CC_missed_events_from_dsar_median_rv.txt'

        path_dates_filt_off = path+sta+'_dates_max_CC_missed_events_from_dsar_median_rv_without_filt_peak.txt'
        path_dates_filt_on = path+sta+'_dates_max_CC_missed_events_from_dsar_median_rv_with_filt_peak.txt'
        #
        if False: # man picked
            path_mp = 'C:\\Users\\aar135\\codes_local_disk\\volc_forecast_tl\\volc_forecast_tl\\features\\lake_data\\'+sta+'\\selection\\'
            path_dates = path_mp+'FWVZ_temp_eruptive_periods.txt'
        #
        date_events = []
        cc_events = []
        max_events = []
        with open(path_dates_filt_on,'r') as fp:
            for ln in fp.readlines():
                _d, _cc, _mx =ln.rstrip().split(',')
                date_events.append(datetimeify(_d))
                cc_events.append(_cc)
                max_events.append(_mx)
            #date_events = [datetimeify(ln.rstrip()) for ln in fp.readlines()]
        #
        date_events_filt_off = []
        cc_events_filt_off = []
        max_events_filt_off = []
        with open(path_dates_filt_off,'r') as fp:
            for ln in fp.readlines():
                _d, _cc, _mx =ln.rstrip().split(',')
                date_events_filt_off.append(datetimeify(_d))
                cc_events_filt_off.append(_cc)
                max_events_filt_off.append(_mx)
            #date_events = [datetimeify(ln.rstrip()) for ln in fp.readlines()]

        # explosive 
        _jan, _feb , _mar, _apr, _may, _jun, _jul, _agu, _sep, _oct, _nov, _dec = 0,0,0,0,0,0,0,0,0,0,0,0
        for dat in date_events: 
            if dat.month == 1:
                _jan += 1
            if dat.month == 2:
                _feb += 1
            if dat.month == 3:
                _mar += 1
            if dat.month == 4:
                _apr += 1
            if dat.month == 5:
                _may += 1
            if dat.month == 6:
                _jun += 1
            if dat.month == 7:
                _jul += 1
            if dat.month == 8:
                _agu += 1
            if dat.month == 9:
                _sep += 1
            if dat.month == 10:
                _oct += 1
            if dat.month == 11:
                _nov += 1
            if dat.month == 12:
                _dec += 1
        # plot
        months = ['jan', 'feb' , 'mar', 'apr', 'may', 'jun', 'jul', 'agu', 'sep', 'oct', 'nov', 'dec']
        months = [1,2,3,4,5,6,7,8,9,10,11,12]
        n_events_expl = [_jan, _feb , _mar, _apr, _may, _jun, _jul, _agu, _sep, _oct, _nov, _dec]

        # non explosive 
        _jan, _feb , _mar, _apr, _may, _jun, _jul, _agu, _sep, _oct, _nov, _dec = 0,0,0,0,0,0,0,0,0,0,0,0
        for dat in date_events_filt_off: 
            if dat.month == 1:
                _jan += 1
            if dat.month == 2:
                _feb += 1
            if dat.month == 3:
                _mar += 1
            if dat.month == 4:
                _apr += 1
            if dat.month == 5:
                _may += 1
            if dat.month == 6:
                _jun += 1
            if dat.month == 7:
                _jul += 1
            if dat.month == 8:
                _agu += 1
            if dat.month == 9:
                _sep += 1
            if dat.month == 10:
                _oct += 1
            if dat.month == 11:
                _nov += 1
            if dat.month == 12:
                _dec += 1
        # 
        months = ['jan', 'feb' , 'mar', 'apr', 'may', 'jun', 'jul', 'agu', 'sep', 'oct', 'nov', 'dec']
        #months = [1,2,3,4,5,6,7,8,9,10,11,12]
        n_events_non_expl = [_jan, _feb , _mar, _apr, _may, _jun, _jul, _agu, _sep, _oct, _nov, _dec]

        # plot 
        x = np.arange(len(months))
        width = 0.35  # the width of the bars
        ax01.bar(x - width/2, n_events_non_expl, width, label='non-explosive')
        ax01.bar(x + width/2, n_events_expl, width, label='explosive')
        ax01.set_xticks(x, months)
        ax01.legend()
        ax01.set_xticks(x)
        ax01.set_xticklabels(months, rotation=90, ha='right')
        #ax01.set_xlabel('month')
        ax01.set_ylabel('# events')
        ax01.set_title('#Number events per month in Ruapehu')

    if False: # plot 02: # Events per year 
        #auto_picked = True # dates elected automaticaly from dsar median and dsar rv correlations (only with FWVZ)
        '''
        '''
        sta = 'FWVZ'#'POS'#'FWVZ'#'COP'
        ## import events
        path = 'C:\\Users\\aar135\\codes_local_disk\\volc_forecast_tl\\volc_forecast_tl\\features\\lake_data\\'+sta+'\\selection\\'
        if auto_picked:
            path = 'C:\\Users\\aar135\\codes_local_disk\\volc_forecast_tl\\volc_forecast_tl\\features\\lake_data\\corr_dsar_ratevar\\'
        dates =[]
        path_dates = path+sta+'_dates_missed_events_from_dsar_median_rv_cc.txt'
        path_dates = path+sta+'_dates_max_CC_missed_events_from_dsar_median_rv.txt'

        path_dates_filt_off = path+sta+'_dates_max_CC_missed_events_from_dsar_median_rv_without_filt_peak.txt'
        path_dates_filt_on = path+sta+'_dates_max_CC_missed_events_from_dsar_median_rv_with_filt_peak.txt'
        #
        if False: # man picked
            path_mp = 'C:\\Users\\aar135\\codes_local_disk\\volc_forecast_tl\\volc_forecast_tl\\features\\lake_data\\'+sta+'\\selection\\'
            path_dates = path_mp+'FWVZ_temp_eruptive_periods.txt'
        #
        date_events = []
        cc_events = []
        max_events = []
        with open(path_dates_filt_on,'r') as fp:
            for ln in fp.readlines():
                _d, _cc, _mx =ln.rstrip().split(',')
                date_events.append(datetimeify(_d))
                cc_events.append(_cc)
                max_events.append(_mx)
            #date_events = [datetimeify(ln.rstrip()) for ln in fp.readlines()]
        #
        date_events_filt_off = []
        cc_events_filt_off = []
        max_events_filt_off = []
        with open(path_dates_filt_off,'r') as fp:
            for ln in fp.readlines():
                _d, _cc, _mx =ln.rstrip().split(',')
                date_events_filt_off.append(datetimeify(_d))
                cc_events_filt_off.append(_cc)
                max_events_filt_off.append(_mx)
            #date_events = [datetimeify(ln.rstrip()) for ln in fp.readlines()]

        _09, _10 , _11, _12, _13, _14, _15, _16, _17, _18, _19, _20, _21 = 0,0,0,0,0,0,0,0,0,0,0,0,0
        for dat in date_events: 
            if dat.year == 2009:
                _09 += 1
            if dat.year == 2010:
                _10 += 1
            if dat.year == 2011:
                _11 += 1
            if dat.year == 2012:
                _12 += 1
            if dat.year == 2013:
                _13 += 1
            if dat.year == 2014:
                _14 += 1
            if dat.year == 2015:
                _15 += 1
            if dat.year == 2016:
                _16 += 1
            if dat.year == 2017:
                _17 += 1
            if dat.year == 2018:
                _18 += 1
            if dat.year == 2019:
                _19 += 1
            if dat.year == 2020:
                _20 += 1
            if dat.year == 2021:
                _21 += 1
        # 
        year = [2009,2010,2011,2012,2013,2014,2015,2016,2017,2018,2019,2020,2021]
        n_events_expl = [_09, _10 , _11, _12, _13, _14, _15, _16, _17, _18, _19, _20, _21]
        
        # non explosive 
        _09, _10 , _11, _12, _13, _14, _15, _16, _17, _18, _19, _20, _21 = 0,0,0,0,0,0,0,0,0,0,0,0,0
        for dat in date_events_filt_off: 
            if dat.year == 2009:
                _09 += 1
            if dat.year == 2010:
                _10 += 1
            if dat.year == 2011:
                _11 += 1
            if dat.year == 2012:
                _12 += 1
            if dat.year == 2013:
                _13 += 1
            if dat.year == 2014:
                _14 += 1
            if dat.year == 2015:
                _15 += 1
            if dat.year == 2016:
                _16 += 1
            if dat.year == 2017:
                _17 += 1
            if dat.year == 2018:
                _18 += 1
            if dat.year == 2019:
                _19 += 1
            if dat.year == 2020:
                _20 += 1
            if dat.year == 2021:
                _21 += 1
        # 
        year = ['2009','2010','2011','2012','2013','2014','2015','2016','2017','2018','2019','2020','2021']
        n_events_non_expl = [_09, _10 , _11, _12, _13, _14, _15, _16, _17, _18, _19, _20, _21]
        
        # plot 
        x = np.arange(len(year))
        width = 0.35  # the width of the bars
        ax02.bar(x - width/2, n_events_non_expl, width, label='non-explosive')
        ax02.bar(x + width/2, n_events_expl, width, label='explosive')
        #ax02.set_xtickslabels(year)
        ax02.set_xticks(x)
        ax02.set_xticklabels(year, rotation=90, ha='right')
        ax02.legend()
        #ax02.set_xlabel('Year')
        ax02.set_ylabel('# events')
        ax02.set_title('Number events per year in Ruapehu')

    if True: # plot 1
        #auto_picked = True # dates elected automaticaly from dsar median and dsar rv correlations (only with FWVZ)
        sta = 'FWVZ'#'FWVZ' 'POS'
        # read results 
        _path = 'C:\\Users\\aar135\\codes_local_disk\\volc_forecast_tl\\volc_forecast_tl\\features\\lake_data\\corr_dsar_ratevar\\hist_fig\\'
        if roll_mean:
            _path = 'C:\\Users\\aar135\\codes_local_disk\\volc_forecast_tl\\volc_forecast_tl\\features\\lake_data\\corr_dsar_ratevar\\hist_fig\\roll_mean\\'

        #
        plot_4days = False
        plot_non_explosive = False
        plot_only_non_explosive = True
        #
        if plot_4days:
            path1 = _path +sta+"_temp_dif_rate_ite100_4days.txt"
            path3 = _path +sta+"_temp_dif_rate_ite100_out_4days.txt"
        #
        if plot_non_explosive or plot_only_non_explosive:
            path1 = _path +sta+"_temp_dif_rate_ite100_20days_nofiltpeak.txt"
            path3 = _path +sta+"_temp_dif_rate_ite100_out_20days_nofiltpeak.txt"
        #    
        path2 = _path +sta+"_temp_dif_rate_ite100_20days.txt"
        path4 = _path +sta+"_temp_dif_rate_ite100_out_20days.txt"
        #
        if True:
            _fls = glob.glob(path1)
            dif_l1 = []
            rate_l1 = []
            rate_days_l1 = []
            for _fl in _fls:
                fl  = np.genfromtxt(_fl, delimiter="\t")
                [dif_l1.append(fl[i][0]) for i in range(len(fl))]
                [rate_l1.append(fl[i][1]) for i in range(len(fl))]
                [rate_days_l1.append(fl[i][2]) for i in range(len(fl))]

            _fls = glob.glob(path2)
            dif_l2 = []
            rate_l2 = []
            rate_days_l2 = []
            for _fl in _fls:
                fl  = np.genfromtxt(_fl, delimiter="\t")
                [dif_l2.append(fl[i][0]) for i in range(len(fl))]
                [rate_l2.append(fl[i][1]) for i in range(len(fl))]
                [rate_days_l2.append(fl[i][2]) for i in range(len(fl))]

            _fls = glob.glob(path3)
            dif_l3 = []
            rate_l3 = []
            rate_days_l3 = []
            for _fl in _fls:
                fl  = np.genfromtxt(_fl, delimiter="\t")
                [dif_l3.append(fl[i][0]) for i in range(len(fl))]
                [rate_l3.append(fl[i][1]) for i in range(len(fl))]
                [rate_days_l3.append(fl[i][2]) for i in range(len(fl))]

            _fls = glob.glob(path4)
            dif_l4 = []
            rate_l4 = []
            rate_days_l4 = []
            for _fl in _fls:
                fl  = np.genfromtxt(_fl, delimiter="\t")
                [dif_l4.append(fl[i][0]) for i in range(len(fl))]
                [rate_l4.append(fl[i][1]) for i in range(len(fl))]
                [rate_days_l4.append(fl[i][2]) for i in range(len(fl))]

        #replace rate > 5
        if False:
            for i, r in enumerate(rate_l1):
                if abs(r)>5:
                    rate_l1[i] = rate_l1[i]/3 

        #dif_l1 = dif_l1[:int(len(dif_l1)/2)]
        #rate_l1 =rate_l1[:int(len(rate_l1)/2)]
        #rate_days_l1 =rate_days_l1[:int(len(rate_days_l1)/2)]
        #
        colors = ['b', 'r', 'gray']#, 'm']
        #_heights, a_bins = np.histogram(pv_samp_in_1)
        
        # select lists 
        #ax.axvline(x=0.05, color = 'k', ls='--', linewidth=1, label = '0.05 threshold')
        if True:
            #multi = [dif_l1, dif_l2, dif_l3]# + dif_l4]
            #colors = ['lightgrey', 'r', 'b']
            if plot_non_explosive:
                labels = ['non-explosive events', 'explosive events', 'out eruption']
                colors = ['lightgrey', 'r', 'b']
                multi = [dif_l1, dif_l2, dif_l3]
            elif plot_only_non_explosive:
                labels = ['possible sealing events', 'random times']
                colors = [ 'b', 'lightgrey']
                multi = [dif_l1, dif_l3]
            else:
                labels = ['4 days back', '20 days back', 'out eruption']
                colors = ['lightgrey', 'r', 'b']
                multi = [dif_l1, dif_l2, dif_l3]
            alpha = [1,1,.5]
            bins = 20#np.linspace(0, 1, 13)
        else:
            multi = [dif_l1, dif_l2]# + dif_l4]
            colors = ['r', 'b']
            labels = ['4 days back', 'out eruption']
            bins = 20#np.linspace(0, 1, 13)
        ax1.hist(multi, bins, color = colors, label=labels, alpha= None, density = True)
        xlim = [-11, 11] #12.5]

        ax1.set_xlim(xlim)
        ax1.set_xlabel('d_temp [°C]')
        ax1.set_ylabel('pdf')
        #ax1.set_title('Ruapehu lake temperature')
        ax1.legend(loc = 1)

    if True: # plot 2
        #auto_picked = True # dates elected automaticaly from dsar median and dsar rv correlations (only with FWVZ)
        sta = 'FWVZ'#'FWVZ' 'POS'
        # read results 
        _path = 'C:\\Users\\aar135\\codes_local_disk\\volc_forecast_tl\\volc_forecast_tl\\features\\lake_data\\corr_dsar_ratevar\\hist_fig\\'
        #
        # path1 = _path +sta+"_level_dif_rate_ite100_4days.txt" 
        # path2 = _path +sta+"_level_dif_rate_ite100_20days.txt"
        # path3 = _path +sta+"_level_dif_rate_ite100_out_4days.txt"
        # path4 = _path +sta+"_level_dif_rate_ite100_out_20days.txt"
        #
        if plot_4days:
            path1 = _path +sta+"_level_dif_rate_ite100_4days.txt"
            path3 = _path +sta+"_level_dif_rate_ite100_out_4days.txt"
        #
        if plot_non_explosive or plot_only_non_explosive:
            path1 = _path +sta+"_level_dif_rate_ite100_20days_nofiltpeak.txt"
            path3 = _path +sta+"_level_dif_rate_ite100_out_20days_nofiltpeak.txt"
        #    
        path2 = _path +sta+"_level_dif_rate_ite100_20days.txt"
        path4 = _path +sta+"_level_dif_rate_ite100_out_20days.txt"
        #
        if True:
            _fls = glob.glob(path1)
            dif_l1 = []
            rate_l1 = []
            rate_days_l1 = []
            for _fl in _fls:
                fl  = np.genfromtxt(_fl, delimiter="\t")
                [dif_l1.append(fl[i][0]) for i in range(len(fl))]
                [rate_l1.append(fl[i][1]) for i in range(len(fl))]
                [rate_days_l1.append(fl[i][2]) for i in range(len(fl))]

            _fls = glob.glob(path2)
            dif_l2 = []
            rate_l2 = []
            rate_days_l2 = []
            for _fl in _fls:
                fl  = np.genfromtxt(_fl, delimiter="\t")
                [dif_l2.append(fl[i][0]) for i in range(len(fl))]
                [rate_l2.append(fl[i][1]) for i in range(len(fl))]
                [rate_days_l2.append(fl[i][2]) for i in range(len(fl))]

            _fls = glob.glob(path3)
            dif_l3 = []
            rate_l3 = []
            rate_days_l3 = []
            for _fl in _fls:
                fl  = np.genfromtxt(_fl, delimiter="\t")
                [dif_l3.append(fl[i][0]) for i in range(len(fl))]
                [rate_l3.append(fl[i][1]) for i in range(len(fl))]
                [rate_days_l3.append(fl[i][2]) for i in range(len(fl))]

            _fls = glob.glob(path4)
            dif_l4 = []
            rate_l4 = []
            rate_days_l4 = []
            for _fl in _fls:
                fl  = np.genfromtxt(_fl, delimiter="\t")
                [dif_l4.append(fl[i][0]) for i in range(len(fl))]
                [rate_l4.append(fl[i][1]) for i in range(len(fl))]
                [rate_days_l4.append(fl[i][2]) for i in range(len(fl))]

        #replace rate > 5
        if False:
            for i, r in enumerate(rate_l1):
                if abs(r)>5:
                    rate_l1[i] = rate_l1[i]/3 

        #dif_l1 = dif_l1[:int(len(dif_l1)/2)]
        #rate_l1 =rate_l1[:int(len(rate_l1)/2)]
        #rate_days_l1 =rate_days_l1[:int(len(rate_days_l1)/2)]
        #
        #_heights, a_bins = np.histogram(pv_samp_in_1)
        
        # select lists 
        #ax.axvline(x=0.05, color = 'k', ls='--', linewidth=1, label = '0.05 threshold')
       # + dif_l4]
        #colors = ['lightgrey', 'r', 'b']
        #labels = ['4 days back', '20 days back', 'out eruption']
        #
        if plot_non_explosive:
            labels = ['non-explosive events', 'explosive events', 'out eruption']
            colors = ['lightgrey', 'r', 'b']
            multi = [dif_l1, dif_l2, dif_l3]
        elif plot_only_non_explosive:
            labels = ['possible sealing events', 'random times']
            colors = [ 'b', 'lightgrey']
            multi = [dif_l1, dif_l3]
        else:
            labels = ['4 days back', '20 days back', 'out eruption']
            colors = ['lightgrey', 'r', 'b']
            multi = [dif_l1, dif_l2, dif_l3]
        #
        bins = 20#np.linspace(0, 1, 13)
        ax2.hist(multi, bins, color = colors, label=labels, density = True)
        xlim = None#[0, 7] #12.5]

        ax2.set_xlabel('d_z [m]')
        ax2.set_xlim([-.35,.35])
        #ax2.set_title('Ruapehu lake level')
        ax2.legend(loc = 1)

    if True: # plot 3
        #auto_picked = True # dates elected automaticaly from dsar median and dsar rv correlations (only with FWVZ)
        sta = 'POS'#'FWVZ' 'POS'
        # read results 
        _path = 'C:\\Users\\aar135\\codes_local_disk\\volc_forecast_tl\\volc_forecast_tl\\features\\lake_data\\corr_dsar_ratevar\\hist_fig\\'
        #
        # path1 = _path +sta+"_temp_dif_rate_ite100_4days.txt"
        # path2 = _path +sta+"_temp_dif_rate_ite100_20days.txt"
        # path3 = _path +sta+"_temp_dif_rate_ite100_out_4days.txt"
        # path4 = _path +sta+"_temp_dif_rate_ite100_out_20days.txt"
        #
        if plot_4days:
            path1 = _path +sta+"_temp_dif_rate_ite100_4days.txt"
            path3 = _path +sta+"_temp_dif_rate_ite100_out_4days.txt"
        #
        if plot_non_explosive or plot_only_non_explosive:
            path1 = _path +sta+"_temp_dif_rate_ite100_20days_nofiltpeak.txt"
            path3 = _path +sta+"_temp_dif_rate_ite100_out_20days_nofiltpeak.txt"
        #    
        path2 = _path +sta+"_temp_dif_rate_ite100_20days.txt"
        path4 = _path +sta+"_temp_dif_rate_ite100_out_20days.txt"
        #
        if True:
            _fls = glob.glob(path1)
            dif_l1 = []
            rate_l1 = []
            rate_days_l1 = []
            for _fl in _fls:
                fl  = np.genfromtxt(_fl, delimiter="\t")
                [dif_l1.append(fl[i][0]) for i in range(len(fl))]
                [rate_l1.append(fl[i][1]) for i in range(len(fl))]
                [rate_days_l1.append(fl[i][2]) for i in range(len(fl))]

            _fls = glob.glob(path2)
            dif_l2 = []
            rate_l2 = []
            rate_days_l2 = []
            for _fl in _fls:
                fl  = np.genfromtxt(_fl, delimiter="\t")
                [dif_l2.append(fl[i][0]) for i in range(len(fl))]
                [rate_l2.append(fl[i][1]) for i in range(len(fl))]
                [rate_days_l2.append(fl[i][2]) for i in range(len(fl))]

            _fls = glob.glob(path3)
            dif_l3 = []
            rate_l3 = []
            rate_days_l3 = []
            for _fl in _fls:
                fl  = np.genfromtxt(_fl, delimiter="\t")
                [dif_l3.append(fl[i][0]) for i in range(len(fl))]
                [rate_l3.append(fl[i][1]) for i in range(len(fl))]
                [rate_days_l3.append(fl[i][2]) for i in range(len(fl))]

            _fls = glob.glob(path4)
            dif_l4 = []
            rate_l4 = []
            rate_days_l4 = []
            for _fl in _fls:
                fl  = np.genfromtxt(_fl, delimiter="\t")
                [dif_l4.append(fl[i][0]) for i in range(len(fl))]
                [rate_l4.append(fl[i][1]) for i in range(len(fl))]
                [rate_days_l4.append(fl[i][2]) for i in range(len(fl))]

        #replace rate > 5
        if False:
            for i, r in enumerate(rate_l1):
                if abs(r)>5:
                    rate_l1[i] = rate_l1[i]/3 

        #dif_l1 = dif_l1[:int(len(dif_l1)/2)]
        #rate_l1 =rate_l1[:int(len(rate_l1)/2)]
        #rate_days_l1 =rate_days_l1[:int(len(rate_days_l1)/2)]
        #
        colors = ['b', 'r', 'gray']#, 'm']
        #_heights, a_bins = np.histogram(pv_samp_in_1)
        
        # select lists 
        #ax.axvline(x=0.05, color = 'k', ls='--', linewidth=1, label = '0.05 threshold')
        # + dif_l4]
        #
        if plot_non_explosive:
            labels = ['non-explosive events', 'explosive events', 'out eruption']
            colors = ['lightgrey', 'r', 'b']
            multi = [dif_l1, dif_l2, dif_l3]
        elif plot_only_non_explosive:
            labels = ['possible sealing events', 'random times']
            colors = [ 'b', 'lightgrey']
            multi = [dif_l1, dif_l3]
        else:
            labels = ['4 days back', '20 days back', 'out eruption']
            colors = ['lightgrey', 'r', 'b']
            multi = [dif_l1, dif_l2, dif_l3]
        #
        alpha = [1,1,.5]
        bins = 12#np.linspace(0, 1, 13)
        ax3.hist(multi, bins, color = colors, label=labels, density = True)
        xlim = None#[0, 7] #12.5]
        #
        ax3.set_xlabel('d_temp [°C]')
        ax3.set_ylabel('pdf')
        #ax3.set_title('Kawa Ijen lake temperature')
        ax3.legend(loc = 1)

    if True: # plot 4
        path =  'C:\\Users\\aar135\\codes_local_disk\\volc_forecast_tl\\volc_forecast_tl\\features\\lake_data\\corr_dsar_ratevar\\hist_fig\\'
        _temp_filt = np.genfromtxt(path+'FWVZ_temp_mean_event_100.txt')
        _temp_nofilt = np.genfromtxt(path+'FWVZ_temp_mean_event_100_nofiltpeak.txt')
        _temp_all = np.genfromtxt(path+'FWVZ_temp_all.txt')
        #
        if plot_only_non_explosive:
            colors = ['salmon', 'lightgrey']#
            multi = [_temp_nofilt, _temp_all]
            labels = ['possible sealing events', 'random times']
        else:
            colors = ['salmon', 'lightgrey']#, 'gray']#
            multi = [_temp_filt, _temp_nofilt]
            labels = ['possible sealing events', 'random times']
        bins = 20
        #
        ax4.hist(multi, bins, color = colors, label=labels, alpha= None, density = True, edgecolor='w', linewidth=1.2)
        ax4.set_xlabel('temp [°C]')
        ax4.set_ylabel('pdf')
        ax4.set_xlim([12,44])
        #ax4.set_title('Ruapehu events temperature')
        ax4.legend(loc = 1)
    #
    ax1.set_title('(b) Ruapehu lake temperature\ndifference before events')
    ax2.set_title('(c) Ruapehu lake level\ndifference before events')
    ax3.set_title('(d) Kawah Ijen lake temperature\ndifference before events')
    ax4.set_title('(a) Ruapehu lake temperature\n during events')
    #####################################################
    plt.tight_layout()
    plt.show()
    plt.close()

def figure_4_alt():
    '''
    Ruapehu 2009, rsam dsar, and lake levels
    '''
    '''
    plot: temperature cycle (1), rsam and dsar before event (2), and lake levels (3)
    '''
    sta = 'FWVZ' 
    if sta == 'FWVZ':
        ffm = False
        server = True # files imported from server to local pc 
        server2 = False # server at uni 
        #plot_erup = False
    #
    look_back = 16
    look_front = 5
    #
    erup_time = datetimeify('2009 07 13 06 30 00')
    #erup_time = datetimeify('2010 09 03 00 00 00')
    #erup_time = datetimeify('2021 03 04 12 00 00')
    #erup_time = datetimeify('2016 11 13 12 00 00')
    #
    #erup_time = datetimeify('2021 09 09 00 00 00')
    #
    day = timedelta(days=1)
    t0 = erup_time - look_back*day#30*day
    t1 = erup_time + look_front*day#hour
    #
    ## plot other data
    temp = True
    level = True
    rainfall = True
    ## 
    plot_erup = True
    # figure
    nrow = 2
    ncol = 1
    #fig, (ax0, ax1, ax2, ax4) = plt.subplots(nrows=nrow, ncols=ncol, figsize=(12,12))#(14,4)) #, ax4)
    #fig, (ax0, ax1, ax2) = plt.subplots(nrows=nrow, ncols=ncol, figsize=(12,8))#(14,4)) #, ax4)
    fig, (ax1, ax2) = plt.subplots(nrows=nrow, ncols=ncol, figsize=(10,6))#(14,4)) #, ax4)
    #
    for ax in [ax1,ax2]:#,ax4]: # plot eruption times 
        # plot event 
        #te = datetimeify('2009-07-07') 
        te = datetimeify('2009-07-13 06:30:00')
        ax.axvline(te+0.12*day, color='gray', alpha = .25, linestyle='-', linewidth=12, zorder = 0)
        ax.plot([], color='gray', alpha = .25,linestyle='-', linewidth=12, label = 'fluid release event')

        #ax.axvline(te+0.22*day, color='r',linestyle='-', linewidth=3, zorder = 4)
        #ax.plot([], color='r', linestyle='-', linewidth=3, label = 'eruption')  
    #####################################################
    # subplot cero
    if False:
        #
        sta = 'FWVZ'#'POS'#'FWVZ'#'COP'
        ## import events
        path = 'C:\\Users\\aar135\\codes_local_disk\\volc_forecast_tl\\volc_forecast_tl\\features\\lake_data\\'+sta+'\\selection\\'
        if auto_picked:
            path = 'C:\\Users\\aar135\\codes_local_disk\\volc_forecast_tl\\volc_forecast_tl\\features\\lake_data\\corr_dsar_ratevar\\'
        dates =[]
        path_dates = path+sta+'_dates_missed_events_from_dsar_median_rv_cc.txt'
        path_dates = path+sta+'_dates_max_CC_missed_events_from_dsar_median_rv.txt'

        path_dates_filt_off = path+sta+'_dates_max_CC_missed_events_from_dsar_median_rv_without_filt_peak.txt'
        path_dates_filt_on = path+sta+'_dates_max_CC_missed_events_from_dsar_median_rv_with_filt_peak.txt'
        #
        #
        date_events = []
        cc_events = []
        max_events = []
        with open(path_dates_filt_on,'r') as fp:
            for ln in fp.readlines():
                _d, _cc, _mx =ln.rstrip().split(',')
                date_events.append(datetimeify(_d))
                cc_events.append(_cc)
                max_events.append(_mx)
            #date_events = [datetimeify(ln.rstrip()) for ln in fp.readlines()]
        #
        date_events_filt_off = []
        cc_events_filt_off = []
        max_events_filt_off = []
        with open(path_dates_filt_off,'r') as fp:
            for ln in fp.readlines():
                _d, _cc, _mx =ln.rstrip().split(',')
                date_events_filt_off.append(datetimeify(_d))
                cc_events_filt_off.append(_cc)
                max_events_filt_off.append(_mx)
        #

        col = ['r','g','b']
        alpha = [.5, 1., 1.]
        thick_line = [1., 3., 3.]
        #
        mov_avg = True # moving average for temp and level data
        utc_0 = True
        # plot temp data
        if True:
            #
            if sta == 'FWVZ':
                #
                ti_e1 = datetimeify('2009 05 10 00 00 00')
                tf_e1 = datetimeify('2010 02 01 00 00 00')
                # import temp data
                path = '..'+os.sep+'data'+os.sep+"RU001_temp_data.csv"
                pd_temp = pd.read_csv(path, index_col=1)
                if utc_0:
                    pd_temp.index = [datetimeify(pd_temp.index[i])-6*hour for i in range(len(pd_temp.index))]
                else:
                    pd_temp.index = [datetimeify(pd_temp.index[i]) for i in range(len(pd_temp.index))]
                # plot data in axis twin axis
                # Trim the data
                temp_e1_tim = pd_temp[ti_e1: tf_e1].index.values

                #temp_e1_tim=to_nztimezone(temp_e1_tim)
                #
                temp_e1_val = pd_temp[ti_e1: tf_e1].loc[:,' t (C)'].values
                # ax2
                #ax2b = ax2.twinx()   
                if mov_avg: # plot moving average
                    n=50
                    #v_plot = (temp_e1_val-np.min(temp_e1_val))/np.max((temp_e1_val-np.min(temp_e1_val)))
                    v_plot = temp_e1_val
                    ax0.plot(temp_e1_tim, v_plot, '-', color='g', label='lake temperature', alpha = 1.)
                    #
                    #ax.plot(temp_e1_tim[:-n+1], moving_average(temp_e1_val, n=n), '--', color='k')#, label='temp. mov. avg.')
                    ax0.plot(temp_e1_tim[n-1-20:-20], moving_average(v_plot[::-1], n=n)[::-1], '--', color='k')#, label='lake temperature')
                else:
                    #ax2b.plot(temp_e1_tim, temp_e1_val, '-', color='g', label='temperature')
                    #ax2.set_ylim([-40,40])
                    #plt.show()
                    v_plot = temp_e1_val
                    ax0.plot(temp_e1_tim, v_plot, '-', color='g', label='lake temperature', alpha = 1.)
                ax0.set_ylabel('Temperature °C')
                ax0.set_ylim([10,50])
                #ax2b.set_ylabel('temperature C')   
                #ax.legend(loc = 2)   
                ## plot event
                for d_events in date_events_filt_off:
                    if d_events > ti_e1 and d_events <= tf_e1:
                        ax0.axvline(x=d_events, color='gray', ls='--', lw = 3)#, lw = 14)
                        #ax.axvline(x=cycle_date_mid[i], color='gray', ls='--')#, lw = 14)
                for d_events in date_events:
                    if d_events > ti_e1 and d_events <= tf_e1:
                        ax0.axvline(x=d_events, color='k', ls='--', lw = 3)#, lw = 14)
                        #ax0.axvline(x=cycle_date_mid[i], color='blue', ls='-')#, lw = 14)
                #
                te = datetimeify('2009 07 13 06 30 00') 
                #ax0.axvline(te+1.1*day, color='r',linestyle='-', linewidth=3, zorder = 4)
                #ax0.plot([], color='r', linestyle='-', linewidth=3, label = 'eruption') 
                #
                ax0.plot([],[], color='gray', ls='--', lw = 3, label = 'non-expl events')
                ax0.plot([],[], color='black', ls='--', lw = 3, label = 'expl events')
                #ax0.plot([],[], color='blue', ls='-', label = 'mid heat/cool cycle')
                ax0.legend(loc = 1)   
                ax0.grid()
                ax0.set_ylim([12,36])
                #plt.show()
    # subplot one: MF, HF, DSAR medians (DSAR yaxis left; MF, HF yaxis rigth). 1/RSAM (normalized)
    if True:
        # features
        fts_yleft = ['zsc2_dsarF__median']
        fts_yrigth = []#['zsc2_dsarF__rate_variance']#['zsc2_dsarF__change_quantiles__f_agg_"var"__isabs_False__qh_0.6__ql_0.4']#['zsc2_mfF__median','zsc2_hfF__median']
        data_yrigth = ['rsam']
        #
        col = ['b','b','r']
        alpha = [1., 1., .5]
        thick_line = [2., 6., 1.]
        ax1b = ax1.twinx() 
        for i, ft in enumerate(fts_yleft):
            if True: # load feature (else: cal feature. median or rv)
                #
                if 'zsc2_dsarF' in ft:
                    ds = ['zsc2_dsarF'] 
                if server:
                    path_feat_serv = 'C:\\Users\\aar135\\codes_local_disk\\volc_forecast_tl\\features_bkp\\features_server\\'
                    fm_e1 = ForecastModel(window=2., overlap=1., station =  sta,
                        look_forward=2., data_streams=ds, 
                        feature_dir=path_feat_serv, 
                        savefile_type='pkl') 
                elif server2:
                    path_feat_serv = 'U:\\Research\\EruptionForecasting\\eruptions\\features\\'
                    fm_e1 = ForecastModel(window=2., overlap=1., station =  sta,
                        look_forward=2., data_streams=ds, 
                        feature_dir=path_feat_serv, 
                        savefile_type='pkl') 
                else:
                    try:
                        fm_e1 = ForecastModel(window=2., overlap=1., station = sta,
                            look_forward=2., data_streams=ds, savefile_type='csv')
                    except:
                        fm_e1 = ForecastModel(window=2., overlap=1., station = sta,
                            look_forward=2., data_streams=ds, savefile_type='pkl')                    
                ##  
                ft = ft.replace("-",'"')
                # adding multiple Axes objects
                ft_e1 = fm_e1.get_features(ti=t0, tf=t1, n_jobs=1, compute_only_features=[ft])
                # extract values to plot 
                ft_e1_t = ft_e1[0].index.values
                ft_e1_v = ft_e1[0].loc[:,ft]
                #
                v_plot = ft_e1_v

                #v_plot = ft_e1_v-np.min(ft_e1_v)/np.max((ft_e1_v-np.min(ft_e1_v)))
                #v_plot = ft_e1_v/np.max(ft_e1_v)
            else: 
                #
                if 'zsc2_dsarF' in ft:
                    ds = 'zsc2_dsarF'
                if 'zsc2_mfF' in ft:
                    ds = 'zsc2_mfF' 
                if 'zsc2_hfF' in ft:
                    ds = 'zsc2_hfF' 
                # 
                #
                day = timedelta(days=1)
                fm = ForecastModel(window=2., overlap=1., station=sta,
                    look_forward=2., data_streams=[ds], 
                    data_dir=r'C:\Users\aar135\codes_local_disk\volc_forecast_tl\volc_forecast_tl\data'
                    )
                #
                N, M = [2,30]
                df = fm.data.df[t0:t1]
                if 'median' in ft:
                    test = df[ds].rolling(N*24*6).median()[N*24*6:]
                if 'rate_variance' in ft:
                    test = df[ds].rolling(N*24*6).apply(chqv)[N*24*6:]
                #
                #out = test.rolling(archtype.shape[0]).apply(partial(conv, (archtype-archtype.mean())/archtype.std()))       
                #out = out.resample('1D').ffill()
                #
                ft_e1_t = test.index
                v_plot = test.values
            #
            if ft == 'zsc2_dsarF__median':
                ft = 'nDSAR median'
            #
            ax1b.plot(ft_e1_t, v_plot, '-', color=col[i], alpha = alpha[i], linewidth = thick_line[i], label=' '+ ft, zorder = 2)
            ax1.plot([], [], '-', color=col[i], alpha = alpha[i], linewidth = thick_line[i], label=' '+ ft, zorder = 2)
            #
            #
            if ffm: # ffm 
                #ax1b = ax1.twinx() 
                #v_plot = data[data_stream].loc[inds]
                inv_rsam = fm_e1.data.get_data(ti=t0, tf=t1)['rsamF']#.loc[ft_e1_t]
                inv_rsam = 1./inv_rsam
                # normalized it to yaxis rigth 
                inv_rsam = inv_rsam/max(inv_rsam)
                inv_rsam = inv_rsam*0.5*max(v_plot)
                #
                ax1.plot(ft_e1_t, inv_rsam, '-', color= 'gray', linewidth=0.5, markersize=0.5, alpha = 1.)
                ax1.plot([], [], '-', color= 'gray', markersize=1, label='1/RSAM', alpha = 1.0)
                #ax1.set_ylim([0,1])
                #ax1.set_yticks([])
            #

            #
            #ax1.legend(loc = 2)
            #
            te = datetimeify(erup_time)#fm_e1.data.tes[int(erup[-1:])-1]
            #ax1.set_xticks([ft_e1[0].index[-1] - 7*day*i for i in range(int(30/7)+1)])#[dat.index.values[0],dat.index.values[-1]])#, ]np.arange(0, len(x)+1, 5))
            
            #ax1b.set_yticks([])
            ax1b.grid()
            ax1b.set_ylabel('nDSAR value')        #ax.set_xticks([ft_e1[0].index[-1]-7*day*i +day for i in range(5)])
            #ax1.set_yscale('log') #ax.set_yscale('log')
            #ax.set_xticks([ft_e1[0].index[-1] - 7*day*i for i in range(int(30/7)+1)])#[dat.index.values[0],dat.index.values[-1]])#, ]np.arange(0, len(x)+1, 5))
            #ax1.set_xticks([t1 - 5*day*i for i in range(int(look_back/5)+1)])#[dat.index.values[0],dat.index.values[-1]])#, ]np.arange(0, len(x)+1, 5))
        #     #
        # except:
        #     pass
        if fts_yrigth:
            #ax1b = ax1.twinx() 
            col = ['r','g']
            alpha = [1., .5]
            thick_line = [2.,1.]
            #try: 
            for i, ft in enumerate(fts_yrigth):
                if 'zsc2_dsarF' in ft:
                    ds = 'zsc2_dsarF'
                if 'zsc2_mfF' in ft:
                    ds = 'zsc2_mfF' 
                if 'zsc2_hfF' in ft:
                    ds = 'zsc2_hfF' 
                # 
                if False: # look feature in the prev cacl features (else: calculate feat from data; only for median and rv)
                    if server:
                        path_feat_serv = 'C:\\Users\\aar135\\codes_local_disk\\volc_forecast_tl\\features_bkp\\features_server\\'
                        fm_e1 = ForecastModel(window=2., overlap=1., station =  sta,
                            look_forward=2., data_streams=ds, 
                            feature_dir=path_feat_serv, 
                            savefile_type='pkl') 
                    elif server2:
                        path_feat_serv = 'U:\\Research\\EruptionForecasting\\eruptions\\features\\'
                        fm_e1 = ForecastModel(window=2., overlap=1., station =  sta,
                            look_forward=2., data_streams=ds, 
                            feature_dir=path_feat_serv, 
                            savefile_type='pkl') 
                    else:
                        try:
                            fm_e1 = ForecastModel(window=2., overlap=1., station = sta,
                                look_forward=2., data_streams=ds, savefile_type='csv')
                        except:
                            fm_e1 = ForecastModel(window=2., overlap=1., station = sta,
                                look_forward=2., data_streams=ds, savefile_type='pkl')                    
                    ##  
                    ft = ft.replace("-",'"')
                    
                    ft_e1 = fm_e1.get_features(ti=t0, tf=t1, n_jobs=1, compute_only_features=[ft])
                    # adding multiple Axes objects

                    # extract values to plot 
                    ft_e1_t = ft_e1[0].index.values
                    ft_e1_v = ft_e1[0].loc[:,ft]
                    #
                    v_plot = ft_e1_v

                else:
                    day = timedelta(days=1)
                    fm = ForecastModel(window=2., overlap=1., station=sta,
                        look_forward=2., data_streams=[ds], 
                        data_dir=r'C:\Users\aar135\codes_local_disk\volc_forecast_tl\volc_forecast_tl\data'
                        )
                    #
                    N, M = [2,30]
                    df = fm.data.df[t0:t1]
                    if 'median' in ft:
                        test = df[ds].rolling(N*24*6).median()[N*24*6:]
                    if 'rate_variance' in ft:
                        test = df[ds].rolling(N*24*6).apply(chqv)[N*24*6:]
                    #
                    #out = test.rolling(archtype.shape[0]).apply(partial(conv, (archtype-archtype.mean())/archtype.std()))       
                    #out = out.resample('1D').ffill()
                    #
                    ft_e1_t = test.index
                    v_plot = test.values

                #v_plot = ft_e1_v-np.min(ft_e1_v)/np.max((ft_e1_v-np.min(ft_e1_v)))
                #v_plot = ft_e1_v/np.max(ft_e1_v)
                #
                if ft == 'zsc2_mfF__median':
                    ft = 'nMF median'
                if ft == 'zsc2_hfF__median':
                    ft = 'nHF median'
                #
                ax1.plot(ft_e1_t, v_plot, '-', color=col[i], alpha = alpha[i],label=' '+ ft, zorder = 4)
                #
                #ax1.legend(loc = 3)
                #
                te = datetimeify(erup_time)#fm_e1.data.tes[int(erup[-1:])-1]
                #ax1.set_xticks([ft_e1[0].index[-1] - 7*day*i for i in range(int(30/7)+1)])#[dat.index.values[0],dat.index.values[-1]])#, ]np.arange(0, len(x)+1, 5))
                
                #ax1b.set_yticks([])
                ax1.grid()
                ax1.set_ylabel('Feature value')        #ax.set_xticks([ft_e1[0].index[-1]-7*day*i +day for i in range(5)])
                #ax.set_xticks([ft_e1[0].index[-1] - 7*day*i for i in range(int(30/7)+1)])#[dat.index.values[0],dat.index.values[-1]])#, ]np.arange(0, len(x)+1, 5))
                #ax1.set_xticks([t1 - 5*day*i for i in range(int(look_back/5)+1)])#[dat.index.values[0],dat.index.values[-1]])#, ]np.arange(0, len(x)+1, 5))
                #
            #except:
            #    pass

        else:
            pass
            if data_yrigth:
                #
                #ax1b = ax1.twinx() 
                #
                td = TremorData(station = sta)
                #td.update(ti=t0, tf=t1)
                data_streams = data_yrigth#['hf','mf', 'rsam']#, 'dsarF']
                label = ['RSAM','MF','HF','DSAR']
                #label = ['1/RSAM']
                inv = False
                if False:
                    data_streams = ['rsam']
                    label = ['RSAM']

                if type(data_streams) is str:
                    data_streams = [data_streams,]
                if any(['_' in ds for ds in data_streams]):
                    td._compute_transforms()
                #ax.set_xlim(*range)
                # plot data for each year
                norm= False
                _range = [t0,t1]
                log =False
                col_def = None
                data = td.get_data(*_range)
                xi = datetime(year=1,month=1,day=1,hour=0,minute=0,second=0)
                cols = ['k','r','g','m',[0.5,0.5,0.5],[0.75,0.75,0.75]]
                if inv:
                    cols = ['k','g','r','m',[0.5,0.5,0.5],[0.75,0.75,0.75]]
                inds = (data.index>=datetimeify(_range[0]))&(data.index<=datetimeify(_range[1]))
                i=0
                for data_stream, col in zip(data_streams,cols):
                    v_plot = data[data_stream].loc[inds]
                    if inv:
                        v_plot = 1/v_plot
                    if log:
                        v_plot = np.log10(v_plot)
                    if norm:
                        v_plot = (v_plot-np.min(v_plot))/np.max((v_plot-np.min(v_plot)))
                    if label:
                        if col_def:
                            ax1.plot(data.index[inds], v_plot, '-', color=col_def, label=label[i], linewidth=1., alpha = 1., zorder = 0)
                        else:
                            #ax1b.plot(data.index[inds], v_plot, '-', color=col, label=label[i], linewidth=1., alpha = 1.0, zorder = 0)
                            ax1.plot(data.index[inds], v_plot, '-', color=col, linewidth=2., alpha = .7, zorder = 0)
                            ax1b.plot([], [], '-', color=col, label=label[i], linewidth=2., alpha = .7, zorder = 0)
                            ax1.plot([], [], '-', color=col, label=label[i], linewidth=2., alpha = .7, zorder = 0)
                    else:
                        ax1.plot(data.index[inds], v_plot, '-', color=col, label=data_stream, linewidth=2., alpha = .7, zorder = 0)
                    i+=1
                for te in td.tes:
                    if [te>=datetimeify(_range[0]) and te<=datetimeify(_range[1])]:
                        pass
                        #ax.axvline(te, color='k', linestyle='--', linewidth=2, zorder = 0)
                #
                ax1.plot([], color='k', linestyle='--', linewidth=2)#, label = 'eruption')
                #ax4.set_xlim(_range)
                #ax1b.legend(loc = 2)
                #ax1b.grid()
                if log:
                    ax1.set_ylabel(' ')
                else:
                    ax1.set_ylabel('RSAM \u03BC m/s')
                #ax4.set_xlabel('Time [month-day hour]')
                #ax4.title.set_text('Station '+td.station+' ('+sta_code[td.station]+'): Tremor data')
                #
                #ax4.set_xticks([te - 5*day*i for i in range(int(look_back/5)+1)])#[dat.index.values[0],dat.index.values[-1]])#, ]np.arange(0, len(x)+1, 5))
                #ax4.set_ylim([1e9,1e13])
                ax1.set_yscale('log')
        #
        ax1.legend(loc = 2)      
        #
    # subplot two: temp data (if any: level and rainfall)
    if True:  
        mov_avg = True # moving average for temp and level data
        # convert to UTC 0
        utc_0 = True
        if utc_0:
            _utc_0 = 0#-13 # hours
        # plot temp data
        if sta == 'FWVZ':
            # plot temperature data 
            if temp:
                try:
                    ti_e1 = t0
                    tf_e1 = t1
                    # import temp data
                    path = '..'+os.sep+'data'+os.sep+"RU001_temp_data.csv"
                    pd_temp = pd.read_csv(path, index_col=1)
                    if utc_0:
                        #pd_temp.index = [datetimeify(pd_temp.index[i])-6*hour for i in range(len(pd_temp.index))]
                        pd_temp.index = [datetimeify(pd_temp.index[i])+_utc_0*hour for i in range(len(pd_temp.index))]
                    else:
                        pd_temp.index = [datetimeify(pd_temp.index[i]) for i in range(len(pd_temp.index))]
                    
                    # plot data in axis twin axis
                    # Trim the data
                    temp_e1_tim = pd_temp[ti_e1: tf_e1].index.values
                    #temp_e1_tim=to_nztimezone(temp_e1_tim)
                    #
                    temp_e1_val = pd_temp[ti_e1: tf_e1].loc[:,' t (C)'].values
                    # ax2
                    #ax2b = ax2.twinx()   
                    if mov_avg: # plot moving average
                        n=30
                        #v_plot = (temp_e1_val-np.min(temp_e1_val))/np.max((temp_e1_val-np.min(temp_e1_val)))
                        v_plot = temp_e1_val
                        ax2.plot(temp_e1_tim, v_plot, '-', color='g', label='lake temperature', alpha = 1.)
                        
                        #
                        #ax.plot(temp_e1_tim[:-n+1], moving_average(temp_e1_val, n=n), '--', color='k')#, label='temp. mov. avg.')
                        ax2.plot(temp_e1_tim[n-1-20:-20], moving_average(v_plot[::-1], n=n)[::-1], '--', color='g')#, label='lake temperature')
                    else:
                        v_plot = temp_e1_val
                        ax2.plot(temp_e1_tim, v_plot, '-', color='g', label='lake temperature', alpha = 1.)
                    ax2.set_ylabel('Temperature °C')
                    #
                    temp_min = min(temp_e1_val)
                    temp_max = max(temp_e1_val)
                    temp_mu = np.mean(temp_e1_tim)
                    temp_sigma = np.std(temp_e1_tim)
                    ax2.set_ylim([temp_mu-3*temp_sigma,temp_mu+3*temp_sigma])
                    #ax2.set_ylabel('temperature C')   
                except:
                    pass

            # plot lake level data
            if level:
                #try:
                if True:
                    ax2b = ax2.twinx()
                    #
                    # import temp data
                    if t0.year >= 2016:# and t0.month >= 3:
                        path = '..'+os.sep+'data'+os.sep+"RU001A_level_data.csv"

                    elif t0.year < 2012:# and t0.month < 7:
                        path = '..'+os.sep+'data'+os.sep+"RU001_level_data.csv"      
                    else:
                        path = '..'+os.sep+'data'+os.sep+"RU001A_level_data_full.csv"
                    #    
                    pd_lev = pd.read_csv(path, index_col=1)
                    if utc_0:
                        #pd_temp.index = [datetimeify(pd_temp.index[i])-6*hour for i in range(len(pd_temp.index))]
                        #pd_temp.index = [datetimeify(pd_temp.index[i])-0*hour for i in range(len(pd_temp.index))]
                        pd_lev.index = [datetimeify(pd_lev.index[i])+_utc_0*hour for i in range(len(pd_lev.index))]
                    else:
                        pd_lev.index = [datetimeify(pd_lev.index[i]) for i in range(len(pd_lev.index))]

                    if t0.year>2010 and t1.year<2016: # rolling median over data
                        N = 2
                        pd_lev = pd_lev[:].rolling(40).median()#[N*24*6:]

                    # plot data in axis twin axis
                    # Trim the data
                    lev_e1_tim = pd_lev[ti_e1: tf_e1].index.values
                    #temp_e1_tim=to_nztimezone(temp_e1_tim)
                    lev_e1_val = pd_lev[ti_e1: tf_e1].loc[:,' z (m)'].values
                    # ax2
                    #ax2b = ax2.twinx()
                    if False:#mov_avg: # plot moving average
                        n=10
                        v_plot = temp_e1_val
                        ax2b.plot(temp_e1_tim, v_plot, '-', color='royalblue', alpha = 1.)
                        ax2.plot([], [], '-', color='royalblue', label='lake level')
                        #ax2b.plot(temp_e1_tim[:-n+1], moving_average(temp_e1_val, n=n), '--', color='k', label='temp. mov. avg.')
                        #v_plot = (temp_e1_val-np.min(temp_e1_val))/np.max((temp_e1_val-np.min(temp_e1_val)))
                        ax2b.plot(temp_e1_tim[n-1-10:-10], moving_average(v_plot[::-1], n=n)[::-1], '--', color='royalblue', label='lake level')
                        #ax2b.plot(temp_e1_tim, v_plot, '-', color='b', alpha = 0.3)
                    else:
                        v_plot = lev_e1_val
                        ax2b.plot(lev_e1_tim, v_plot, '-', color='royalblue', label='lake level')
                    #
                    ax2b.set_ylabel('Lake level cm') 
                    ax2.plot([], [], '-', color='royalblue', label='lake level')

                #except:
                #    pass
            
            # plot rainfall data
            if rainfall:
                try:
                    ti_e1 = t0#datetimeify(t0)
                    tf_e1 = t1#datetimeify(t1)
                    #
                    # import temp data
                    path = '..'+os.sep+'data'+os.sep+"_chateau_rain.csv"
                    pd_rf = pd.read_csv(path, index_col=1)
                    pd_rf.index = pd.to_datetime(pd_rf.index, format='%Y%m%d:%H%M')
                    if utc_0:
                        pd_rf.index = [pd_rf.index[i]+_utc_0*hour for i in range(len(pd_rf.index))]

                    # Trim the data
                    rf_e2_tim = pd_rf[ti_e1: tf_e1].index#.values
                    rf_e2_val = pd_rf[ti_e1: tf_e1].loc[:,'Amount(mm)'].values /3
                    # ax2
                    #ax2b = ax2.twinx()
                    v_plot = (rf_e2_val-np.min(rf_e2_val))/np.max((rf_e2_val-np.min(rf_e2_val)))
                    #v_plot = v_plot*5 + 14
                    if temp_max:
                        v_plot = v_plot*(temp_max-temp_min)*0.6 + temp_min
                    ax2.plot(rf_e2_tim, v_plot, '-', color='c', label='rain fall', alpha = 0.6)
                    #ax2b.set_ylabel('temperature C')
                    #ax2b.legend(loc = 1)
                except:
                    pass

        if sta == 'DAM' or sta == 'POS':
            lake = False # no data
            rainfall = False # no data
            try:
                if temp:
                    ti_e1 = t0
                    tf_e1 = t1
                    # import temp data
                    path = '..'+os.sep+'data'+os.sep+"DAM_temp_data.csv"
                    pd_temp = pd.read_csv(path, index_col=1)

                    if utc_0:
                        pd_temp.index = [datetimeify(pd_temp.index[i])-6*hour for i in range(len(pd_temp.index))]
                    else:
                        pd_temp.index = [datetimeify(pd_temp.index[i]) for i in range(len(pd_temp.index))]
                    # plot data in axis twin axis
                    # Trim the data
                    temp_e1_tim = pd_temp[ti_e1: tf_e1].index.values
                    #temp_e1_tim=to_nztimezone(temp_e1_tim)
                    #
                    temp_e1_val = pd_temp[ti_e1: tf_e1].loc[:,' t (C)'].values
                    # ax2
                    #ax2b = ax2.twinx()   
                    if mov_avg: # plot moving average
                        n=50
                        #v_plot = (temp_e1_val-np.min(temp_e1_val))/np.max((temp_e1_val-np.min(temp_e1_val)))
                        v_plot = temp_e1_val
                        ax2.plot(temp_e1_tim, v_plot, '-', color='g', label='lake temperature', alpha = 1.)
                        #
                        #ax.plot(temp_e1_tim[:-n+1], moving_average(temp_e1_val, n=n), '--', color='k')#, label='temp. mov. avg.')
                        _x = temp_e1_tim[n-1-20:-20]
                        _y = moving_average(v_plot[::-1], n=n)[::-1]
                        ax2.plot(_x, _y, '--', color='k')#, label='lake temperature')
                    else:
                        v_plot = temp_e1_val
                        ax2.plot(temp_e1_tim, v_plot, '-', color='g', label='lake temperature', alpha = 1.)
                    ax2.set_ylabel('Temperature °C')
                    
                    _ylim = [min(_y)-1,max(_y)+1] 
                    ax2.set_ylim(_ylim)
                    #ax2.set_ylabel('temperature C')   
            except:
                pass
            if plot_erup: # plot vertical lines
                te = datetimeify(erup_time)#fm_e1.data.tes[int(erup[-1:])-1]
                ax2.axvline(te, color='k',linestyle='--', linewidth=2, zorder = 4)
                #ax2.plot([], color='k', linestyle='--', linewidth=2, label = 'eruption')

        ax2.legend(loc = 2)   
        ax2.grid()
        #ax2.set_xticks([t1 - 5*day*i for i in range(int(look_back/5)+1)])#[dat.index.values[0],dat.index.values[-1]])#, ]np.arange(0, len(x)+1, 5))
    # subplot three: filtered  RSAM, MF, HF datastreams
    if False:
        #
        td = TremorData(station = sta)
        #td.update(ti=t0, tf=t1)
        data_streams = ['rsam','hf', 'mf']#, 'dsarF']
        label = ['nRSAM','nHF','nMF','nDSAR']
        #label = ['1/RSAM']
        inv = False
        if False:
            data_streams = ['hf', 'mf', 'rsam', 'lf', 'vlf']
            label = ['HF','MF','RSAM','LF', 'VLF']

        if type(data_streams) is str:
            data_streams = [data_streams,]
        if any(['_' in ds for ds in data_streams]):
            td._compute_transforms()
        #ax.set_xlim(*range)
        # plot data for each year
        norm= False
        _range = [t0,t1]
        log =False
        col_def = None
        data = td.get_data(*_range)
        xi = datetime(year=1,month=1,day=1,hour=0,minute=0,second=0)
        cols = ['r','g','gray','m',[0.5,0.5,0.5],[0.75,0.75,0.75]]
        if inv:
            cols = ['gray','g','r','m',[0.5,0.5,0.5],[0.75,0.75,0.75]]
        inds = (data.index>=datetimeify(_range[0]))&(data.index<=datetimeify(_range[1]))
        i=0
        for data_stream, col in zip(data_streams,cols):
            v_plot = data[data_stream].loc[inds]
            if inv:
                v_plot = 1/v_plot
            if log:
                v_plot = np.log10(v_plot)
            if norm:
                v_plot = (v_plot-np.min(v_plot))/np.max((v_plot-np.min(v_plot)))
            if label:
                if col_def:
                    ax3.plot(data.index[inds], v_plot, '-', color=col_def, label=label[i], alpha = 1.0)
                else:
                    ax3.plot(data.index[inds], v_plot, '-', color=col, label=label[i], alpha = 1.0)
            else:
                ax3.plot(data.index[inds], v_plot, '-', color=col, label=data_stream, alpha = 1.0)
            i+=1
        for te in td.tes:
            if [te>=datetimeify(_range[0]) and te<=datetimeify(_range[1])]:
                pass
                #ax.axvline(te, color='k', linestyle='--', linewidth=2, zorder = 0)
        #
        #ax.plot([], color='k', linestyle='--', linewidth=2, label = 'eruption')
        #ax3.set_xlim(_range)
        ax3.legend(loc = 2)
        ax3.grid()
        if log:
            ax3.set_ylabel(' ')
        else:
            ax3.set_ylabel('\u03BC m/s')
        #ax3.set_xlabel('Time [month-day hour]')
        #ax3.title.set_text('Station '+td.station+' ('+sta_code[td.station]+'): Tremor data')
        #
        if plot_erup: # plot vertical lines
            te = datetimeify(erup_time)#fm_e1.data.tes[int(erup[-1:])-1]
            ax3.axvline(te, color='k',linestyle='--', linewidth=2, zorder = 4)
            ax3.plot([], color='k', linestyle='--', linewidth=2, label = 'event')
        
        #
        te = datetimeify(erup_time)#fm_e1.data.tes[int(erup[-1:])-1]
        #ax3.set_xticks([te - 5*day*i for i in range(int(look_back/5)+1)])#[dat.index.values[0],dat.index.values[-1]])#, ]np.arange(0, len(x)+1, 5))

        ax3.set_yscale('log')
        lims = [10e-2,10e2]#[np.mean(np.log(v_plot))-3*np.std(np.log(v_plot)), np.mean(np.log(v_plot))+6*np.std(np.log(v_plot))]
        if sta == 'COP':
            lims = None
        ax3.set_ylim(lims)
    # subplot four: non filtered  RSAM, MF, HF datastreams
    if False:
        #
        td = TremorData(station = sta)
        #td.update(ti=t0, tf=t1)
        data_streams = ['hf','mf', 'rsam']#, 'dsarF']
        label = ['HF','MF','RSAM','DSAR']
        #label = ['1/RSAM']
        inv = False
        if False:
            data_streams = ['hf', 'mf', 'rsam', 'lf', 'vlf']
            label = ['HF','MF','RSAM','LF', 'VLF']

        if type(data_streams) is str:
            data_streams = [data_streams,]
        if any(['_' in ds for ds in data_streams]):
            td._compute_transforms()
        #ax.set_xlim(*range)
        # plot data for each year
        norm= False
        _range = [t0,t1]
        log =False
        col_def = None
        data = td.get_data(*_range)
        xi = datetime(year=1,month=1,day=1,hour=0,minute=0,second=0)
        cols = ['g','r','gray','m',[0.5,0.5,0.5],[0.75,0.75,0.75]]
        if inv:
            cols = ['gray','g','r','m',[0.5,0.5,0.5],[0.75,0.75,0.75]]
        inds = (data.index>=datetimeify(_range[0]))&(data.index<=datetimeify(_range[1]))
        i=0
        for data_stream, col in zip(data_streams,cols):
            v_plot = data[data_stream].loc[inds]
            if inv:
                v_plot = 1/v_plot
            if log:
                v_plot = np.log10(v_plot)
            if norm:
                v_plot = (v_plot-np.min(v_plot))/np.max((v_plot-np.min(v_plot)))
            if label:
                if col_def:
                    ax4.plot(data.index[inds], v_plot, '-', color=col_def, label=label[i], linewidth=1., alpha = 1.0)
                else:
                    ax4.plot(data.index[inds], v_plot, '-', color=col, label=label[i], linewidth=1., alpha = 1.0)
            else:
                ax4.plot(data.index[inds], v_plot, '-', color=col, label=data_stream, linewidth=1., alpha = 1.0)
            i+=1
        for te in td.tes:
            if [te>=datetimeify(_range[0]) and te<=datetimeify(_range[1])]:
                pass
                #ax.axvline(te, color='k', linestyle='--', linewidth=2, zorder = 0)
        #
        #ax.plot([], color='k', linestyle='--', linewidth=2, label = 'eruption')
        #ax4.set_xlim(_range)
        ax4.legend(loc = 2)
        ax4.grid()
        if log:
            ax4.set_ylabel(' ')
        else:
            ax4.set_ylabel('\u03BC m/s')
        #ax4.set_xlabel('Time [month-day hour]')
        #ax4.title.set_text('Station '+td.station+' ('+sta_code[td.station]+'): Tremor data')
        #
        te = datetimeify(erup_time)#fm_e1.data.tes[int(erup[-1:])-1]
        #ax4.set_xticks([te - 5*day*i for i in range(int(look_back/5)+1)])#[dat.index.values[0],dat.index.values[-1]])#, ]np.arange(0, len(x)+1, 5))
        #ax4.set_ylim([1e9,1e13])
        ax4.set_yscale('log')
    #
    if False: # plot vertical lines
        te = datetimeify(erup_time)#fm_e1.data.tes[int(erup[-1:])-1]
        ax1.axvline(te, color='k',linestyle='--', linewidth=2, zorder = 4)
        ax2.axvline(te, color='k',linestyle='--', linewidth=2, zorder = 4)
        ax3.axvline(te, color='k',linestyle='--', linewidth=2, zorder = 4)
        ax1.plot([], color='k', linestyle='--', linewidth=2, label = 'event')
    #
    if False:#save_png_path:
        dat = erup_time.strftime('%Y-%m-%d')
        title =  sta+'_'+dat+'_'+'look_back'+str(look_back)
        ax1.set_title(title)
        plt.savefig(save_png_path+sta+'_'+dat+'_'+'look_back'+str(look_back)+'.png')
    #
    ax1.set_xticks([t1 - 4*day*i for i in range(int((look_back+look_front)/4)+1)])
    ax2.set_xticks([t1 - 4*day*i for i in range(int((look_back+look_front)/4)+1)])
    #ax3.set_xticks([t1 - 2*day*i for i in range(int((look_back+look_front)/2)+1)])
    #ax4.set_xticks([t1 - 4*day*i for i in range(int((look_back+look_front)/4)+1)])
    #    
    ax1.set_xlim([t0+2*day,t1])
    ax2.set_xlim([t0+2*day,t1])
    #ax2b.set_ylim([0.2,0.5])
    #ax2b.set_ylim([0.1,0.5])

    #ax3.set_xlim([t0+2*day,t1])
    #ax4.set_xlim([t0+2*day,t1])
    #
    #ax0.set_title('(a) Ruapehu 2009 temperature cycle and hydrothermal event on 07/13')
    ax1.set_title('(a) Ruapehu DSAR median and RSAM before hydrothermal event on 2009/07/13')
    ax2.set_title('(b) Ruapehu Lake temperature and level before hydrothermal event on 2009/07/13')
    #ax4.set_title('Seismic datastreams before hydrothermal event on 07/13')
    #
    plt.tight_layout()
    plt.show()
    plt.close('all')

def figure_5_alt_ruap_kawa_copa():  # man selected events RSAM and DSAR
    '''
    plot events: seismic from multiple events (rsam and dsar)
    '''
    sta = 'FWVZ' 
    if sta == 'FWVZ':
        ffm = False
        server = False # files imported from server to local pc 
        server2 = True # server at uni 
        #plot_erup = False
    #
    look_back = 0
    look_front = 1
    #
    day = timedelta(days=1)
    #
    ## plot other data
    temp = True
    level = True
    rainfall = True
    ## 
    plot_erup = True
    # figure
    #nrow = 5
    #ncol = 2
    #fig, (ax0, ax1, ax2, ax4) = plt.subplots(nrows=nrow, ncols=ncol, figsize=(12,12))#(14,4)) #, ax4)
    #fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6), (ax7, ax8), (ax9, ax10)) = plt.subplots(nrows=nrow, ncols=ncol, figsize=(20,20))#(14,4)) #, ax4)
    nrow = 3
    ncol = 2
    fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(nrows=nrow, ncols=ncol, figsize=(14,8))#(14,4)) #, ax4)

    # linear fit 
    import matplotlib.dates as mdates
    l_fit = []
    #####################################################
    # First column (first and second row)
    if True:
        #
        erup_times = [datetimeify('2006 10 04 09 30 00'), 
                        datetimeify('2009 07 13 06 30 00'), 
                        datetimeify('2010 09 03 16 00 00')]
        #
        erup_times = [datetimeify('2006 10 04 11 45 00'), 
                        datetimeify('2009 07 13 08 00 00'), 
                        datetimeify('2010 09 03 16 45 00')]
        #
        for j,ax in enumerate([ax1, ax3, ax5]):
            #
            t0 = erup_times[j] - look_back*day#30*day
            t1 = erup_times[j] + look_front*day#hour
            # subplot one: MF, HF, DSAR medians (DSAR yaxis left; MF, HF yaxis rigth). 1/RSAM (normalized)
            if True:
                # features
                fts_yleft = []#['zsc2_dsarF__median']
                fts_yrigth = []#['zsc2_dsarF__rate_variance']#['zsc2_dsarF__change_quantiles__f_agg_"var"__isabs_False__qh_0.6__ql_0.4']#['zsc2_mfF__median','zsc2_hfF__median']
                data_yrigth = ['rsam']
                
                #
                col = ['b','b','r']
                alpha = [1., 1., .5]
                thick_line = [2., 6., 1.]
                axb = ax.twinx() 
                for i, ft in enumerate(fts_yleft):
                    if True: # load feature (else: cal feature. median or rv)
                        if 'zsc2_dsarF' in ft:
                            ds = ['zsc2_dsarF'] 
                        if server:
                            path_feat_serv = 'C:\\Users\\aar135\\codes_local_disk\\volc_forecast_tl\\features_bkp\\features_server\\'
                            fm_e1 = ForecastModel(window=2., overlap=1., station =  sta,
                                look_forward=2., data_streams=ds, 
                                feature_dir=path_feat_serv, 
                                savefile_type='pkl') 
                        elif server2:
                            path_feat_serv = 'U:\\Research\\EruptionForecasting\\eruptions\\features\\'
                            fm_e1 = ForecastModel(window=2., overlap=1., station =  sta,
                                look_forward=2., data_streams=ds, 
                                feature_dir=path_feat_serv, 
                                savefile_type='pkl') 
                        else:
                            try:
                                fm_e1 = ForecastModel(window=2., overlap=1., station = sta,
                                    look_forward=2., data_streams=ds, savefile_type='csv')
                            except:
                                fm_e1 = ForecastModel(window=2., overlap=1., station = sta,
                                    look_forward=2., data_streams=ds, savefile_type='pkl')                    
                        ##  
                        ft = ft.replace("-",'"')
                        # adding multiple Axes objects
                        ft_e1 = fm_e1.get_features(ti=t0, tf=t1, n_jobs=1, compute_only_features=[ft])
                        # extract values to plot 
                        ft_e1_t = ft_e1[0].index.values
                        ft_e1_v = ft_e1[0].loc[:,ft]
                        #
                        v_plot = ft_e1_v

                        #v_plot = ft_e1_v-np.min(ft_e1_v)/np.max((ft_e1_v-np.min(ft_e1_v)))
                        #v_plot = ft_e1_v/np.max(ft_e1_v)
                    else: 
                        #
                        if 'zsc2_dsarF' in ft:
                            ds = 'zsc2_dsarF'
                        if 'zsc2_mfF' in ft:
                            ds = 'zsc2_mfF' 
                        if 'zsc2_hfF' in ft:
                            ds = 'zsc2_hfF' 
                        # 
                        #
                        day = timedelta(days=1)
                        fm = ForecastModel(window=2., overlap=1., station=sta,
                            look_forward=2., data_streams=[ds], 
                            data_dir=r'C:\Users\aar135\codes_local_disk\volc_forecast_tl\volc_forecast_tl\data'
                            )
                        #
                        N, M = [2,30]
                        df = fm.data.df[t0:t1]
                        if 'median' in ft:
                            test = df[ds].rolling(N*24*6).median()[N*24*6:]
                        if 'rate_variance' in ft:
                            test = df[ds].rolling(N*24*6).apply(chqv)[N*24*6:]
                        #
                        #out = test.rolling(archtype.shape[0]).apply(partial(conv, (archtype-archtype.mean())/archtype.std()))       
                        #out = out.resample('1D').ffill()
                        #
                        ft_e1_t = test.index
                        v_plot = test.values
                    #
                    if ft == 'zsc2_dsarF__median':
                        ft = 'nDSAR median'
                    #
                    axb.plot(ft_e1_t, v_plot, '-', color=col[i], alpha = alpha[i], linewidth = thick_line[i], label=' '+ ft, zorder = 2)

                    #
                    if ffm: # ffm 
                        #ax1b = ax1.twinx() 
                        #v_plot = data[data_stream].loc[inds]
                        inv_rsam = fm_e1.data.get_data(ti=t0, tf=t1)['rsamF']#.loc[ft_e1_t]
                        inv_rsam = 1./inv_rsam
                        # normalized it to yaxis rigth 
                        inv_rsam = inv_rsam/max(inv_rsam)
                        inv_rsam = inv_rsam*0.5*max(v_plot)
                        #
                        ax.plot(ft_e1_t, inv_rsam, '-', color= 'gray', linewidth=0.5, markersize=0.5, alpha = 1.)
                        ax.plot([], [], '-', color= 'gray', markersize=1, label='1/RSAM', alpha = 1.0)
                        #ax1.set_ylim([0,1])
                        #ax1.set_yticks([])
                    #
                    if plot_erup: # plot vertical lines
                        te = datetimeify(erup_times[j])#fm_e1.data.tes[int(erup[-1:])-1]
                        if j == 0:
                            ax.axvline(te, color='red', alpha = .25, linestyle='-', linewidth=12, zorder = 0)
                            axb.plot([], color='red', alpha = .25, linestyle='-', linewidth=12, label = 'eruption')
                        else:
                            ax.axvline(te, color='gray', alpha = .25, linestyle='-', linewidth=12, zorder = 0)
                            axb.plot([], color='gray', alpha = .25, linestyle='-', linewidth=12, label = 'fluid release event')
                    #
                    #ax1.legend(loc = 2)
                    #
                    te = datetimeify(erup_times[j])#fm_e1.data.tes[int(erup[-1:])-1]
                    #ax1.set_xticks([ft_e1[0].index[-1] - 7*day*i for i in range(int(30/7)+1)])#[dat.index.values[0],dat.index.values[-1]])#, ]np.arange(0, len(x)+1, 5))
                    
                    #ax1b.set_yticks([])
                    axb.grid()
                    axb.set_ylabel('nDSAR value')        #ax.set_xticks([ft_e1[0].index[-1]-7*day*i +day for i in range(5)])
                    #ax1.set_yscale('log') #ax.set_yscale('log')
                    #ax.set_xticks([ft_e1[0].index[-1] - 7*day*i for i in range(int(30/7)+1)])#[dat.index.values[0],dat.index.values[-1]])#, ]np.arange(0, len(x)+1, 5))
                    #ax1.set_xticks([t1 - 5*day*i for i in range(int(look_back/5)+1)])#[dat.index.values[0],dat.index.values[-1]])#, ]np.arange(0, len(x)+1, 5))
                #     #
                # except:
                #     pass
                if fts_yrigth:
                    #ax1b = ax1.twinx() 
                    col = ['r','g']
                    alpha = [1., .5]
                    thick_line = [2.,1.]
                    #try: 
                    for i, ft in enumerate(fts_yrigth):
                        if 'zsc2_dsarF' in ft:
                            ds = 'zsc2_dsarF'
                        if 'zsc2_mfF' in ft:
                            ds = 'zsc2_mfF' 
                        if 'zsc2_hfF' in ft:
                            ds = 'zsc2_hfF' 
                        # 
                        if False: # look feature in the prev cacl features (else: calculate feat from data; only for median and rv)
                            if server:
                                path_feat_serv = 'C:\\Users\\aar135\\codes_local_disk\\volc_forecast_tl\\features_bkp\\features_server\\'
                                fm_e1 = ForecastModel(window=2., overlap=1., station =  sta,
                                    look_forward=2., data_streams=ds, 
                                    feature_dir=path_feat_serv, 
                                    savefile_type='pkl') 
                            elif server2:
                                path_feat_serv = 'U:\\Research\\EruptionForecasting\\eruptions\\features\\'
                                fm_e1 = ForecastModel(window=2., overlap=1., station =  sta,
                                    look_forward=2., data_streams=ds, 
                                    feature_dir=path_feat_serv, 
                                    savefile_type='pkl') 
                            else:
                                try:
                                    fm_e1 = ForecastModel(window=2., overlap=1., station = sta,
                                        look_forward=2., data_streams=ds, savefile_type='csv')
                                except:
                                    fm_e1 = ForecastModel(window=2., overlap=1., station = sta,
                                        look_forward=2., data_streams=ds, savefile_type='pkl')                    
                            ##  
                            ft = ft.replace("-",'"')
                            
                            ft_e1 = fm_e1.get_features(ti=t0, tf=t1, n_jobs=1, compute_only_features=[ft])
                            # adding multiple Axes objects

                            # extract values to plot 
                            ft_e1_t = ft_e1[0].index.values
                            ft_e1_v = ft_e1[0].loc[:,ft]
                            #
                            v_plot = ft_e1_v

                        else:
                            day = timedelta(days=1)
                            fm = ForecastModel(window=2., overlap=1., station=sta,
                                look_forward=2., data_streams=[ds], 
                                data_dir=r'C:\Users\aar135\codes_local_disk\volc_forecast_tl\volc_forecast_tl\data'
                                )
                            #
                            N, M = [2,30]
                            df = fm.data.df[t0:t1]
                            if 'median' in ft:
                                test = df[ds].rolling(N*24*6).median()[N*24*6:]
                            if 'rate_variance' in ft:
                                test = df[ds].rolling(N*24*6).apply(chqv)[N*24*6:]
                            #
                            #out = test.rolling(archtype.shape[0]).apply(partial(conv, (archtype-archtype.mean())/archtype.std()))       
                            #out = out.resample('1D').ffill()
                            #
                            ft_e1_t = test.index
                            v_plot = test.values

                        #v_plot = ft_e1_v-np.min(ft_e1_v)/np.max((ft_e1_v-np.min(ft_e1_v)))
                        #v_plot = ft_e1_v/np.max(ft_e1_v)
                        #
                        if ft == 'zsc2_mfF__median':
                            ft = 'nMF median'
                        if ft == 'zsc2_hfF__median':
                            ft = 'nHF median'
                        #
                        ax.plot(ft_e1_t, v_plot, '-', color=col[i], alpha = alpha[i],label=' '+ ft, zorder = 4)
                        #
                        ax.legend(loc = 3)
                        #
                        te = datetimeify(erup_time)#fm_e1.data.tes[int(erup[-1:])-1]
                        #ax1.set_xticks([ft_e1[0].index[-1] - 7*day*i for i in range(int(30/7)+1)])#[dat.index.values[0],dat.index.values[-1]])#, ]np.arange(0, len(x)+1, 5))
                        
                        #ax1b.set_yticks([])
                        #ax.grid()
                        ax.set_ylabel('Feature value')        #ax.set_xticks([ft_e1[0].index[-1]-7*day*i +day for i in range(5)])
                        #ax.set_xticks([ft_e1[0].index[-1] - 7*day*i for i in range(int(30/7)+1)])#[dat.index.values[0],dat.index.values[-1]])#, ]np.arange(0, len(x)+1, 5))
                        #ax1.set_xticks([t1 - 5*day*i for i in range(int(look_back/5)+1)])#[dat.index.values[0],dat.index.values[-1]])#, ]np.arange(0, len(x)+1, 5))
                        #
                    #except:
                    #    pass

                else:
                    pass
                    if data_yrigth:
                        #
                        #ax1b = ax1.twinx() 
                        #
                        td = TremorData(station = sta)
                        #td.update(ti=t0, tf=t1)
                        data_streams = data_yrigth#['hf','mf', 'rsam']#, 'dsarF']
                        label = ['RSAM','MF','HF','DSAR']
                        #label = ['1/RSAM']
                        inv = False
                        if False:
                            data_streams = ['rsam']
                            label = ['RSAM']

                        if type(data_streams) is str:
                            data_streams = [data_streams,]
                        if any(['_' in ds for ds in data_streams]):
                            td._compute_transforms()
                        #ax.set_xlim(*range)
                        # plot data for each year
                        norm= False
                        _range = [t0,t1]
                        log =False
                        col_def = None
                        data = td.get_data(*_range)
                        xi = datetime(year=1,month=1,day=1,hour=0,minute=0,second=0)
                        cols = ['k','r','g','m',[0.5,0.5,0.5],[0.75,0.75,0.75]]
                        if inv:
                            cols = ['k','g','r','m',[0.5,0.5,0.5],[0.75,0.75,0.75]]
                        inds = (data.index>=datetimeify(_range[0]))&(data.index<=datetimeify(_range[1]))
                        i=0
                        for data_stream, col in zip(data_streams,cols):
                            v_plot = data[data_stream].loc[inds]
                            if inv:
                                v_plot = 1/v_plot
                            if log:
                                v_plot = np.log10(v_plot)
                            if norm:
                                v_plot = (v_plot-np.min(v_plot))/np.max((v_plot-np.min(v_plot)))
                            if label:
                                if col_def:
                                    ax.plot(data.index[inds], v_plot, '-', color=col_def, label=label[i], linewidth=1., alpha = 1., zorder = 0)
                                else:
                                    #ax1b.plot(data.index[inds], v_plot, '-', color=col, label=label[i], linewidth=1., alpha = 1.0, zorder = 0)
                                    _dd = data.index[inds]
                                    _dd = np.arange(v_plot.shape[0])/6
                                    ax.plot(_dd, v_plot, '-', color=col, linewidth=1., alpha = .7, zorder = 3)
                                    axb.plot([], [], '-', color=col, label=label[i], linewidth=1., alpha = .7, zorder = 3)
                            else:
                                ax.plot(data.index[inds], v_plot, '-', color=col, label=data_stream, linewidth=1., alpha = .7, zorder = 3)
                            i+=1
                        for te in td.tes:
                            if [te>=datetimeify(_range[0]) and te<=datetimeify(_range[1])]:
                                pass
                                #ax.axvline(te, color='k', linestyle='--', linewidth=2, zorder = 0)
                        #
                        ax.plot([], color='k', linestyle='--', linewidth=1, label = 'eruption')
                        #ax4.set_xlim(_range)
                        #ax1b.legend(loc = 2)
                        #ax1b.grid()
                        #
                        if True:
                            ## polyfit 1
                            #x = data.index[inds] 
                            #x = mdates.date2num(data.index[inds])
                            y = np.log10(v_plot.values)
                            x = np.arange(y.shape[0])/6
                            z = np.polyfit(x, y, 1)
                            p = np.poly1d(z)
                            #
                            #xx = np.linspace(x.min(), x.max(), 100)
                            #dd = mdates.num2date(xx)
                            #
                            l_fit.append(z)
                            #axb.plot([],[], '-g', label = 'linear fit (\{tau}_0='+str(round(l_fit[0][0],2))+')')
                            label = r'\tau_0 = '+ str(round(l_fit[0][0],2))
                            #ax.text(0., 0., label)#, fontsize='medium', verticalalignment='top', fontfamily='serif',
                                #bbox=dict(facecolor='0.7', edgecolor='none', pad=3.0))
                            #dt = (dd[-1]-dd[0]).seconds/3600 # hours
                            #dy = p(xx[0]) - p(xx[-1])
                            #tau = dt/dy
                            tau = -(x[-1]-x[0])/(y[-1]-y[0])
                            #
                            #_yy = 10**p(x)#(xx)
                            #_dd = np.arange(_yy.shape[0])
                            #
                            ax.plot(x, 10**p(x), '-g', linewidth=3, label = 'linear fit (\{tau}_0='+str(round(tau,2))+')')
                            axb.plot([],[], '-g', linewidth=3, label = r'linear fit ($\tau_d$='+str(round(tau,1))+' hrs)')

                            ## polyfit 2
                            if j in [0,2]:
                                t0 = erup_times[j] - look_back*day#30*day
                                t1 = erup_times[j] + 2/8*day#hour
                                _range = [t0,t1]
                                #
                                inds = (data.index>=datetimeify(_range[0]))&(data.index<=datetimeify(_range[1]))
                                v_plot = data[data_stream].loc[inds]
                                #
                                #x = mdates.date2num(data.index[inds])
                                y = np.log10(v_plot.values)
                                x = np.arange(y.shape[0])/6
                                z = np.polyfit(x, y, 1)
                                p = np.poly1d(z)
                                #
                                #xx = np.linspace(x.min(), x.max(), 100)
                                #dd = mdates.num2date(xx)
                                #
                                l_fit.append(z)
                                #axb.plot([],[], '-g', label = 'linear fit (\{tau}_0='+str(round(l_fit[0][0],2))+')')
                                label = r'\tau_0 = '+ str(round(l_fit[0][0],2))
                                #ax.text(0., 0., label)#, fontsize='medium', verticalalignment='top', fontfamily='serif',
                                    #bbox=dict(facecolor='0.7', edgecolor='none', pad=3.0))
                                #dt = (dd[-1]-dd[0]).seconds/3600 # hours
                                #dy = p(xx[0]) - p(xx[-1])
                                #tau = dt/dy
                                tau = -(x[-1]-x[0])/(y[-1]-y[0])
                                #
                                #ax.plot(dd, 10**p(xx), '--b', linewidth=3, label = 'linear fit (\{tau}_0='+str(round(tau,2))+')')
                                ax.plot(x, 10**p(x),'--b', linewidth=3, label = 'linear fit (\{tau}_0='+str(round(tau,2))+')')
                                axb.plot([],[], '--b', linewidth=3, label = r'linear fit ($\tau_d$='+str(round(tau,1))+' hrs)')
                                ##
                                if log:
                                    ax.set_ylabel(' ')
                                else:
                                    ax.set_ylabel('RSAM \u03BC m/s')
                                #ax4.set_xlabel('Time [month-day hour]')
                                #ax4.title.set_text('Station '+td.station+' ('+sta_code[td.station]+'): Tremor data')
                                #
                                #ax.set_xticks([te - 5*day*i for i in range(int(look_back/5)+1)])#[dat.index.values[0],dat.index.values[-1]])#, ]np.arange(0, len(x)+1, 5))
                                #ax.set_xticks([i for i in range(24)])#[dat.index.values[0],dat.index.values[-1]])#, ]np.arange(0, len(x)+1, 5))
                                #ax4.set_ylim([1e9,1e13])
                                #ax4.set_xticks([te - 5*day*i for i in range(int(look_back/5)+1)])#[dat.index.values[0],dat.index.values[-1]])#, ]np.arange(0, len(x)+1, 5))
                                #ax.set_yscale('log')
                #if j == 0:
                axb.legend(loc = 1, prop={'size': 10})     
                axb.grid(False)
                ax.grid(color='gray', linestyle='-', linewidth=.5, alpha = 0.5)
                #
                axb.set_yticklabels([])
                ax.set_xlabel('Time [hours]')        #ax.set_xticks([ft_e1[0].index[-1]-7*day*i +day for i in range(5)])

                #
        # _d = 5 
        # t1 = erup_times[0] + look_front*day#hour
        # ax1.set_xticks([t1 - _d*day*i for i in range(int((look_back+look_front)/_d)+1)])
        # t3 = erup_times[1] + look_front*day#hour
        # ax3.set_xticks([t3 - _d*day*i for i in range(int((look_back+look_front)/_d)+1)])
        # t5 = erup_times[2] + look_front*day#hour
        # ax5.set_xticks([t5 - _d*day*i for i in range(int((look_back+look_front)/_d)+1)])

    #####################################################
    # Second column (first and second row)
    if True:
        erup_times = [datetimeify('2007 09 25 08 20 00'), 
                        datetimeify('2016 11 13 12 00 00'),
                        datetimeify('2021 03 04 12 00 00')]
        erup_times = [datetimeify('2007 09 25 09 45 00'), 
                        datetimeify('2016 11 13 12 00 00'),
                        datetimeify('2021 03 04 14 00 00')]
        #
        for j,ax in enumerate([ax2, ax4, ax6]):
            #
            t0 = erup_times[j] - look_back*day#30*day
            t1 = erup_times[j] + look_front*day#hour
            # subplot one: MF, HF, DSAR medians (DSAR yaxis left; MF, HF yaxis rigth). 1/RSAM (normalized)
            if True:
                # features
                fts_yleft = []#['zsc2_dsarF__median']
                fts_yrigth = []#['zsc2_dsarF__rate_variance']#['zsc2_dsarF__change_quantiles__f_agg_"var"__isabs_False__qh_0.6__ql_0.4']#['zsc2_mfF__median','zsc2_hfF__median']
                data_yrigth = ['rsam']
                
                #
                col = ['b','b','r']
                alpha = [1., 1., .5]
                thick_line = [2., 6., 1.]
                axb = ax.twinx() 
                for i, ft in enumerate(fts_yleft):
                    if True: # load feature (else: cal feature. median or rv)
                        if 'zsc2_dsarF' in ft:
                            ds = ['zsc2_dsarF'] 
                        if server:
                            path_feat_serv = 'C:\\Users\\aar135\\codes_local_disk\\volc_forecast_tl\\features_bkp\\features_server\\'
                            fm_e1 = ForecastModel(window=2., overlap=1., station =  sta,
                                look_forward=2., data_streams=ds, 
                                feature_dir=path_feat_serv, 
                                savefile_type='pkl') 
                        elif server2:
                            path_feat_serv = 'U:\\Research\\EruptionForecasting\\eruptions\\features\\'
                            fm_e1 = ForecastModel(window=2., overlap=1., station =  sta,
                                look_forward=2., data_streams=ds, 
                                feature_dir=path_feat_serv, 
                                savefile_type='pkl') 
                        else:
                            try:
                                fm_e1 = ForecastModel(window=2., overlap=1., station = sta,
                                    look_forward=2., data_streams=ds, savefile_type='csv')
                            except:
                                fm_e1 = ForecastModel(window=2., overlap=1., station = sta,
                                    look_forward=2., data_streams=ds, savefile_type='pkl')                    
                        ##  
                        ft = ft.replace("-",'"')
                        # adding multiple Axes objects
                        ft_e1 = fm_e1.get_features(ti=t0, tf=t1, n_jobs=1, compute_only_features=[ft])
                        # extract values to plot 
                        ft_e1_t = ft_e1[0].index.values
                        ft_e1_v = ft_e1[0].loc[:,ft]
                        #
                        v_plot = ft_e1_v

                        #v_plot = ft_e1_v-np.min(ft_e1_v)/np.max((ft_e1_v-np.min(ft_e1_v)))
                        #v_plot = ft_e1_v/np.max(ft_e1_v)
                    else: 
                        #
                        if 'zsc2_dsarF' in ft:
                            ds = 'zsc2_dsarF'
                        if 'zsc2_mfF' in ft:
                            ds = 'zsc2_mfF' 
                        if 'zsc2_hfF' in ft:
                            ds = 'zsc2_hfF' 
                        # 
                        #
                        day = timedelta(days=1)
                        fm = ForecastModel(window=2., overlap=1., station=sta,
                            look_forward=2., data_streams=[ds], 
                            data_dir=r'C:\Users\aar135\codes_local_disk\volc_forecast_tl\volc_forecast_tl\data'
                            )
                        #
                        N, M = [2,30]
                        df = fm.data.df[t0:t1]
                        if 'median' in ft:
                            test = df[ds].rolling(N*24*6).median()[N*24*6:]
                        if 'rate_variance' in ft:
                            test = df[ds].rolling(N*24*6).apply(chqv)[N*24*6:]
                        #
                        #out = test.rolling(archtype.shape[0]).apply(partial(conv, (archtype-archtype.mean())/archtype.std()))       
                        #out = out.resample('1D').ffill()
                        #
                        ft_e1_t = test.index
                        v_plot = test.values
                    #
                    if ft == 'zsc2_dsarF__median':
                        ft = 'nDSAR median'
                    #
                    axb.plot(ft_e1_t, v_plot, '-', color=col[i], alpha = alpha[i], linewidth = thick_line[i], label=' '+ ft, zorder = 2)

                    #
                    if ffm: # ffm 
                        #ax1b = ax1.twinx() 
                        #v_plot = data[data_stream].loc[inds]
                        inv_rsam = fm_e1.data.get_data(ti=t0, tf=t1)['rsamF']#.loc[ft_e1_t]
                        inv_rsam = 1./inv_rsam
                        # normalized it to yaxis rigth 
                        inv_rsam = inv_rsam/max(inv_rsam)
                        inv_rsam = inv_rsam*0.5*max(v_plot)
                        #
                        ax.plot(ft_e1_t, inv_rsam, '-', color= 'gray', linewidth=0.5, markersize=0.5, alpha = 1.)
                        ax.plot([], [], '-', color= 'gray', markersize=1, label='1/RSAM', alpha = 1.0)
                        #ax1.set_ylim([0,1])
                        #ax1.set_yticks([])
                    #
                    if plot_erup: # plot vertical lines
                        te = datetimeify(erup_times[j])#fm_e1.data.tes[int(erup[-1:])-1]
                        if j == 0:
                            ax.axvline(te, color='red', alpha = .25, linestyle='-', linewidth=12, zorder = 0)
                            axb.plot([], color='red', alpha = .25, linestyle='-', linewidth=12, label = 'eruption')
                        else:
                            ax.axvline(te, color='gray', alpha = .25, linestyle='-', linewidth=12, zorder = 0)
                            axb.plot([], color='gray', alpha = .25, linestyle='-', linewidth=12, label = 'fluid release event')
                    #
                    #ax1.legend(loc = 2)
                    #
                    te = datetimeify(erup_times[j])#fm_e1.data.tes[int(erup[-1:])-1]
                    #ax1.set_xticks([ft_e1[0].index[-1] - 7*day*i for i in range(int(30/7)+1)])#[dat.index.values[0],dat.index.values[-1]])#, ]np.arange(0, len(x)+1, 5))
                    
                    #ax1b.set_yticks([])
                    axb.grid()
                    axb.set_ylabel('nDSAR value')        #ax.set_xticks([ft_e1[0].index[-1]-7*day*i +day for i in range(5)])
                    #ax1.set_yscale('log') #ax.set_yscale('log')
                    #ax.set_xticks([ft_e1[0].index[-1] - 7*day*i for i in range(int(30/7)+1)])#[dat.index.values[0],dat.index.values[-1]])#, ]np.arange(0, len(x)+1, 5))
                    #ax1.set_xticks([t1 - 5*day*i for i in range(int(look_back/5)+1)])#[dat.index.values[0],dat.index.values[-1]])#, ]np.arange(0, len(x)+1, 5))
                #     #
                # except:
                #     pass
                if fts_yrigth:
                    #ax1b = ax1.twinx() 
                    col = ['r','g']
                    alpha = [1., .5]
                    thick_line = [2.,1.]
                    #try: 
                    for i, ft in enumerate(fts_yrigth):
                        if 'zsc2_dsarF' in ft:
                            ds = 'zsc2_dsarF'
                        if 'zsc2_mfF' in ft:
                            ds = 'zsc2_mfF' 
                        if 'zsc2_hfF' in ft:
                            ds = 'zsc2_hfF' 
                        # 
                        if False: # look feature in the prev cacl features (else: calculate feat from data; only for median and rv)
                            if server:
                                path_feat_serv = 'C:\\Users\\aar135\\codes_local_disk\\volc_forecast_tl\\features_bkp\\features_server\\'
                                fm_e1 = ForecastModel(window=2., overlap=1., station =  sta,
                                    look_forward=2., data_streams=ds, 
                                    feature_dir=path_feat_serv, 
                                    savefile_type='pkl') 
                            elif server2:
                                path_feat_serv = 'U:\\Research\\EruptionForecasting\\eruptions\\features\\'
                                fm_e1 = ForecastModel(window=2., overlap=1., station =  sta,
                                    look_forward=2., data_streams=ds, 
                                    feature_dir=path_feat_serv, 
                                    savefile_type='pkl') 
                            else:
                                try:
                                    fm_e1 = ForecastModel(window=2., overlap=1., station = sta,
                                        look_forward=2., data_streams=ds, savefile_type='csv')
                                except:
                                    fm_e1 = ForecastModel(window=2., overlap=1., station = sta,
                                        look_forward=2., data_streams=ds, savefile_type='pkl')                    
                            ##  
                            ft = ft.replace("-",'"')
                            
                            ft_e1 = fm_e1.get_features(ti=t0, tf=t1, n_jobs=1, compute_only_features=[ft])
                            # adding multiple Axes objects

                            # extract values to plot 
                            ft_e1_t = ft_e1[0].index.values
                            ft_e1_v = ft_e1[0].loc[:,ft]
                            #
                            v_plot = ft_e1_v

                        else:
                            day = timedelta(days=1)
                            fm = ForecastModel(window=2., overlap=1., station=sta,
                                look_forward=2., data_streams=[ds], 
                                data_dir=r'C:\Users\aar135\codes_local_disk\volc_forecast_tl\volc_forecast_tl\data'
                                )
                            #
                            N, M = [2,30]
                            df = fm.data.df[t0:t1]
                            if 'median' in ft:
                                test = df[ds].rolling(N*24*6).median()[N*24*6:]
                            if 'rate_variance' in ft:
                                test = df[ds].rolling(N*24*6).apply(chqv)[N*24*6:]
                            #
                            #out = test.rolling(archtype.shape[0]).apply(partial(conv, (archtype-archtype.mean())/archtype.std()))       
                            #out = out.resample('1D').ffill()
                            #
                            ft_e1_t = test.index
                            v_plot = test.values

                        #v_plot = ft_e1_v-np.min(ft_e1_v)/np.max((ft_e1_v-np.min(ft_e1_v)))
                        #v_plot = ft_e1_v/np.max(ft_e1_v)
                        #
                        if ft == 'zsc2_mfF__median':
                            ft = 'nMF median'
                        if ft == 'zsc2_hfF__median':
                            ft = 'nHF median'
                        #
                        ax.plot(ft_e1_t, v_plot, '-', color=col[i], alpha = alpha[i],label=' '+ ft, zorder = 4)
                        #
                        ax.legend(loc = 3)
                        #
                        te = datetimeify(erup_time)#fm_e1.data.tes[int(erup[-1:])-1]
                        #ax1.set_xticks([ft_e1[0].index[-1] - 7*day*i for i in range(int(30/7)+1)])#[dat.index.values[0],dat.index.values[-1]])#, ]np.arange(0, len(x)+1, 5))
                        
                        #ax1b.set_yticks([])
                        #ax.grid()
                        ax.set_ylabel('Feature value')        #ax.set_xticks([ft_e1[0].index[-1]-7*day*i +day for i in range(5)])
                        #ax.set_xticks([ft_e1[0].index[-1] - 7*day*i for i in range(int(30/7)+1)])#[dat.index.values[0],dat.index.values[-1]])#, ]np.arange(0, len(x)+1, 5))
                        #ax1.set_xticks([t1 - 5*day*i for i in range(int(look_back/5)+1)])#[dat.index.values[0],dat.index.values[-1]])#, ]np.arange(0, len(x)+1, 5))
                        #
                    #except:
                    #    pass

                else:
                    pass
                    if data_yrigth:
                        #
                        #ax1b = ax1.twinx() 
                        #
                        td = TremorData(station = sta)
                        #td.update(ti=t0, tf=t1)
                        data_streams = data_yrigth#['hf','mf', 'rsam']#, 'dsarF']
                        label = ['RSAM','MF','HF','DSAR']
                        #label = ['1/RSAM']
                        inv = False
                        if False:
                            data_streams = ['rsam']
                            label = ['RSAM']

                        if type(data_streams) is str:
                            data_streams = [data_streams,]
                        if any(['_' in ds for ds in data_streams]):
                            td._compute_transforms()
                        #ax.set_xlim(*range)
                        # plot data for each year
                        norm= False
                        _range = [t0,t1]
                        log =False
                        col_def = None
                        data = td.get_data(*_range)
                        xi = datetime(year=1,month=1,day=1,hour=0,minute=0,second=0)
                        cols = ['k','r','g','m',[0.5,0.5,0.5],[0.75,0.75,0.75]]
                        if inv:
                            cols = ['k','g','r','m',[0.5,0.5,0.5],[0.75,0.75,0.75]]
                        inds = (data.index>=datetimeify(_range[0]))&(data.index<=datetimeify(_range[1]))
                        i=0
                        for data_stream, col in zip(data_streams,cols):
                            v_plot = data[data_stream].loc[inds]
                            if inv:
                                v_plot = 1/v_plot
                            if log:
                                v_plot = np.log10(v_plot)
                            if norm:
                                v_plot = (v_plot-np.min(v_plot))/np.max((v_plot-np.min(v_plot)))
                            if label:
                                if col_def:
                                    ax.plot(data.index[inds], v_plot, '-', color=col_def, label=label[i], linewidth=1., alpha = 1., zorder = 0)
                                else:
                                    #ax1b.plot(data.index[inds], v_plot, '-', color=col, label=label[i], linewidth=1., alpha = 1.0, zorder = 0)
                                    _dd = data.index[inds]
                                    _dd = np.arange(v_plot.shape[0])/6
                                    ax.plot(_dd, v_plot, '-', color=col, linewidth=1., alpha = .7, zorder = 3)
                                    axb.plot([], [], '-', color=col, label=label[i], linewidth=1., alpha = .7, zorder = 3)
                            else:
                                ax.plot(data.index[inds], v_plot, '-', color=col, label=data_stream, linewidth=1., alpha = .7, zorder = 3)
                            i+=1
                        for te in td.tes:
                            if [te>=datetimeify(_range[0]) and te<=datetimeify(_range[1])]:
                                pass
                                #ax.axvline(te, color='k', linestyle='--', linewidth=2, zorder = 0)
                        #
                        ax.plot([], color='k', linestyle='--', linewidth=1, label = 'eruption')
                        #ax4.set_xlim(_range)
                        #ax1b.legend(loc = 2)
                        #ax1b.grid()
                        #
                        if True:
                            ## polyfit 1
                            #x = data.index[inds] 
                            #x = mdates.date2num(data.index[inds])
                            y = np.log10(v_plot.values)
                            x = np.arange(y.shape[0])/6
                            z = np.polyfit(x, y, 1)
                            p = np.poly1d(z)
                            #
                            #xx = np.linspace(x.min(), x.max(), 100)
                            #dd = mdates.num2date(xx)
                            #
                            l_fit.append(z)
                            #axb.plot([],[], '-g', label = 'linear fit (\{tau}_0='+str(round(l_fit[0][0],2))+')')
                            label = r'\tau_0 = '+ str(round(l_fit[0][0],2))
                            #ax.text(0., 0., label)#, fontsize='medium', verticalalignment='top', fontfamily='serif',
                                #bbox=dict(facecolor='0.7', edgecolor='none', pad=3.0))
                            #dt = (dd[-1]-dd[0]).seconds/3600 # hours
                            #dy = p(xx[0]) - p(xx[-1])
                            #tau = dt/dy
                            tau = -(x[-1]-x[0])/(y[-1]-y[0])
                            #
                            #_yy = 10**p(x)#(xx)
                            #_dd = np.arange(_yy.shape[0])
                            #
                            ax.plot(x, 10**p(x), '-g', linewidth=3, label = 'linear fit (\{tau}_0='+str(round(tau,2))+')')
                            axb.plot([],[], '-g', linewidth=3, label = r'linear fit ($\tau_d$='+str(round(tau,1))+' hrs)')

                            ## polyfit 2
                            if j in [0,1]:
                                t0 = erup_times[j] - look_back*day#30*day
                                t1 = erup_times[j] + 2/8*day#hour
                                _range = [t0,t1]
                                #
                                inds = (data.index>=datetimeify(_range[0]))&(data.index<=datetimeify(_range[1]))
                                v_plot = data[data_stream].loc[inds]
                                #
                                #x = mdates.date2num(data.index[inds])
                                y = np.log10(v_plot.values)
                                x = np.arange(y.shape[0])/6
                                z = np.polyfit(x, y, 1)
                                p = np.poly1d(z)
                                #
                                #xx = np.linspace(x.min(), x.max(), 100)
                                #dd = mdates.num2date(xx)
                                #
                                l_fit.append(z)
                                #axb.plot([],[], '-g', label = 'linear fit (\{tau}_0='+str(round(l_fit[0][0],2))+')')
                                label = r'\tau_0 = '+ str(round(l_fit[0][0],2))
                                #ax.text(0., 0., label)#, fontsize='medium', verticalalignment='top', fontfamily='serif',
                                    #bbox=dict(facecolor='0.7', edgecolor='none', pad=3.0))
                                #dt = (dd[-1]-dd[0]).seconds/3600 # hours
                                #dy = p(xx[0]) - p(xx[-1])
                                #tau = dt/dy
                                tau = -(x[-1]-x[0])/(y[-1]-y[0])
                                #
                                #ax.plot(dd, 10**p(xx), '--b', linewidth=3, label = 'linear fit (\{tau}_0='+str(round(tau,2))+')')
                                ax.plot(x, 10**p(x),'--b', linewidth=3, label = 'linear fit (\{tau}_0='+str(round(tau,2))+')')
                                axb.plot([],[], '--b', linewidth=3, label = r'linear fit ($\tau_d$='+str(round(tau,1))+' hrs)')
                                ##
                                if log:
                                    ax.set_ylabel(' ')
                                else:
                                    ax.set_ylabel('RSAM \u03BC m/s')
                                #ax4.set_xlabel('Time [month-day hour]')
                                #ax4.title.set_text('Station '+td.station+' ('+sta_code[td.station]+'): Tremor data')
                                #
                                #ax.set_xticks([te - 5*day*i for i in range(int(look_back/5)+1)])#[dat.index.values[0],dat.index.values[-1]])#, ]np.arange(0, len(x)+1, 5))
                                #ax.set_xticks([i for i in range(24)])#[dat.index.values[0],dat.index.values[-1]])#, ]np.arange(0, len(x)+1, 5))
                                #ax4.set_ylim([1e9,1e13])
                                #ax4.set_xticks([te - 5*day*i for i in range(int(look_back/5)+1)])#[dat.index.values[0],dat.index.values[-1]])#, ]np.arange(0, len(x)+1, 5))
                                #ax.set_yscale('log')
                #if j == 0:
                axb.legend(loc = 1, prop={'size': 10})     
                axb.grid(False)
                ax.grid(color='gray', linestyle='-', linewidth=.5, alpha = 0.5)
                #
                axb.set_yticklabels([])
                ax.set_xlabel('Time [hours]')        #ax.set_xticks([ft_e1[0].index[-1]-7*day*i +day for i in range(5)])

        # _d = 5 
        # t2 = erup_times[0] + look_front*day#hour
        # ax2.set_xticks([t2 - _d*day*i for i in range(int((look_back+look_front)/_d)+1)])
        # t4 = erup_times[1] + look_front*day#hour
        # ax4.set_xticks([t4 - _d*day*i for i in range(int((look_back+look_front)/_d)+1)])
        # t6 = erup_times[2] + look_front*day#hour
        # ax6.set_xticks([t6 - _d*day*i for i in range(int((look_back+look_front)/_d)+1)])

        if False: # plot vertical lines
            te = datetimeify(erup_time)#fm_e1.data.tes[int(erup[-1:])-1]
            ax1.axvline(te, color='k',linestyle='--', linewidth=2, zorder = 4)
            ax2.axvline(te, color='k',linestyle='--', linewidth=2, zorder = 4)
            ax3.axvline(te, color='k',linestyle='--', linewidth=2, zorder = 4)
            ax1.plot([], color='k', linestyle='--', linewidth=2, label = 'event')
        #
        if False:#save_png_path:
            dat = erup_time.strftime('%Y-%m-%d')
            title =  sta+'_'+dat+'_'+'look_back'+str(look_back)
            ax1.set_title(title)
            plt.savefig(save_png_path+sta+'_'+dat+'_'+'look_back'+str(look_back)+'.png')
        #
        # for j,ax in enumerate([ax1, ax3, ax5, ax2, ax4, ax6]):
        #     pass
        #     #ax.set_ylim([1e0,1e4])
        
    #####################################################
    ax1.set_yscale('log')
    ax2.set_yscale('log')
    ax3.set_yscale('log')
    ax4.set_yscale('log')
    ax5.set_yscale('log')
    ax6.set_yscale('log')
    #
    ax1.set_title('(a) 24 hours after 2006/10/04 Ruapehu eruption')
    ax2.set_title('(b) 24 hours after 2007/09/25 Ruapehu eruption')
    #
    ax3.set_title('(c) 24 hours after 2009/07/13 Ruapehu fluid release event')
    ax5.set_title('(e) 24 hours after 2010/09/03 Ruapehu fluid release event')
    ax4.set_title('(d) 24 hours after 2016/11/13 Ruapehu fluid release event')
    ax6.set_title('(f) 24 hours after 2021/03/04 Ruapehu possible sealing and fluid release event')
    #
    fig.tight_layout()
    plt.show()
    #plt.close('all')
    #
    if True:
        #nrow = 2
        #ncol = 2
        #fig, ((ax7, ax8), (ax9, ax10)) = plt.subplots(nrows=nrow, ncols=ncol, figsize=(17,8))#(14,4)) #, ax4)
        nrow = 3
        ncol = 2
        fig, ((ax7, ax8), (ax9, ax10), (_ax, _ax_)) = plt.subplots(nrows=nrow, ncols=ncol, figsize=(14,8))#(14,4)) #, ax4)

        # Third row 
        sta = 'POS' 
        if sta == 'POS':
            ffm = False
            server = False # files imported from server to local pc 
            server2 = True # server at uni 

        if True: # Kawah Ijen eruption 
            erup_times = [datetimeify('2013 01 24 00 00 00'), datetimeify('2013 03 31 22 45 00')]
            erup_times = [datetimeify('2013 01 24 00 00 00'), datetimeify('2013 04 01 02 45 00')]

            # subplot one: MF, HF, DSAR medians (DSAR yaxis left; MF, HF yaxis rigth). 1/RSAM (normalized)
            if True:
                for j,ax in enumerate([ax7, ax9]):
                    #
                    t0 = erup_times[j] - look_back*day#30*day
                    t1 = erup_times[j] + look_front*day#hour
                    # subplot one: MF, HF, DSAR medians (DSAR yaxis left; MF, HF yaxis rigth). 1/RSAM (normalized)
                    if True:
                        # features
                        fts_yleft = []#['zsc2_dsarF__median']
                        fts_yrigth = []#['zsc2_dsarF__rate_variance']#['zsc2_dsarF__change_quantiles__f_agg_"var"__isabs_False__qh_0.6__ql_0.4']#['zsc2_mfF__median','zsc2_hfF__median']
                        data_yrigth = ['rsam']
                        
                        #
                        col = ['b','b','r']
                        alpha = [1., 1., .5]
                        thick_line = [2., 6., 1.]
                        axb = ax.twinx() 
                        for i, ft in enumerate(fts_yleft):
                            if True: # load feature (else: cal feature. median or rv)
                                if 'zsc2_dsarF' in ft:
                                    ds = ['zsc2_dsarF'] 
                                if server:
                                    path_feat_serv = 'C:\\Users\\aar135\\codes_local_disk\\volc_forecast_tl\\features_bkp\\features_server\\'
                                    fm_e1 = ForecastModel(window=2., overlap=1., station =  sta,
                                        look_forward=2., data_streams=ds, 
                                        feature_dir=path_feat_serv, 
                                        savefile_type='pkl') 
                                elif server2:
                                    path_feat_serv = 'U:\\Research\\EruptionForecasting\\eruptions\\features\\'
                                    fm_e1 = ForecastModel(window=2., overlap=1., station =  sta,
                                        look_forward=2., data_streams=ds, 
                                        feature_dir=path_feat_serv, 
                                        savefile_type='pkl') 
                                else:
                                    try:
                                        fm_e1 = ForecastModel(window=2., overlap=1., station = sta,
                                            look_forward=2., data_streams=ds, savefile_type='csv')
                                    except:
                                        fm_e1 = ForecastModel(window=2., overlap=1., station = sta,
                                            look_forward=2., data_streams=ds, savefile_type='pkl')                    
                                ##  
                                ft = ft.replace("-",'"')
                                # adding multiple Axes objects
                                ft_e1 = fm_e1.get_features(ti=t0, tf=t1, n_jobs=1, compute_only_features=[ft])
                                # extract values to plot 
                                ft_e1_t = ft_e1[0].index.values
                                ft_e1_v = ft_e1[0].loc[:,ft]
                                #
                                v_plot = ft_e1_v

                                #v_plot = ft_e1_v-np.min(ft_e1_v)/np.max((ft_e1_v-np.min(ft_e1_v)))
                                #v_plot = ft_e1_v/np.max(ft_e1_v)
                            else: 
                                #
                                if 'zsc2_dsarF' in ft:
                                    ds = 'zsc2_dsarF'
                                if 'zsc2_mfF' in ft:
                                    ds = 'zsc2_mfF' 
                                if 'zsc2_hfF' in ft:
                                    ds = 'zsc2_hfF' 
                                # 
                                #
                                day = timedelta(days=1)
                                fm = ForecastModel(window=2., overlap=1., station=sta,
                                    look_forward=2., data_streams=[ds], 
                                    data_dir=r'C:\Users\aar135\codes_local_disk\volc_forecast_tl\volc_forecast_tl\data'
                                    )
                                #
                                N, M = [2,30]
                                df = fm.data.df[t0:t1]
                                if 'median' in ft:
                                    test = df[ds].rolling(N*24*6).median()[N*24*6:]
                                if 'rate_variance' in ft:
                                    test = df[ds].rolling(N*24*6).apply(chqv)[N*24*6:]
                                #
                                #out = test.rolling(archtype.shape[0]).apply(partial(conv, (archtype-archtype.mean())/archtype.std()))       
                                #out = out.resample('1D').ffill()
                                #
                                ft_e1_t = test.index
                                v_plot = test.values
                            #
                            if ft == 'zsc2_dsarF__median':
                                ft = 'nDSAR median'
                            #
                            axb.plot(ft_e1_t, v_plot, '-', color=col[i], alpha = alpha[i], linewidth = thick_line[i], label=' '+ ft, zorder = 2)

                            #
                            if ffm: # ffm 
                                #ax1b = ax1.twinx() 
                                #v_plot = data[data_stream].loc[inds]
                                inv_rsam = fm_e1.data.get_data(ti=t0, tf=t1)['rsamF']#.loc[ft_e1_t]
                                inv_rsam = 1./inv_rsam
                                # normalized it to yaxis rigth 
                                inv_rsam = inv_rsam/max(inv_rsam)
                                inv_rsam = inv_rsam*0.5*max(v_plot)
                                #
                                ax.plot(ft_e1_t, inv_rsam, '-', color= 'gray', linewidth=0.5, markersize=0.5, alpha = 1.)
                                ax.plot([], [], '-', color= 'gray', markersize=1, label='1/RSAM', alpha = 1.0)
                                #ax1.set_ylim([0,1])
                                #ax1.set_yticks([])
                            #
                            if plot_erup: # plot vertical lines
                                te = datetimeify(erup_times[j])#fm_e1.data.tes[int(erup[-1:])-1]
                                if j == 0:
                                    ax.axvline(te, color='red', alpha = .25, linestyle='-', linewidth=12, zorder = 0)
                                    axb.plot([], color='red', alpha = .25, linestyle='-', linewidth=12, label = 'eruption')
                                else:
                                    ax.axvline(te, color='gray', alpha = .25, linestyle='-', linewidth=12, zorder = 0)
                                    axb.plot([], color='gray', alpha = .25, linestyle='-', linewidth=12, label = 'fluid release event')
                            #
                            #ax1.legend(loc = 2)
                            #
                            te = datetimeify(erup_times[j])#fm_e1.data.tes[int(erup[-1:])-1]
                            #ax1.set_xticks([ft_e1[0].index[-1] - 7*day*i for i in range(int(30/7)+1)])#[dat.index.values[0],dat.index.values[-1]])#, ]np.arange(0, len(x)+1, 5))
                            
                            #ax1b.set_yticks([])
                            axb.grid()
                            axb.set_ylabel('nDSAR value')        #ax.set_xticks([ft_e1[0].index[-1]-7*day*i +day for i in range(5)])
                            #ax1.set_yscale('log') #ax.set_yscale('log')
                            #ax.set_xticks([ft_e1[0].index[-1] - 7*day*i for i in range(int(30/7)+1)])#[dat.index.values[0],dat.index.values[-1]])#, ]np.arange(0, len(x)+1, 5))
                            #ax1.set_xticks([t1 - 5*day*i for i in range(int(look_back/5)+1)])#[dat.index.values[0],dat.index.values[-1]])#, ]np.arange(0, len(x)+1, 5))
                        #     #
                        # except:
                        #     pass
                        if fts_yrigth:
                            #ax1b = ax1.twinx() 
                            col = ['r','g']
                            alpha = [1., .5]
                            thick_line = [2.,1.]
                            #try: 
                            for i, ft in enumerate(fts_yrigth):
                                if 'zsc2_dsarF' in ft:
                                    ds = 'zsc2_dsarF'
                                if 'zsc2_mfF' in ft:
                                    ds = 'zsc2_mfF' 
                                if 'zsc2_hfF' in ft:
                                    ds = 'zsc2_hfF' 
                                # 
                                if False: # look feature in the prev cacl features (else: calculate feat from data; only for median and rv)
                                    if server:
                                        path_feat_serv = 'C:\\Users\\aar135\\codes_local_disk\\volc_forecast_tl\\features_bkp\\features_server\\'
                                        fm_e1 = ForecastModel(window=2., overlap=1., station =  sta,
                                            look_forward=2., data_streams=ds, 
                                            feature_dir=path_feat_serv, 
                                            savefile_type='pkl') 
                                    elif server2:
                                        path_feat_serv = 'U:\\Research\\EruptionForecasting\\eruptions\\features\\'
                                        fm_e1 = ForecastModel(window=2., overlap=1., station =  sta,
                                            look_forward=2., data_streams=ds, 
                                            feature_dir=path_feat_serv, 
                                            savefile_type='pkl') 
                                    else:
                                        try:
                                            fm_e1 = ForecastModel(window=2., overlap=1., station = sta,
                                                look_forward=2., data_streams=ds, savefile_type='csv')
                                        except:
                                            fm_e1 = ForecastModel(window=2., overlap=1., station = sta,
                                                look_forward=2., data_streams=ds, savefile_type='pkl')                    
                                    ##  
                                    ft = ft.replace("-",'"')
                                    
                                    ft_e1 = fm_e1.get_features(ti=t0, tf=t1, n_jobs=1, compute_only_features=[ft])
                                    # adding multiple Axes objects

                                    # extract values to plot 
                                    ft_e1_t = ft_e1[0].index.values
                                    ft_e1_v = ft_e1[0].loc[:,ft]
                                    #
                                    v_plot = ft_e1_v

                                else:
                                    day = timedelta(days=1)
                                    fm = ForecastModel(window=2., overlap=1., station=sta,
                                        look_forward=2., data_streams=[ds], 
                                        data_dir=r'C:\Users\aar135\codes_local_disk\volc_forecast_tl\volc_forecast_tl\data'
                                        )
                                    #
                                    N, M = [2,30]
                                    df = fm.data.df[t0:t1]
                                    if 'median' in ft:
                                        test = df[ds].rolling(N*24*6).median()[N*24*6:]
                                    if 'rate_variance' in ft:
                                        test = df[ds].rolling(N*24*6).apply(chqv)[N*24*6:]
                                    #
                                    #out = test.rolling(archtype.shape[0]).apply(partial(conv, (archtype-archtype.mean())/archtype.std()))       
                                    #out = out.resample('1D').ffill()
                                    #
                                    ft_e1_t = test.index
                                    v_plot = test.values

                                #v_plot = ft_e1_v-np.min(ft_e1_v)/np.max((ft_e1_v-np.min(ft_e1_v)))
                                #v_plot = ft_e1_v/np.max(ft_e1_v)
                                #
                                if ft == 'zsc2_mfF__median':
                                    ft = 'nMF median'
                                if ft == 'zsc2_hfF__median':
                                    ft = 'nHF median'
                                #
                                ax.plot(ft_e1_t, v_plot, '-', color=col[i], alpha = alpha[i],label=' '+ ft, zorder = 4)
                                #
                                ax.legend(loc = 3)
                                #
                                te = datetimeify(erup_time)#fm_e1.data.tes[int(erup[-1:])-1]
                                #ax1.set_xticks([ft_e1[0].index[-1] - 7*day*i for i in range(int(30/7)+1)])#[dat.index.values[0],dat.index.values[-1]])#, ]np.arange(0, len(x)+1, 5))
                                
                                #ax1b.set_yticks([])
                                #ax.grid()
                                ax.set_ylabel('Feature value')        #ax.set_xticks([ft_e1[0].index[-1]-7*day*i +day for i in range(5)])
                                #ax.set_xticks([ft_e1[0].index[-1] - 7*day*i for i in range(int(30/7)+1)])#[dat.index.values[0],dat.index.values[-1]])#, ]np.arange(0, len(x)+1, 5))
                                #ax1.set_xticks([t1 - 5*day*i for i in range(int(look_back/5)+1)])#[dat.index.values[0],dat.index.values[-1]])#, ]np.arange(0, len(x)+1, 5))
                                #
                            #except:
                            #    pass

                        else:
                            pass
                            if data_yrigth:
                                #
                                #ax1b = ax1.twinx() 
                                #
                                td = TremorData(station = sta)
                                #td.update(ti=t0, tf=t1)
                                data_streams = data_yrigth#['hf','mf', 'rsam']#, 'dsarF']
                                label = ['RSAM','MF','HF','DSAR']
                                #label = ['1/RSAM']
                                inv = False
                                if False:
                                    data_streams = ['rsam']
                                    label = ['RSAM']

                                if type(data_streams) is str:
                                    data_streams = [data_streams,]
                                if any(['_' in ds for ds in data_streams]):
                                    td._compute_transforms()
                                #ax.set_xlim(*range)
                                # plot data for each year
                                norm= False
                                _range = [t0,t1]
                                log =False
                                col_def = None
                                data = td.get_data(*_range)
                                xi = datetime(year=1,month=1,day=1,hour=0,minute=0,second=0)
                                cols = ['k','r','g','m',[0.5,0.5,0.5],[0.75,0.75,0.75]]
                                if inv:
                                    cols = ['k','g','r','m',[0.5,0.5,0.5],[0.75,0.75,0.75]]
                                inds = (data.index>=datetimeify(_range[0]))&(data.index<=datetimeify(_range[1]))
                                i=0
                                for data_stream, col in zip(data_streams,cols):
                                    v_plot = data[data_stream].loc[inds]
                                    if inv:
                                        v_plot = 1/v_plot
                                    if log:
                                        v_plot = np.log10(v_plot)
                                    if norm:
                                        v_plot = (v_plot-np.min(v_plot))/np.max((v_plot-np.min(v_plot)))
                                    if label:
                                        if col_def:
                                            ax.plot(data.index[inds], v_plot, '-', color=col_def, label=label[i], linewidth=1., alpha = 1., zorder = 0)
                                        else:
                                            #ax1b.plot(data.index[inds], v_plot, '-', color=col, label=label[i], linewidth=1., alpha = 1.0, zorder = 0)
                                            _dd = data.index[inds]
                                            _dd = np.arange(v_plot.shape[0])/6
                                            ax.plot(_dd, v_plot, '-', color=col, linewidth=1., alpha = .7, zorder = 3)
                                            axb.plot([], [], '-', color=col, label=label[i], linewidth=1., alpha = .7, zorder = 3)
                                    else:
                                        ax.plot(data.index[inds], v_plot, '-', color=col, label=data_stream, linewidth=1., alpha = .7, zorder = 3)
                                    i+=1
                                for te in td.tes:
                                    if [te>=datetimeify(_range[0]) and te<=datetimeify(_range[1])]:
                                        pass
                                        #ax.axvline(te, color='k', linestyle='--', linewidth=2, zorder = 0)
                                #
                                ax.plot([], color='k', linestyle='--', linewidth=1, label = 'eruption')
                                #ax4.set_xlim(_range)
                                #ax1b.legend(loc = 2)
                                #ax1b.grid()
                                #
                                if True:
                                    ## polyfit 1
                                    #x = data.index[inds] 
                                    #x = mdates.date2num(data.index[inds])
                                    y = np.log10(v_plot.values)
                                    x = np.arange(y.shape[0])/6
                                    z = np.polyfit(x, y, 1)
                                    p = np.poly1d(z)
                                    #
                                    #xx = np.linspace(x.min(), x.max(), 100)
                                    #dd = mdates.num2date(xx)
                                    #
                                    l_fit.append(z)
                                    #axb.plot([],[], '-g', label = 'linear fit (\{tau}_0='+str(round(l_fit[0][0],2))+')')
                                    label = r'\tau_0 = '+ str(round(l_fit[0][0],2))
                                    #ax.text(0., 0., label)#, fontsize='medium', verticalalignment='top', fontfamily='serif',
                                        #bbox=dict(facecolor='0.7', edgecolor='none', pad=3.0))
                                    #dt = (dd[-1]-dd[0]).seconds/3600 # hours
                                    #dy = p(xx[0]) - p(xx[-1])
                                    #tau = dt/dy
                                    tau = -(x[-1]-x[0])/(y[-1]-y[0])
                                    #
                                    #_yy = 10**p(x)#(xx)
                                    #_dd = np.arange(_yy.shape[0])
                                    #
                                    ax.plot(x, 10**p(x), '-g', linewidth=3, label = 'linear fit (\{tau}_0='+str(round(tau,2))+')')
                                    axb.plot([],[], '-g', linewidth=3, label = r'linear fit ($\tau_d$='+str(round(tau,1))+' hrs)')

                                    ## polyfit 2
                                    if j in [0,1]:
                                        t0 = erup_times[j] - look_back*day#30*day
                                        t1 = erup_times[j] + 2/8*day#hour
                                        _range = [t0,t1]
                                        #
                                        inds = (data.index>=datetimeify(_range[0]))&(data.index<=datetimeify(_range[1]))
                                        v_plot = data[data_stream].loc[inds]
                                        #
                                        #x = mdates.date2num(data.index[inds])
                                        y = np.log10(v_plot.values)
                                        x = np.arange(y.shape[0])/6
                                        z = np.polyfit(x, y, 1)
                                        p = np.poly1d(z)
                                        #
                                        #xx = np.linspace(x.min(), x.max(), 100)
                                        #dd = mdates.num2date(xx)
                                        #
                                        l_fit.append(z)
                                        #axb.plot([],[], '-g', label = 'linear fit (\{tau}_0='+str(round(l_fit[0][0],2))+')')
                                        label = r'\tau_0 = '+ str(round(l_fit[0][0],2))
                                        #ax.text(0., 0., label)#, fontsize='medium', verticalalignment='top', fontfamily='serif',
                                            #bbox=dict(facecolor='0.7', edgecolor='none', pad=3.0))
                                        #dt = (dd[-1]-dd[0]).seconds/3600 # hours
                                        #dy = p(xx[0]) - p(xx[-1])
                                        #tau = dt/dy
                                        tau = -(x[-1]-x[0])/(y[-1]-y[0])
                                        #
                                        #ax.plot(dd, 10**p(xx), '--b', linewidth=3, label = 'linear fit (\{tau}_0='+str(round(tau,2))+')')
                                        ax.plot(x, 10**p(x),'--b', linewidth=3, label = 'linear fit (\{tau}_0='+str(round(tau,2))+')')
                                        axb.plot([],[], '--b', linewidth=3, label = r'linear fit ($\tau_d$='+str(round(tau,1))+' hrs)')
                                        ##
                                        if log:
                                            ax.set_ylabel(' ')
                                        else:
                                            ax.set_ylabel('RSAM \u03BC m/s')
                                        #ax4.set_xlabel('Time [month-day hour]')
                                        #ax4.title.set_text('Station '+td.station+' ('+sta_code[td.station]+'): Tremor data')
                                        #
                                        #ax.set_xticks([te - 5*day*i for i in range(int(look_back/5)+1)])#[dat.index.values[0],dat.index.values[-1]])#, ]np.arange(0, len(x)+1, 5))
                                        #ax.set_xticks([i for i in range(24)])#[dat.index.values[0],dat.index.values[-1]])#, ]np.arange(0, len(x)+1, 5))
                                        #ax4.set_ylim([1e9,1e13])
                                        #ax4.set_xticks([te - 5*day*i for i in range(int(look_back/5)+1)])#[dat.index.values[0],dat.index.values[-1]])#, ]np.arange(0, len(x)+1, 5))
                                        #ax.set_yscale('log')
                        #if j == 0:
                        axb.legend(loc = 1, prop={'size': 10})     
                        axb.grid(False)
                        ax.grid(color='gray', linestyle='-', linewidth=.5, alpha = 0.5)
                        #
                        axb.set_yticklabels([])
                        ax.set_xlabel('Time [hours]')        #ax.set_xticks([ft_e1[0].index[-1]-7*day*i +day for i in range(5)])

        if True: # Copahue eruption 
            #
            sta = 'COP' 
            if sta == 'COP':
                ffm = False
                server = True # files imported from server to local pc 
                server2 = False # server at uni 
            #
            #erup_times = [datetimeify('2020 07 14 06 00 00'), datetimeify('2020 08 06 13 00 00')]
            #erup_times = [datetimeify('2020 06 16 00 00 00')]
            erup_times = [datetimeify('2020 07 16 06 00 00'), datetimeify('2020 08 06 09 00 00')]
            #look_back = 0
            #look_front = 1
            #
            col = ['b','b','r']
            alpha = [1., 1., .5]
            thick_line = [2., 6., 1.]
            #
            N, M = [2,1]
            #
            # subplot one: MF, HF, DSAR medians (DSAR yaxis left; MF, HF yaxis rigth). 1/RSAM (normalized)
            if True:
                for j,ax in enumerate([ax8, ax10]):
                    #
                    if True: # RSAM
                        ## DSAR median 
                        day = timedelta(days=1)
                        #sta_arch = 'WIZ'
                        dt = 'zsc2_rsamF'
                        fm = ForecastModel(window=2., overlap=1., station=sta,
                            look_forward=2., data_streams=[dt], 
                            data_dir=r'C:\Users\aar135\codes_local_disk\volc_forecast_tl\volc_forecast_tl\data'
                            )
                        #
                        te = erup_times[j]#fm.data.tes[erup] 
                        if True:
                            # rolling median and signature length window
                            #N, M = [2,15]
                            #l_forw = 0
                            # time
                            k = fm.data.df.index
                            # median 
                            #df = fm.data.df[(k>(te-(M+N)*day))&(k<te+(M+N)*day)]
                            df = fm.data.df[(k>(te-0*day))&(k<te+1*day)]

                            #archtype = df[dt].rolling(N*24*6).median()[N*24*6:]
                            archtype = df[dt]#.rolling(N*24*6).median()[N*24*6:]
                            #
                            #
                            _times = archtype.index

                            _val = archtype.values
                            _val = np.exp(_val)
                            _times = np.arange(_val.shape[0])/6
                            _val_max = max(_val)
                        else:
                            k = fm.data.df.index
                            df = fm.data.df[(k>(te-(M+N)*day))&(k<te+look_front*day)]
                            _times = np.arange(df.shape[0])/6
                            _val = df[dt].values

                        #
                        ft = 'RSAM'
                        ax.plot(_times, _val, '-', color='k', linewidth=1., alpha = .7,label=' '+ ft, zorder = 4) #'-', color='k', alpha = 0.8, linewidth=thick_line[0], label=' '+ ft,zorder=1)
                        #ax.plot([], [], '-', color='k', alpha = .7, linewidth=1., label=' '+ ft,zorder=1)
                        # lim
                        # polyfit
                        if True:
                            #x = data.index[inds] 
                            #x = mdates.date2num(_times)
                            y = np.log10(_val)
                            x = np.arange(y.shape[0])/6
                            z = np.polyfit(x, y, 1)
                            p = np.poly1d(z)
                            #
                            #xx = np.linspace(x.min(), x.max(), 100)
                            #dd = mdates.num2date(xx)
                            #
                            l_fit.append(z)
                            #ax.plot(dd, 10**p(xx), '-g')#, label = 'linear fit')
                            #
                            #dt = (dd[-1]-dd[0]).seconds/3600 # hours
                            #dy = p(xx[0]) - p(xx[-1])
                            #tau = dt/dy
                            tau = -(x[-1]-x[0])/(y[-1]-y[0])
                            #
                            #ax.plot(dd, 10**p(xx), '-g', label = 'linear fit (\{tau}_0='+str(round(tau,2))+')')
                            #ax.plot([],[], '-g', linewidth=3, label = r'linear fit ($\tau_d$='+str(round(tau,1))+' hrs)')
                            #ax.plot([],[], '-g', label = r'linear fit (tau='+str(round(-1./z[0],2))+' hrs)')
                            #axb.plot([],[], '-g', label = 'linear fit')
                            #ax.set_ylim([0,np.mean(_val)+3*np.std(_val)])
                            ax.plot(x, 10**p(x), '-g', linewidth=3, label = r'linear fit ($\tau_d$='+str(round(tau,1))+' hrs)')
                            axb.plot([],[], '-g', linewidth=3, label = r'linear fit ($\tau_d$='+str(round(tau,1))+' hrs)')

                            ## polyfit 2
                            if j in [0]:
                                df = fm.data.df[(k>(te-0*day))&(k<te+3.5/8*day)]
                                #archtype = df[dt].rolling(N*24*6).median()[N*24*6:]
                                dt = 'zsc2_rsamF'
                                archtype = df[dt]#.rolling(N*24*6).median()[N*24*6:]
                                #
                                #
                                _times = archtype.index
                                _val = archtype.values
                                _val = np.exp(_val)
                                #
                                #x = mdates.date2num(_times)
                                y = np.log10(_val)
                                x = np.arange(y.shape[0])/6
                                z = np.polyfit(x, y, 1)
                                p = np.poly1d(z)
                                #
                                #xx = np.linspace(x.min(), x.max(), 100)
                                #dd = mdates.num2date(xx)
                                #
                                l_fit.append(z)
                                #axb.plot([],[], '-g', label = 'linear fit (\{tau}_0='+str(round(l_fit[0][0],2))+')')
                                #label = r'\tau_0 = '+ str(round(l_fit[0][0],2))
                                #ax.text(0., 0., label)#, fontsize='medium', verticalalignment='top', fontfamily='serif',
                                #bbox=dict(facecolor='0.7', edgecolor='none', pad=3.0))
                                #dt = (dd[-1]-dd[0]).seconds/3600 # hours
                                #dy = p(xx[0]) - p(xx[-1])
                                #tau = dt/dy
                                tau = -(x[-1]-x[0])/(y[-1]-y[0])
                                #
                                ax.plot(x, 10**p(x), '--b', linewidth=3)#, label = 'linear fit (\{tau}_0='+str(round(tau,2))+')')
                                ax.plot([],[], '--b', linewidth=3, label = r'linear fit ($\tau_d$='+str(round(tau,1))+' hrs)')
                            ##
                    #
                    if False: # DSAR
                        axb = ax.twinx()
                        ## DSAR median 
                        day = timedelta(days=1)
                        #sta_arch = 'WIZ'
                        dt = 'zsc2_dsarF'
                        fm = ForecastModel(window=2., overlap=1., station=sta,
                            look_forward=2., data_streams=[dt], 
                            data_dir=r'C:\Users\aar135\codes_local_disk\volc_forecast_tl\volc_forecast_tl\data'
                            )
                        #
                        te = erup_times[j]#fm.data.tes[erup] 
                        # rolling median and signature length window
                        #N, M = [2,15]
                        #l_forw = 0
                        # time
                        k = fm.data.df.index
                        # median 
                        df = fm.data.df[(k>(te-(M+N)*day))&(k<te+look_front*day)]
                        archtype = df[dt].rolling(N*24*6).median()[N*24*6:]
                        #
                        _times = archtype.index
                        _val = archtype.values
                        _val_max = max(_val)
                        #
                        ft = 'nDSAR median'
                        axb.plot(_times, _val, '-', color='b', alpha = None, linewidth = 2, label=' '+ ft, zorder = 2) # '-', color='b', alpha = alpha[0], linewidth=thick_line[0], label=' '+ ft,zorder=1)
                        ax.plot([], [], '-', color='b', alpha = None, linewidth = 2, label=' '+ ft, zorder = 2)# color='b', alpha = alpha[0], linewidth=thick_line[0], label=' '+ ft,zorder=1)
                        #ax.plot([], [], '-', color='w', alpha = 0.1, linewidth=thick_line[0], label=str(te.year)+' '+str(te.mo
                        #
                        if plot_erup: # plot vertical lines
                            te = datetimeify(erup_times[j])#fm_e1.data.tes[int(erup[-1:])-1]
                            if j == 1:
                                ax.axvline(te, color='red', alpha = .25, linestyle='-', linewidth=12, zorder = 0)
                                axb.plot([], color='red', alpha = .25, linestyle='-', linewidth=12, label = 'eruption')
                            else:
                                ax.axvline(te, color='gray', alpha = .25, linestyle='-', linewidth=12, zorder = 0)
                                axb.plot([], color='gray', alpha = .25, linestyle='-', linewidth=12, label = 'fluid release event')
                        #
                    #
                    #ax1.legend(loc = 2)
                    #
                    te = datetimeify(erup_times[j])#fm_e1.data.tes[int(erup[-1:])-1]
                    #ax1.set_xticks([ft_e1[0].index[-1] - 7*day*i for i in range(int(30/7)+1)])#[dat.index.values[0],dat.index.values[-1]])#, ]np.arange(0, len(x)+1, 5))
                    #ax1b.set_yticks([])
                    #axb.grid()
                    #axb.set_ylabel('nDSAR value')        #ax.set_xticks([ft_e1[0].index[-1]-7*day*i +day for i in range(5)])
                    ax.set_ylabel('RSAM \u03BC m/s')        #ax.set_xticks([ft_e1[0].index[-1]-7*day*i +day for i in range(5)])
                    ax.set_xlabel('Time [hours]')        #ax.set_xticks([ft_e1[0].index[-1]-7*day*i +day for i in range(5)])
                    #
                    #ax.set_xticklabels(ax.get_xticks(), rotation = 30)
                    #ax1.set_yscale('log') #ax.set_yscale('log')
                    #ax.set_xticks([ft_e1[0].index[-1] - 7*day*i for i in range(int(30/7)+1)])#[dat.index.values[0],dat.index.values[-1]])#, ]np.arange(0, len(x)+1, 5))
                    #ax1.set_xticks([t1 - 5*day*i for i in range(int(look_back/5)+1)])#[dat.index.values[0],dat.index.values[-1]])#, ]np.arange(0, len(x)+1, 5))
                    #     #
                    # except:
                    ax.legend(loc = 1, prop={'size': 10})     
                    ax.grid(False)
                    ax.grid(color='gray', linestyle='-', linewidth=.5, alpha = 0.5)
                    #
                    #axb.set_yticklabels([])
                    #
                    # _d = 5 
                    # t1 = erup_times[0] + look_front*day#hour
                    # ax.set_xticks([t1 - _d*day*i for i in range(int((look_back+look_front)/_d)+1)])
                    #
        #
        #ax.set_xlim([t0+2*day,t1])
        #ax1.set_ylim([10**.4,10**2.1])
        #ax3.set_ylim([10**.5,10**2])
        #ax4.set_ylim([1,100])
        #ax7.set_ylim([1,100])
        #ax2b.set_ylim([0.1,0.5])
        #ax3.set_xlim([t0+2*day,t1])
        #ax4.set_xlim([t0+2*day,t1])
        #
        # ax1.set_yscale('log')
        # ax2.set_yscale('log')
        # ax3.set_yscale('log')
        # ax4.set_yscale('log')
        # ax5.set_yscale('log')
        # ax6.set_yscale('log')
        ax7.set_yscale('log')
        ax8.set_yscale('log')
        ax9.set_yscale('log')
        ax10.set_yscale('log')
        #
        if False:
            ax7.set_ylim([10**9,10**13])
            ax9.set_ylim([2* 10**10,10**11])       
            #
            ax8.set_ylim([0.1,10**7])
            ax10.set_ylim([1,10**10])
        #
        # ax1.set_title('(a) 24 hours after 2006/10/04 Ruapehu eruption')
        # ax2.set_title('(b) 24 hours after 2007/09/25 Ruapehu eruption')
        # #
        # ax3.set_title('(c) 24 hours after 2009/07/13 Ruapehu fluid release event')
        # ax5.set_title('(e) 24 hours after 2010/09/03 Ruapehu fluid release event')
        # ax4.set_title('(d) 24 hours after 2016/11/13 Ruapehu fluid release event')
        # ax6.set_title('(f) 24 hours after 2021/03/04 Ruapehu possible sealing and fluid release event')
        #
        ax7.set_title('(g) 24 hours after 2013/01/23 Kawah Ijen fluid release event')
        ax9.set_title('(h) 24 hours after 2013/04/01 Kawah Ijen fluid release event')
        ax8.set_title('(i) 24 hours after 2020/07/16 Copahue fluid release event')
        ax10.set_title('(j) 24 hours after 2020/08/06 Copahue fluid release event')
        #
        plt.tight_layout()
        plt.show()
        plt.close('all')
        #

##########################################################
def main():
    ##
    #plot_temp_data()
    #plot_seismic_temp_data()

    ## lake temperature correlation
    cc_over_record()
    #locate_missed_events_seismic()
    #plot_located_events_compress()

    # temp_erup_ana()
    # temp_dif_rate_stats()
    # plot_temp_erup_ana()
    #temp_level_change_corr_ana()
    
    ##comb_stat_test_temp_dif_rate()
    ##pval_comb_stat_temp_dif_rate()
    # scat_plot_cc_dT()

    ## lake level correlation
    # level_erup_ana()
    # level_dif_rate_stats()
    # plot_level_erup_ana()

    # acoustic data
    #download_acoustic()
    #plot_acoustic()

    ## temperature cycles and events
    #map_events_in_temp_cycles()

    # figures paper
    #figure_2()
    #figure_3()
    #figure_4()

    #figure_sup_ruapehu_events() # events in seismic and lake levels 
    #figure_sup_copahue_events()
    #figure_sup_kawahijen_events()
    #figure_5_ruapehu()

    # current
    #figure_1()
    #figure_3_alt_ruap_kawa_copa() #man selected events RSAM and DSAR
    #figure_2_alt() # histograms
    #figure_4_alt()
    #figure_5_alt_ruap_kawa_copa() #man selected events RSAM, zoom in after events 

if __name__ == "__main__":
    main()