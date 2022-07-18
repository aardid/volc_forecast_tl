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
month = timedelta(days=365.25/12)
day = timedelta(days=1)
hour = day/24
textsize = 13.
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
            'BELO_1': 'Bezymiany 2007a',
            'BELO_2': 'Bezymiany 2007b',
            'BELO_3': 'Bezymiany 2007c',
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
            'MEA01_8': 'Merapi 2019a'
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
            'VNSS_1': '2-3',
            'VNSS_2': '3',
            'TBTN_1': ' ',
            'TBTN_2': ' '
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
            'MEA01': 'Merapi'
            }

# eruption times
erup_times = {'WIZ_1': '2012 08 04 16 52 00',
            'WIZ_2': '2013 08 19 22 23 00',
            'WIZ_3': '2013 10 11 07 09 00',#'2013 10 03 12 35 00',#
            'WIZ_4': '2016 04 27 09 37 00',
            'WIZ_5': '2019 12 09 01 11 00',
            'FWVZ_1': '2006 10 04 09 30 00',
            'FWVZ_2': '2007 09 25 08 20 00',
            'FWVZ_3': '2009 07 13 06 30 00',
            'KRVZ_1': '2012 08 06 11 50 00',
            'KRVZ_2': '2012 11 21 00 20 00',
            'BELO_1': '2007 09 25 08 30 00',
            'BELO_2': '2007 10 14 14 27 00',
            'BELO_3': '2007 11 05 08 43 00',
            'PVV_1': '2014 05 31 17 22 00',
            'PVV_2': '2014 11 13 00 00 00',
            'PVV_3': '2016 03 28 01 33 00',
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
            'MEA01_8': '2019 10 14 12 00 00'
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
def to_nztimezone(t):
    """ Routine to convert UTC to NZ time zone.
    """
    from dateutil import tz
    utctz = tz.gettz('UTC')
    nztz = tz.gettz('Pacific/Auckland')
    return [ti.replace(tzinfo=utctz).astimezone(nztz) for ti in pd.to_datetime(t)]

##
def plot_data_ratios():
    #
    # define station and time
    sta = 'GOD'
    erup = 1
    erup_time = datetimeify(erup_times[sta+'_'+str(erup)])
    t0 = erup_time - 28*day
    t1 = erup_time #+ 1*day#hour
    win = 2.
    if False: # other dates
        t1 = datetimeify("2021-12-01")
        t0 = t1 - 25*day
        plot_periods =  False
        ffm = False
        server = False
    #
    if sta  == 'WIZ': #True: # WIZ5
        if erup == 5:
            plot_periods = False#['2019 11 11 00 00 00','2019 11 23 00 00 00', '2019 12 02 00 00 00', '2019 12 06 12 00 00']
            plot_periods_label = ['magma-geothermal system interaction', 'Pulsating gas flux', 
                                    'Sealing consolidation', 'Pressurization and eruption']
            plot_periods_col = ['gray', 'gray', 'gray', 'gray']
            #
            ffm = False
            server = True
    #
    if sta  == 'MEA01': #True: # WIZ5
        #if erup == 7:
        plot_periods = False#['2019 11 11 00 00 00','2019 11 23 00 00 00', '2019 12 02 00 00 00', '2019 12 06 12 00 00']
        plot_periods_label = ['magma-geothermal system interaction', 'Pulsating gas flux', 
                                'Sealing consolidation', 'Pressurization and eruption']
        plot_periods_col = ['gray', 'gray', 'gray', 'gray']
        #
        ffm = False
        server = False
    #
    if sta  == 'FWVZ': #True: # WIZ5
        if erup == 2:
            plot_periods = False#['2019 11 11 00 00 00','2019 11 23 00 00 00', '2019 12 02 00 00 00', '2019 12 06 12 00 00']
            plot_periods_label = ['magma-geothermal system interaction', 'Pulsating gas flux', 
                                    'Sealing consolidation', 'Pressurization and eruption']
            plot_periods_col = ['gray', 'gray', 'gray', 'gray']
            #
            ffm = False
            server = True
    #
    def _plt_intp(sta, t0, t1):
        # figure
        nrow = 4
        ncol = 1
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(nrows=nrow, ncols=ncol,figsize=(14,8))#(14,4))
        #####################################################
        # subplot one:  features
        if True:
            # features
            fts = ['zsc2_vlfF__median','zsc2_lfF__median']
                    
            col = ['r','g','b']
            alpha = [.5, 1., 1.]
            thick_line = [3., 3., 6.]
            #
            for i, ft in enumerate(fts):
                if 'zsc2_vlfF' in ft:
                    ds = ['zsc2_vlfF'] 
                if 'zsc2_lfF' in ft:
                    ds = ['zsc2_lfF']
                if 'zsc2_rsamF' in ft:
                    ds = ['zsc2_rsamF']
                #
                if server:
                    path_feat_serv = 'C:\\Users\\aar135\\codes_local_disk\\volc_forecast_tl\\features_bkp\\features_server\\'
                    fm_e1 = ForecastModel(window=win, overlap=1., station =  sta,
                        look_forward=2., data_streams=ds, 
                        feature_dir=path_feat_serv, 
                        savefile_type='pkl') 
                else:
                    fm_e1 = ForecastModel(window=win, overlap=1., station = sta,
                        look_forward=2., data_streams=ds, savefile_type='csv')
                ##
                ft = ft.replace("-",'"')
                # adding multiple Axes objects
                ft_e1 = fm_e1.get_features(ti=t0, tf=t1, n_jobs=1, compute_only_features=[ft])
                # extract values to plot 
                ft_e1_t = ft_e1[0].index.values
                ft_e1_v = ft_e1[0].loc[:,ft]
                #
                #v_plot = ft_e1_v-np.min(ft_e1_v)/np.max((ft_e1_v-np.min(ft_e1_v)))
                #v_plot = ft_e1_v/np.max(ft_e1_v)
                v_plot =  ft_e1_v
                #
                #if ft == 'zsc2_dsarF__change_quantiles__f_agg_"var"__isabs_False__qh_0.6__ql_0.4':
                #    ft = 'nDSAR rate variance'#'DSAR change quantiles (.6-.4) variance'

                ax1.plot(ft_e1_t, v_plot, '-', color=col[i], alpha = alpha[i],label='Feature: '+ ft)
                #
            #
            if ffm: # ffm 
                ax1b = ax1.twinx() 
                #v_plot = data[data_stream].loc[inds]
                inv_rsam = fm_e1.data.get_data(ti=t0, tf=t1)['rsamF']#.loc[ft_e1_t]
                inv_rsam = 1./inv_rsam
                ax1b.plot(ft_e1_t, inv_rsam/0.0035, '-', color= 'gray', linewidth=0.5, markersize=0.5, alpha = 1.)
                ax1.plot([], [], '-', color= 'gray', markersize=1, label='1/RSAM', alpha = 1.0)
                ax1b.set_ylim([0,1])
                ax1b.set_yticks([])
                #
                if False:#mov_avg: # plot moving average
                    n=50
                    v_plot = (inv_rsam-np.min(inv_rsam))/np.max((inv_rsam-np.min(inv_rsam)))
                    #ax2b.plot(temp_e1_tim[:-n+1], moving_average(temp_e1_val, n=n), '--', color='k', label='temp. mov. avg.')
                    ax1b.plot(ft_e1_t[n-1-15:-15], moving_average(v_plot[::-1], n=n)[::-1], '-', color='g', label='lake temperature')
                    ax1b.set_yticks([])
            #
            if plot_periods:
                for k, t in enumerate(plot_periods):
                    te = datetimeify(t)#fm_e1.data.tes[int(erup[-1:])-1]
                    ax1.axvline(te, color=plot_periods_col[k], linestyle='-', linewidth=20, alpha = 0.2, zorder = 4)
                    #ax3.plot([], color='k', linestyle='--', linewidth=2, label = 'event')
            #
            if erup_time: # plot vertical lines
                te = datetimeify(erup_time)#fm_e1.data.tes[int(erup[-1:])-1]
                ax1.axvline(te, color='k',linestyle='--', linewidth=2, zorder = 4)
                ax1.plot([], color='k', linestyle='--', linewidth=2, label = 'eruption')

            #
            ax1.legend(loc = 2)
            #ax1.set_ylim([0,1])
            #
            te = datetimeify(erup_time)#fm_e1.data.tes[int(erup[-1:])-1]
            #ax1.set_xticks([te - 3*day*i for i in range(int(30/3)+1)])#[dat.index.values[0],dat.index.values[-1]])#, ]np.arange(0, len(x)+1, 5))
            #ax1.set_xticks([ft_e1[0].index[-1] - 7*day*i for i in range(int(30/7)+1)])#[dat.index.values[0],dat.index.values[-1]])#, ]np.arange(0, len(x)+1, 5))
            #ax1.set_yticks([])
            
            #ax1b.set_yticks([])
            ax1.grid(color='gray', linestyle='-', linewidth=.3, alpha = 0.5)

            ax1.set_ylabel('feature value')        #ax.set_xticks([ft_e1[0].index[-1]-7*day*i +day for i in range(5)])
            #ax.set_xticks([ft_e1[0].index[-1] - 7*day*i for i in range(int(30/7)+1)])#[dat.index.values[0],dat.index.values[-1]])#, ]np.arange(0, len(x)+1, 5))
        #####################################################
        # subplot one: normalize features
        if True:
            # features
            fts = ['zsc2_rsamF__median','zsc2_mfF__median','zsc2_hfF__median']
                    
            col = ['r','g','b']
            alpha = [.5, 1., 1.]
            thick_line = [3., 3., 6.]
            #
            for i, ft in enumerate(fts):
                if 'zsc2_mfF' in ft:
                    ds = ['zsc2_mfF'] 
                if 'zsc2_hfF' in ft:
                    ds = ['zsc2_hfF']
                if 'zsc2_rsamF' in ft:
                    ds = ['zsc2_rsamF']
                #
                if server:
                    path_feat_serv = 'C:\\Users\\aar135\\codes_local_disk\\volc_forecast_tl\\features_bkp\\features_server\\'
                    fm_e1 = ForecastModel(window=win, overlap=1., station =  sta,
                        look_forward=2., data_streams=ds, 
                        feature_dir=path_feat_serv, 
                        savefile_type='pkl') 
                else:
                    fm_e1 = ForecastModel(window=win, overlap=1., station = sta,
                        look_forward=2., data_streams=ds, savefile_type='csv')
                ##
                ft = ft.replace("-",'"')
                # adding multiple Axes objects
                ft_e1 = fm_e1.get_features(ti=t0, tf=t1, n_jobs=1, compute_only_features=[ft])
                # extract values to plot 
                ft_e1_t = ft_e1[0].index.values
                ft_e1_v = ft_e1[0].loc[:,ft]
                #
                v_plot = ft_e1_v-np.min(ft_e1_v)/np.max((ft_e1_v-np.min(ft_e1_v)))
                v_plot = ft_e1_v/np.max(ft_e1_v)
                #v_plot =  ft_e1_v
                #
                #if ft == 'zsc2_dsarF__change_quantiles__f_agg_"var"__isabs_False__qh_0.6__ql_0.4':
                #    ft = 'nDSAR rate variance'#'DSAR change quantiles (.6-.4) variance'

                ax2.plot(ft_e1_t, v_plot, '-', color=col[i], alpha = alpha[i],label='Feature: '+ ft)
                #
            #
            if ffm: # ffm 
                ax1b = ax1.twinx() 
                #v_plot = data[data_stream].loc[inds]
                inv_rsam = fm_e1.data.get_data(ti=t0, tf=t1)['rsamF']#.loc[ft_e1_t]
                inv_rsam = 1./inv_rsam
                ax1b.plot(ft_e1_t, inv_rsam/0.0035, '-', color= 'gray', linewidth=0.5, markersize=0.5, alpha = 1.)
                ax1.plot([], [], '-', color= 'gray', markersize=1, label='1/RSAM', alpha = 1.0)
                ax1b.set_ylim([0,1])
                ax1b.set_yticks([])
                #
                if False:#mov_avg: # plot moving average
                    n=50
                    v_plot = (inv_rsam-np.min(inv_rsam))/np.max((inv_rsam-np.min(inv_rsam)))
                    #ax2b.plot(temp_e1_tim[:-n+1], moving_average(temp_e1_val, n=n), '--', color='k', label='temp. mov. avg.')
                    ax1b.plot(ft_e1_t[n-1-15:-15], moving_average(v_plot[::-1], n=n)[::-1], '-', color='g', label='lake temperature')
                    ax1b.set_yticks([])
            #
            if plot_periods:
                for k, t in enumerate(plot_periods):
                    te = datetimeify(t)#fm_e1.data.tes[int(erup[-1:])-1]
                    ax1.axvline(te, color=plot_periods_col[k], linestyle='-', linewidth=20, alpha = 0.2, zorder = 4)
                    #ax3.plot([], color='k', linestyle='--', linewidth=2, label = 'event')
            #
            if erup_time: # plot vertical lines
                te = datetimeify(erup_time)#fm_e1.data.tes[int(erup[-1:])-1]
                ax2.axvline(te, color='k',linestyle='--', linewidth=2, zorder = 4)
                ax2.plot([], color='k', linestyle='--', linewidth=2, label = 'eruption')

            #
            ax2.legend(loc = 2)
            #ax1.set_ylim([0,1])
            #
            te = datetimeify(erup_time)#fm_e1.data.tes[int(erup[-1:])-1]
            #ax1.set_xticks([te - 3*day*i for i in range(int(30/3)+1)])#[dat.index.values[0],dat.index.values[-1]])#, ]np.arange(0, len(x)+1, 5))
            #ax1.set_xticks([ft_e1[0].index[-1] - 7*day*i for i in range(int(30/7)+1)])#[dat.index.values[0],dat.index.values[-1]])#, ]np.arange(0, len(x)+1, 5))
            #ax1.set_yticks([])
            
            #ax1b.set_yticks([])
            ax2.grid(color='gray', linestyle='-', linewidth=.3, alpha = 0.5)

            ax2.set_ylabel('normalized value')        #ax.set_xticks([ft_e1[0].index[-1]-7*day*i +day for i in range(5)])        
        #####################################################
        # subplot three: features median ratios
        # features
        if True:
            fts2 = ['zsc2_vlarF__median','zsc2_lrarF__median','zsc2_rmarF__median','zsc2_dsarF__median']
            
            col = ['r','g','b','m']
            alpha = [1., 1., 1., 1.]
            thick_line = [3., 3., 3., 3.]
            for i, ft in enumerate(fts2):
                if 'zsc2_vlarF' in ft:
                    ds = ['zsc2_vlarF']
                if 'zsc2_lrarF' in ft:
                    ds = ['zsc2_lrarF']
                if 'zsc2_rmarF' in ft:
                    ds = ['zsc2_rmarF']
                if 'zsc2_dsarF' in ft:
                    ds = ['zsc2_dsarF']
                if server:
                    path_feat_serv = 'C:\\Users\\aar135\\codes_local_disk\\volc_forecast_tl\\features_bkp\\features_server\\'
                    fm_e2 = ForecastModel(window=win, overlap=1., station =  sta,
                        look_forward=2., data_streams=ds, 
                        feature_dir=path_feat_serv, 
                        savefile_type='pkl') 
                else:
                    fm_e2 = ForecastModel(window=win, overlap=1., station = sta,
                        look_forward=2., data_streams=ds, savefile_type='csv')
                #
                ft = ft.replace("-",'"')
                # adding multiple Axes objects
                fm_e2 = fm_e2.get_features(ti=t0, tf=t1, n_jobs=1, compute_only_features=[ft])
                # extract values to plot 
                fm_e2_t = fm_e2[0].index.values
                fm_e2_v = fm_e2[0].loc[:,ft]
                #
                # import datastream to plot 
                ## ax1 and ax2
                if len(fts2) == 1:
                    v_plot = fm_e2_v
                else:
                    v_plot = fm_e2_v#(ft_e1_v-np.min(ft_e1_v))/np.max((ft_e1_v-np.min(ft_e1_v)))
                    #v_plot = ft_e1_v
                if False:
                    if rank == 262:
                        v_plot = v_plot*40
                        v_plot = v_plot - np.mean(v_plot) +.5
                #
                #if ft == 'zsc2_mfF__median':
                #    ft = 'nMF median'
                #
                ax3.plot(fm_e2_t, v_plot, '-', color=col[i], alpha = alpha[i],label='Feature: '+ ft)
            #
            if erup_time: # plot vertical lines
                te = datetimeify(erup_time)#fm_e1.data.tes[int(erup[-1:])-1]
                ax3.axvline(te, color='k',linestyle='--', linewidth=2, zorder = 4)
                ax3.plot([], color='k', linestyle='--', linewidth=2, label = 'eruption')

            if plot_periods:
                for k, t in enumerate(plot_periods):
                    te = datetimeify(t)#fm_e1.data.tes[int(erup[-1:])-1]
                    ax2.axvline(te, color=plot_periods_col[k], linestyle='-', linewidth=20, alpha = 0.2, zorder = 4)
                    #ax3.plot([], color='k', linestyle='--', linewidth=2, label = 'event')
            #
            ax3.set_ylabel('feature data')
            ax3.legend(loc = 2)
            te = datetimeify(erup_time)#fm_e1.data.tes[int(erup[-1:])-1]
            #ax2.set_xticks([te - 3*day*i for i in range(int(30/3)+1)])#[dat.index.values[0],dat.index.values[-1]])#, ]np.arange(0, len(x)+1, 5))
            #ax2.set_ylim([1,2])
            #ax2.set_xticks([ft_e1[0].index[-1] - 7*day*i for i in range(int(30/7))])#[dat.index.values[0],dat.index.values[-1]])#, ]np.arange(0, len(x)+1, 5))
            #ax.set_yticks([])
            ax3.grid(color='gray', linestyle='-', linewidth=.3, alpha = 0.5)
        #####################################################
        # subplot four: features median ratios
        # features
        if True:
            fts2 = ['zsc2_vlarF__median','zsc2_lrarF__median','zsc2_rmarF__median','zsc2_dsarF__median']
            
            col = ['r','g','b','m']
            alpha = [1., 1., 1., 1.]
            thick_line = [3., 3., 3., 3.]
            for i, ft in enumerate(fts2):
                if 'zsc2_vlarF' in ft:
                    ds = ['zsc2_vlarF']
                if 'zsc2_lrarF' in ft:
                    ds = ['zsc2_lrarF']
                if 'zsc2_rmarF' in ft:
                    ds = ['zsc2_rmarF']
                if 'zsc2_dsarF' in ft:
                    ds = ['zsc2_dsarF']
                if server:
                    path_feat_serv = 'C:\\Users\\aar135\\codes_local_disk\\volc_forecast_tl\\features_bkp\\features_server\\'
                    fm_e2 = ForecastModel(window=win, overlap=1., station =  sta,
                        look_forward=2., data_streams=ds, 
                        feature_dir=path_feat_serv, 
                        savefile_type='pkl') 
                else:
                    fm_e2 = ForecastModel(window=win, overlap=1., station = sta,
                        look_forward=2., data_streams=ds, savefile_type='csv')
                #
                ft = ft.replace("-",'"')
                # adding multiple Axes objects
                fm_e2 = fm_e2.get_features(ti=t0, tf=t1, n_jobs=1, compute_only_features=[ft])
                # extract values to plot 
                fm_e2_t = fm_e2[0].index.values
                fm_e2_v = fm_e2[0].loc[:,ft]
                #
                # import datastream to plot 
                ## ax1 and ax2
                if False:#len(fts2) == 1:
                    v_plot = fm_e2_v
                else:
                    v_plot = fm_e2_v / np.max(fm_e2_v)#(ft_e1_v-np.min(ft_e1_v))/np.max((ft_e1_v-np.min(ft_e1_v)))
                    #v_plot = ft_e1_v
                if False:
                    if rank == 262:
                        v_plot = v_plot*40
                        v_plot = v_plot - np.mean(v_plot) +.5
                #
                #if ft == 'zsc2_mfF__median':
                #    ft = 'nMF median'
                #
                ax4.plot(fm_e2_t, v_plot, '-', color=col[i], alpha = alpha[i],label='Feature: '+ ft)
            #
            if erup_time: # plot vertical lines
                te = datetimeify(erup_time)#fm_e1.data.tes[int(erup[-1:])-1]
                ax4.axvline(te, color='k',linestyle='--', linewidth=2, zorder = 4)
                ax4.plot([], color='k', linestyle='--', linewidth=2, label = 'eruption')

            if plot_periods:
                for k, t in enumerate(plot_periods):
                    te = datetimeify(t)#fm_e1.data.tes[int(erup[-1:])-1]
                    ax4.axvline(te, color=plot_periods_col[k], linestyle='-', linewidth=20, alpha = 0.2, zorder = 4)
                    #ax3.plot([], color='k', linestyle='--', linewidth=2, label = 'event')
            #
            ax4.set_ylabel('normalized data')
            #ax4.legend(loc = 2)
            te = datetimeify(erup_time)#fm_e1.data.tes[int(erup[-1:])-1]
            #ax2.set_xticks([te - 3*day*i for i in range(int(30/3)+1)])#[dat.index.values[0],dat.index.values[-1]])#, ]np.arange(0, len(x)+1, 5))
            #ax2.set_ylim([1,2])
            #ax2.set_xticks([ft_e1[0].index[-1] - 7*day*i for i in range(int(30/7))])#[dat.index.values[0],dat.index.values[-1]])#, ]np.arange(0, len(x)+1, 5))
            #ax.set_yticks([])
            ax4.grid(color='gray', linestyle='-', linewidth=.3, alpha = 0.5)
        #####################################################

        ax1.set_xlim([t0,t1])
        ax2.set_xlim([t0,t1])
        ax3.set_xlim([t0,t1])
        ax4.set_xlim([t0,t1])
        #
        #ax1.grid(color='gray', linestyle='-', linewidth=.3, alpha = 0.5)
        #ax2.grid(color='gray', linestyle='-', linewidth=.3, alpha = 0.5)
        #ax3.grid(color='gray', linestyle='-', linewidth=.3, alpha = 0.5)
        #ax4.grid(color='gray', linestyle='-', linewidth=.3, alpha = 0.5)
        #####################################################
        fig.suptitle(sta_code[sta]+': '+str(t0)+' to '+str(t1))#'Feature: '+ ft_nm_aux, ha='center')
        plt.tight_layout()
        path = '..'+os.sep+'features'+os.sep+'correlations'+os.sep+'comb_feat_analysis'+os.sep
        #plt.savefig(path+erup+'_'+ft_id+'.png')
        plt.show()
        plt.close()
    ##
    _plt_intp(sta,t0, t1)
    # 
def plot_ratios():
    #
    # define station and time
    sta = 'FWVZ'
    erup = 2
    win = 2.
    erup_time = datetimeify(erup_times[sta+'_'+str(erup)])
    d_back = 30
    t0 = erup_time - d_back*day
    t1 = erup_time #- 2*day
    if True: # other dates
        t1 = datetimeify("2021-12-01")
        t0 = t1 - 25*day
        plot_periods =  False
        ffm = False
        server = False
    #
    # select just one
    plot_inv_rat = False
    plot_data = True
    #
    if sta  == 'MEA01': #True: # WIZ5
        #if erup == 7:
        plot_periods = False#['2019 11 11 00 00 00','2019 11 23 00 00 00', '2019 12 02 00 00 00', '2019 12 06 12 00 00']
        plot_periods_label = ['magma-geothermal system interaction', 'Pulsating gas flux', 
                                'Sealing consolidation', 'Pressurization and eruption']
        plot_periods_col = ['gray', 'gray', 'gray', 'gray']
        #
        ffm = False
        server = False
    #
    if sta  == 'WIZ': #True: # WIZ5
        if erup == 5:
            plot_periods = False#['2019 11 11 00 00 00','2019 11 23 00 00 00', '2019 12 02 00 00 00', '2019 12 06 12 00 00']
            plot_periods_label = ['magma-geothermal system interaction', 'Pulsating gas flux', 
                                    'Sealing consolidation', 'Pressurization and eruption']
            plot_periods_col = ['gray', 'gray', 'gray', 'gray']
            #
            ffm = False
            server = True

        if erup == 4:
            plot_periods = False#['2019 11 11 00 00 00','2019 11 23 00 00 00', '2019 12 02 00 00 00', '2019 12 06 12 00 00']
            plot_periods_label = ['magma-geothermal system interaction', 'Pulsating gas flux', 
                                    'Sealing consolidation', 'Pressurization and eruption']
            plot_periods_col = ['gray', 'gray', 'gray', 'gray']
            #
            ffm = False
            server = True
        if erup == 1:
            plot_periods = False#['2019 11 11 00 00 00','2019 11 23 00 00 00', '2019 12 02 00 00 00', '2019 12 06 12 00 00']
            plot_periods_label = ['magma-geothermal system interaction', 'Pulsating gas flux', 
                                    'Sealing consolidation', 'Pressurization and eruption']
            plot_periods_col = ['gray', 'gray', 'gray', 'gray']
            #
            ffm = False
            server = True
        if erup == 2:
            plot_periods = False#['2019 11 11 00 00 00','2019 11 23 00 00 00', '2019 12 02 00 00 00', '2019 12 06 12 00 00']
            plot_periods_label = ['magma-geothermal system interaction', 'Pulsating gas flux', 
                                    'Sealing consolidation', 'Pressurization and eruption']
            plot_periods_col = ['gray', 'gray', 'gray', 'gray']
            #
            ffm = False
            server = True
        if erup == 3:
            plot_periods = False#['2019 11 11 00 00 00','2019 11 23 00 00 00', '2019 12 02 00 00 00', '2019 12 06 12 00 00']
            plot_periods_label = ['magma-geothermal system interaction', 'Pulsating gas flux', 
                                    'Sealing consolidation', 'Pressurization and eruption']
            plot_periods_col = ['gray', 'gray', 'gray', 'gray']
            #
            ffm = False
            server = True
    #
    if sta  == 'VNSS':
        if erup == 2:
            plot_periods = False#['2019 11 11 00 00 00','2019 11 23 00 00 00', '2019 12 02 00 00 00', '2019 12 06 12 00 00']
            plot_periods_label = ['magma-geothermal system interaction', 'Pulsating gas flux', 
                                    'Sealing consolidation', 'Pressurization and eruption']
            plot_periods_col = ['gray', 'gray', 'gray', 'gray']
            #
            ffm = False
            server = True  
    #
    if sta  == 'FWVZ':
        if erup in [1,2,3]:
            plot_periods = False#['2019 11 11 00 00 00','2019 11 23 00 00 00', '2019 12 02 00 00 00', '2019 12 06 12 00 00']
            plot_periods_label = ['magma-geothermal system interaction', 'Pulsating gas flux', 
                                    'Sealing consolidation', 'Pressurization and eruption']
            plot_periods_col = ['gray', 'gray', 'gray', 'gray']
            #
            ffm = False
            server = True    
    #
    if sta  == 'PVV':
        if erup in [1,2,3]:
            plot_periods = False#['2019 11 11 00 00 00','2019 11 23 00 00 00', '2019 12 02 00 00 00', '2019 12 06 12 00 00']
            plot_periods_label = ['magma-geothermal system interaction', 'Pulsating gas flux', 
                                    'Sealing consolidation', 'Pressurization and eruption']
            plot_periods_col = ['gray', 'gray', 'gray', 'gray']
            #
            ffm = False
            server = True    
    #
    if sta  == 'KRVZ':
        if erup in [1,2]:
            plot_periods = False#['2019 11 11 00 00 00','2019 11 23 00 00 00', '2019 12 02 00 00 00', '2019 12 06 12 00 00']
            plot_periods_label = ['magma-geothermal system interaction', 'Pulsating gas flux', 
                                    'Sealing consolidation', 'Pressurization and eruption']
            plot_periods_col = ['gray', 'gray', 'gray', 'gray']
            #
            ffm = False
            server = True    
    #
    if sta  == 'BELO':
        if erup in [1,2,3]:
            plot_periods = False#['2019 11 11 00 00 00','2019 11 23 00 00 00', '2019 12 02 00 00 00', '2019 12 06 12 00 00']
            plot_periods_label = ['magma-geothermal system interaction', 'Pulsating gas flux', 
                                    'Sealing consolidation', 'Pressurization and eruption']
            plot_periods_col = ['gray', 'gray', 'gray', 'gray']
            #
            ffm = False
            server = True    
    #
    def _plt_intp(sta, t0, t1):
        # figure
        nrow = 5
        ncol = 1
        fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(nrows=nrow, ncols=ncol,figsize=(14,8))#(14,4))
        #####################################################
        # subplot one:  features
        if True:
            # features
            fts = ['zsc2_vlarF__median']
            #     
            col = ['r','g','b']
            col = ['r']
            alpha = [1., 1., 1.]
            thick_line = [3., 3., 6.]
            #
            for i, ft in enumerate(fts):
                if 'zsc2_vlarF' in ft:
                    ds = ['zsc2_vlarF'] 
                #
                if server:
                    path_feat_serv = 'C:\\Users\\aar135\\codes_local_disk\\volc_forecast_tl\\features_bkp\\features_server\\'
                    fm_e1 = ForecastModel(window=win, overlap=1., station =  sta,
                        look_forward=2., data_streams=ds, 
                        feature_dir=path_feat_serv, 
                        savefile_type='pkl') 
                else:
                    fm_e1 = ForecastModel(window=win, overlap=1., station = sta,
                        look_forward=2., data_streams=ds, savefile_type='csv')
                ##
                ft = ft.replace("-",'"')
                # adding multiple Axes objects
                ft_e1 = fm_e1.get_features(ti=t0, tf=t1, n_jobs=1, compute_only_features=[ft])
                # extract values to plot 
                ft_e1_t = ft_e1[0].index.values
                ft_e1_v = ft_e1[0].loc[:,ft]
                #
                #v_plot = ft_e1_v-np.min(ft_e1_v)/np.max((ft_e1_v-np.min(ft_e1_v)))
                #v_plot = ft_e1_v/np.max(ft_e1_v)
                v_plot =  ft_e1_v
                #
                #if ft == 'zsc2_dsarF__change_quantiles__f_agg_"var"__isabs_False__qh_0.6__ql_0.4':
                #    ft = 'nDSAR rate variance'#'DSAR change quantiles (.6-.4) variance'

                ax1.plot(ft_e1_t, v_plot, '-', color=col[i], alpha = alpha[i],label='Feature: '+ ft)
                #
                if plot_inv_rat:
                    ax1b = ax1.twinx()
                    ax1b.plot(ft_e1_t, 1/v_plot, '-', color='gray', alpha = alpha[i],label='Feature: 1/'+ ft)
                    ax1b.legend(loc = 3)
            #
            if ffm: # ffm 
                ax1b = ax1.twinx() 
                #v_plot = data[data_stream].loc[inds]
                inv_rsam = fm_e1.data.get_data(ti=t0, tf=t1)['rsamF']#.loc[ft_e1_t]
                inv_rsam = 1./inv_rsam
                ax1b.plot(ft_e1_t, inv_rsam/0.0035, '-', color= 'gray', linewidth=0.5, markersize=0.5, alpha = 1.)
                ax1.plot([], [], '-', color= 'gray', markersize=1, label='1/RSAM', alpha = 1.0)
                ax1b.set_ylim([0,1])
                ax1b.set_yticks([])
                #
                if False:#mov_avg: # plot moving average
                    n=50
                    v_plot = (inv_rsam-np.min(inv_rsam))/np.max((inv_rsam-np.min(inv_rsam)))
                    #ax2b.plot(temp_e1_tim[:-n+1], moving_average(temp_e1_val, n=n), '--', color='k', label='temp. mov. avg.')
                    ax1b.plot(ft_e1_t[n-1-15:-15], moving_average(v_plot[::-1], n=n)[::-1], '-', color='g', label='lake temperature')
                    ax1b.set_yticks([])
            #
            if plot_periods:
                for k, t in enumerate(plot_periods):
                    te = datetimeify(t)#fm_e1.data.tes[int(erup[-1:])-1]
                    ax1.axvline(te, color=plot_periods_col[k], linestyle='-', linewidth=20, alpha = 0.2, zorder = 4)
                    #ax3.plot([], color='k', linestyle='--', linewidth=2, label = 'event')
            #
            if erup_time: # plot vertical lines
                te = datetimeify(erup_time)#fm_e1.data.tes[int(erup[-1:])-1]
                ax1.axvline(te, color='k',linestyle='--', linewidth=2, zorder = 4)
                ax1.plot([], color='k', linestyle='--', linewidth=2, label = 'eruption')
            #
            if plot_data:
                # features
                ax1b = ax1.twinx()
                fts = ['zsc2_vlfF__median','zsc2_lfF__median']
                        
                col = ['gray','gray','b']
                alpha = [1., 1.]
                thick_line = [1., 1.]
                linestyles = ['-', '--']
                #
                for i, ft in enumerate(fts):
                    if 'zsc2_vlfF' in ft:
                        ds = ['zsc2_vlfF'] 
                    if 'zsc2_lfF' in ft:
                        ds = ['zsc2_lfF']
                    if 'zsc2_rsamF' in ft:
                        ds = ['zsc2_rsamF']
                    #
                    if server:
                        path_feat_serv = 'C:\\Users\\aar135\\codes_local_disk\\volc_forecast_tl\\features_bkp\\features_server\\'
                        fm_e1 = ForecastModel(window=win, overlap=1., station =  sta,
                            look_forward=2., data_streams=ds, 
                            feature_dir=path_feat_serv, 
                            savefile_type='pkl') 
                    else:
                        fm_e1 = ForecastModel(window=win, overlap=1., station = sta,
                            look_forward=2., data_streams=ds, savefile_type='csv')
                    ##
                    ft = ft.replace("-",'"')
                    # adding multiple Axes objects
                    ft_e1 = fm_e1.get_features(ti=t0, tf=t1, n_jobs=1, compute_only_features=[ft])
                    # extract values to plot 
                    ft_e1_t = ft_e1[0].index.values
                    ft_e1_v = ft_e1[0].loc[:,ft]
                    #
                    #v_plot = ft_e1_v-np.min(ft_e1_v)/np.max((ft_e1_v-np.min(ft_e1_v)))
                    #v_plot = ft_e1_v/np.max(ft_e1_v)
                    v_plot =  ft_e1_v
                    #
                    #if ft == 'zsc2_dsarF__change_quantiles__f_agg_"var"__isabs_False__qh_0.6__ql_0.4':
                    #    ft = 'nDSAR rate variance'#'DSAR change quantiles (.6-.4) variance'

                    ax1b.plot(ft_e1_t, v_plot, linestyle = linestyles[i], color=col[i], alpha = alpha[i],label='Feature: '+ ft)
                    ax1b.legend(loc = 3)

            #
            ax1.legend(loc = 2)
            #ax1.set_ylim([0,1])
            #
            te = datetimeify(erup_time)#fm_e1.data.tes[int(erup[-1:])-1]
            #ax1.set_xticks([te - 3*day*i for i in range(int(30/3)+1)])#[dat.index.values[0],dat.index.values[-1]])#, ]np.arange(0, len(x)+1, 5))
            #ax1.set_xticks([ft_e1[0].index[-1] - 7*day*i for i in range(int(30/7)+1)])#[dat.index.values[0],dat.index.values[-1]])#, ]np.arange(0, len(x)+1, 5))
            #ax1.set_yticks([])
            
            #ax1b.set_yticks([])
            ax1.grid(color='gray', linestyle='-', linewidth=.3, alpha = 0.5)

            ax1.set_ylabel('feature value')        #ax.set_xticks([ft_e1[0].index[-1]-7*day*i +day for i in range(5)])
            #ax.set_xticks([ft_e1[0].index[-1] - 7*day*i for i in range(int(30/7)+1)])#[dat.index.values[0],dat.index.values[-1]])#, ]np.arange(0, len(x)+1, 5))
        #####################################################
        # subplot one: normalize features
        if True:
            # features
            fts = ['zsc2_lrarF__median']
                    
            col = ['r','g','b']
            col = ['r','g','b','m']
            col = ['g']
            alpha = [1., 1., 1.]
            thick_line = [3., 3., 6.]
            #
            for i, ft in enumerate(fts):
                if 'zsc2_lrarF' in ft:
                    ds = ['zsc2_lrarF'] 
                #
                if server:
                    path_feat_serv = 'C:\\Users\\aar135\\codes_local_disk\\volc_forecast_tl\\features_bkp\\features_server\\'
                    fm_e1 = ForecastModel(window=win, overlap=1., station =  sta,
                        look_forward=2., data_streams=ds, 
                        feature_dir=path_feat_serv, 
                        savefile_type='pkl') 
                else:
                    fm_e1 = ForecastModel(window=win, overlap=1., station = sta,
                        look_forward=2., data_streams=ds, savefile_type='csv')
                ##
                ft = ft.replace("-",'"')
                # adding multiple Axes objects
                ft_e1 = fm_e1.get_features(ti=t0, tf=t1, n_jobs=1, compute_only_features=[ft])
                # extract values to plot 
                ft_e1_t = ft_e1[0].index.values
                ft_e1_v = ft_e1[0].loc[:,ft]
                #
                v_plot = ft_e1_v-np.min(ft_e1_v)/np.max((ft_e1_v-np.min(ft_e1_v)))
                v_plot = ft_e1_v/np.max(ft_e1_v)
                v_plot =  ft_e1_v
                #
                #if ft == 'zsc2_dsarF__change_quantiles__f_agg_"var"__isabs_False__qh_0.6__ql_0.4':
                #    ft = 'nDSAR rate variance'#'DSAR change quantiles (.6-.4) variance'

                ax2.plot(ft_e1_t, v_plot, '-', color=col[i], alpha = alpha[i],label='Feature: '+ ft)
                if plot_inv_rat:
                    ax2b = ax2.twinx()
                    ax2b.plot(ft_e1_t, 1/v_plot, '-', color='gray', alpha = alpha[i],label='Feature: 1/'+ ft)
                    ax2b.legend(loc = 3)
                #
            #
            if ffm: # ffm 
                ax1b = ax1.twinx() 
                #v_plot = data[data_stream].loc[inds]
                inv_rsam = fm_e1.data.get_data(ti=t0, tf=t1)['rsamF']#.loc[ft_e1_t]
                inv_rsam = 1./inv_rsam
                ax1b.plot(ft_e1_t, inv_rsam/0.0035, '-', color= 'gray', linewidth=0.5, markersize=0.5, alpha = 1.)
                ax1.plot([], [], '-', color= 'gray', markersize=1, label='1/RSAM', alpha = 1.0)
                ax1b.set_ylim([0,1])
                ax1b.set_yticks([])
                #
                if False:#mov_avg: # plot moving average
                    n=50
                    v_plot = (inv_rsam-np.min(inv_rsam))/np.max((inv_rsam-np.min(inv_rsam)))
                    #ax2b.plot(temp_e1_tim[:-n+1], moving_average(temp_e1_val, n=n), '--', color='k', label='temp. mov. avg.')
                    ax1b.plot(ft_e1_t[n-1-15:-15], moving_average(v_plot[::-1], n=n)[::-1], '-', color='g', label='lake temperature')
                    ax1b.set_yticks([])
            #
            if plot_periods:
                for k, t in enumerate(plot_periods):
                    te = datetimeify(t)#fm_e1.data.tes[int(erup[-1:])-1]
                    ax1.axvline(te, color=plot_periods_col[k], linestyle='-', linewidth=20, alpha = 0.2, zorder = 4)
                    #ax3.plot([], color='k', linestyle='--', linewidth=2, label = 'event')
            #
            if erup_time: # plot vertical lines
                te = datetimeify(erup_time)#fm_e1.data.tes[int(erup[-1:])-1]
                ax2.axvline(te, color='k',linestyle='--', linewidth=2, zorder = 4)
                ax2.plot([], color='k', linestyle='--', linewidth=2, label = 'eruption')
            #
            if plot_data:
                # features
                ax2b = ax2.twinx()
                fts = ['zsc2_lfF__median','zsc2_rsamF__median']
                        
                col = ['gray','gray','b']
                alpha = [1., 1.]
                thick_line = [1., 1.]
                linestyles = ['-', '--']
                #
                for i, ft in enumerate(fts):
                    if 'zsc2_vlfF' in ft:
                        ds = ['zsc2_vlfF'] 
                    if 'zsc2_lfF' in ft:
                        ds = ['zsc2_lfF']
                    if 'zsc2_rsamF' in ft:
                        ds = ['zsc2_rsamF']
                    #
                    if server:
                        path_feat_serv = 'C:\\Users\\aar135\\codes_local_disk\\volc_forecast_tl\\features_bkp\\features_server\\'
                        fm_e1 = ForecastModel(window=win, overlap=1., station =  sta,
                            look_forward=2., data_streams=ds, 
                            feature_dir=path_feat_serv, 
                            savefile_type='pkl') 
                    else:
                        fm_e1 = ForecastModel(window=win, overlap=1., station = sta,
                            look_forward=2., data_streams=ds, savefile_type='csv')
                    ##
                    ft = ft.replace("-",'"')
                    # adding multiple Axes objects
                    ft_e1 = fm_e1.get_features(ti=t0, tf=t1, n_jobs=1, compute_only_features=[ft])
                    # extract values to plot 
                    ft_e1_t = ft_e1[0].index.values
                    ft_e1_v = ft_e1[0].loc[:,ft]
                    #
                    #v_plot = ft_e1_v-np.min(ft_e1_v)/np.max((ft_e1_v-np.min(ft_e1_v)))
                    #v_plot = ft_e1_v/np.max(ft_e1_v)
                    v_plot =  ft_e1_v
                    #
                    #if ft == 'zsc2_dsarF__change_quantiles__f_agg_"var"__isabs_False__qh_0.6__ql_0.4':
                    #    ft = 'nDSAR rate variance'#'DSAR change quantiles (.6-.4) variance'

                    ax2b.plot(ft_e1_t, v_plot, linestyle = linestyles[i], color=col[i], alpha = alpha[i],label='Feature: '+ ft)
                    ax2b.legend(loc = 3)
            #
            ax2.legend(loc = 2)
            #ax1.set_ylim([0,1])
            #
            te = datetimeify(erup_time)#fm_e1.data.tes[int(erup[-1:])-1]
            #ax1.set_xticks([te - 3*day*i for i in range(int(30/3)+1)])#[dat.index.values[0],dat.index.values[-1]])#, ]np.arange(0, len(x)+1, 5))
            #ax1.set_xticks([ft_e1[0].index[-1] - 7*day*i for i in range(int(30/7)+1)])#[dat.index.values[0],dat.index.values[-1]])#, ]np.arange(0, len(x)+1, 5))
            #ax1.set_yticks([])
            
            #ax1b.set_yticks([])
            ax2.grid(color='gray', linestyle='-', linewidth=.3, alpha = 0.5)

            ax2.set_ylabel('feature value')        #ax.set_xticks([ft_e1[0].index[-1]-7*day*i +day for i in range(5)])        
        #####################################################
        # subplot three: features median ratios
        # features
        if True:
            fts2 = ['zsc2_rmarF__median']
            
            col = ['r','g','b','m']
            col = ['b']
            alpha = [1., 1., 1., 1.]
            thick_line = [3., 3., 3., 3.]
            for i, ft in enumerate(fts2):
                if 'zsc2_rmarF' in ft:
                    ds = ['zsc2_rmarF']
                if server:
                    path_feat_serv = 'C:\\Users\\aar135\\codes_local_disk\\volc_forecast_tl\\features_bkp\\features_server\\'
                    fm_e2 = ForecastModel(window=win, overlap=1., station =  sta,
                        look_forward=2., data_streams=ds, 
                        feature_dir=path_feat_serv, 
                        savefile_type='pkl') 
                else:
                    fm_e2 = ForecastModel(window=win, overlap=1., station = sta,
                        look_forward=2., data_streams=ds, savefile_type='csv')
                #
                ft = ft.replace("-",'"')
                # adding multiple Axes objects
                fm_e2 = fm_e2.get_features(ti=t0, tf=t1, n_jobs=1, compute_only_features=[ft])
                # extract values to plot 
                fm_e2_t = fm_e2[0].index.values
                fm_e2_v = fm_e2[0].loc[:,ft]
                #
                # import datastream to plot 
                ## ax1 and ax2
                if len(fts2) == 1:
                    v_plot = fm_e2_v
                else:
                    v_plot = fm_e2_v#(ft_e1_v-np.min(ft_e1_v))/np.max((ft_e1_v-np.min(ft_e1_v)))
                    #v_plot = ft_e1_v
                if False:
                    if rank == 262:
                        v_plot = v_plot*40
                        v_plot = v_plot - np.mean(v_plot) +.5
                #
                #if ft == 'zsc2_mfF__median':
                #    ft = 'nMF median'
                #
                ax3.plot(fm_e2_t, v_plot, '-', color=col[i], alpha = alpha[i],label='Feature: '+ ft)
                if plot_inv_rat:
                    ax3b = ax3.twinx()
                    ax3b.plot(fm_e2_t, 1/v_plot, '-', color='gray', alpha = alpha[i],label='Feature: 1/'+ ft)
                    ax3b.legend(loc = 3)
            #
            if erup_time: # plot vertical lines
                te = datetimeify(erup_time)#fm_e1.data.tes[int(erup[-1:])-1]
                ax3.axvline(te, color='k',linestyle='--', linewidth=2, zorder = 4)
                ax3.plot([], color='k', linestyle='--', linewidth=2, label = 'eruption')

            if plot_periods:
                for k, t in enumerate(plot_periods):
                    te = datetimeify(t)#fm_e1.data.tes[int(erup[-1:])-1]
                    ax2.axvline(te, color=plot_periods_col[k], linestyle='-', linewidth=20, alpha = 0.2, zorder = 4)
                    #ax3.plot([], color='k', linestyle='--', linewidth=2, label = 'event')
            #
            #
            if plot_data:
                # features
                ax3b = ax3.twinx()
                fts = ['zsc2_rsamF__median','zsc2_mfF__median']
                        
                col = ['gray','gray','b']
                alpha = [1., 1.]
                thick_line = [1., 1.]
                linestyles = ['-', '--']
                #
                for i, ft in enumerate(fts):
                    if 'zsc2_rsamF' in ft:
                        ds = ['zsc2_rsamF'] 
                    if 'zsc2_mfF' in ft:
                        ds = ['zsc2_mfF']
                    #
                    if server:
                        path_feat_serv = 'C:\\Users\\aar135\\codes_local_disk\\volc_forecast_tl\\features_bkp\\features_server\\'
                        fm_e1 = ForecastModel(window=win, overlap=1., station =  sta,
                            look_forward=2., data_streams=ds, 
                            feature_dir=path_feat_serv, 
                            savefile_type='pkl') 
                    else:
                        fm_e1 = ForecastModel(window=win, overlap=1., station = sta,
                            look_forward=2., data_streams=ds, savefile_type='csv')
                    ##
                    ft = ft.replace("-",'"')
                    # adding multiple Axes objects
                    ft_e1 = fm_e1.get_features(ti=t0, tf=t1, n_jobs=1, compute_only_features=[ft])
                    # extract values to plot 
                    ft_e1_t = ft_e1[0].index.values
                    ft_e1_v = ft_e1[0].loc[:,ft]
                    #
                    #v_plot = ft_e1_v-np.min(ft_e1_v)/np.max((ft_e1_v-np.min(ft_e1_v)))
                    #v_plot = ft_e1_v/np.max(ft_e1_v)
                    v_plot =  ft_e1_v
                    #
                    #if ft == 'zsc2_dsarF__change_quantiles__f_agg_"var"__isabs_False__qh_0.6__ql_0.4':
                    #    ft = 'nDSAR rate variance'#'DSAR change quantiles (.6-.4) variance'

                    ax3b.plot(ft_e1_t, v_plot, linestyle = linestyles[i], color=col[i], alpha = alpha[i],label='Feature: '+ ft)
                    ax3b.legend(loc = 3)
            ax3.set_ylabel('feature data')
            ax3.legend(loc = 2)
            te = datetimeify(erup_time)#fm_e1.data.tes[int(erup[-1:])-1]
            #ax2.set_xticks([te - 3*day*i for i in range(int(30/3)+1)])#[dat.index.values[0],dat.index.values[-1]])#, ]np.arange(0, len(x)+1, 5))
            #ax2.set_ylim([1,2])
            #ax2.set_xticks([ft_e1[0].index[-1] - 7*day*i for i in range(int(30/7))])#[dat.index.values[0],dat.index.values[-1]])#, ]np.arange(0, len(x)+1, 5))
            #ax.set_yticks([])
            ax3.grid(color='gray', linestyle='-', linewidth=.3, alpha = 0.5)
        #####################################################
        # subplot four: features median ratios
        # features
        if True:
            fts2 = ['zsc2_dsarF__median']
            
            col = ['r','g','b','m']
            col = ['m']
            alpha = [1., 1., 1., 1.]
            thick_line = [3., 3., 3., 3.]
            for i, ft in enumerate(fts2):
                if 'zsc2_vlarF' in ft:
                    ds = ['zsc2_vlarF']
                if 'zsc2_lrarF' in ft:
                    ds = ['zsc2_lrarF']
                if 'zsc2_rmarF' in ft:
                    ds = ['zsc2_rmarF']
                if 'zsc2_dsarF' in ft:
                    ds = ['zsc2_dsarF']
                if server:
                    path_feat_serv = 'C:\\Users\\aar135\\codes_local_disk\\volc_forecast_tl\\features_bkp\\features_server\\'
                    fm_e2 = ForecastModel(window=win, overlap=1., station =  sta,
                        look_forward=2., data_streams=ds, 
                        feature_dir=path_feat_serv, 
                        savefile_type='pkl') 
                else:
                    fm_e2 = ForecastModel(window=win, overlap=1., station = sta,
                        look_forward=2., data_streams=ds, savefile_type='csv')
                #
                ft = ft.replace("-",'"')
                # adding multiple Axes objects
                fm_e2 = fm_e2.get_features(ti=t0, tf=t1, n_jobs=1, compute_only_features=[ft])
                # extract values to plot 
                fm_e2_t = fm_e2[0].index.values
                fm_e2_v = fm_e2[0].loc[:,ft]
                #
                # import datastream to plot 
                ## ax1 and ax2
                if True:#len(fts2) == 1:
                    v_plot = fm_e2_v
                else:
                    v_plot = fm_e2_v / np.max(fm_e2_v)#(ft_e1_v-np.min(ft_e1_v))/np.max((ft_e1_v-np.min(ft_e1_v)))
                    #v_plot = ft_e1_v
                if False:
                    if rank == 262:
                        v_plot = v_plot*40
                        v_plot = v_plot - np.mean(v_plot) +.5
                #
                #if ft == 'zsc2_mfF__median':
                #    ft = 'nMF median'
                #
                ax4.plot(fm_e2_t, v_plot, '-', color=col[i], alpha = alpha[i],label='Feature: '+ ft)
                if plot_inv_rat:
                    ax4b = ax4.twinx()
                    ax4b.plot(fm_e2_t, 1/v_plot, '-', color='gray', alpha = alpha[i],label='Feature: 1/'+ ft)
                    ax4b.legend(loc = 3)
            #
            if erup_time: # plot vertical lines
                te = datetimeify(erup_time)#fm_e1.data.tes[int(erup[-1:])-1]
                ax4.axvline(te, color='k',linestyle='--', linewidth=2, zorder = 4)
                ax4.plot([], color='k', linestyle='--', linewidth=2, label = 'eruption')

            if plot_periods:
                for k, t in enumerate(plot_periods):
                    te = datetimeify(t)#fm_e1.data.tes[int(erup[-1:])-1]
                    ax4.axvline(te, color=plot_periods_col[k], linestyle='-', linewidth=20, alpha = 0.2, zorder = 4)
                    #ax3.plot([], color='k', linestyle='--', linewidth=2, label = 'event')
            #
            #
            if plot_data:
                # features
                ax4b = ax4.twinx()
                fts = ['zsc2_mfF__median','zsc2_hfF__median']
                        
                col = ['gray','gray','b']
                alpha = [1., 1.]
                thick_line = [1., 1.]
                linestyles = ['-', '--']
                #
                for i, ft in enumerate(fts):
                    if 'zsc2_mfF' in ft:
                        ds = ['zsc2_mfF'] 
                    if 'zsc2_hfF' in ft:
                        ds = ['zsc2_hfF']
                    #
                    if server:
                        path_feat_serv = 'C:\\Users\\aar135\\codes_local_disk\\volc_forecast_tl\\features_bkp\\features_server\\'
                        fm_e1 = ForecastModel(window=win, overlap=1., station =  sta,
                            look_forward=2., data_streams=ds, 
                            feature_dir=path_feat_serv, 
                            savefile_type='pkl') 
                    else:
                        fm_e1 = ForecastModel(window=win, overlap=1., station = sta,
                            look_forward=2., data_streams=ds, savefile_type='csv')
                    ##
                    ft = ft.replace("-",'"')
                    # adding multiple Axes objects
                    ft_e1 = fm_e1.get_features(ti=t0, tf=t1, n_jobs=1, compute_only_features=[ft])
                    # extract values to plot 
                    ft_e1_t = ft_e1[0].index.values
                    ft_e1_v = ft_e1[0].loc[:,ft]
                    #
                    #v_plot = ft_e1_v-np.min(ft_e1_v)/np.max((ft_e1_v-np.min(ft_e1_v)))
                    #v_plot = ft_e1_v/np.max(ft_e1_v)
                    v_plot =  ft_e1_v
                    #
                    #if ft == 'zsc2_dsarF__change_quantiles__f_agg_"var"__isabs_False__qh_0.6__ql_0.4':
                    #    ft = 'nDSAR rate variance'#'DSAR change quantiles (.6-.4) variance'

                    ax4b.plot(ft_e1_t, v_plot, linestyle = linestyles[i], color=col[i], alpha = alpha[i],label='Feature: '+ ft)
                    ax4b.legend(loc = 3)
                    #
            ax4.set_ylabel('feature data')
            ax4.legend(loc = 2)
            te = datetimeify(erup_time)#fm_e1.data.tes[int(erup[-1:])-1]
            #ax2.set_xticks([te - 3*day*i for i in range(int(30/3)+1)])#[dat.index.values[0],dat.index.values[-1]])#, ]np.arange(0, len(x)+1, 5))
            #ax2.set_ylim([1,2])
            #ax2.set_xticks([ft_e1[0].index[-1] - 7*day*i for i in range(int(30/7))])#[dat.index.values[0],dat.index.values[-1]])#, ]np.arange(0, len(x)+1, 5))
            #ax.set_yticks([])
            ax4.grid(color='gray', linestyle='-', linewidth=.3, alpha = 0.5)
        #####################################################
        #####################################################
        # subplot five: features median ratios normalized
        # features
        if True:
            fts2 = ['zsc2_vlarF__median','zsc2_lrarF__median','zsc2_rmarF__median','zsc2_dsarF__median']
            
            col = ['r','g','b','m']
            alpha = [1., 1., 1., 1.]
            thick_line = [3., 3., 3., 3.]
            for i, ft in enumerate(fts2):
                if 'zsc2_vlarF' in ft:
                    ds = ['zsc2_vlarF']
                if 'zsc2_lrarF' in ft:
                    ds = ['zsc2_lrarF']
                if 'zsc2_rmarF' in ft:
                    ds = ['zsc2_rmarF']
                if 'zsc2_dsarF' in ft:
                    ds = ['zsc2_dsarF']
                if server:
                    path_feat_serv = 'C:\\Users\\aar135\\codes_local_disk\\volc_forecast_tl\\features_bkp\\features_server\\'
                    fm_e2 = ForecastModel(window=win, overlap=1., station =  sta,
                        look_forward=2., data_streams=ds, 
                        feature_dir=path_feat_serv, 
                        savefile_type='pkl') 
                else:
                    fm_e2 = ForecastModel(window=win, overlap=1., station = sta,
                        look_forward=2., data_streams=ds, savefile_type='csv')
                #
                ft = ft.replace("-",'"')
                # adding multiple Axes objects
                fm_e2 = fm_e2.get_features(ti=t0, tf=t1, n_jobs=1, compute_only_features=[ft])
                # extract values to plot 
                fm_e2_t = fm_e2[0].index.values
                fm_e2_v = fm_e2[0].loc[:,ft]
                #
                # import datastream to plot 
                ## ax1 and ax2
                if False:#len(fts2) == 1:
                    v_plot = fm_e2_v
                else:
                    v_plot = fm_e2_v / np.max(fm_e2_v)#(ft_e1_v-np.min(ft_e1_v))/np.max((ft_e1_v-np.min(ft_e1_v)))
                    #v_plot = ft_e1_v
                if False:
                    if rank == 262:
                        v_plot = v_plot*40
                        v_plot = v_plot - np.mean(v_plot) +.5
                #
                #if ft == 'zsc2_mfF__median':
                #    ft = 'nMF median'
                #
                ax5.plot(fm_e2_t, v_plot, '-', color=col[i], alpha = alpha[i],label='Feature: '+ ft)
            #
            if erup_time: # plot vertical lines
                te = datetimeify(erup_time)#fm_e1.data.tes[int(erup[-1:])-1]
                ax5.axvline(te, color='k',linestyle='--', linewidth=2, zorder = 4)
                ax5.plot([], color='k', linestyle='--', linewidth=2, label = 'eruption')

            if plot_periods:
                for k, t in enumerate(plot_periods):
                    te = datetimeify(t)#fm_e1.data.tes[int(erup[-1:])-1]
                    ax5.axvline(te, color=plot_periods_col[k], linestyle='-', linewidth=20, alpha = 0.2, zorder = 4)
                    #ax3.plot([], color='k', linestyle='--', linewidth=2, label = 'event')
            #
            ax5.set_ylabel('normalized data')
            ax5.legend(loc = 2)
            te = datetimeify(erup_time)#fm_e1.data.tes[int(erup[-1:])-1]
            #ax2.set_xticks([te - 3*day*i for i in range(int(30/3)+1)])#[dat.index.values[0],dat.index.values[-1]])#, ]np.arange(0, len(x)+1, 5))
            #ax2.set_ylim([1,2])
            #ax2.set_xticks([ft_e1[0].index[-1] - 7*day*i for i in range(int(30/7))])#[dat.index.values[0],dat.index.values[-1]])#, ]np.arange(0, len(x)+1, 5))
            #ax.set_yticks([])
            ax5.grid(color='gray', linestyle='-', linewidth=.3, alpha = 0.5)
        #
        ax1.set_xlim([t0-int(d_back/4.2)*day,t1])
        ax2.set_xlim([t0-int(d_back/4.2)*day,t1])
        ax3.set_xlim([t0-int(d_back/4.2)*day,t1])
        ax4.set_xlim([t0-int(d_back/4.2)*day,t1])
        ax5.set_xlim([t0-int(d_back/4.2)*day,t1])
        #
        #ax1.grid(color='gray', linestyle='-', linewidth=.3, alpha = 0.5)
        #ax2.grid(color='gray', linestyle='-', linewidth=.3, alpha = 0.5)
        #ax3.grid(color='gray', linestyle='-', linewidth=.3, alpha = 0.5)
        #ax4.grid(color='gray', linestyle='-', linewidth=.3, alpha = 0.5)
        #####################################################
        fig.suptitle(sta_code[sta]+': '+str(t0)+' to '+str(t1))#'Feature: '+ ft_nm_aux, ha='center')
        plt.tight_layout()
        path = '..'+os.sep+'features'+os.sep+'correlations'+os.sep+'comb_feat_analysis'+os.sep
        #plt.savefig(path+erup+'_'+ft_id+'.png')
        plt.show()
        plt.close()
    ##
    _plt_intp(sta,t0, t1)
    # 
#
def exp_ratios():
    '''
    plot 4 ratios median calc as moving average
    '''
    # plot 
    nrow = 4
    ncol = 1
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(nrows=nrow, ncols=ncol,figsize=(12,12))#(14,4))
    # subplot one: normalize features
    col = ['b','g','r','m']
    alpha = [1., .5, .5]
    thick_line = [2., 1., 1.]
    l_forw = 0
    N, M = [2,180]
    #sta_arch = 'FWVZ'
    sta_arch = 'WIZ'
    #sta_arch = 'GOD'
    #sta_arch = 'VONK'
    # sta_arch = 'PVV'
    #sta_arch = 'VNSS'
    # sta_arch = 'REF'
    # sta_arch = 'AUS'
    # sta_arch = 'MEA01'
    # sta_arch = 'FWVZ'
    #sta_arch = 'GOD'
    #
    if sta_arch == 'WIZ':
        erup = -1
    if sta_arch == 'GOD':
        erup = 1
    if sta_arch == 'VONK':
        erup = 0
    if sta_arch == 'PVV':
        erup = 2
    if sta_arch == 'VNSS':
        erup = 1
    if sta_arch == 'REF':
        erup = 0
    if sta_arch == 'AUS':
        erup = 2
    if sta_arch == 'MEA01':
        erup = 5
    if sta_arch == 'FWVZ':
        erup = 0
    if sta_arch == 'BELO':
        erup = 3#1
    #
    date = False
    if date: # select date
        ref_time = datetimeify('2022-05-11 00:00:00')

    
    # plot 1: WIZ (precursor, reference)
    if True: # DSAR
        ## DSAR median 
        day = timedelta(days=1)
        #sta_arch = 'WIZ'
        dt = 'zsc2_dsarF'
        fm = ForecastModel(window=2., overlap=1., station=sta_arch,
            look_forward=2., data_streams=[dt], 
            data_dir=r'C:\Users\aar135\codes_local_disk\volc_forecast_tl\volc_forecast_tl\data'
            )
        #
        te = fm.data.tes[erup]
        if date:
            te = ref_time
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
        ft = 'nDSAR median'
        ax1.plot(_times, _val, '-', color=col[0], alpha = alpha[0],linewidth=thick_line[0], label=' '+ ft,zorder=1)

    if True: # RMAR
        ## DSAR median 
        day = timedelta(days=1)
        #sta_arch = 'WIZ'
        dt = 'zsc2_rmarF'
        fm = ForecastModel(window=2., overlap=1., station=sta_arch,
            look_forward=2., data_streams=[dt], 
            data_dir=r'C:\Users\aar135\codes_local_disk\volc_forecast_tl\volc_forecast_tl\data'
            )
        #
        te = fm.data.tes[erup]
        if date:
            te = ref_time
        # rolling median and signature length window
        #N, M = [2,15]
        #l_forw = 2
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
        ft = 'nRMAR median'
        ax2.plot(_times, _val, '-', color=col[1], alpha = alpha[0],linewidth=thick_line[0], label=' '+ ft,zorder=1)

    if True: # LRAR
        ## DSAR median 
        day = timedelta(days=1)
        #sta_arch = 'WIZ'
        dt = 'zsc2_lrarF'
        fm = ForecastModel(window=2., overlap=1., station=sta_arch,
            look_forward=2., data_streams=[dt], 
            data_dir=r'C:\Users\aar135\codes_local_disk\volc_forecast_tl\volc_forecast_tl\data'
            )
        #
        te = fm.data.tes[erup]
        if date:
            te = ref_time
        # rolling median and signature length window
        #N, M = [2,15]
        #l_forw = 2
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
        ft = 'nLRAR median'
        ax3.plot(_times, _val, '-', color=col[2], alpha = alpha[0],linewidth=thick_line[0], label=' '+ ft,zorder=1)

    if True: # VLAR
        ## D
        day = timedelta(days=1)
        #sta_arch = 'WIZ'
        dt = 'zsc2_vlarF'
        fm = ForecastModel(window=2., overlap=1., station=sta_arch,
            look_forward=2., data_streams=[dt], 
            data_dir=r'C:\Users\aar135\codes_local_disk\volc_forecast_tl\volc_forecast_tl\data'
            )
        #
        te = fm.data.tes[erup]
        if date:
            te = ref_time
        # rolling median and signature length window
        #N, M = [2,15]
        #l_forw = 2
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
        ft = 'nVLAR median'
        ax4.plot(_times, _val, '-', color=col[3], alpha = alpha[0],linewidth=thick_line[0], label=' '+ ft,zorder=1)

    # plot eruption 
    for ax in [ax1, ax2, ax3, ax4]:
        ax.axvline(te, color='r',linestyle='-', linewidth=3, zorder = 0)
        ax.plot([], color='r', linestyle='-', linewidth=3, label = 'eruption')

    ## 
    ax1.legend(loc=2)
    ax2.legend(loc=2)
    ax3.legend(loc=2)
    ax4.legend(loc=2)
    #
    ax1.set_ylabel('nDSAR median')
    ax2.set_ylabel('nRMAR median')
    ax3.set_ylabel('nLRAR median')
    ax4.set_ylabel('nVLAR median')
    #
    ax1.grid(color='gray', linestyle='-', linewidth=.3, alpha = 0.5)
    ax2.grid(color='gray', linestyle='-', linewidth=.3, alpha = 0.5)
    ax3.grid(color='gray', linestyle='-', linewidth=.3, alpha = 0.5)
    ax4.grid(color='gray', linestyle='-', linewidth=.3, alpha = 0.5)
    #
    plt.show()

def ratios_mult_erup():
    '''
    plot 5 eruptions  ratios median calc as moving average
    '''
    # plot 
    # nrow = 5
    # ncol = 1
    # fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(nrows=nrow, ncols=ncol,figsize=(12,12))#(14,4))
    # subplot one: normalize features
    col = ['b','g','r','m']
    alpha = [1., 1., 1., 1,]
    thick_line = [1., 1., 1., 1.]
    l_forw = 0
    N, M = [2,65]
    max_plot = 2
    #
    #erup_2plot = [['WIZ',-1],['FWVZ',1],['FWVZ',2],['GOD',1],['VONK',0]]
    #titles = ['Whakaari 2019', 'Ruapehu 2006', 'Ruapehu 2007', 'Eyjafjallajökull 2010a', 'Holuhraun 2014a']
    erup_2plot = [['GOD',1],['VONK',0],['REF',0],['FWVZ',0],['FWVZ',1]]
    titles = ['Eyjafjallajökull 2010a', 'Holuhraun 2014a' , 'Redoubt 2010','Ruapehu 2006','Ruapehu 2007']
    #erup_2plot = [['WIZ',-1],['WIZ',-2],['WIZ',-3],['WIZ',-4],['WIZ',-5]]
    #titles = ['Whakaari 2019', 'Whakaari 2016', 'Whakaari 2013', 'Whakaari 2013', 'Whakaari 2012']
    # ref_time = datetimeify('2022-05-11 00:00:00')
    # plot 
    nrow = len(erup_2plot)
    ncol = 1
    fig, axs = plt.subplots(nrows=nrow, ncols=ncol)
    #
    for i, ax in enumerate(axs.reshape(-1)): 
        #for erup in erup_2plot:
        #
        sta_arch = erup_2plot[i][0]
        erup = erup_2plot[i][1] 
        #
        # plot 1: WIZ (precursor, reference)
        if True: # DSAR
            ## DSAR median 
            day = timedelta(days=1)
            #sta_arch = 'WIZ'
            dt = 'zsc2_dsarF'
            fm = ForecastModel(window=2., overlap=1., station=sta_arch,
                look_forward=2., data_streams=[dt], 
                data_dir=r'C:\Users\aar135\codes_local_disk\volc_forecast_tl\volc_forecast_tl\data'
                )
            #
            te = fm.data.tes[erup] 
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
            ft = 'nDSAR median'
            ax.plot(_times, _val, '-', color=col[0], alpha = alpha[0], linewidth=thick_line[0], label=' '+ ft,zorder=1)

        if True: # RMAR
            ## DSAR median 
            day = timedelta(days=1)
            #sta_arch = 'WIZ'
            dt = 'zsc2_rmarF'
            fm = ForecastModel(window=2., overlap=1., station=sta_arch,
                look_forward=2., data_streams=[dt], 
                data_dir=r'C:\Users\aar135\codes_local_disk\volc_forecast_tl\volc_forecast_tl\data'
                )
            #
            te = fm.data.tes[erup]  
            # rolling median and signature length window
            #N, M = [2,15]
            #l_forw = 2
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
            ft = 'nRMAR median'
            if _val_max > max_plot:
                twin1 = ax.twinx()
                twin1.plot(_times, _val, '-', color=col[1], alpha = alpha[0],linewidth=thick_line[0], label=' '+ ft, zorder=1)
                ax.plot([], [], '-', color=col[1], alpha = alpha[0],linewidth=thick_line[0], label=' '+ ft, zorder=1)

        if True: # LRAR
            ## DSAR median 
            day = timedelta(days=1)
            #sta_arch = 'WIZ'
            dt = 'zsc2_lrarF'
            fm = ForecastModel(window=2., overlap=1., station=sta_arch,
                look_forward=2., data_streams=[dt], 
                data_dir=r'C:\Users\aar135\codes_local_disk\volc_forecast_tl\volc_forecast_tl\data'
                )
            #
            te = fm.data.tes[erup]  
            # rolling median and signature length window
            #N, M = [2,15]
            #l_forw = 2
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
            ft = 'nLRAR median'
            if _val_max > max_plot:
                twin2 = ax.twinx()
                twin2.plot(_times, _val, '-', color=col[2], alpha = alpha[0],linewidth=thick_line[0], label=' '+ ft,zorder=1)
                ax.plot([], [], '-', color=col[2], alpha = alpha[0],linewidth=thick_line[0], label=' '+ ft, zorder=1)

        if True: # VLAR
            ## 
            day = timedelta(days=1)
            #sta_arch = 'WIZ'
            dt = 'zsc2_vlarF'
            fm = ForecastModel(window=2., overlap=1., station=sta_arch,
                look_forward=2., data_streams=[dt], 
                data_dir=r'C:\Users\aar135\codes_local_disk\volc_forecast_tl\volc_forecast_tl\data'
                )
            #
            te = fm.data.tes[erup]  
            # rolling median and signature length window
            #N, M = [2,15]
            #l_forw = 2
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
            ft = 'nVLAR median'
            if _val_max > max_plot:
                twin3 = ax.twinx()
                twin3.plot(_times, _val, '-', color=col[3], alpha = alpha[0],linewidth=thick_line[0], label=' '+ ft,zorder=1)
                ax.plot([], [], '-', color=col[3], alpha = alpha[0],linewidth=thick_line[0], label=' '+ ft, zorder=1)

        # plot eruption 
        te = fm.data.tes[erup]
        ax.axvline(te, color='r',linestyle='-', linewidth=3, zorder = 0)
        ax.plot([], color='r', linestyle='-', linewidth=3, label = 'eruption')
        ## 
        ax.legend(loc=2)
        #
        ax.set_ylabel('Ratio median')
        #
        ax.grid(color='gray', linestyle='-', linewidth=.3, alpha = 0.5)
        #
        ax.set_title(titles[i])
        #
    #plt.tight_layout()
    plt.show()
    
    #
    asdf    

def plot_medians():
    '''
    plot dsar, hf and mf medians
    '''
    # plot 
    nrow = 2
    ncol = 1
    fig, (ax1, ax2) = plt.subplots(nrows=nrow, ncols=ncol,figsize=(12,12))#(14,4))
    # subplot one: normalize features
    col = ['b','g','r','m']
    alpha = [1., .5, .5]
    thick_line = [2., 1., 1.]
    l_forw = 0
    N, M = [2,28]
    sta_arch = 'WIZ'
    #
    if sta_arch == 'WIZ':
        erup = -1

    # plot 1: WIZ (precursor, reference)
    if True: # DSAR
        ## DSAR median 
        day = timedelta(days=1)
        #sta_arch = 'WIZ'
        dt = 'zsc2_dsarF'
        fm = ForecastModel(window=2., overlap=1., station=sta_arch,
            look_forward=2., data_streams=[dt], 
            data_dir=r'C:\Users\aar135\codes_local_disk\volc_forecast_tl\volc_forecast_tl\data'
            )
        #
        te = fm.data.tes[erup]
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
        ft = 'nDSAR median'
        ax1.plot(_times, _val, '-', color='g', alpha = alpha[0],linewidth=thick_line[0]+1, label=' '+ ft,zorder=1)

    if True: # HF and MF
        ax1b = ax1.twinx()
        ## MF median 
        day = timedelta(days=1)
        #sta_arch = 'WIZ'
        dt = 'zsc2_mfF'
        fm = ForecastModel(window=2., overlap=1., station=sta_arch,
            look_forward=2., data_streams=[dt], 
            data_dir=r'C:\Users\aar135\codes_local_disk\volc_forecast_tl\volc_forecast_tl\data'
            )
        #
        te = fm.data.tes[erup]
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
        ft = 'MF median'
        ax1b.plot(_times, _val, '-', color='r', alpha = alpha[0],linewidth=thick_line[0], label=' '+ ft,zorder=1)
        ax1.plot([], [], '-', color='r', alpha = alpha[0],linewidth=thick_line[0], label=' '+ ft,zorder=1)

        ## HF median 
        day = timedelta(days=1)
        #sta_arch = 'WIZ'
        dt = 'zsc2_hfF'
        fm = ForecastModel(window=2., overlap=1., station=sta_arch,
            look_forward=2., data_streams=[dt], 
            data_dir=r'C:\Users\aar135\codes_local_disk\volc_forecast_tl\volc_forecast_tl\data'
            )
        #
        te = fm.data.tes[erup]
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
        ft = 'HF median'
        ax1b.plot(_times, _val, '-', color='b', alpha = alpha[0],linewidth=thick_line[0], label=' '+ ft,zorder=1)
        ax1.plot([], [], '-', color='b', alpha = alpha[0],linewidth=thick_line[0], label=' '+ ft,zorder=1)

    # plot data
    if True:          
        td = TremorData(station = sta_arch)
        #td.update(ti=t0, tf=t1)
        data_streams = ['rsam', 'mf', 'hf']#, 'dsarF']
        label = ['RSAM','MF','HF']#,'DSAR']
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
        _range = [te - M*day,te + l_forw*day]
        log =False
        col_def = None
        data = td.get_data(*_range)
        xi = datetime(year=1,month=1,day=1,hour=0,minute=0,second=0)
        cols = ['gray','r','b','m',[0.5,0.5,0.5],[0.75,0.75,0.75]]
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
                    ax2.plot(data.index[inds], v_plot, '-', color=col_def, label=label[i], alpha = 1.0)
                else:
                    ax2.plot(data.index[inds], v_plot, '-', color=col, label=label[i], alpha = 1.0)
            else:
                ax2.plot(data.index[inds], v_plot, '-', color=col, label=data_stream, alpha = 1.0)
            i+=1
        #
        ax2.set_yscale('log')

    # plot eruption 
    for ax in [ax1, ax2]:
        ax.axvline(te, color='k',linestyle='--', linewidth=3, zorder = 0)
        ax.plot([], color='k', linestyle='--', linewidth=3, label = 'eruption')

    ## 
    ax1.legend(loc=2)
    ax2.legend(loc=2)
    #
    ax1.set_ylabel('DSAR')
    ax1b.set_ylabel('HF and MF')
    ax2.set_ylabel(r'$\mu$m/s')
    #
    ax1.set_xticks([])
    ax2.set_xticks([])

    ax1.set_xticks([te-0*day,te-7*day,te-14*day,te-21*day])
    ax2.set_xticks([te-0*day,te-7*day,te-14*day,te-21*day])
    #
    ax1.grid(color='gray', linestyle='-', linewidth=.3, alpha = 0.5)
    ax2.grid(color='gray', linestyle='-', linewidth=.3, alpha = 0.5)
    #
    plt.show()
#
def main():
    #
    #plot_data_ratios()
    #plot_ratios()
    # 
    #exp_ratios()
    #ratios_mult_erup()
    #
    ## fig marsden
    plot_medians()
    

if __name__ == "__main__":
    main()

