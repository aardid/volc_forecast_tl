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
            'GOD_2' : 'Eyjafjallajökull 2010b',
            'VONK_1' : 'Holuhraun 2014a'
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
            'DAM' : 'Kawa Ijen',
            'VONK' : 'Holuhraun',
            'BOR' : 'Piton de la Fournaise',
            'VRLE' : 'Rincon de la Vega',
            'T01' : 'Tungurahua',
            'COP' : 'Copahue'
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
            'POS_1' : '2013 03 20 12 00 00',
            'DAM_1' : '2013 03 20 12 00 00',
            'VONK_1': '2014 08 29 12 00 00',
            'T01_1': '2015 8 15 12 00 00',
            'COP_1' : '2020 06 16 12 00 00'
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
def conv_set_arch(_archetype_set, feat = 'median'):
    '''
    Function that convolutes a set of archetype over the records of multiple stations.
    input: archtype set [ite, [count, dt, _archtype, sta, te, stas]]
        [n, dt, , archtype, sta, te, stas]
        n: ref number
        dt: datastream and feature
        archetype: short signal to be convoluted
        sta: archetype station 
        te: time of the end of the archetype 
        stas: list of station for convolutions (records) 
        ind_samp: pop out one day every 30, so independent samples 

    output: cc, pv save to a temporal txt 

    '''
    # whole record to convolute with 
    pv_samp_in = []
    cc_samp_in  = []
    _pv_samp_in = [] 
    _cc_samp_in = [] 
    #
    ite_ref =  _archetype_set[0] + 1
    archetype_set = _archetype_set[1] # [count, dt, _archtype, sta, te, stas]

    for _archetype in archetype_set:
        _n = _archetype[0]
        dt = _archetype[1]
        feat = _archetype[2]
        archtype = _archetype[3]
        sta = _archetype[4]
        te = _archetype[5]
        stas = _archetype[6]
        ind_samp = _archetype[7]
        #
        cc_te = []
        cc_non_te = []
        ds = [dt]
        # loop over volcanoes and extract cc and pvals
        pop_in = []
        pop_out = []
        tests = []
        archetypes = []
        archetypes_out = []
        #
        for sta in stas: 
            #
            if sta == 'WIZ':
                endtime = datetimeify("2021-06-30")
                years_back = 10 # WIZ 10; VNSS 7 ; FWVZ 14
                look_back = years_back*365 + 3 # days, years back from endtime (day by day)
            if sta == 'FWVZ':
                endtime = datetimeify("2020-12-31")#("2020-12-31")
                years_back = 14 # WIZ 10; VNSS 7 ; FWVZ 14
                look_back = years_back*365 # days, years back from endtime (day by day) 
            if sta == 'KRVZ':
                endtime = datetimeify("2020-12-31")#("2020-12-31")
                years_back = 15 # WIZ 10; VNSS 7 ; FWVZ 14
                look_back = years_back*365 # days, years back from endtime (day by day) 
            if sta == 'VNSS':
                endtime = datetimeify("2019-12-31")# 
                years_back = 3 # WIZ 10; VNSS 7 ; FWVZ 14
                look_back = years_back*365 - 181 # days, years back from endtime (day by day) 
            if sta == 'PVV':
                endtime = datetimeify("2016-06-30")# 
                years_back = 2 # WIZ 10; VNSS 7 ; FWVZ 14
                look_back = years_back*365 + 120 # days, years back from endtime (day by day) 
            if sta == 'BELO':
                endtime = datetimeify("2008-05-07")# 
                years_back = 0.4 # WIZ 10; VNSS 7 ; FWVZ 14
                look_back = int(years_back*365) # days, years back from endtime (day by day) 
                        
            # vector of days
            _vec_days = [endtime - day*i for i in range(look_back)]
            #
            if ind_samp:
                win_d_length = 30 #
                _vec_days = [endtime - day*win_d_length*i for i in range(int(look_back/win_d_length))]
            #

            if False:
                fm = ForecastModel(window=2., overlap=1., station = sta,
                look_forward=2., data_streams=ds, savefile_type='csv')
            else:
                path_feat_serv = 'C:\\Users\\aar135\\codes_local_disk\\volc_forecast_tl\\features_bkp\\features_server\\'
                fm = ForecastModel(window=2., overlap=1., station = sta,
                    look_forward=2., data_streams=ds, 
                    feature_dir=path_feat_serv, 
                    savefile_type='pkl')

            #
            dt = ds[0]#'zsc2_dsarF'
            # rolling median and signature length window
            N, M = [2,30]
            # time
            j = fm.data.df.index
            # construct signature
            df = fm.data.df[(j>(te-(M+N)*day))&(j<te)]
            # convolve over the station data
            df = fm.data.df[:]#[(j<te)]
            #test = df[dt].rolling(N*24*6).median()[N*24*6:]
            #out = test.rolling(archtype.shape[0]).apply(partial(conv, (archtype-archtype.mean())/archtype.std()))
            #
            if feat == 'median':
                test = df[dt].rolling(N*24*6).median()[N*24*6:]
            if feat == 'rate_var':
                test = df[dt].rolling(N*24*6).apply(chqv)[N*24*6:]
            out = test.rolling(archtype.shape[0]).apply(partial(conv, (archtype-archtype.mean())/archtype.std()))
            #
            # cc in eruption times
            _cc_te = []
            #_samp_in = [samp_in[k] for k in range(len(samp_in)) if k != l]
            for _te in fm.data.tes[:]:
                _cc_te.append(out[out.index.get_loc(_te, method='nearest')])
            _cc_te = np.array(_cc_te)
            #
            cc_te =  np.concatenate((cc_te, _cc_te), axis=0)
            #
            # save non-eruptive cc values 
            _pop_out = []
            # 
            pop_rej = []
            for e in fm.data.tes[:]:
                #e = datetime.date(e.year, e.month, e.day)
                _e = e.replace(hour=00, minute=00)
                n_days_before = 60
                _vdays = [_e - n_days_before*day + i*day for i in range(n_days_before+30)] # 2 weeks before and 1 week after   
                pop_rej =  pop_rej + _vdays

            # construct out of eruption population
            for d in _vec_days: 
                if d not in pop_rej:
                    _pop_out.append(d)
            # cc in eruption times
            _cc_non_te = []
            #_samp_in = [samp_in[k] for k in range(len(samp_in)) if k != l]
            for _te in _pop_out:
                _cc_non_te.append(out[out.index.get_loc(_te, method='nearest')])
            _cc_non_te = np.array(_cc_non_te)
            #
            cc_non_te =  np.concatenate((cc_non_te, _cc_non_te), axis=0)
            #
    
        # correct cc_te by removing the value of the archetypr with itself 
        cc_te = cc_te[cc_te < 0.95]
        _cc_non_te = _cc_non_te[_cc_non_te < 0.95]
        ## (4) calulate p-value for the sample 
        # 2-sample Kolmogorov Smirnov test for difference in underlying distributions
        from scipy.stats import kstest
        #a = out.iloc[archtype.shape[0]::24*6].values
        #pv = kstest(cc_te, out.iloc[archtype.shape[0]::24*6].values).pvalue
        pv = kstest(cc_te, _cc_non_te).pvalue
        #
        _pv_samp_in.append(pv)
        [_cc_samp_in.append(cc_te[i]) for i in range(len(cc_te))]
        #  
    #
    pv_samp_in = pv_samp_in + _pv_samp_in
    cc_samp_in = cc_samp_in + _cc_samp_in

    path = '..'+os.sep+'features'+os.sep+'correlations'+os.sep+'stat_sign_per_hig_corr_feat'+os.sep+'_temp'+os.sep
    with open(path+str(ite_ref)+'_pv'+'.txt', 'w') as f:
        for k in range(len(pv_samp_in)):
            f.write(str(k+1)+'\t'+str(pv_samp_in[k])+'\n')
    # write cc from samples
    path = '..'+os.sep+'features'+os.sep+'correlations'+os.sep+'stat_sign_per_hig_corr_feat'+os.sep+'_temp'+os.sep
    with open(path+str(ite_ref)+'_cc'+'.txt', 'w') as f:
        for k in range(len(cc_samp_in)):
            f.write(str(k+1)+'\t'+str(cc_samp_in[k])+'\n')

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
def import_data():
    '''
    Download tremor data from an specific station. 
    Available stations : 
    'PVV','VNSS','SSLW','OKWR','REF','BELO','CRPO','VTUN','KRVZ','FWVZ','WIZ','AUR','AUS','IVGP'
    '''
    if False: # plot raw vel data
        #
        from obspy import UTCDateTime
        #
        from obspy.clients.fdsn import Client
        client = Client("GEONET")
        from obspy import UTCDateTime
        #t = UTCDateTime("2012-02-27T00:00:00.000")
        starttime = UTCDateTime("2021-06-15")
        endtime = UTCDateTime("2021-07-15")
        inventory = client.get_stations(network="AV", station="PVV", starttime=starttime, endtime=endtime)
        st = client.get_waveforms(network = "AV", station = "PVV", location = None, channel = "EHZ", starttime=starttime, endtime=endtime)
        st.plot()  
        asdf
    # 
    if True: # download between two dates or update
        sta = 'FWVZ'
        #t0 = "2021-01-01"
        #t1 = "2022-03-22"
        td = TremorData(station = sta)
        #td.update(ti=t0, tf=t1, n_jobs = 4)
        td.update()
    #
    if False:  # download between multiple pair of  dates
        sta = 'OGDI'
        times = [['2017-06-13','2017-07-15'],
                ['2016-12-31','2017-02-01'],
                ['2016-08-10','2016-09-12'],
                ['2016-04-25','2016-05-27'],
                ['2015-01-03','2015-02-05'],
                ['2014-05-20','2014-06-22']]
        for t in times:
            try:
                t0,t1 = t
                td = TremorData(station = sta)
                td.update(ti=t0, tf=t1, n_jobs = 2)
            except:
                print('dont worked \n')

    #td.update()

def data_Q_assesment():
    '''
    Check section of interpolated data
    '''
    # plot data
    if True:
        # constants
        month = timedelta(days=365.25/12)
        day = timedelta(days=1)
        station = 'POS'#'MEA01'#'FWVZ'#'WIZ'#'FWVZ'
        # read raw data
        td = TremorData(station = station)
        #t0 = "2007-08-22"
        #t1 = "2007-09-22"
        #td.update(ti=t0, tf=t1)

        # plot data 
        #td.plot( data_streams = ['rsamF'])#(ti=t0, tf=t1)
        t0 = "2012-04-01"#"2009-06-11"#"2019-11-11"#"2021-09-07"#"2010-03-15"#"2021-09-07"
        t1 = "2012-02-01"#"2009-07-13"#"2019-12-09"#"2021-09-09"#"2010-06-07"#"2021-09-09" 
        #td.update(ti=t0, tf=t1)
        data_streams = ['hfF', 'mfF', 'rsamF', 'dsarF']
        label = ['HF','MF','RSAM','DSAR']
        td.plot_zoom(data_streams = data_streams, range = [t0,t1], label = label, norm= True, log =True)# label = 'HF data, Whakaari 2019', col_def = 'm')
        plt.show()
        # interpolated data 
        #t0 = "2015-01-01"
        #t1 = "2015-02-01"
        #td.plot_intp_data(range_dates = None)
        #td.plot_intp_data(range_dates = [t0,t1])

    # statistics
    if False:
        # constants
        month = timedelta(days=365.25/12)
        day = timedelta(days=1)
        station = 'WIZ'#'FWVZ'
        # read raw data
        td = TremorData(station = station)
        t0 = "2011-12-01"#"2021-09-07"
        t1 = "2020-01-01"#"2021-09-09" 
        #
        td.get_data(ti=t0, tf=t1)
        #td._compute_transforms()
        vec = td.df['dsarF'].values
        # log data
        log10dat = np.log10(vec)
        mean, std = np.mean(log10dat), np.std(log10dat) 
        # normalized data
        normdat = (log10dat - mean) / std 
        #vec = 10**vec
        # stats
        mean, std = np.mean(normdat), np.std(normdat) 
        asdaf

def calc_feature_pre_erup():
    ''' 
    Load data and calculate features in parallel for multiple stations, multiple datastreams, and multiple window sizes.
    Overlap is set to 1.0 (max) 
    '''
    ## data streams
    #ds = ['zsc2_rsamF','zsc2_mfF','zsc2_hfF','zsc2_dsarF','diff_zsc2_rsamF','diff_zsc2_mfF','diff_zsc2_hfF','diff_zsc2_dsarF',
    #    'log_zsc2_rsamF','log_zsc2_mfF','log_zsc2_hfF','log_zsc2_dsarF']
    ds = ['zsc2_rsamF', 'zsc2_hfF','zsc2_mfF','zsc2_dsarF']
    #ds = ['zsc2_vlarF', 'zsc2_lrarF','zsc2_rmarF','zsc2_dsarF']
    ## stations
    #ss = ['PVV','VNSS','SSLW','OKWR','REF','BELO','CRPO','VTUN','KRVZ','FWVZ','WIZ','AUR']
    ss = ['PVV','VNSS','KRVZ','FWVZ','WIZ','BELO'] # ,'SSLW'
    ss = ['FWVZ']#,'DAM']
    ## days looking backward from eruptions 
    lbs = [30]
    ## Run parallelization 
    ps = []
    #
    if True: # serial 
        for s in ss:
            print(s)
            for d in ds:
                for lb in lbs:
                    p = [lb, s, d]
                    calc_one(p)
    else: # parallel`
        print('Calculating features')
        for s in ss:
            for d in ds:
                for lb in lbs:
                    ps.append([lb,s,d])
        n_jobs = 2 # number of cores
        p = Pool(n_jobs)
        p.map(calc_one, ps)

def calc_one(p):
    ''' p = [weeks before eruption, station, datastream] 
    Load HQ data (by calculating features if need) before (p[0] days) every eruption in station given in p[1] for datastreams p[2]. 
    (auxiliary function for parallelization)
    '''
    lb,s,d = p
    #fm = ForecastModel(window=2., overlap=1., station = s,
    #    look_forward=2., data_streams=[d], feature_dir='/media/eruption_forecasting/eruptions/features/', savefile_type='pkl') 
    fm = ForecastModel(window=2., overlap=1., station = s,
        look_forward=2., data_streams=[d], savefile_type='csv')
    if False: # run pre define eruptions
        a = fm.data.tes
        for etime in fm.data.tes:
            ti = etime - lb*day
            tf = etime + 2*day
            fm._load_data(ti, tf, None)
    else: # select time of eruption 
        ti = datetimeify("2022-01-01")
        tf = datetimeify("2022-04-09")#tf - 1200*day
        print(ti)
        print(tf)
        fm._load_data(ti, tf, None)

def plot_feats():
    #
    t0 = "2009-02-20"#"2019-11-10"#'2009-06-14'#'2009-06-15'#'2019-11-09'#"2021-03-01"#"2005-11-10"#"2021-08-10"#'2021-09-18'
    t1 = "2009-03-15"#"2019-12-09"#'2009-07-13'#'2009-07-13'#'2019-12-09'#"2021-03-31"#"2005-12-16"#"2021-09-09"#'2021-10-17'
    #
    sta = 'REF'
    #
    # features
    fts = ['zsc2_hfF__fft_coefficient__coeff_38__attr_"real"',
            'zsc2_dsarF__change_quantiles__f_agg_"var"__isabs_False__qh_0.6__ql_0.4',
            'zsc2_dsarF__median'
            ]
    fts = ['zsc2_mfF__median', 'zsc2_hfF__median', 'zsc2_rsamF__median']
    fts = ['zsc2_hfF__fft_coefficient__coeff_38__attr_"real"', 
            'zsc2_mfF__fft_coefficient__coeff_38__attr_"real"', 
            'zsc2_rsamF__fft_coefficient__coeff_38__attr_"real"']
    fts = ['zsc2_dsarF__median']
    #
    def _plt_erup_mult_feat(sta, t0, t1, fts):
        #
        nrow = 1
        ncol = 1
        fig, ax = plt.subplots(nrows=nrow, ncols=ncol,figsize=(10,2))#(14,4))
        #
        
        if len(fts) == 1:
            if fts == ['zsc2_dsarF__median']:
                col = ['b']
            if fts == ['zsc2_dsarF__change_quantiles__f_agg_"var"__isabs_False__qh_0.6__ql_0.4']:
                col = ['g']
        else:
            col = ['r','g','b']

        alpha = [.5, .5, .5]
        thick_line = [3., 3., 3.]
        for i, ft in enumerate(fts):
            if 'zsc2_dsarF' in ft:
                ds = ['zsc2_dsarF']
            if 'zsc2_rsamF' in ft:
                ds = ['zsc2_rsamF']
            if 'zsc2_mfF' in ft:
                ds = ['zsc2_mfF']
            if 'zsc2_hfF' in ft:
                ds = ['zsc2_hfF']
            if False:
                path_feat_serv = 'C:\\Users\\aar135\\codes_local_disk\\volc_forecast_tl\\features_bkp\\features_server\\'
                fm_e1 = ForecastModel(window=2., overlap=1., station =  sta,
                    look_forward=2., data_streams=ds, 
                    feature_dir=path_feat_serv, 
                    savefile_type='pkl') 
            else:
                fm_e1 = ForecastModel(window=2., overlap=1., station = sta,
                    look_forward=2., data_streams=ds, savefile_type='csv')
            #
            ft = ft.replace("-",'"')
            # adding multiple Axes objects
            ft_e1 = fm_e1.get_features(ti=t0, tf=t1, n_jobs=1, compute_only_features=[ft])
            # extract values to plot 
            ft_e1_t = ft_e1[0].index.values
            ft_e1_v = ft_e1[0].loc[:,ft]
            #
            # import datastream to plot 
            ## ax1 and ax2
            if len(fts) == 1:
                v_plot = ft_e1_v
            else:
                v_plot = ft_e1_v#(ft_e1_v-np.min(ft_e1_v))/np.max((ft_e1_v-np.min(ft_e1_v)))
                #v_plot = ft_e1_v
            if False:
                if rank == 262:
                    v_plot = v_plot*40
                    v_plot = v_plot - np.mean(v_plot) +.5
            #
            if ft == 'zsc2_dsarF__change_quantiles__f_agg_"var"__isabs_False__qh_0.6__ql_0.4':
                ft = 'DSAR rate variance'#'DSAR change quantiles (.6-.4) variance'
            if ft == 'zsc2_dsarF__median':
                ft = 'DSAR median'
            if ft == 'zsc2_hfF__fft_coefficient__coeff_38__attr_"real"':
                ft = '75-minute HF harmonic' #'HF Fourier coefficient 38'
            if ft == 'zsc2_mfF__fft_coefficient__coeff_38__attr_"real"':
                ft = '75-minute MF harmonic' #'HF Fourier coefficient 38'
            if ft == 'zsc2_rsamF__fft_coefficient__coeff_38__attr_"real"':
                ft = '75-minute RSAM harmonic' #'HF Fourier coefficient 38'
            #
            if ft == 'zsc2_mfF__median':
                ft = 'MF median'
            if ft == 'zsc2_hfF__median':
                ft = 'HF median'
            if ft == 'zsc2_rsamF__median':
                ft = 'RSAM median'
            #
            ax.plot(ft_e1_t, v_plot, '-', color=col[i], alpha = alpha[i],label='Feature: '+ ft)
        # plot data on the back
        if False: 
            pass
        #
        if False: # plot vertical lines
            #te = datetimeify("2021 10 17 00 00 00")#fm_e1.data.tes[int(erup[-1:])-1]
            te = datetimeify("2009 07 13 00 00 00")#fm_e1.data.tes[int(erup[-1:])-1]
            #te = datetimeify("2021 09 07 22 11 00")#fm_e1.data.tes[int(erup[-1:])-1]
            #te = datetimeify("2019 12 09 01 11 00")#fm_e1.data.tes[int(erup[-1:])-1]
            ax.axvline(te, color='k',linestyle='--', linewidth=2, zorder = 4)
            ax.plot([], color='k', linestyle='--', linewidth=2, label = 'event')
            # te = datetimeify("2005 12 10 00 00 00")#fm_e1.data.tes[int(erup[-1:])-1]
            # ax.axvline(te, color='k',linestyle='--', linewidth=2, zorder = 4)
            # te = datetimeify("2005 12 12 00 00 00")#fm_e1.data.tes[int(erup[-1:])-1]
            # ax.axvline(te, color='k',linestyle='--', linewidth=2, zorder = 4)
            # te = datetimeify("2005 12 15 00 00 00")#fm_e1.data.tes[int(erup[-1:])-1]
            # ax.axvline(te, color='k',linestyle='--', linewidth=2, zorder = 4)
            # ax.plot([], color='k', linestyle='--', linewidth=2, label = 'rsam peak')
        #
        if False: # ffm 
            ax2 = ax.twinx() 
            #v_plot = data[data_stream].loc[inds]
            inv_rsam = fm_e1.data.get_data(ti=t0, tf=t1)['rsamF']#.loc[ft_e1_t]
            inv_rsam = 1./inv_rsam
            ax2.plot(ft_e1_t, inv_rsam, '-', color= 'gray', markersize=1, alpha = 0.6)
            ax.plot([], [], '-', color= 'gray', markersize=1, label='1/rsam', alpha = 0.6)
            #
            if False:#mov_avg: # plot moving average
                n=50
                v_plot = (inv_rsam-np.min(inv_rsam))/np.max((inv_rsam-np.min(inv_rsam)))
                #ax2b.plot(temp_e1_tim[:-n+1], moving_average(temp_e1_val, n=n), '--', color='k', label='temp. mov. avg.')
                ax2.plot(ft_e1_t[n-1-15:-15], moving_average(v_plot[::-1], n=n)[::-1], '-', color='g', label='lake temperature')
            
        if False: # plot extras     
            pass
        #
        #te = datetimeify("2021 09 08 14 00 00")#fm_e1.data.tes[int(erup[-1:])-1]
        #ax.axvline(te, color='k',linestyle='-', linewidth=2, zorder = 4)
        #ax.plot([], color='k', linestyle='-', linewidth=2, label = 'reported activity')
        
        ax.legend(loc = 2)
        #ax.set_ylim([-.0,1.3])
        #ax.set_yticks([])
        ax.grid()
        #ax.set_ylabel('feature value')        #ax.set_xticks([ft_e1[0].index[-1]-7*day*i +day for i in range(5)])
        #ax.set_xticks([ft_e1[0].index[-1] - 7*day*i for i in range(int(30/7)+1)])#[dat.index.values[0],dat.index.values[-1]])#, ]np.arange(0, len(x)+1, 5))
        #
        #ax.set_xticks([ft_e1[0].index[-1] - 2*day*i for i in range(int(30/2)+1)])#[dat.index.values[0],dat.index.values[-1]])#, ]np.arange(0, len(x)+1, 5))
        fig.suptitle(sta_code[sta]+': '+str(t0)+' to '+str(t1))#'Feature: '+ ft_nm_aux, ha='center')
        #plt.tight_layout()
        #plt.show()
        path = '..'+os.sep+'features'+os.sep+'correlations'+os.sep+'comb_feat_analysis'+os.sep
        #plt.savefig(path+erup+'_'+ft_id+'.png')
        plt.show()
        plt.close()
    #
    _plt_erup_mult_feat(sta,t0, t1, fts)
    # 

def corr_feat_calc():
    ''' Correlation analysis between features calculated for multiple volcanoes 
        considering 1 month before their eruptions.
        Correlations are performed between multiple eruptions (for several stations)
        for common features derived from multple data streams.
        Correlations are perfomed using pandas Pearson method.
        Results are saved as .csv files and .png in ../features/correlations/
    '''
    # method 
    kendall = False # need to be change in calc_one_corr() too. 
    spearman = True # need to be change in calc_one_corr() too. 
    ## stations (volcanoes)
    ss = ['WIZ','FWVZ','KRVZ','PVV','VNSS','BELO'] # ,'SSLW'
    ## data streams
    ds = ['log_zsc2_rsamF', 'zsc2_hfF','zsc2_mfF','zsc2_dsarF']
    ds = ['zsc2_dsarF']
    ## days looking backward from eruptions 
    lbs = 30
    # auxiliary df to extract feature names 
    fm = ForecastModel(window=2., overlap=1., station = 'WIZ',
        look_forward=2., data_streams=ds, savefile_type='csv')
    
    ti = fm.data.tes[0]
    tf = fm.data.tes[0] + day
    fm.get_features(ti=ti, tf=tf, n_jobs=2, drop_features=[], compute_only_features=[])
    
    # drop features
    # remove duplicate linear features (because correlated), unhelpful fourier compoents
    # and fourier harmonics too close to Nyquist
    ## ref: DD removes
    ##drop_features = ['linear_trend_timewise','agg_linear_trend','*attr_"imag"*','*attr_"real"*',
    ##    '*attr_"angle"*']  #*'isabs_True'*
    ##freq_max = fm.dtw//fm.dt//4
    ##drop_features += ['*fft_coefficient__coeff_{:d}*'.format(i) for i in range(freq_max+1, 2*freq_max+2)]
    # 
    drop_features = ['linear_trend_timewise','*attr_"imag"*','*isabs_True*','*attr_"angle"*'] 
    fm.fM = fm._drop_features(fm.fM, drop_features)
    # list of features names
    ftns = fm.fM.columns[:] 

    # directory to be saved
    try:
        if kendall:
            os.mkdir('..'+os.sep+'features'+os.sep+'correlations_kendall')
        elif spearman:
            os.mkdir('..'+os.sep+'features'+os.sep+'correlations_spearman')
        else:
            os.mkdir('..'+os.sep+'features'+os.sep+'correlations')
    except:
        pass
    # timming
    import timeit
    tic=timeit.default_timer()
    #
    if False: # serial 
        # loop over features 
        for j in range(4):#range(len(ftns)):
            print(str(j)+'/'+str(len(ftns))+' : '+ftns[j])
            p = j, ftns[j], ss, ds, lbs
            calc_one_corr(p)
    else: # parallel
        n_jobs = 4 # number of cores
        if True:
            print('Parallel')
            print('Number of features: '+str(len(ftns)))
            print('Time when run:')
            print(datetime.now())
            print('Estimated run time (hours):')
            print(len(ftns)/n_jobs * (231/3600))
        #
        ps = []
        for j in range(len(ftns)):
            p = j, ftns[j], ss, ds, lbs
            ps.append(p)
        #
        p = Pool(n_jobs)
        p.map(calc_one_corr, ps)
    # end timming
    toc=timeit.default_timer()
    print(toc - tic) # print elapsed time in seconds
    
def calc_one_corr(p):
    'auxiliary funtion for parallelization inside function corr_ana_feat()'
    # method 
    kendall = False
    spearman =  True
    j, ftn, ss, ds, lbs = p 
    # path to new files 
    if kendall:
        path = '..'+os.sep+'features'+os.sep+'correlations_kendall'+os.sep+'corr_'+str(j+1)+'_'+ftn.replace('"','-')
    elif spearman:
        path = '..'+os.sep+'features'+os.sep+'correlations_spearman'+os.sep+'corr_'+str(j+1)+'_'+ftn.replace('"','-')
    else:
        path = '..'+os.sep+'features'+os.sep+'correlations'+os.sep+'corr_'+str(j+1)+'_'+ftn.replace('"','-')

    if os.path.isfile(path+'.csv'):
        return
    # aux funtion 
    def _load_feat_erup_month(fm_a, ft_n, i):
        ti = fm_a.data.tes[i] - lbs*day # intial time
        tf = fm_a.data.tes[i] # final time
        fm_a.get_features(ti=ti, tf=tf, n_jobs=None, drop_features=[], compute_only_features=[])
        #fn = fm_a.fM.columns[0]   # name feature one
        #f2c = fm_a.fM[f2n]       # whole column feature one (df)
        fv = fm_a.fM[ft_n].values      # whole column feature one (np array)
        return fv
    #
    data = []
    lab = []
    # loop over stations 
    for s in ss:
        fm_aux = ForecastModel(window=2., overlap=1., station = s,
            look_forward=2., data_streams=ds, savefile_type='csv')
        # loop over eruptions of station
        for e in range(len(fm_aux.data.tes)):
            data.append(_load_feat_erup_month(fm_aux, ftn, e))
            lab.append(fm_aux.station+'_'+str(e+1))
    data = np.vstack((data)).T
    # create aux pandas obj for correlation
    df = pd.DataFrame(data=data, index=fm_aux.fM.index, columns=lab, dtype=None, copy=None)
    # find the correlation among the columns
    #print(df.corr(method ='pearson'))
    # create heatmap figure for feature
    fig, ax = plt.subplots(figsize=(10, 10))
    # correlate and save csv
    if kendall:
        df_corr = df.corr(method='kendall')
    elif spearman:
        df_corr = df.corr(method='spearman')
    else:
        df_corr = df.corr(method='pearson')
    df_corr.to_csv(path+'.csv')#, index=fm_aux.fM.index)
    # save png
    sns.heatmap(df_corr, vmin=-1.0, vmax=1.0, annot=True, fmt='.2f', 
                cmap=plt.get_cmap('coolwarm'), cbar=True, ax=ax)
    ax.set_title(ftn)
    ax.set_yticklabels(ax.get_yticklabels(), rotation="horizontal")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(path+'.png', 
        bbox_inches='tight', pad_inches=0.0, dpi = 100)
    del df, df_corr, data, lab, fm_aux
    plt.close('all')

def corr_feat_analysis():
    ' Analysis of feature correlation results'
    ## heatmap of feature correlations between eruptions (single)
    if False:
        # create heatmap figure for feature
        fig, ax = plt.subplots(figsize=(10, 10))
        # correlate and save csv
        # files list (list)
        #fl_nm = 'corr_1379_zsc2_mfF__fft_coefficient__coeff_74__attr_-abs-'
        #ft_nm = 'zsc2_mfF__fft_coefficient__coeff_74__attr_-abs-'
        fl_nm = 'corr_1673_zsc2_dsarF__change_quantiles__f_agg_-var-__isabs_False__qh_0.6__ql_0.4'
        ft_nm = 'zsc2_dsarF__change_quantiles__f_agg_-var-__isabs_False__qh_0.6__ql_0.4'
        #
        fl_nm = 'corr_2014_zsc2_dsarF__median'
        ft_nm = 'zsc2_dsarF__median'
        path_files = '..'+os.sep+'features'+os.sep+'correlations'+os.sep+fl_nm+'.csv'
        #
        df_aux = pd.read_csv(path_files, index_col=0)
        df_aux = df_aux.abs()
        # save png
        #sns.heatmap(df_aux, vmin=-1.0, vmax=1.0, annot=True, fmt='.2f', 
        #            cmap=plt.get_cmap('coolwarm'), cbar=True, ax=ax)
        sns.heatmap(df_aux, vmin=0., vmax=.8, annot=True, fmt='.2f', 
            cmap=plt.get_cmap('YlGnBu'), cbar=True, ax=ax)
        ax.set_title('Feature: '+ft_nm)
        _ticks = [erup_dict[df_aux.index.values[i]] for i in range(len(df_aux.index.values))]
        ax.set_yticklabels(_ticks, rotation="horizontal")
        ax.set_xticklabels(_ticks, rotation=90)
        #ax.set_yticklabels(ax.get_yticklabels(), rotation="horizontal")
        #plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(path_files[:-4]+'_2.png', 
            bbox_inches='tight', pad_inches=0.0, dpi = 300)
        plt.show()
        del df_aux
        plt.close('all')         

    ## Marginal sum of cc between eruptions (sum of corr for every feature)
    if False: 
        # remove previous 'csv' output
        try:
            os.remove('..'+os.sep+'features'+os.sep+'correlations'+os.sep+'corr_0_sum_cum.csv')
        except:
            pass
        # files list (list)
        path_files = '..'+os.sep+'features'+os.sep+'correlations'+os.sep+'*.csv'
        pos_ast = path_files.find('*')
        file_dir = glob.glob(path_files)
        if 'corr_0_rank_erup_cc.csv' in file_dir[0]:
            file_dir = file_dir[1:]
        # import first file
        df_sum = pd.read_csv(file_dir[0], index_col=0)
        # loop over files (starting from the second) (.csv created for each feature correlation)
        count_nan = 0
        for i in range(2,len(file_dir),1):
            # add values from file to cummulative dataframe
            df_aux = pd.read_csv(file_dir[i], index_col=0)
            # sum df that aren't nan
            if df_aux.isnull().values.any():# math.isnan(df_aux.iat[0,0]): 
                #print('DataFrame is empty: '+ file_dir[i])
                count_nan+=1
            else:
                df_sum = abs(df_sum) + abs(df_aux)
        # modified diagonal of df_sum
        #np.fill_diagonal(df_sum.values, 1)
        np.fill_diagonal(df_sum.values, np.nan)
        df_sum = df_sum / (len(file_dir)-count_nan)
        # save csv and png
        path = '..'+os.sep+'features'+os.sep+'correlations'+os.sep+'corr_0_sum_cum'
        # create heatmap figure for feature
        fig, ax = plt.subplots(figsize=(12, 12))
        sns.heatmap(df_sum, annot=True, fmt='.2f', 
                    cmap=plt.get_cmap('YlGnBu'), cbar=True, ax=ax)
        ax.set_title('Cumulative average correlation for '+str(len(file_dir)-count_nan)+' features')
        ax.set_yticklabels(ax.get_yticklabels(), rotation="horizontal")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(path+'.png', 
            bbox_inches='tight', pad_inches=0.0, dpi = 100)
        # modified diagonal of df_sum
        np.fill_diagonal(df_sum.values, 0)
        df_sum.to_csv(path+'.csv')#, index=fm_aux.fM.index)
        del df_sum, df_aux
        plt.close('all')

    ## Sum of cc by feature
    if False:
        # remove previous 'csv' output
        try:
            os.remove('..'+os.sep+'features'+os.sep+'correlations'+os.sep+'corr_0_feat_sum_rank.csv')
        except:
            pass
        # files list (list)
        path_files = '..'+os.sep+'features'+os.sep+'correlations'+os.sep+'*.csv'

        pos_ast = path_files.find('*')
        file_dir = glob.glob(path_files)
        # remove cum sum .csv file if loaded
        if file_dir[0].split('\\')[-1] == 'corr_0_sum_cum.csv':
            file_dir = file_dir[1:]

        ## write txt with feature code and name 
        def _ft_name(a):
            '''Aux function to split the file name into the ft name.
            input (example):
            '..\\features\\correlations\\corr_1000_zsc2_hfF__fft_coefficient__coeff_14__attr_-imag-.csv'
            output:
            [1000, 'zsc2_hfF fft_coefficient coeff_14 attr_-imag-']
            '''
            a = a.split('\\')[-1]
            a = a.replace('.csv','')
            a = a.strip()
            a = a.split("__")
            n = int(a[0].split("_")[1])
            b = a[0].split("_")[2:]
            b = '_'.join(b)
            a[0] = b
            a = ' '.join(a)
            return n, a
        ft_id_nm = [_ft_name(file_dir[i]) for i in range(len(file_dir))]
        ft_id_nm = sorted(ft_id_nm, key=lambda x: x[0], reverse=False)
        # 
        path_aux = '..'+os.sep+'features'+os.sep+'correlations'+os.sep+'corr_0_feat_code_name'
        with open(path_aux+'.txt', 'w') as f:
            f.write("feature code, feature name\n")
            for line in ft_id_nm:
                f.write(str(line[0])+", ")
                f.write(line[1]+"\n")
        # dict of feat id and name 
        ft_id_nm_dic = {}
        for i in range(len(ft_id_nm)):
            ft_id_nm_dic[ft_id_nm[i][0]] = ft_id_nm[i][1]    

        # loop over files (starting from the second) (.csv created for each feature correlation)
        count_nan = 0
        # array of containing correlation mean and std for each feature
        feat_sum = []
        for i in range(0,len(file_dir),1):
            ## add values from file to cummulative dataframe
            #read feature file
            df_aux = pd.read_csv(file_dir[i], index_col=0)
            # sum df that aren't nan
            if df_aux.isnull().values.any():# math.isnan(df_aux.iat[0,0]): 
                #print('DataFrame is empty: '+ file_dir[i])
                count_nan+=1
            else:
                # import name (first cell)
                ftn = ft_id_nm[i][0]#str(i)
                #
                np.fill_diagonal(df_aux.values, 0)
                np_aux = np.abs(df_aux.to_numpy())
                feat_sum.append([ftn, np.mean(np_aux), np.std(np_aux)])
        
        if True: # sort list by mean value
            feat_sum = sorted(feat_sum, key=lambda x: x[1], reverse=True)
        # save csv and png
        path = '..'+os.sep+'features'+os.sep+'correlations'+os.sep+'corr_0_feat_sum_rank'
        with open(path+'.txt', 'w') as f:
            f.write("feature, mean, std \n")
            for line in feat_sum:
                f.write(str(line[0])+", ")
                f.write(str(line[1])+", ")
                f.write(str(line[2])+"\n")  

        ## figure 1: top features with name
        fig, (ax1) = plt.subplots(1, 1, figsize=(8, 8))
        ## ax2: create figure of top features
        # features to plot (cluster of features higher than 0.2)
        _ = [feat_sum[j][1] for j in range(len(file_dir)-count_nan-1)]
        n_to_plot = sum(i>.22 for i in _) #20
        means = [feat_sum[i][1] for i in range(n_to_plot)]
        ftns = [ft_id_nm_dic[int(feat_sum[i][0])] for i in range(n_to_plot)] # feature name
        #ftns = [str(feat_sum[i][0]) for i in range(n_to_plot)] # code number
        ftns_n = [n_to_plot - i for i in range(n_to_plot)]
        stds = [feat_sum[i][2]/10. for i in range(n_to_plot)]
        ax1.errorbar(means, ftns_n, xerr =stds, fmt='o')
        #ax.text(0.05, 18, ftns[0], horizontalalignment='center',
        #    verticalalignment='center', transform=ax.transAxes)
        [ax1.text(means[i]-0.002, ftns_n[i]+0.2, ftns[i], size = None) for i in range(n_to_plot)]
        #
        ax1.set_yticklabels([])
        ax1.set_xlim([0.2,.5])
        #plt.yticks(ftns)
        # ticks feat_sum[:,0]
        ax1.set_title('Ranking top correlated features')
        ax1.set_xlabel('Correlation coeficient (CC)')
        #plt.savefig(path+'_zoom_'+str(n_to_plot)+'.png', 
        #    bbox_inches='tight', pad_inches=0.0, dpi = 300)
        plt.tight_layout()
        plt.savefig(path+'_name.png', 
            bbox_inches='tight', pad_inches=0.0, dpi = 300)    
    
        ## figure 2: ranking all features and top with id
        fig, (ax2,ax1) = plt.subplots(1, 2, figsize=(8, 8))
        ## ax2: create figure of top features
        # features to plot (cluster of features higher than 0.2)
        _ = [feat_sum[j][1] for j in range(len(file_dir)-count_nan-1)]
        n_to_plot = sum(i>.22 for i in _) #20
        means = [feat_sum[i][1] for i in range(n_to_plot)]
        #ftns = [ft_id_nm_dic[int(feat_sum[i][0])] for i in range(n_to_plot)] # feature name
        ftns = [str(feat_sum[i][0]) for i in range(n_to_plot)] # code number
        ftns_n = [n_to_plot - i for i in range(n_to_plot)]
        stds = [feat_sum[i][2]/10. for i in range(n_to_plot)]
        ax1.errorbar(means, ftns_n, xerr =stds, fmt='o')
        #ax.text(0.05, 18, ftns[0], horizontalalignment='center',
        #    verticalalignment='center', transform=ax.transAxes)
        [ax1.text(means[i]-0.002, ftns_n[i]+0.2, ftns[i], size = None) for i in range(n_to_plot)]
        #
        ax1.set_yticklabels([])
        #plt.yticks(ftns)
        # ticks feat_sum[:,0]
        ax1.set_title('Ranking top correlated features')
        ax1.set_xlabel('Correlation coeficient (CC)')
        #plt.savefig(path+'_zoom_'+str(n_to_plot)+'.png', 
        #    bbox_inches='tight', pad_inches=0.0, dpi = 300)

        ## ax1: create figure of all features
        #fig, ax = plt.subplots(figsize=(5, 10))
        n_to_plot_full = len(file_dir)-count_nan-1
        means_f = [feat_sum[i][1] for i in range(n_to_plot_full)]
        ftns_f = [str(feat_sum[i][0]) for i in range(n_to_plot_full)]
        ftns_n_f = [n_to_plot_full - i for i in range(n_to_plot_full)]
        stds_f = [feat_sum[i][2] for i in range(n_to_plot_full)]
        ax2.errorbar(means_f, ftns_n_f, xerr =stds_f, fmt='-', alpha = 0.1, color = 'skyblue', label = 'Feature CC std. dev.')
        ax2.plot(means_f, ftns_n_f, 'o', color = 'skyblue', zorder = 3, label = 'Feature CC average')
        #ax.text(0.05, 18, ftns[0], horizontalalignment='center',
        #    verticalalignment='center', transform=ax.transAxes)
        #[ax.text(means[i]-0.002, ftns_n[i]+0.2, ftns[i], size = 'large') for i in range(n_to_plot)]
        ax2.errorbar(means, ftns_n_f[0:len(means)], xerr =stds, fmt='-', alpha = 0.1, color = 'b')#, label = 'Std. Dev.')
        ax2.plot(means, ftns_n_f[0:len(means)], 'bo', zorder = 3, label = 'Top feature by CC average')        
        #
        ax2.set_yticklabels([])
        ax2.set_xlabel('Correlation coeficient (CC)')
        ax2.set_ylabel('Features')
        #plt.yticks(ftns)
        # ticks feat_sum[:,0]
        ax2.set_title('Ranking all features')
        ax2.legend(loc = 4)
        #ax.set_yticklabels(ax.get_yticklabels(), rotation="horizontal")
        #plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(path+'.png', 
            bbox_inches='tight', pad_inches=0.0, dpi = 300)
        #plt.show()
        del df_aux
        plt.close('all')

    ## Sum of cc bv data streem
    if False:
        ## histrogram for each data stream 
        # lists to fill
        rsam_l = []
        mf_l = []
        hf_l = []
        dsar_l = []
        # import list of files 
        # files list (list)
        path_files = '..'+os.sep+'features'+os.sep+'correlations'+os.sep+'*.csv'
        pos_ast = path_files.find('*')
        file_dir = glob.glob(path_files)
        # remove cum sum .csv file if loaded
        if file_dir[0].split('\\')[-1] == 'corr_0_feat_sum_rank.csv':
            file_dir = file_dir[1:]
        if file_dir[0].split('\\')[-1] == 'corr_0_sum_cum.csv':
            file_dir = file_dir[1:]
        # loop over cc resutls files (importing them inside)
        count_rsam = 0
        count_mf = 0
        count_hf = 0
        count_dsar = 0
        # aux function to import cc values from a certain feat cc matrix
        #def _cc_mat
        max = False # import the max values from each matrix

        for i in range(0,len(file_dir),1):
            #read feature file
            df_aux = pd.read_csv(file_dir[i], index_col=0) 
            # sum df that aren't nan
            if df_aux.isnull().values.any():# math.isnan(df_aux.iat[0,0]): 
                #print('DataFrame is empty: '+ file_dir[i])
                #count_nan+=1
                pass
            else: 
                np.fill_diagonal(df_aux.values, 0)
                if 'rsam' in file_dir[i]:
                    if not max:
                        rsam_l.append(np.mean(np.abs(df_aux.to_numpy())))
                    else:
                        rsam_l.append(np.max(np.abs(df_aux.to_numpy())))
                    count_rsam += 1
                elif 'mf' in file_dir[i]:
                    if not max:
                        mf_l.append(np.mean(np.abs(df_aux.to_numpy())))
                    else:
                        mf_l.append(np.max(np.abs(df_aux.to_numpy())))
                    count_mf += 1
                elif 'hf' in file_dir[i]:
                    if not max:
                        hf_l.append(np.mean(np.abs(df_aux.to_numpy())))
                    else:
                        hf_l.append(np.max(np.abs(df_aux.to_numpy())))
                    count_hf += 1
                elif 'dsar' in file_dir[i]:
                    if not max:
                        dsar_l.append(np.mean(np.abs(df_aux.to_numpy())))
                    else:
                        dsar_l.append(np.max(np.abs(df_aux.to_numpy())))
                    count_dsar += 1
                else:
                    raise

        # plot histograms 
        hist_half = True

        fig, (ax1, ax2, ax3, ax4) = plt.subplots(nrows = 4, ncols = 1, figsize = (4,12))
        xlim = None#[0,0.25]
        ylim = None#[0,70]
        if hist_half:
            xlim = [0.075, 0.25]
            xlim = None
        colors = ['b', 'r', 'g', 'm']

        if hist_half:
            per = 20 # %, percentile cut -> 100
            med = np.percentile(rsam_l,per)
            rsam_l = [val for val in rsam_l if val> med]
            med = np.percentile(mf_l,per)
            mf_l = [val for val in mf_l if val> med]
            med = np.percentile(hf_l,per)
            hf_l = [val for val in hf_l if val> med]
            med = np.percentile(dsar_l,per)
            dsar_l = [val for val in dsar_l if val> med]
        
        # rsam
        n_bins = int(np.sqrt(len(rsam_l)))
        ax1.hist(rsam_l, n_bins, histtype='bar', color = colors[0], edgecolor='#E6E6E6', label = 'rsam')
        ax1.set_xlabel('cc', fontsize=textsize)
        ax1.set_ylabel('samples', fontsize=textsize)
        ax1.grid(True, which='both', linewidth=0.1)
        ax1.set_xlim(xlim)
        ax1.set_ylim(ylim)
        #ax1.set_title('RSAM', fontsize = textsize)
        ax1.plot([np.median(rsam_l), np.median(rsam_l)],[0, count_rsam+ count_rsam*.1],'k--', label = 'median: '+str(round(np.median(rsam_l),2)))
        ax1.legend(loc = 1)

        # mf
        n_bins = int(np.sqrt(len(mf_l)))
        ax2.hist(mf_l, n_bins, histtype='bar', color = colors[1], edgecolor='#E6E6E6', label = 'mf')
        ax2.set_xlabel('cc', fontsize=textsize)
        ax2.set_ylabel('samples', fontsize=textsize)
        ax2.grid(True, which='both', linewidth=0.1)
        ax2.set_xlim(xlim)
        ax2.set_ylim(ylim)
        #ax2.set_title('MF', fontsize = textsize)
        ax2.plot([np.median(mf_l), np.median(mf_l)],[0, count_mf + count_mf*.1],'k--', label = 'median: '+str(round(np.median(mf_l),2)))
        ax2.legend(loc = 1)

        # hf
        n_bins = int(np.sqrt(len(hf_l)))
        ax3.hist(hf_l, n_bins, histtype='bar', color = colors[2], edgecolor='#E6E6E6', label = 'hf')
        ax3.set_xlabel('cc', fontsize=textsize)
        ax3.set_ylabel('samples', fontsize=textsize)
        ax3.grid(True, which='both', linewidth=0.1)
        ax3.set_xlim(xlim)
        ax3.set_ylim(ylim)
        #ax3.set_title('HF', fontsize = textsize)
        ax3.plot([np.median(hf_l), np.median(hf_l)],[0, count_hf+ count_hf*.1],'k--', label = 'median: '+str(round(np.median(hf_l),2)))
        ax3.legend(loc = 1)

        # dsar
        n_bins = int(np.sqrt(len(dsar_l)))
        ax4.hist(dsar_l, n_bins, histtype='bar', color = colors[3], edgecolor='#E6E6E6', label = 'dsar')
        ax4.set_xlabel('cc', fontsize=textsize)
        ax4.set_ylabel('samples', fontsize=textsize)
        ax4.grid(True, which='both', linewidth=0.1)
        ax4.set_xlim(xlim)
        ax4.set_ylim(ylim)
        #ax3.set_title('DSAR', fontsize = textsize)
        ax4.plot([np.median(dsar_l), np.median(dsar_l)],[0,  count_dsar+ count_dsar*.1],'k--', label = 'median: '+str(round(np.median(dsar_l),2)))
        ax4.legend(loc = 1)

        # save fig
        plt.tight_layout()
        path = '..'+os.sep+'features'+os.sep+'correlations'+os.sep+'corr_0_datastream_hist_cc'
        if hist_half:
            fig.suptitle('Top correlation by data stream '+str(100-per)+'%', fontsize=textsize)
            plt.tight_layout()
            if not max:
                plt.savefig(path+'_'+str(100-per)+'.png', 
                    bbox_inches='tight', pad_inches=0.0, dpi = 300)
            else:
                plt.savefig(path+'_max_'+str(100-per)+'.png', 
                        bbox_inches='tight', pad_inches=0.0, dpi = 300)
        else:
            fig.suptitle('Correlation by data stream', fontsize=textsize)
            plt.tight_layout()
            if not max:
                plt.savefig(path+'.png', 
                    bbox_inches='tight', pad_inches=0.0, dpi = 300)
            else:
                plt.savefig(path+'_max.png', 
                    bbox_inches='tight', pad_inches=0.0, dpi = 300)
        #plt.show()
        del df_aux
        plt.close('all')

    # .csv with rank of cc eruptions (sample max from each feat matrix)
    if False:
        #
        # files list (list)
        path_files = '..'+os.sep+'features'+os.sep+'correlations'+os.sep+'*.csv'
        pos_ast = path_files.find('*')
        file_dir = glob.glob(path_files)
        # remove cum sum .csv file if loaded
        if file_dir[0].split('\\')[-1] == 'corr_0_feat_sum_rank.csv':
            file_dir = file_dir[1:]
        if file_dir[0].split('\\')[-1] == 'corr_0_rank_erup_cc.csv':
            file_dir = file_dir[1:]
        if file_dir[0].split('\\')[-1] == 'corr_0_sum_cum.csv':
            file_dir = file_dir[1:]
        #
        rank_l = [] # cols: rank, e1, e2, cc, feat name
        for i in range(len(file_dir)):
            #read feature file
            df_aux = pd.read_csv(file_dir[i], index_col=0) 
            np.fill_diagonal(df_aux.values, 0)
            # sum df that aren't nan
            if df_aux.isnull().values.any():# math.isnan(df_aux.iat[0,0]): 
                #print('DataFrame is empty: '+ file_dir[i])
                #count_nan+=1
                pass
            else:
                ## fill list. cols: rank, e1, e2, cc, feat id, feat name
                def _ft_name(a):
                    '''Aux function to split the file name into the ft name.
                    input (example):
                    '..\\features\\correlations\\corr_1000_zsc2_hfF__fft_coefficient__coeff_14__attr_-imag-.csv'
                    output:
                    [1000, 'zsc2_hfF fft_coefficient coeff_14 attr_-imag-']
                    '''
                    a = a.split('\\')[-1]
                    a = a.replace('.csv','')
                    a = a.strip()
                    a = a.split("__")
                    n = int(a[0].split("_")[1])
                    b = a[0].split("_")[2:]
                    b = '_'.join(b)
                    a[0] = b
                    a = ' '.join(a)
                    return n, a
                if False: # just save the maximum value per feature
                    aux = np.abs(df_aux).idxmax(axis = 0)
                    e1, e2 = aux[0], aux[1]
                    ft_id_nm = _ft_name(file_dir[i]) 
                    rank_l.append([0, e1, e2, abs(df_aux.loc[e1, e2]), ft_id_nm[0], ft_id_nm[1]])
                elif True: # save every combination element in feature
                    N = len(df_aux)
                    ft_id_nm = _ft_name(file_dir[i]) 
                    for j, column in enumerate(df_aux):
                        for k in range(j+1,N):
                            e1, e2 = df_aux.columns[j], df_aux.columns[k]
                            a = df_aux.iloc[j][k]
                            rank_l.append([0, e1, e2, abs(df_aux.loc[e1, e2]), ft_id_nm[0], ft_id_nm[1]])

        rank_l = sorted(rank_l,key=lambda x: x[3], reverse = True)
        for i in range(len(rank_l)):
            rank_l[i][0] = i+1
        # 
        path = '..'+os.sep+'features'+os.sep+'correlations'+os.sep+'corr_0_rank_erup_cc'
        with open(path+'.txt', 'w') as f:
            f.write("rank,erup1,erup2,cc,featID,featNM\n")
            for line in rank_l:
                f.write(str(line[0])+",")
                f.write(str(line[1])+",")
                f.write(str(line[2])+",")  
                f.write(str(line[3])+",")
                f.write(str(line[4])+",")
                f.write(str(line[5])+"\n") 
        
        del df_aux
    
        if False: # change name of file: add eruption at the end of the file name 
            # import file as pandas
            path = '..'+os.sep+'features'+os.sep+'correlations'+os.sep+'corr_0_rank_erup_cc.csv'
            pd_rank = pd.read_csv(path)
            i = 0
            #
            # files list (list)
            path_files = '..'+os.sep+'features'+os.sep+'correlations'+os.sep+'plot_ranking_features'+os.sep+'*.png'
            file_dir = glob.glob(path_files)

            for i, row in pd_rank.iterrows():
                #
                r = row['rank']
                e1 = row['erup1'] # name of eruptions to plot
                e2 = row['erup2']
                f = row['featNM']
                # get the name of the file 
                for fl in file_dir:
                    if 'feat_'+str(r)+'_' in fl:
                        fl_s  = fl
                        os.rename(fl, fl[:-4]+'_'+e1+'_'+e2+'.png')
    
    # download geonet lake temperature and level data 
    if False:
        ## download temperature data. Save in data folder
        # Download data for Ruapehu crater lake from GEONET(siteID: WI201)
        if True: 
            ruapehu = True
            whakaari = False   
            import json # package to read json code
            import requests # package to get data from an API
            import datetime # package to deal with time
            # Define the longitude and latitude of the point of interest for Ruapehu
            if ruapehu:
                point_long = 175.565
                point_lat = -39.281
                box_size = 0.1
            if whakaari:
                point_long = 177.182
                point_lat = -37.521
                box_size = 0.1
            #
            long_max = str(point_long + box_size) 
            long_min = str(point_long - box_size)
            lat_max  = str(point_lat - box_size)
            lat_min  = str(point_lat + box_size)
            #
            poly = ("POLYGON((" +long_max + " " +lat_max
                    +","+long_max+" "+lat_min
                    +","+long_min+" "+lat_min
                    +","+long_min+" "+lat_max
                    +","+long_max+" "+lat_max +"))")
            # set url
            base_url = "https://fits.geonet.org.nz/"
            endpoint = "site"
            url = base_url + endpoint
            #
            parameters ={'within':poly}
            # get site data
            sites = requests.get(url, params=parameters)
            # read the json file 
            data = sites.json()['features'] 
            # Initialize this data frame
            dat = pd.DataFrame() #empty dataframe
            # Add the site data to this data frame
            for i, val in enumerate(data):
                geometry = val['geometry']
                lon = geometry['coordinates'][0]
                lat = geometry['coordinates'][1]
                properties = val['properties']
                siteID = properties['siteID']
                height = properties['height']
                name = properties['name']
                #append these to df
                dat = dat.append({'siteID': siteID, 'lon': lon, 'lat': lat, 'height': height, 'name': name}, ignore_index=True)
            # save data
            if ruapehu:
                dat.to_csv('..'+os.sep+'data'+os.sep+"ruapehu_sites_temp.csv") 
            if whakaari:
                dat.to_csv('..'+os.sep+'data'+os.sep+"whakaari_sites_temp.csv") 
            #
            def get_volcano_data(site,typeID):
                """
                This function takes a site ID and type ID as a strings
                and returns a dataframe with all the observation of that type for that site
                """
                #Setup
                base_url = "https://fits.geonet.org.nz/"
                endpoint = "observation"
                url = base_url + endpoint
                
                #Set query parameters
                parameters ={"typeID": typeID, "siteID": site}
                
                #Get data
                request = requests.get(url, params=parameters)
                
                #Unpack data
                data = (request.content)
                
                #If there are data points
                if len(data) > 50:
                    #run volcano_dataframe on it
                    df = volcano_dataframe(data.decode("utf-8"))
                    #print some info on it 
                    print(site,"has", typeID, "data and has", len(df.index), 
                        "data points from ",df['date-time'][1]," to ", df['date-time'][len(df.index)])
                    #retrun it
                    return df
            #
            def volcano_dataframe(data):
                """
                This function turns the string of volcano data received by requests.get
                into a data frame with volcano data correctly formatted.
                """
                # splits data on the new line symbol
                data = data.split("\n") 
                
                # For each data point
                for i in range(0, len(data)):
                    data[i]= data[i].split(",")# splits data ponits on the , symbol
                
                # For each data point 
                for i in range(1, (len(data)-1)):
                    data[i][0] = datetime.datetime.strptime(data[i][0], '%Y-%m-%dT%H:%M:%S.%fZ') #make 1st value into a datetime object
                    data[i][1] = float(data[i][1]) #makes 2nd value into a decimal number
                    data[i][2] = float(data[i][2]) #makes 3rd value into a decimal number
                    
                #make the list into a data frame
                df = pd.DataFrame(data[1:-1],index = range(1, (len(data)-1)), columns=data[0]) #make the list into a data frame
                
                #Return this data frame
                return df
            #
            def get_method(typeID):
                """
                This function takes a type ID as a strings
                and returns all methods used for this type
                """
                
                #Setup
                base_url = "https://fits.geonet.org.nz/"
                endpoint = "method"
                url = base_url + endpoint
                
                #Set query parameters
                parameters ={"typeID": typeID}
                
                #Get data
                request = requests.get(url, params=parameters)
                
                #Unpack data
                data = request.json()['method']
                
                #run make_method_df on data
                df =  make_method_df(data)
                
                return df
            #
            def make_method_df(data):
                """
                This function takes method data as a list
                and returns a dataframe with all the method data.
                """
                #Initialize this data frame
                df = pd.DataFrame()
                
                #add data to the data frame
                for i, val in enumerate(data):
                    methodID = val['methodID']
                    name = val['name']
                    description = val['description']
                    reference = val['reference']
                    #append these to df
                    df = df.append({'name': name, 'methodID': methodID, 'description': description, 'reference':reference}, 
                                ignore_index=True)
                
                #Return this data frame
                return df  
            #
            # Set the type to the type ID for temperature
            typeID = "t" # temperature 
            typeID = "z" # lake level
            #typeID = "nve" # number of volcanic-tectonic earthquakes recorded per day (empty)
            typeID = "ph" # degree of acidity or alkalinity of a sample
            typeID = "tl" # angle of tilt relative to the horizontal
            typeID =  "u" # displacement from initial position
            #typeID =  "u_rf" # displacement from initial position (empty)
            typeID = "Cl-w" # chloride in water sample
            typeID = "SO4-w" # Sulphate in water sample

            # Get the methods for this type ID
            methods = get_method(typeID)
            # Get all temperature data from all these sites
            #Initialize a list to put the data in later
            t={}
            #loop over each site ID
            for i, site in enumerate(dat["siteID"]):
                #use the get_volcano_data funtion to get the data and save it with the key of the site's ID
                t.update({site:get_volcano_data(site,typeID)})
            # Save as CSV file
            if ruapehu:
                siteID = 'RU001'
                if typeID ==  "u" or typeID ==  "u_rf": 
                    siteID = 'VGOB'
            if whakaari:
                siteID = 'WI201'
            if typeID == "t":
                t[siteID].to_csv('..'+os.sep+'data'+os.sep+siteID+"_temp_data.csv") 
            if typeID == "z":
                t[siteID].to_csv('..'+os.sep+'data'+os.sep+siteID+"_level_data.csv") 
            if typeID == "nvte":
                t[siteID].to_csv('..'+os.sep+'data'+os.sep+siteID+"_nvte_data.csv") 
            if typeID == "ph":
                t[siteID].to_csv('..'+os.sep+'data'+os.sep+siteID+"_ph_data.csv") 
            if typeID == "tl":
                t[siteID].to_csv('..'+os.sep+'data'+os.sep+siteID+"_tilt_data.csv") 
            if typeID == "u":
                t[siteID].to_csv('..'+os.sep+'data'+os.sep+siteID+"_u_disp_abs_data.csv") 
            if typeID == "u_rf":
                t[siteID].to_csv('..'+os.sep+'data'+os.sep+siteID+"_u_disp_reg_filt_data.csv")
            if typeID == "Cl-w":
                t[siteID].to_csv('..'+os.sep+'data'+os.sep+siteID+"_cl_data.csv")
            if typeID == "SO4-w":
                t[siteID].to_csv('..'+os.sep+'data'+os.sep+siteID+"_so4_data.csv")
            #
            if True:  # plot temperature data
                start = datetime.datetime(2019, 6, 1)
                end = datetime.datetime(2019, 12, 31)
                # Trim the data
                df = t[siteID].loc[t[siteID]['date-time']<end]
                df = df.loc[t[siteID]['date-time']>start]
                #
                plot2 = df.plot(x='date-time', y= ' t (C)', 
                    title = 'Temperature')
                plt.show()

        # plot temp data against features for Ruapehu

    # plot feature correlation (from .csv of rank of cc eruptions)
    if False:
        import numpy as np
        # import file as pandas
        path = '..'+os.sep+'features'+os.sep+'correlations'+os.sep+'corr_0_rank_erup_cc.csv'
        try:
            pd_rank = pd.read_csv(path, index_col=0)
        except: # convert txt to csv
            read_file = pd.read_csv ('..'+os.sep+'features'+os.sep+'correlations'+os.sep+'corr_0_rank_erup_cc.txt')
            read_file.to_csv(path, index=None)
            #
            pd_rank = pd.read_csv(path, index_col=0)
            del read_file
        # locate index in rank for cc cut value of 0.5
        idx, _ = find_nearest(pd_rank.loc[:, 'cc'].values, 0.5)
        # for plotting: import features of eruptions per rank
        ranks = np.arange(130,idx,1)
        #
        plot_2_pairs_erups = False # set to true to plot a third correlated eruption
        if plot_2_pairs_erups:
            ranks = np.arange(130, idx, 1)
        #
        # plot data stream
        plt_ds = False # plot datastream 

        for rank in [262]:#[262]: #ranks:262   2325  120   5865
            # name of eruptions to plot
            e1 = pd_rank.loc[rank, 'erup1'] # name of eruptions to plot
            e2 = pd_rank.loc[rank, 'erup2']
            # feature to plot
            ft_nm = pd_rank.loc[rank, 'featNM']
            ft_id = pd_rank.loc[rank, 'featID']
            cc = pd_rank.loc[rank, 'cc']
            # look for the third eruption mostly correlated with the e1 or e2 in ft_nm
            #try:
                #
            if plot_2_pairs_erups:
                # loop over rank and find first apperance 
                for r in range(rank, len(pd_rank), 1):#ranks[rank:]:
                    naux = pd_rank.loc[r, 'featNM']
                    if ft_nm == pd_rank.loc[r, 'featNM']:
                        e1a, e2a = pd_rank.loc[r, 'erup1'], pd_rank.loc[r, 'erup2']
                        if not ((e1a in [e1,e2]) or (e2a in [e1,e2])):
                            ranka = r
                            cca = pd_rank.loc[r, 'cc']
                            ft_ida = pd_rank.loc[r, 'featID']
                            break

            ## import features
            # proper name of feature
            ft_nm_aux = ft_nm.replace(" ","__")
            ft_nm_aux = ft_nm_aux.replace("-",'"')
            # name of the data stream 
            ds = ['log_zsc2_rsamF', 'zsc2_hfF','zsc2_mfF','zsc2_dsarF']
            for d in ds:
                if d in ft_nm_aux:
                    ds = [d]
                    break
            # create objects for each station 
            fm_e1 = ForecastModel(window=2., overlap=1., station = e1[:-2],
                look_forward=2., data_streams=ds, savefile_type='csv')
            fm_e2 = ForecastModel(window=2., overlap=1., station = e2[:-2],
                look_forward=2., data_streams=ds, savefile_type='csv')
            #
            if plot_2_pairs_erups:
                fm_e1a = ForecastModel(window=2., overlap=1., station = e1a[:-2],
                look_forward=2., data_streams=ds, savefile_type='csv')
                fm_e2a = ForecastModel(window=2., overlap=1., station = e2a[:-2],
                look_forward=2., data_streams=ds, savefile_type='csv')
        
            # initial and final time of interest for each station
            tf_e1 = fm_e1.data.tes[int(e1[-1:])-1]
            ti_e1 = tf_e1 - 30*day #month
            tf_e2 = fm_e2.data.tes[int(e2[-1:])-1]
            ti_e2 = tf_e2 - 30*day
            #
            if plot_2_pairs_erups:
                tf_e1a = fm_e1a.data.tes[int(e1a[-1:])-1]
                ti_e1a = tf_e1a - 30*day #month
                tf_e2a = fm_e2a.data.tes[int(e2a[-1:])-1]
                ti_e2a = tf_e2a - 30*day  
            #
            ft_e1 = fm_e1.get_features(ti=ti_e1, tf=tf_e1, n_jobs=1, compute_only_features=[ft_nm_aux])
            ft_e2 = fm_e2.get_features(ti=ti_e2, tf=tf_e2, n_jobs=1, compute_only_features=[ft_nm_aux])
            # extract values to plot 
            ft_e1_t = ft_e1[0].index.values
            ft_e1_v = ft_e1[0].loc[:,ft_nm_aux]
            ft_e2_t = ft_e2[0].index.values
            ft_e2_v = ft_e2[0].loc[:,ft_nm_aux]
            #
            if plot_2_pairs_erups:
                ft_e1a = fm_e1a.get_features(ti=ti_e1a, tf=tf_e1a, n_jobs=1, compute_only_features=[ft_nm_aux])
                ft_e2a = fm_e2a.get_features(ti=ti_e2a, tf=tf_e2a, n_jobs=1, compute_only_features=[ft_nm_aux])
                # extract values to plot 
                ft_e1a_t = ft_e1a[0].index.values
                ft_e1a_v = ft_e1a[0].loc[:,ft_nm_aux]
                ft_e2a_t = ft_e2a[0].index.values
                ft_e2a_v = ft_e2a[0].loc[:,ft_nm_aux]

            # adding multiple Axes objects  
            if not plot_2_pairs_erups:
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10,6))
            else:
                fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(10,12))

            # ax1
            #ax1.plot(ft_e1_t, ft_e1_v, '-', color='b', label=erup_dict[e1])
            ax1.plot(ft_e1_t, ft_e1_v, '-', color='b', label=erup_dict[e1]+', vei: '+erup_vei_dict[e1])
            te = fm_e1.data.tes[int(e1[-1:])-1]
            ax1.axvline(te, color='k', linestyle='--', linewidth=2, zorder = 0)
            ax1.plot([], color='k', linestyle='--', linewidth=2, label = 'eruption')
            
            ax1.grid()
            ax1.set_ylabel('feature value')
            ax1.set_xlabel('time')
            #plt.xticks(rotation=45)
            
            # ax2
            #ax2.plot(ft_e2_t, ft_e2_v, '-', color='r', label=erup_dict[e2])
            ax2.plot(ft_e2_t, ft_e2_v, '-', color='r', alpha = 0.5, label=erup_dict[e2]+', vei: '+erup_vei_dict[e2])
            te = fm_e2.data.tes[int(e2[-1:])-1]
            ax2.axvline(te, color='k', linestyle='--', linewidth=2, zorder = 0)
            ax2.plot([], color='k', linestyle='--', linewidth=2, label = 'eruption')
            
            ax2.grid()
            ax2.set_ylabel('feature value')
            ax2.set_xlabel('time')
            #plt.xticks(rotation=45)

            if plt_ds:
                # plot datastream e1
                dat = fm_e1.data.get_data(ti = ti_e1, tf = tf_e1)
                nm = ft_nm_aux.split('__')[0]
                #nm = '_'.join(nm)
                ax1b = ax1.twinx()  # instantiate a second axes that shares the same x-axis
                if 'rsam' in nm:
                    _nm = 'rsam'
                if 'mf' in nm:
                    _nm = 'mf'
                if 'hf' in nm:
                    _nm = 'hf'
                if 'dsar' in nm:
                    _nm = 'dsar'
                ax1b.plot(dat.index.values, dat[nm].values, color='c', linestyle='-', linewidth=1, 
                    label = _nm+' data' , alpha = .3)
                try:
                    ax1b.set_ylim([-.5*np.max(dat[nm].values),np.max(dat[nm].values)])
                except:
                    ax1b.set_ylim([-.5*np.max(dat[nm].values),np.max(dat[nm].values)])
                #ax1b.set_ylabel(nm)
                ax1b.legend(loc = 1)
                ax1b.set_yticks([])

                # plot datastream e2
                dat = fm_e2.data.get_data(ti = ti_e2, tf = tf_e2)
                nm = ft_nm_aux.split('__')[0]
                if 'rsam' in nm:
                    _nm = 'rsam'
                if 'mf' in nm:
                    _nm = 'mf'
                if 'hf' in nm:
                    _nm = 'hf'
                if 'dsar' in nm:
                    _nm = 'dsar'
                ax2b = ax2.twinx()  # instantiate a second axes that shares the same x-axis
                ax2b.plot(dat.index.values, dat[nm].values, color='c', linestyle='-', linewidth=1, 
                    label = _nm+' data', alpha = .3)
                try:
                    ax2b.set_ylim([-.5*np.max(dat[nm].values),np.max(dat[nm].values)])
                except:
                    ax2b.set_ylim([-.5*np.max(dat[nm].values),np.max(dat[nm].values)])
                #ax1b.set_ylabel(nm.split('_')[-1])
                ax2b.legend(loc = 1)
                ax2b.set_yticks([])

            if plot_2_pairs_erups:

                # ax3
                ax3.plot(ft_e1a_t, ft_e1a_v, '-', color='b', label=erup_dict[e1a])
                te = fm_e1a.data.tes[int(e1a[-1:])-1]
                ax3.axvline(te, color='k', linestyle='--', linewidth=2, zorder = 0)
                ax3.plot([], color='k', linestyle='--', linewidth=2, label = 'eruption')
                ax3.legend(loc = 2)
                ax3.grid()
                ax3.set_ylabel('feature value')
                ax3.set_xlabel('time')
                #plt.xticks(rotation=45)
                
                # ax4
                ax4.plot(ft_e2a_t, ft_e2a_v, '-', color='r', label=erup_dict[e2a])
                te = fm_e2a.data.tes[int(e2a[-1:])-1]
                ax4.axvline(te, color='k', linestyle='--', linewidth=2, zorder = 0)
                ax4.plot([], color='k', linestyle='--', linewidth=2, label = 'eruption')
                ax4.legend(loc = 2)
                ax4.grid()
                ax4.set_ylabel('feature value')
                ax4.set_xlabel('time')
                #plt.xticks(rotation=45).

                if plt_ds:
                    # plot datastream e1
                    dat = fm_e1a.data.get_data(ti = ti_e1a, tf = tf_e1a)
                    nm = ft_nm_aux.split('__')[0]
                    #nm = '_'.join(nm)
                    ax3b = ax3.twinx()  # instantiate a second axes that shares the same x-axis
                    if 'rsam' in nm:
                        _nm = 'rsam'
                    if 'mf' in nm:
                        _nm = 'mf'
                    if 'hf' in nm:
                        _nm = 'hf'
                    if 'dsar' in nm:
                        _nm = 'dsar'
                    ax3b.plot(dat.index.values, dat[nm].values, color='c', linestyle='-', linewidth=1, 
                        label = _nm+' data' , alpha = .3)
                    try:
                        ax3b.set_ylim([-.5*np.max(dat[nm].values),np.max(dat[nm].values)])
                    except:
                        ax3b.set_ylim([-.5*np.max(dat[nm].values),np.max(dat[nm].values)])
                    #ax1b.set_ylabel(nm)
                    ax3b.legend(loc = 4)
                    ax3b.set_yticks([])
                    # title for second pair 
                    ax3b.set_title('Rank: '+str(ranka)+' (cc:'+str(round(cca,2))+')  Eruptions: '+ e1a+' and '+ e2a)#, ha='center')
                    
                    # plot datastream e2
                    dat = fm_e2a.data.get_data(ti = ti_e2a, tf = tf_e2a)
                    nm = ft_nm_aux.split('__')[0]
                    if 'rsam' in nm:
                        _nm = 'rsam'
                    if 'mf' in nm:
                        _nm = 'mf'
                    if 'hf' in nm:
                        _nm = 'hf'
                    if 'dsar' in nm:
                        _nm = 'dsar'
                    ax4b = ax4.twinx()  # instantiate a second axes that shares the same x-axis
                    ax4b.plot(dat.index.values, dat[nm].values, color='c', linestyle='-', linewidth=1, 
                        label = _nm+' data', alpha = .3)
                    try:
                        ax4b.set_ylim([-.5*np.max(dat[nm].values),np.max(dat[nm].values)])
                    except:
                        ax4b.set_ylim([-.5*np.max(dat[nm].values),np.max(dat[nm].values)])
                    #ax1b.set_ylabel(nm.split('_')[-1])
                    ax4b.legend(loc = 4)
                    ax4b.set_yticks([])
            
            ## plot other data
            temp = True
            level = True
            rainfall = True
            ph = False
            u = False
            cl = False
            so4 = False
            #
            mov_avg = True # moving average for temp and level data
            #
            utc_0 = True
            if temp or level or rainfall or u or ph or cl or so4:
                ax1b = ax1.twinx()
                ax2b = ax2.twinx()
                if e2[:-2] == 'FWVZ':
                    #ax2.plot(ft_e2_t, ft_e2_v, '-', color='w')
                    #ax2.plot(ft_e2_t, ft_e2_v, '-', color='r', alpha = 0.8)
                    pass
            # plot temp data
            if temp:
                if e1[:-2] == 'FWVZ':
                    # import temp data
                    path = '..'+os.sep+'data'+os.sep+"RU001_temp_data.csv"
                    pd_temp = pd.read_csv(path, index_col=1)
                    #pd_temp.index = [datetimeify(pd_temp.index[i]) for i in range(len(pd_temp.index))]
                    #
                    if utc_0:
                        pd_temp.index = [datetimeify(pd_temp.index[i])-6*hour for i in range(len(pd_temp.index))]
                    else:
                        pd_temp.index = [datetimeify(pd_temp.index[i]) for i in range(len(pd_temp.index))]
                    # plot data in axis twin axis
                    # Trim the data
                    temp_e1_tim = pd_temp[ti_e1: tf_e1].index.values
                    temp_e1_val = pd_temp[ti_e1: tf_e1].loc[:,' t (C)'].values
                    # ax2
                    #ax2b = ax2.twinx()   
                    if mov_avg: # plot moving average
                        n=50
                        v_plot = (temp_e1_val-np.min(temp_e1_val))/np.max((temp_e1_val-np.min(temp_e1_val)))
                        #ax2b.plot(temp_e1_tim[:-n+1], moving_average(temp_e1_val, n=n), '--', color='k', label='temp. mov. avg.')
                        ax1b.plot(temp_e1_tim[n-1-15:-15], moving_average(v_plot[::-1], n=n)[::-1], '-', color='g', label='lake temperature')
                        #ax2b.plot(temp_e1_tim, v_plot, '-', color='g', label='lake temperature', alpha = 0.3)
                    else:
                        #ax2b.plot(temp_e1_tim, temp_e1_val, '-', color='g', label='temperature')
                        #ax2.set_ylim([-40,40])
                        #plt.show()
                        v_plot = (temp_e1_val-np.min(temp_e1_val))/np.max((temp_e1_val-np.min(temp_e1_val)))
                        ax1b.plot(temp_e1_tim, v_plot, '-', color='g', label='lake temperature')
                    #ax2b.set_ylabel('temperature C')
                if e2[:-2] == 'FWVZ':
                    # import temp data
                    path = '..'+os.sep+'data'+os.sep+"RU001_temp_data.csv"
                    pd_temp = pd.read_csv(path, index_col=1)
                    pd_temp.index = [datetimeify(pd_temp.index[i]) for i in range(len(pd_temp.index))]
                    # plot data in axis twin axis
                    # Trim the data
                    temp_e1_tim = pd_temp[ti_e2: tf_e2].index.values
                    temp_e1_val = pd_temp[ti_e2: tf_e2].loc[:,' t (C)'].values
                    # ax2
                    #ax2b = ax2.twinx()   
                    if mov_avg: # plot moving average
                        v_plot = (temp_e1_val-np.min(temp_e1_val))/np.max((temp_e1_val-np.min(temp_e1_val)))
                        ax2b.plot(temp_e1_tim, v_plot, '-', color='g', label='lake temperature')
                        n=50
                        v_plot = (temp_e1_val-np.min(temp_e1_val))/np.max((temp_e1_val-np.min(temp_e1_val)))
                        #ax2b.plot(temp_e1_tim[:-n+1], moving_average(temp_e1_val, n=n), '--', color='k', label='temp. mov. avg.')
                        ax2b.plot(temp_e1_tim[n-1-20:-20], moving_average(v_plot[::-1], n=n)[::-1], '--', color='k')#, label='lake temperature')
                        #ax2b.plot(temp_e1_tim, v_plot, '-', color='g', label='lake temperature', alpha = 0.3)
                    else:
                        v_plot = (temp_e1_val-np.min(temp_e1_val))/np.max((temp_e1_val-np.min(temp_e1_val)))
                        #v_plot = temp_e1_val
                        ax2b.plot(temp_e1_tim, v_plot, '-', color='g', label='lake temperature')
                        #ax2b.set_ylabel('Temperature °C')
                        #plt.show()
                
                    #ax2b.set_ylabel('temperature C')   
                    #ax2b.legend(loc = 3)         
            # plot lake level data
            if level:
                if e1[:-2] == 'FWVZ':
                    # import temp data
                    path = '..'+os.sep+'data'+os.sep+"RU001_level_data.csv"
                    pd_temp = pd.read_csv(path, index_col=1)
                    pd_temp.index = [datetimeify(pd_temp.index[i]) for i in range(len(pd_temp.index))]
                    # plot data in axis twin axis
                    # Trim the data
                    temp_e1_tim = pd_temp[ti_e1: tf_e1].index.values
                    temp_e1_val = pd_temp[ti_e1: tf_e1].loc[:,' z (m)'].values
                    # ax2
                    #ax2b = ax2.twinx()
                    if mov_avg: # plot moving average
                        n=30
                        #ax2b.plot(temp_e1_tim[:-n+1], moving_average(temp_e1_val, n=n), '--', color='k', label='temp. mov. avg.')
                        v_plot = (temp_e1_val-np.min(temp_e1_val))/np.max((temp_e1_val-np.min(temp_e1_val)))
                        ax1b.plot(temp_e1_tim[n-1-20:-20], moving_average(v_plot[::-1], n=n)[::-1], '-', color='b', label='lake level')
                        #ax2b.plot(temp_e1_tim, v_plot, '-', color='b', alpha = 0.3)
                    else:
                        #ax2b.plot(temp_e1_tim, temp_e1_val, '-', color='b', label='level')
                        #ax2.set_ylim([-40,40])
                        #plt.show()
                        v_plot = (temp_e1_val-np.min(temp_e1_val))/np.max((temp_e1_val-np.min(temp_e1_val)))
                        ax1b.plot(temp_e1_tim, v_plot, '-', color='b', label='lake level')

                    #ax2b.set_ylabel('temperature C')
                    #ax2b.legend(loc = 3)  
                if e2[:-2] == 'FWVZ':
                    # import temp data
                    path = '..'+os.sep+'data'+os.sep+"RU001_level_data.csv"
                    pd_temp = pd.read_csv(path, index_col=1)
                    #pd_temp.index = [datetimeify(pd_temp.index[i]) for i in range(len(pd_temp.index))]
                    #
                    if utc_0:
                        pd_temp.index = [datetimeify(pd_temp.index[i])-6*hour for i in range(len(pd_temp.index))]
                    else:
                        pd_temp.index = [datetimeify(pd_temp.index[i]) for i in range(len(pd_temp.index))]
                    # plot data in axis twin axis
                    # Trim the data
                    temp_e1_tim = pd_temp[ti_e2: tf_e2].index.values
                    temp_e1_val = pd_temp[ti_e2: tf_e2].loc[:,' z (m)'].values
                    # ax2
                    #ax2b = ax2.twinx()
                    if mov_avg: # plot moving average
                        v_plot = (temp_e1_val-np.min(temp_e1_val))/np.max((temp_e1_val-np.min(temp_e1_val)))
                        ax2b.plot(temp_e1_tim, v_plot, '-', color='b', label='lake level')
                        n=30
                        #ax2b.plot(temp_e1_tim[:-n+1], moving_average(temp_e1_val, n=n), '--', color='k', label='temp. mov. avg.')
                        v_plot = (temp_e1_val-np.min(temp_e1_val))/np.max((temp_e1_val-np.min(temp_e1_val)))
                        ax2b.plot(temp_e1_tim[n-1-15:-15], moving_average(v_plot[::-1], n=n)[::-1], '--', color='k')#, label='lake level')
                        #ax2b.plot(temp_e1_tim, v_plot, '-', color='b', alpha = 0.3)
                        if True: # plot derivative moving average 
                            val_y =  moving_average(v_plot[::-1], n=n)[::-1]
                            val_x = temp_e1_tim[n-1-15:-15]
                            #
                            #der = val_y[:-1] - val_y[1:] 
                            #der = moving_average(der[::-1], n=3)[::-1]
                            der = [(val_y[i+1]-val_y[i])/1. *20. for i in range(len(val_y)-1)] # float(val_x[i+1].sec-val_x[i].sec)
                            der = moving_average(der[::-1], n=6)[::-1]
                            der = np.abs(der)
                            #
                            ax2b.plot(val_x[6:], der*2., '-', color='m',alpha = 0.8, label='lake flowrate')
                    else:
                        v_plot = (temp_e1_val-np.min(temp_e1_val))/np.max((temp_e1_val-np.min(temp_e1_val)))
                        #v_plot = temp_e1_val
                        ax2b.plot(temp_e1_tim, v_plot*100, '-', color='b', label='lake level')
                        #ax2b.set_ylabel('Lake level cm')
                        #plt.show()

                    #ax2b.set_ylabel('temperature C')
                    #ax2b.legend(loc = 3)   
            # plot rainfall data
            if rainfall:
                if e1[:-2] == 'FWVZ':
                    # import temp data
                    path = '..'+os.sep+'data'+os.sep+"ruapehu_chateau_rainfall_data.csv"
                    pd_rf = pd.read_csv(path, index_col=1)
                    pd_rf.index = pd.to_datetime(pd_rf.index, format='%Y%m%d:%H%M')
                    #pd_rf.index = [datetimeify(pd_temp.index[i]) for i in range(len(pd_temp.index))]
                    pd_rf.index = [datetimeify(pd_rf.index[i]) for i in range(len(pd_rf.index))]
                    # plot data in axis twin axis
                    # Trim the data
                    rf_e2_tim = pd_rf[ti_e1: tf_e1].index.values
                    rf_e2_val = pd_rf[ti_e1: tf_e1].loc[:,'Amount(mm)'].values /4
                    # ax2
                    #ax2b = ax2.twinx()
                    v_plot = (rf_e2_val-np.min(rf_e2_val))/np.max((rf_e2_val-np.min(rf_e2_val)))
                    v_plot = v_plot/8
                    ax1b.plot(rf_e2_tim, v_plot, '-', color='c', label='rain fall', alpha = 0.8)
                    #ax2b.set_ylabel('temperature C')
                    #ax2b.legend(loc = 1)
                if e2[:-2] == 'FWVZ':
                    # import temp data
                    path = '..'+os.sep+'data'+os.sep+"ruapehu_chateau_rainfall_data.csv"
                    pd_rf = pd.read_csv(path, index_col=1)
                    pd_rf.index = pd.to_datetime(pd_rf.index, format='%Y%m%d:%H%M')
                    #pd_rf.index = [datetimeify(pd_temp.index[i]) for i in range(len(pd_temp.index))]
                    if utc_0:
                        pd_rf.index = [datetimeify(pd_rf.index[i])-12*hour for i in range(len(pd_rf.index))]
                    else:
                        pd_rf.index = [datetimeify(pd_rf.index[i]) for i in range(len(pd_rf.index))]
                    #pd_rf.index = [datetimeify(pd_rf.index[i]) for i in range(len(pd_rf.index))]
                    # plot data in axis twin axis
                    # Trim the data
                    rf_e2_tim = pd_rf[ti_e2: tf_e2].index.values
                    rf_e2_val = pd_rf[ti_e2: tf_e2].loc[:,'Amount(mm)'].values /4
                    # ax2
                    #ax2b = ax2.twinx()
                    v_plot = (rf_e2_val-np.min(rf_e2_val))/np.max((rf_e2_val-np.min(rf_e2_val)))
                    v_plot = v_plot/6
                    ax2b.plot(rf_e2_tim, v_plot, '-', color='c', linewidth=2, label='rain fall', alpha = 1., zorder = 5)
                    #ax2b.set_ylabel('temperature C')
                #ax2b.legend(loc = 3)
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

            if temp or level or rainfall or u or ph:
                ax2.set_ylim([-40,40])
                ax2b.set_ylim([-0.01,1])
                ax2b.legend(loc = 3)        
            #
            ax1.legend(loc = 2)
            ax2.legend(loc = 2)
            #
            if True: # xticks every week
                dat = fm_e1.data.get_data(ti = ti_e1, tf = tf_e1)
                ax1.set_xticks([dat.index[-1]-7*day*i for i in range(5)])
                dat = fm_e2.data.get_data(ti = ti_e2, tf = tf_e2)
                ax2.set_xticks([dat.index[-1]-7*day*i for i in range(5)])
                if plot_2_pairs_erups:
                    dat = fm_e1a.data.get_data(ti = ti_e1a, tf = tf_e1a)
                    ax3.set_xticks([dat.index[-1]-7*day*i for i in range(5)])
                    dat = fm_e2a.data.get_data(ti = ti_e2a, tf = tf_e2a)
                    ax4.set_xticks([dat.index[-1]-7*day*i for i in range(5)])
            #
            ax1.set_ylim([-100, 100])
            ax2.set_ylim([-40,40])
            #ax3.set_ylim([0,160])
            #ax4.set_ylim([0,12.5])

            #fig.suptitle('Rank: '+str(rank)+' (cc:'+str(round(cc,2))+')  Eruptions: '+ erup_dict[e1]+' and '+ erup_dict[e2]
            #    +'\n Feature: '+ ft_nm_aux+' (id:'+str(ft_id)+')')#, ha='center')
            plt.tight_layout()
            plt.show()
            if False: # save 
                if os.path.isdir('..'+os.sep+'features'+os.sep+'correlations'+os.sep+'plot_ranking_features'): 
                    pass
                else:
                    os.mkdir('..'+os.sep+'features'+os.sep+'correlations'+os.sep+'plot_ranking_features')
                #
                #path = '..'+os.sep+'features'+os.sep+'correlations'+os.sep+'plot_ranking_features'+os.sep+'corr_0_rank_erup_cc_feat_'
                path = '..'+os.sep+'features'+os.sep+'correlations'+os.sep+'plot_ranking_features'+os.sep+'feat_'
                if plot_2_pairs_erups:
                    if os.path.isdir('..'+os.sep+'features'+os.sep+'correlations'+os.sep+'plot_ranking_features'+os.sep+'plot_ranking_features_2pairs'):
                        pass
                    else:
                        os.mkdir('..'+os.sep+'features'+os.sep+'correlations'+os.sep+'plot_ranking_features'+os.sep+'plot_ranking_features_2pairs')
                    #
                    #path = '..'+os.sep+'features'+os.sep+'correlations'+os.sep+'plot_ranking_features'+os.sep+'corr_0_rank_erup_cc_feat_'
                    path = '..'+os.sep+'features'+os.sep+'correlations'+os.sep+'plot_ranking_features'+os.sep+'plot_ranking_features_2pairs'+os.sep+'feat_'
                plt.savefig(path+str(rank)+'_'+ft_nm+'.png', dpi = 300)
            print(rank)
            del fm_e1, fm_e2
            plt.close('all')
                #except:
            #    pass

    # hist of ranking per eruptions and per volcanoes 
    if False: 
        # import csv
        # import file as pandas
        path = '..'+os.sep+'features'+os.sep+'correlations'+os.sep+'corr_0_rank_erup_cc.csv'
        pd_rank = pd.read_csv(path, index_col=0)
        
        if True: # count per eruptions
            ## count euruption occurency for cc > .5 in rank file
            w_1, w_2, w_3, w_4, w_5 = 0,0,0,0,0 
            r_1, r_2, r_3 = 0,0,0
            t_1, t_2 = 0,0
            b_1, b_2, b_3 = 0,0,0
            v_1, v_2 = 0,0
            p_1, p_2, p_3 = 0,0,0
            # by volcano 
            vo_1, vo_2, vo_3, vo_4, vo_5, vo_6 = 0,0,0,0,0,0
            # set low bound for 'cc'
            cc_l = 0.5         
            # loop over rank (index)
            for index, row in pd_rank.iterrows():
                e1 = pd_rank.loc[index, 'erup1'] # name of eruptions to plot
                e2 = pd_rank.loc[index, 'erup2']
                cc = pd_rank.loc[index, 'cc']
                if cc > cc_l:
                    #
                    if e1 == 'WIZ_1' or e2 == 'WIZ_1':
                        w_1 += 1
                        vo_1 += 1
                    if e1 == 'WIZ_2' or e2 == 'WIZ_2':
                        w_2 += 1
                        vo_1 += 1
                    if e1 == 'WIZ_3' or e2 == 'WIZ_3':
                        w_3 += 1
                        vo_1 += 1
                    if e1 == 'WIZ_4' or e2 == 'WIZ_4':
                        w_4 += 1
                        vo_1 += 1
                    if e1 == 'WIZ_5' or e2 == 'WIZ_5':
                        w_5 += 1
                        vo_1 += 1
                    #
                    if e1 == 'FWVZ_1' or e2 == 'FWVZ_1':
                        r_1 += 1
                        vo_2 += 1
                    if e1 == 'FWVZ_2' or e2 == 'FWVZ_2':
                        r_2 += 1
                        vo_2 += 1
                    if e1 == 'FWVZ_3' or e2 == 'FWVZ_3':
                        r_3 += 1
                        vo_2 += 1
                    #
                    if e1 == 'KRVZ_1' or e2 == 'KRVZ_1':
                        t_1 += 1
                        vo_3 += 1
                    if e1 == 'KRVZ_2' or e2 == 'KRVZ_2':
                        t_2 += 1
                        vo_3 += 1
                    #
                    if e1 == 'BELO_1' or e2 == 'BELO_1':
                        b_1 += 1
                        vo_4 += 1
                    if e1 == 'BELO_2' or e2 == 'BELO_2':
                        b_2 += 1
                        vo_4 += 1
                    #
                    if e1 == 'VNSS_1' or e2 == 'VNSS_1':
                        v_1 += 1
                        vo_5 += 1
                    if e1 == 'VNSS_2' or e2 == 'VNSS_2':
                        v_2 += 1
                        vo_5 += 1
                    #
                    if e1 == 'PVV_1' or e2 == 'PVV_1':
                        p_1 += 1
                        vo_6 += 1
                    if e1 == 'PVV_2' or e2 == 'PVV_2':
                        p_2 += 1
                        vo_6 += 1
                    if e1 == 'PVV_3' or e2 == 'PVV_3':
                        p_3 += 1
                        vo_6 += 1
            
            ## plot results for eruptions
            data = [w_1, w_2,  w_3, w_4,  w_5,
                    r_1, r_2, r_3, t_1, t_2,
                    b_1, b_2, v_1, v_2, p_1, p_2, p_3]
            columns = ('WIZ_1', 'WIZ_2', 'WIZ_3', 'WIZ_4', 'WIZ_5',
                'FWVZ_1', 'FWVZ_2', 'FWVZ_3', 'KRVZ_1', 'KRVZ_2',
                'BELO_1', 'BELO_2', 'VNSS_1', 'VNSS_2', 
                'PVV_1', 'PVV_2', 'PVV_3')
            # adding multiple Axes objects  
            fig, (ax1) = plt.subplots(1, 1, figsize=(8,4))
            # ax1
            a =np.arange(0,len(data),1)
            ax1.bar(a, data)
            #ax1.text(x = a, y = data, s = [str(d) for d in data], zorder = 3, fontsize=12)
            plt.xticks(rotation=90)
            plt.xticks(a, columns)

            ax1.set_ylabel('frequency')
            ax1.set_xlabel('eruption')
            ax1.set_title('Eruption ocurrence in correlation \n(cc low bound: '+str(cc_l)+')')
            plt.tight_layout()
            path = '..'+os.sep+'features'+os.sep+'correlations'+os.sep+'corr_0_ocurrence_eruption_cc_'+str(cc_l)
            plt.savefig(path+'.png', dpi = 300)

            ## plot results for volcano
            if True: # normalize by amount of eruptions in teh volcano 
                data = [vo_1/5, vo_2/3, vo_3/2, vo_4/2, vo_5/2, vo_6/3]
            else:
                data = [vo_1, vo_2, vo_3, vo_4, vo_5, vo_6]
            columns = ('WIZ', 'FWVZ', 'KRVZ', 'BELO', 'VNSS', 'PVV')
            # adding multiple Axes objects  
            fig, (ax1) = plt.subplots(1, 1, figsize=(4,4))
            # ax1
            a =np.arange(0,len(data),1)
            ax1.bar(a, data)
            #ax1.text(x = a, y = data, s = [str(d) for d in data], zorder = 3, fontsize=12)
            plt.xticks(rotation=90)
            plt.xticks(a, columns)
            ax1.set_ylabel('frequency')
            ax1.set_xlabel('eruption')
            ax1.set_title('Volcano ocurrence in correlation \n(cc low bound: '+str(cc_l)+')')
            plt.tight_layout()
            path = '..'+os.sep+'features'+os.sep+'correlations'+os.sep+'corr_0_ocurrence_volcano_cc_'+str(cc_l)
            plt.savefig(path+'.png', dpi = 300)

            del pd_rank
            #plt.show()
            plt.close('all')

    # plot feature along multiple datastreams (Figure 3 paper)
    if False:
        import numpy as np 
        # 
        plot_2_pairs_erups =  False
        # import file as pandas
        path = '..'+os.sep+'features'+os.sep+'correlations'+os.sep+'corr_0_rank_erup_cc.csv'
        try:
            pd_rank = pd.read_csv(path, index_col=0)
        except: # convert txt to csv
            read_file = pd.read_csv ('..'+os.sep+'features'+os.sep+'correlations'+os.sep+'corr_0_rank_erup_cc.txt')
            read_file.to_csv(path, index=None)
            #
            pd_rank = pd.read_csv(path, index_col=0)
            del read_file
        # locate index in rank for cc cut value of 0.5
        idx, _ = find_nearest(pd_rank.loc[:, 'cc'].values, 0.5)
        # for plotting: import features of eruptions per rank
        ranks = np.arange(0,idx,1)
        #
        for rank in [262]:#[262]: #ranks:262   2325  120
            # name of eruptions to plot
            e1 = pd_rank.loc[rank, 'erup1'] # name of eruptions to plot
            e2 = pd_rank.loc[rank, 'erup2']
            # feature to plot
            ft_nm = pd_rank.loc[rank, 'featNM']
            ft_id = pd_rank.loc[rank, 'featID']
            cc = pd_rank.loc[rank, 'cc']
            #
            if plot_2_pairs_erups:
                # loop over rank and find first apperance 
                for r in range(rank, len(pd_rank), 1):#ranks[rank:]:
                    naux = pd_rank.loc[r, 'featNM']
                    if ft_nm == pd_rank.loc[r, 'featNM']:
                        e1a, e2a = pd_rank.loc[r, 'erup1'], pd_rank.loc[r, 'erup2']
                        if not ((e1a in [e1,e2]) or (e2a in [e1,e2])):
                            ranka = r
                            cca = pd_rank.loc[r, 'cc']
                            ft_ida = pd_rank.loc[r, 'featID']
                            break
            ## import features
            # proper name of feature
            ft_nm_aux = ft_nm.replace(" ","__")
            ft_nm_aux = ft_nm_aux.replace("-",'"')
            # name of the data stream 
            ds = ['log_zsc2_rsamF', 'zsc2_hfF','zsc2_mfF','zsc2_dsarF']
            for d in ds:
                if d in ft_nm_aux:
                    ds = [d]
                    break
            # create objects for each station 
            fm_e1 = ForecastModel(window=2., overlap=1., station = e1[:-2],
                look_forward=2., data_streams=ds, savefile_type='csv')
            fm_e2 = ForecastModel(window=2., overlap=1., station = e2[:-2],
                look_forward=2., data_streams=ds, savefile_type='csv')
            #
            if plot_2_pairs_erups:
                fm_e1a = ForecastModel(window=2., overlap=1., station = e1a[:-2],
                look_forward=2., data_streams=ds, savefile_type='csv')
                fm_e2a = ForecastModel(window=2., overlap=1., station = e2a[:-2],
                look_forward=2., data_streams=ds, savefile_type='csv')
        
            # initial and final time of interest for each station
            tf_e1 = fm_e1.data.tes[int(e1[-1:])-1]
            ti_e1 = tf_e1 - 30*day #month
            tf_e2 = fm_e2.data.tes[int(e2[-1:])-1]
            ti_e2 = tf_e2 - 30*day
            #
            if plot_2_pairs_erups:
                tf_e1a = fm_e1a.data.tes[int(e1a[-1:])-1]
                ti_e1a = tf_e1a - 30*day #month
                tf_e2a = fm_e2a.data.tes[int(e2a[-1:])-1]
                ti_e2a = tf_e2a - 30*day
            #
            ft_e1 = fm_e1.get_features(ti=ti_e1, tf=tf_e1, n_jobs=1, compute_only_features=[ft_nm_aux])
            ft_e2 = fm_e2.get_features(ti=ti_e2, tf=tf_e2, n_jobs=1, compute_only_features=[ft_nm_aux])
            # extract values to plot 
            ft_e1_t = ft_e1[0].index.values
            ft_e1_v = ft_e1[0].loc[:,ft_nm_aux]
            ft_e2_t = ft_e2[0].index.values
            ft_e2_v = ft_e2[0].loc[:,ft_nm_aux]
            #
            if plot_2_pairs_erups:
                ft_e1a = fm_e1a.get_features(ti=ti_e1a, tf=tf_e1a, n_jobs=1, compute_only_features=[ft_nm_aux])
                ft_e2a = fm_e2a.get_features(ti=ti_e2a, tf=tf_e2a, n_jobs=1, compute_only_features=[ft_nm_aux])
                # extract values to plot 
                ft_e1a_t = ft_e1a[0].index.values
                ft_e1a_v = ft_e1a[0].loc[:,ft_nm_aux]
                ft_e2a_t = ft_e2a[0].index.values
                ft_e2a_v = ft_e2a[0].loc[:,ft_nm_aux]

            # import datastream to plot 

            # adding multiple Axes objects  
            if not plot_2_pairs_erups:
                fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12,4))
            else:
                fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6), (ax7, ax8)) = plt.subplots(4, 2, figsize=(10,6))
            
            ## ax1 and ax2
            ax1.plot(ft_e1_t, ft_e1_v, '-', color='b', label=erup_dict[e1])
            te = fm_e1.data.tes[int(e1[-1:])-1]
            ax1.axvline(te, color='k', linestyle='--', linewidth=2, zorder = 1)
            ax1.plot([], color='k', linestyle='--', linewidth=2, label = 'eruption')
            ax1.grid()
            ax1.set_ylabel('feature value')
            #ax1.set_xlabel('time')
            ax1.legend(loc = 2)
            # plot data related to feature
            # plot datastream e1
            dat = fm_e1.data.get_data(ti = ti_e1, tf = tf_e1)
            # hf
            nm = 'dsarF'
            _nm = 'dsar'
            #
            ax1b = ax1.twinx() 
            ax1b.plot(dat.index.values, dat[nm].values, color='c', linestyle='-', linewidth=1, 
                label = _nm+' data' , alpha = .5, zorder = 0)
            #ax1b.set_ylabel(nm)
            #ax1b.set_ylim([0,1050])
            ax1b.legend(loc = 3)
            ax1.set_xticks([dat.index[-1]-7*day*i for i in range(5)])#[dat.index.values[0],dat.index.values[-1]])#, ]np.arange(0, len(x)+1, 5))

            #plt.xticks(rotation=45)
            if True: # plot data streams
                # plot datastream e1
                dat = fm_e1.data.get_data(ti = ti_e1, tf = tf_e1)
                # hf
                nm = 'hfF'
                _nm = 'hf'
                ax2.plot(dat.index.values, dat[nm].values, color='b', linestyle='-', linewidth=1, 
                    label = _nm+' data' , alpha = .8, zorder = 1)
                #ax1.set_ylabel(nm)
                ax2.set_ylim([0,1050])
                #ax2.legend(loc = 2)
                #ax2.set_yticks([])
                # mf
                nm = 'mfF'
                _nm = 'mf'
                #ax2b = ax2.twinx()  # instantiate a second axes that shares the same x-axis
                ax2.plot(dat.index.values, dat[nm].values, color='r', linestyle='-', linewidth=1, 
                    label = _nm+' data' , alpha = .8, zorder = 0)
                #ax2b.set_ylim([0,1050])
                #ax1b.set_ylabel(nm)
                #ax2b.legend(loc = 3)
                #ax2b.set_yticks([])
                ax2.grid()
                ax2.legend(loc = 2)
                ax2.axvline(te, color='k', linestyle='--', linewidth=2, zorder = 1)
                ax2.set_xticks([dat.index[-1]-7*day*i for i in range(5)])#[dat.index.values[0],dat.index.values[-1]])#, ]np.arange(0, len(x)+1, 5))

            ## ax3 and ax4
            ax3.plot(ft_e2_t, ft_e2_v, '-', color='b', label=erup_dict[e2])
            te = fm_e2.data.tes[int(e2[-1:])-1]
            ax3.axvline(te, color='k', linestyle='--', linewidth=2, zorder = 1)
            ax3.plot([], color='k', linestyle='--', linewidth=2, label = 'eruption')
            ax3.grid()
            ax3.set_ylabel('feature value')
            #ax3.set_xlabel('time')
            ax3.legend(loc = 2)
            # plot data related to feature
            # plot datastream e1
            dat = fm_e2.data.get_data(ti = ti_e2, tf = tf_e2)
            # hf
            nm = 'dsarF'
            _nm = 'dsar'
            #
            ax3b = ax3.twinx() 
            ax3b.plot(dat.index.values, dat[nm].values, color='c', linestyle='-', linewidth=1, 
                label = _nm+' data' , alpha = .5, zorder = 0)
            #ax1b.set_ylabel(nm)
            #ax1b.set_ylim([0,1050])
            ax3b.legend(loc = 3)
            ax3.set_xticks([dat.index[-1]-7*day*i for i in range(5)])#[dat.index.values[0],dat.index.values[-1]])#, ]np.arange(0, len(x)+1, 5))

            #plt.xticks(rotation=45)
            if True: # plot data streams
                # plot datastream e1
                dat = fm_e2.data.get_data(ti = ti_e2, tf = tf_e2)
                # hf
                nm = 'hfF'
                _nm = 'hf'
                ax4.plot(dat.index.values, dat[nm].values, color='b', linestyle='-', linewidth=1, 
                    label = _nm+' data' , alpha = .8, zorder = 0)
                #ax1.set_ylabel(nm)
                #ax4.set_ylim([0,1050])
                ax4.legend(loc = 2)
                #ax2.set_yticks([])
                # mf
                nm = 'mfF'
                _nm = 'mf'
                #ax4b = ax4.twinx()  # instantiate a second axes that shares the same x-axis
                ax4.plot(dat.index.values, dat[nm].values, color='r', linestyle='-', linewidth=1, 
                    label = _nm+' data' , alpha = .8, zorder = 1)
                #ax4b.set_ylim([0,1050])
                #ax1b.set_ylabel(nm)
                #ax4b.legend(loc = 3)
                #ax2b.set_yticks([])
                ax4.grid()
                ax4.legend(loc = 2)
                ax4.axvline(te, color='k', linestyle='--', linewidth=2, zorder = 2)
                ax4.set_xticks([dat.index[-1]-7*day*i for i in range(5)])#[dat.index.values[0],dat.index.values[-1]])#, ]np.arange(0, len(x)+1, 5))
            #
            if plot_2_pairs_erups:
                ## ax5 and ax6
                ax5.plot(ft_e1a_t, ft_e1a_v, '-', color='b', label=erup_dict[e1a])
                te = fm_e1a.data.tes[int(e1a[-1:])-1]
                ax5.axvline(te, color='k', linestyle='--', linewidth=2, zorder = 1)
                ax5.plot([], color='k', linestyle='--', linewidth=2, label = 'eruption')
                ax5.grid()
                ax5.set_ylabel('feature value')
                #ax5.set_xlabel('time')
                ax5.legend(loc = 2)
                # plot data related to feature
                # plot datastream e1
                dat = fm_e1a.data.get_data(ti = ti_e1a, tf = tf_e1a)
                # hf
                nm = 'dsarF'
                _nm = 'dsar'
                #
                ax5b = ax5.twinx() 
                ax5b.plot(dat.index.values, dat[nm].values, color='c', linestyle='-', linewidth=1, 
                    label = _nm+' data' , alpha = .5, zorder = 0)
                #ax1b.set_ylabel(nm)
                #ax1b.set_ylim([0,1050])
                ax5b.legend(loc = 3)
                ax5.set_xticks([dat.index[-1]-7*day*i for i in range(5)])#[dat.index.values[0],dat.index.values[-1]])#, ]np.arange(0, len(x)+1, 5))

                #plt.xticks(rotation=45)
                if True: # plot data streams
                    # plot datastream e1
                    dat = fm_e1a.data.get_data(ti = ti_e1a, tf = tf_e1a)
                    # hf
                    nm = 'hfF'
                    _nm = 'hf'
                    ax6.plot(dat.index.values, dat[nm].values, color='b', linestyle='-', linewidth=1, 
                        label = _nm+' data' , alpha = .8, zorder = 1)
                    #ax6.set_ylabel(nm)
                    ax6.set_ylim([0,1050])
                    #ax6.legend(loc = 2)
                    #ax6.set_yticks([])
                    # mf
                    nm = 'mfF'
                    _nm = 'mf'
                    #ax6b = ax6.twinx()  # instantiate a second axes that shares the same x-axis
                    ax6.plot(dat.index.values, dat[nm].values, color='r', linestyle='-', linewidth=1, 
                        label = _nm+' data' , alpha = .8, zorder = 0)
                    ax6.set_ylim([0,700])
                    #ax6b.set_ylabel(nm)
                    #ax6b.legend(loc = 3)
                    #ax6b.set_yticks([])
                    ax6.grid()
                    ax6.legend(loc = 2)
                    ax6.axvline(te, color='k', linestyle='--', linewidth=2, zorder = 1)
                    ax6.set_xticks([dat.index[-1]-7*day*i for i in range(5)])#[dat.index.values[0],dat.index.values[-1]])#, ]np.arange(0, len(x)+1, 5))


                ## ax7 and ax8
                ax7.plot(ft_e2a_t, ft_e2a_v, '-', color='b', label=erup_dict[e2a])
                te = fm_e2a.data.tes[int(e2a[-1:])-1]
                ax7.axvline(te, color='k', linestyle='--', linewidth=2, zorder = 1)
                ax7.plot([], color='k', linestyle='--', linewidth=2, label = 'eruption')
                ax7.grid()
                ax7.set_ylabel('feature value')
                #ax7.set_xlabel('time')
                ax7.legend(loc = 2)
                # plot data related to feature
                # plot datastream e1
                dat = fm_e2a.data.get_data(ti = ti_e2a, tf = tf_e2a)
                # hf
                nm = 'dsarF'
                _nm = 'dsar'
                #
                ax7b = ax7.twinx() 
                ax7b.plot(dat.index.values, dat[nm].values, color='c', linestyle='-', linewidth=1, 
                    label = _nm+' data' , alpha = .5, zorder = 0)
                #ax1b.set_ylabel(nm)
                #ax1b.set_ylim([0,1050])
                ax7b.legend(loc = 3)
                ax7.set_xticks([dat.index[-1]-7*day*i for i in range(5)])#[dat.index.values[0],dat.index.values[-1]])#, ]np.arange(0, len(x)+1, 5))
                #plt.xticks(rotation=45)
                if True: # plot data streams
                    # plot datastream e1
                    dat = fm_e2a.data.get_data(ti = ti_e2a, tf = tf_e2a)
                    # hf
                    nm = 'hfF'
                    _nm = 'hf'
                    ax8.plot(dat.index.values, dat[nm].values, color='b', linestyle='-', linewidth=1, 
                        label = _nm+' data' , alpha = .8, zorder = 0)
                    #ax1.set_ylabel(nm)
                    #ax4.set_ylim([0,1050])
                    ax8.legend(loc = 2)
                    #ax2.set_yticks([])
                    # mf
                    nm = 'mfF'
                    _nm = 'mf'
                    #ax4b = ax4.twinx()  # instantiate a second axes that shares the same x-axis
                    ax8.plot(dat.index.values, dat[nm].values, color='r', linestyle='-', linewidth=1, 
                        label = _nm+' data' , alpha = .8, zorder = 1)
                    ax8.set_ylim([0,300])
                    #ax1b.set_ylabel(nm)
                    #ax4b.legend(loc = 3)
                    #ax2b.set_yticks([])
                    ax8.grid()
                    ax8.legend(loc = 2)
                    ax8.axvline(te, color='k', linestyle='--', linewidth=2, zorder = 2)
                    ax8.set_xticks([dat.index[-1]-7*day*i for i in range(5)])#[dat.index.values[0],dat.index.values[-1]])#, ]np.arange(0, len(x)+1, 5))


            fig.suptitle('Feature: '+ ft_nm_aux, ha='center')
            plt.tight_layout()
            plt.show()

    # plot 4 ratios together for the same feature (dsar, rmar, lrar, vlar)
    if False: 
        # 
        plot_2_pairs_erups =  True
        # import file as pandas
        path = '..'+os.sep+'features'+os.sep+'correlations'+os.sep+'corr_0_rank_erup_cc.csv'
        try:
            pd_rank = pd.read_csv(path, index_col=0)
        except: # convert txt to csv
            read_file = pd.read_csv ('..'+os.sep+'features'+os.sep+'correlations'+os.sep+'corr_0_rank_erup_cc.txt')
            read_file.to_csv(path, index=None)
            #
            pd_rank = pd.read_csv(path, index_col=0)
            del read_file
        # locate index in rank for cc cut value of 0.5
        idx, _ = find_nearest(pd_rank.loc[:, 'cc'].values, 0.5)
        # for plotting: import features of eruptions per rank
        ranks = np.arange(0,idx,1)
        #
        for rank in [713]:#[]: #ranks:713, 373
            # name of eruptions to plot
            e1 = pd_rank.loc[rank, 'erup1'] # name of eruptions to plot
            e2 = pd_rank.loc[rank, 'erup2']
            # feature to plot
            ft_nm = pd_rank.loc[rank, 'featNM']
            ft_id = pd_rank.loc[rank, 'featID']
            cc = pd_rank.loc[rank, 'cc']
            ## import features
            # proper name of feature
            ft_nm_aux = ft_nm.replace(" ","__")
            ft_nm_aux = ft_nm_aux.replace("-",'"')
            # adding multiple Axes objects
            one_erup = False  
            look_back = 30*day
            if False: # modified second eruption to plot
                e2 = 'WIZ_1'
            #    
            if one_erup: # one eruption 
                fig, ((ax1, ax3, ax5, ax7)) = plt.subplots(4, 1, figsize=(5,6))
            else: # two eruption
                fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6), (ax7, ax8)) = plt.subplots(4, 2, figsize=(10,6))
            if True: # dsar
                # feature to plot
                ft_nm = pd_rank.loc[rank, 'featNM']
                ft_id = pd_rank.loc[rank, 'featID']
                cc = pd_rank.loc[rank, 'cc']
                ## import features
                # proper name of feature
                ft_nm_aux = ft_nm.replace(" ","__")
                ft_nm_aux = ft_nm_aux.replace("-",'"')
                # name of the data stream 

                ds = ['zsc2_dsarF']
                # create objects for each station 
                fm_e1 = ForecastModel(window=2., overlap=1., station = e1[:-2],
                    look_forward=2., data_streams=ds, savefile_type='csv')
                fm_e2 = ForecastModel(window=2., overlap=1., station = e2[:-2],
                    look_forward=2., data_streams=ds, savefile_type='csv')
                #
                # create objects for each station 
                #path_feat_serv = 'C:\\Users\\aar135\\codes_local_disk\\volc_forecast_tl\\features_bkp\\features_server\\'
                #fm_e1 = ForecastModel(window=2., overlap=1., station = e1[:-2],
                #    look_forward=2., data_streams=ds, 
                #    feature_dir=path_feat_serv, 
                #    savefile_type='pkl')
                #fm_e2 = ForecastModel(window=2., overlap=1., station = e2[:-2],
                #    look_forward=2., data_streams=ds, 
                #    feature_dir=path_feat_serv, 
                #    savefile_type='pkl')
                #
                # initial and final time of interest for each station
                tf_e1 = fm_e1.data.tes[int(e1[-1:])-1]
                ti_e1 = tf_e1 - look_back #month
                tf_e2 = fm_e2.data.tes[int(e2[-1:])-1]
                ti_e2 = tf_e2 - look_back
                #
                ft_e1 = fm_e1.get_features(ti=ti_e1, tf=tf_e1, n_jobs=1, compute_only_features=[ft_nm_aux])
                ft_e2 = fm_e2.get_features(ti=ti_e2, tf=tf_e2, n_jobs=1, compute_only_features=[ft_nm_aux])
                # extract values to plot 
                ft_e1_t = ft_e1[0].index.values
                ft_e1_v = ft_e1[0].loc[:,ft_nm_aux]
                ft_e2_t = ft_e2[0].index.values
                ft_e2_v = ft_e2[0].loc[:,ft_nm_aux]
                #
                # import datastream to plot 
                ## ax1 and ax2
                ax1.plot(ft_e1_t, ft_e1_v, '-', color='b', label=erup_dict[e1])
                te = fm_e1.data.tes[int(e1[-1:])-1]
                ax1.axvline(te, color='k', linestyle='--', linewidth=2, zorder = 1)
                ax1.plot([], color='k', linestyle='--', linewidth=2, label = 'eruption')
                ax1.grid()
                ax1.set_ylabel('feature value')
                #ax1.set_xlabel('time')
                #ax1.legend(loc = 2)
                # plot data related to feature
                # plot datastream e1
                dat = fm_e1.data.get_data(ti = ti_e1, tf = tf_e1)
                # hf
                nm = 'dsarF'
                _nm = 'dsar'
                #
                ax1b = ax1.twinx() 
                ax1b.plot(dat.index.values, dat[nm].values, color='c', linestyle='-', linewidth=1, 
                    label = _nm+' data' , alpha = .5, zorder = 0)
                #ax1b.set_ylabel(nm)
                #ax1b.set_ylim([0,1050])
                ax1b.legend(loc = 2)
                ax1.set_xticks([dat.index[-1]-7*day*i for i in range(5)])#[dat.index.values[0],dat.index.values[-1]])#, ]np.arange(0, len(x)+1, 5))

                ax1.set_title('Eruption: '+ erup_dict[e1]+'\n Feature: '+ft_nm_aux.split('_')[-1])

                ## ax3 and ax4
                if not one_erup:
                    ax2.plot(ft_e2_t, ft_e2_v, '-', color='b', label=erup_dict[e2])
                    te = fm_e2.data.tes[int(e2[-1:])-1]
                    ax2.axvline(te, color='k', linestyle='--', linewidth=2, zorder = 1)
                    ax2.plot([], color='k', linestyle='--', linewidth=2, label = 'eruption')
                    ax2.grid()
                    ax2.set_ylabel('feature value')
                    #ax3.set_xlabel('time')
                    #ax2.legend(loc = 2)
                    # plot data related to feature
                    # plot datastream e1
                    dat = fm_e2.data.get_data(ti = ti_e2, tf = tf_e2)
                    # hf
                    nm = 'dsarF'
                    _nm = 'dsar'
                    #
                    ax2b = ax2.twinx() 
                    ax2b.plot(dat.index.values, dat[nm].values, color='c', linestyle='-', linewidth=1, 
                        label = _nm+' data' , alpha = .5, zorder = 0)
                    #ax1b.set_ylabel(nm)
                    #ax1b.set_ylim([0,1050])
                    ax2b.legend(loc = 2)
                    ax2.set_xticks([dat.index[-1]-7*day*i for i in range(5)])#[dat.index.values[0],dat.index.values[-1]])#, ]np.arange(0, len(x)+1, 5))

                    ax2.set_title('Eruption: '+ erup_dict[e2]+'\n Feature: '+ft_nm_aux.split('_')[-1])
                    #plt.xticks(rotation=45)

            if True: # rmar
                # feature to plot
                ft_nm = pd_rank.loc[rank, 'featNM']
                ft_id = pd_rank.loc[rank, 'featID']
                cc = pd_rank.loc[rank, 'cc']
                ## import features
                # proper name of feature
                ft_nm_aux = ft_nm.replace(" ","__")
                ft_nm_aux = ft_nm_aux.replace("-",'"')
                #
                ft_nm_aux = ft_nm_aux.replace('dsar', 'rmar') #string.replace(old, new, count)
                # name of the data stream 
                ds = ['zsc2_rmarF']
                # create objects for each station 
                path_feat_serv = 'C:\\Users\\aar135\\codes_local_disk\\volc_forecast_tl\\features_bkp\\features_server\\'
                fm_e1 = ForecastModel(window=2., overlap=1., station = e1[:-2],
                    look_forward=2., data_streams=ds, 
                    feature_dir=path_feat_serv, 
                    savefile_type='pkl')
                fm_e2 = ForecastModel(window=2., overlap=1., station = e2[:-2],
                    look_forward=2., data_streams=ds, 
                    feature_dir=path_feat_serv, 
                    savefile_type='pkl')
                #
                # initial and final time of interest for each station
                tf_e1 = fm_e1.data.tes[int(e1[-1:])-1]
                ti_e1 = tf_e1 - look_back #month
                tf_e2 = fm_e2.data.tes[int(e2[-1:])-1]
                ti_e2 = tf_e2 - look_back
                #
                ft_e1 = fm_e1.get_features(ti=ti_e1, tf=tf_e1, n_jobs=1, compute_only_features=[ft_nm_aux])
                ft_e2 = fm_e2.get_features(ti=ti_e2, tf=tf_e2, n_jobs=1, compute_only_features=[ft_nm_aux])
                # extract values to plot 
                ft_e1_t = ft_e1[0].index.values
                ft_e1_v = ft_e1[0].loc[:,ft_nm_aux]
                ft_e2_t = ft_e2[0].index.values
                ft_e2_v = ft_e2[0].loc[:,ft_nm_aux]
                #
                # import datastream to plot 
                ## ax1 and ax2
                ax3.plot(ft_e1_t, ft_e1_v, '-', color='b', label=erup_dict[e1])
                te = fm_e1.data.tes[int(e1[-1:])-1]
                ax3.axvline(te, color='k', linestyle='--', linewidth=2, zorder = 1)
                ax3.plot([], color='k', linestyle='--', linewidth=2, label = 'eruption')
                ax3.grid()
                ax3.set_ylabel('feature value')
                #ax1.set_xlabel('time')
                #ax3.legend(loc = 2)
                # plot data related to feature
                # plot datastream e1
                dat = fm_e1.data.get_data(ti = ti_e1, tf = tf_e1)
                # hf
                nm = 'rmarF'
                _nm = 'rmar'
                #
                ax3b = ax3.twinx() 
                ax3b.plot(dat.index.values, dat[nm].values, color='c', linestyle='-', linewidth=1, 
                    label = _nm+' data' , alpha = .5, zorder = 0)
                #ax1b.set_ylabel(nm)
                #ax1b.set_ylim([0,1050])
                ax3b.legend(loc = 2)
                ax3.set_xticks([dat.index[-1]-7*day*i for i in range(5)])#[dat.index.values[0],dat.index.values[-1]])#, ]np.arange(0, len(x)+1, 5))

                ## ax3 and ax4
                if not one_erup:
                    ax4.plot(ft_e2_t, ft_e2_v, '-', color='b', label=erup_dict[e2])
                    te = fm_e2.data.tes[int(e2[-1:])-1]
                    ax4.axvline(te, color='k', linestyle='--', linewidth=2, zorder = 1)
                    ax4.plot([], color='k', linestyle='--', linewidth=2, label = 'eruption')
                    ax4.grid()
                    ax4.set_ylabel('feature value')
                    #ax3.set_xlabel('time')
                    #ax4.legend(loc = 2)
                    # plot data related to feature
                    # plot datastream e1
                    dat = fm_e2.data.get_data(ti = ti_e2, tf = tf_e2)
                    # hf
                    nm = 'rmarF'
                    _nm = 'rmar'
                    #
                    ax4b = ax4.twinx() 
                    ax4b.plot(dat.index.values, dat[nm].values, color='c', linestyle='-', linewidth=1, 
                        label = _nm+' data' , alpha = .5, zorder = 0)
                    #ax1b.set_ylabel(nm)
                    #ax1b.set_ylim([0,1050])
                    ax4b.legend(loc = 2)
                    ax4.set_xticks([dat.index[-1]-7*day*i for i in range(5)])#[dat.index.values[0],dat.index.values[-1]])#, ]np.arange(0, len(x)+1, 5))

                    #plt.xticks(rotation=45)

            if True: # lrar
                #
                ft_nm_aux = ft_nm_aux.replace('rmar', 'lrar') #string.replace(old, new, count)

                ds = ['zsc2_lrarF']
                # create objects for each station 
                path_feat_serv = 'C:\\Users\\aar135\\codes_local_disk\\volc_forecast_tl\\features_bkp\\features_server\\'
                fm_e1 = ForecastModel(window=2., overlap=1., station = e1[:-2],
                    look_forward=2., data_streams=ds, 
                    feature_dir=path_feat_serv, 
                    savefile_type='pkl')
                fm_e2 = ForecastModel(window=2., overlap=1., station = e2[:-2],
                    look_forward=2., data_streams=ds, 
                    feature_dir=path_feat_serv, 
                    savefile_type='pkl')
                #
                # initial and final time of interest for each station
                tf_e1 = fm_e1.data.tes[int(e1[-1:])-1]
                ti_e1 = tf_e1 - look_back #month
                tf_e2 = fm_e2.data.tes[int(e2[-1:])-1]
                ti_e2 = tf_e2 - look_back
                #
                ft_e1 = fm_e1.get_features(ti=ti_e1, tf=tf_e1, n_jobs=1, compute_only_features=[ft_nm_aux])
                ft_e2 = fm_e2.get_features(ti=ti_e2, tf=tf_e2, n_jobs=1, compute_only_features=[ft_nm_aux])
                # extract values to plot 
                ft_e1_t = ft_e1[0].index.values
                ft_e1_v = ft_e1[0].loc[:,ft_nm_aux]
                ft_e2_t = ft_e2[0].index.values
                ft_e2_v = ft_e2[0].loc[:,ft_nm_aux]
                #
                # import datastream to plot 
                ## ax1 and ax2
                ax5.plot(ft_e1_t, ft_e1_v, '-', color='b', label=erup_dict[e1])
                te = fm_e1.data.tes[int(e1[-1:])-1]
                ax5.axvline(te, color='k', linestyle='--', linewidth=2, zorder = 1)
                ax5.plot([], color='k', linestyle='--', linewidth=2, label = 'eruption')
                ax5.grid()
                ax5.set_ylabel('feature value')
                #ax1.set_xlabel('time')
                #ax7.legend(loc = 2)
                # plot data related to feature
                # plot datastream e1
                dat = fm_e1.data.get_data(ti = ti_e1, tf = tf_e1)
                # hf
                nm = 'lrarF'
                _nm = 'lrar'
                #
                ax5b = ax5.twinx() 
                ax5b.plot(dat.index.values, dat[nm].values, color='c', linestyle='-', linewidth=1, 
                    label = _nm+' data' , alpha = .5, zorder = 0)
                #ax1b.set_ylabel(nm)
                #ax1b.set_ylim([0,1050])
                ax5b.legend(loc = 2)
                ax5.set_xticks([dat.index[-1]-7*day*i for i in range(5)])#[dat.index.values[0],dat.index.values[-1]])#, ]np.arange(0, len(x)+1, 5))

                ## ax3 and ax4
                if not one_erup:
                    ax6.plot(ft_e2_t, ft_e2_v, '-', color='b', label=erup_dict[e2])
                    te = fm_e2.data.tes[int(e2[-1:])-1]
                    ax6.axvline(te, color='k', linestyle='--', linewidth=2, zorder = 1)
                    ax6.plot([], color='k', linestyle='--', linewidth=2, label = 'eruption')
                    ax6.grid()
                    ax6.set_ylabel('feature value')
                    #ax3.set_xlabel('time')
                    #ax8.legend(loc = 2)
                    # plot data related to feature
                    # plot datastream e1
                    dat = fm_e2.data.get_data(ti = ti_e2, tf = tf_e2)
                    # hf
                    nm = 'lrarF'
                    _nm = 'lrar'
                    #
                    ax6b = ax8.twinx() 
                    ax6b.plot(dat.index.values, dat[nm].values, color='c', linestyle='-', linewidth=1, 
                        label = _nm+' data' , alpha = .5, zorder = 0)
                    #ax1b.set_ylabel(nm)
                    #ax1b.set_ylim([0,1050])
                    ax6b.legend(loc = 2)
                    ax6.set_xticks([dat.index[-1]-7*day*i for i in range(5)])#[dat.index.values[0],dat.index.values[-1]])#, ]np.arange(0, len(x)+1, 5))
                    #plt.xticks(rotation=45)

            if True: # vlar
                #
                ft_nm_aux = ft_nm_aux.replace('lrar', 'vlar') #string.replace(old, new, count)
                # name of the data stream 
                ds = ['zsc2_vlarF']
                # create objects for each station 
                path_feat_serv = 'C:\\Users\\aar135\\codes_local_disk\\volc_forecast_tl\\features_bkp\\features_server\\'
                fm_e1 = ForecastModel(window=2., overlap=1., station = e1[:-2],
                    look_forward=2., data_streams=ds, 
                    feature_dir=path_feat_serv, 
                    savefile_type='pkl')
                fm_e2 = ForecastModel(window=2., overlap=1., station = e2[:-2],
                    look_forward=2., data_streams=ds, 
                    feature_dir=path_feat_serv, 
                    savefile_type='pkl')
                #
                # initial and final time of interest for each station
                tf_e1 = fm_e1.data.tes[int(e1[-1:])-1]
                ti_e1 = tf_e1 - look_back #month
                tf_e2 = fm_e2.data.tes[int(e2[-1:])-1]
                ti_e2 = tf_e2 - look_back
                #
                ft_e1 = fm_e1.get_features(ti=ti_e1, tf=tf_e1, n_jobs=1, compute_only_features=[ft_nm_aux])
                ft_e2 = fm_e2.get_features(ti=ti_e2, tf=tf_e2, n_jobs=1, compute_only_features=[ft_nm_aux])
                # extract values to plot 
                ft_e1_t = ft_e1[0].index.values
                ft_e1_v = ft_e1[0].loc[:,ft_nm_aux]
                ft_e2_t = ft_e2[0].index.values
                ft_e2_v = ft_e2[0].loc[:,ft_nm_aux]
                #
                # import datastream to plot 
                ## ax1 and ax2
                ax7.plot(ft_e1_t, ft_e1_v, '-', color='b', label=erup_dict[e1])
                te = fm_e1.data.tes[int(e1[-1:])-1]
                ax7.axvline(te, color='k', linestyle='--', linewidth=2, zorder = 1)
                ax7.plot([], color='k', linestyle='--', linewidth=2, label = 'eruption')
                ax7.grid()
                ax7.set_ylabel('feature value')
                #ax1.set_xlabel('time')
                #ax5.legend(loc = 2)
                # plot data related to feature
                # plot datastream e1
                dat = fm_e1.data.get_data(ti = ti_e1, tf = tf_e1)
                # hf
                nm = 'vlarF'
                _nm = 'vlar'
                #
                ax7b = ax7.twinx() 
                ax7b.plot(dat.index.values, dat[nm].values, color='c', linestyle='-', linewidth=1, 
                    label = _nm+' data' , alpha = .5, zorder = 0)
                #ax1b.set_ylabel(nm)
                #ax1b.set_ylim([0,1050])
                ax7b.legend(loc = 2)
                ax7.set_xticks([dat.index[-1]-7*day*i for i in range(5)])#[dat.index.values[0],dat.index.values[-1]])#, ]np.arange(0, len(x)+1, 5))

                ## ax3 and ax4
                if not one_erup:
                    ax8.plot(ft_e2_t, ft_e2_v, '-', color='b', label=erup_dict[e2])
                    te = fm_e2.data.tes[int(e2[-1:])-1]
                    ax8.axvline(te, color='k', linestyle='--', linewidth=2, zorder = 1)
                    ax8.plot([], color='k', linestyle='--', linewidth=2, label = 'eruption')
                    ax8.grid()
                    ax8.set_ylabel('feature value')
                    #ax3.set_xlabel('time')
                    #ax6.legend(loc = 2)
                    # plot data related to feature
                    # plot datastream e1
                    dat = fm_e2.data.get_data(ti = ti_e2, tf = tf_e2)
                    # hf
                    nm = 'vlarF'
                    _nm = 'vlar'
                    #
                    ax8b = ax8.twinx() 
                    ax8b.plot(dat.index.values, dat[nm].values, color='c', linestyle='-', linewidth=1, 
                        label = _nm+' data' , alpha = .5, zorder = 0)
                    of
                    ax1b.set_ylabel(nm)
                    #ax1b.set_ylim([0,1050])
                    ax8b.legend(loc = 2)
                    ax8.set_xticks([dat.index[-1]-7*day*i for i in range(5)])#[dat.index.values[0],dat.index.values[-1]])#, ]np.arange(0, len(x)+1, 5))

                    #plt.xticks(rotation=45)

            #fig.suptitle('Feature: '+ ft_nm_aux, ha='center')
            plt.tight_layout()
            #
            plt.show()

    # plot same feature for all the eruptions (and subsets)
    if False: 
        import numpy as np
        # import file as pandas
        path = '..'+os.sep+'features'+os.sep+'correlations'+os.sep+'corr_0_rank_erup_cc.csv'
        try:
            pd_rank = pd.read_csv(path, index_col=0)
        except: # convert txt to csv
            read_file = pd.read_csv ('..'+os.sep+'features'+os.sep+'correlations'+os.sep+'corr_0_rank_erup_cc.txt')
            read_file.to_csv(path, index=None)
            #
            pd_rank = pd.read_csv(path, index_col=0)
            del read_file
        # locate index in rank for cc cut value of 0.5
        idx, _ = find_nearest(pd_rank.loc[:, 'cc'].values, 0.5)
        # for plotting: import features of eruptions per rank
        ranks = np.arange(0,idx,1)
        #
        rank = 713#373#713#262
        lb = 30
        look_back = lb*day
        #
        if False: # plot all eruptions
            # list of eruptions
            erup_list = ['WIZ_1','WIZ_2','WIZ_3','WIZ_4','WIZ_5',
                'FWVZ_1','FWVZ_2','FWVZ_3',
                'KRVZ_1','KRVZ_2',
                'BELO_1','BELO_2','BELO_3',
                'PVV_1','PVV_2','PVV_3','VNSS_1','VNSS_2'
                ]  
            nrow = int(len(erup_list)/2)
            ncol = 2
            fig, axs = plt.subplots(nrows=nrow, ncols=ncol,figsize=(15,20))

            for i, ax in enumerate(axs.reshape(-1)): 
                #
                ds = ['zsc2_dsarF']
                if rank == 262:
                    ds = ['zsc2_hfF']
                if rank == 252:
                    ds = ['log_zsc2_rsamF']
                # create objects for each station

                if erup_list[i] in ['FWVZ_3']:
                        path_feat_serv = 'C:\\Users\\aar135\\codes_local_disk\\volc_forecast_tl\\features_bkp\\features_server\\'
                        fm_e1 = ForecastModel(window=2., overlap=1., station =  erup_list[i][:-2],
                        look_forward=2., data_streams=ds, 
                        feature_dir=path_feat_serv, 
                        savefile_type='pkl') 
                else:
                    fm_e1 = ForecastModel(window=2., overlap=1., station = erup_list[i][:-2],
                        look_forward=2., data_streams=ds, savefile_type='csv')
                # initial and final time of interest for each station
                tf_e1 = fm_e1.data.tes[int(erup_list[i][-1:])-1]
                ti_e1 = tf_e1 - look_back #month
                #
                # feature to plot
                ft_nm = pd_rank.loc[rank, 'featNM']
                ft_id = pd_rank.loc[rank, 'featID']
                cc = pd_rank.loc[rank, 'cc']
                ## import features
                # proper name of feature
                ft_nm_aux = ft_nm.replace(" ","__")
                ft_nm_aux = ft_nm_aux.replace("-",'"')
                # adding multiple Axes objects
                ft_e1 = fm_e1.get_features(ti=ti_e1, tf=tf_e1, n_jobs=1, compute_only_features=[ft_nm_aux])
                # extract values to plot 
                ft_e1_t = ft_e1[0].index.values
                ft_e1_v = ft_e1[0].loc[:,ft_nm_aux]
                #
                # import datastream to plot 
                ## ax1 and ax2
                ax.plot(ft_e1_t, ft_e1_v, '-', color='b', label=erup_dict[erup_list[i]])
                te = fm_e1.data.tes[int(erup_list[i][-1:])-1]
                ax.axvline(te, color='k', linestyle='--', linewidth=2, zorder = 1)
                ax.plot([], color='k', linestyle='--', linewidth=2, label = 'eruption')
                ax.grid()
                ax.set_ylabel('feature value')
                ax.legend(loc = 2)
                #ax.set_xticks([ft_e1_t[-1] - look_back, ft_e1_t[-1]])#[dat.index.values[0],dat.index.values[-1]])#, ]np.arange(0, len(x)+1, 5))

                if True: # ffm 
                    ax2 = ax.twinx() 
                    #v_plot = data[data_stream].loc[inds]
                    inv_rsam = fm_e1.data.get_data(ti=ti_e1, tf=tf_e1)['rsamF']#.loc[ft_e1_t]
                    inv_rsam = 1./inv_rsam
                    ax2.plot(ft_e1_t, inv_rsam, 'o', color= 'k', markersize=1, label='1/rsam')
                
                ax.set_xticks([ft_e1[0].index[-1] - 2*day*i for i in range(int(lb/2))])#[dat.index.values[0],dat.index.values[-1]])#, ]np.arange(0, len(x)+1, 5))

        if True: # plot selected eruptions 
            # Figure for dsarF median
            if rank == 713:
                #            
                nrow = 8
                ncol = 1
                fig, axs = plt.subplots(nrows=nrow, ncols=ncol,figsize=(12,10))
                e_list = ['WIZ_5',
                            'VNSS_1',
                            'WIZ_4',
                            'WIZ_3',
                            'FWVZ_1',
                            'KRVZ_1',
                            'PVV_1',
                            'BELO_3',
                            ]
                #
                ds = ['zsc2_dsarF']
                nrow = 5
                ncol = 1
                fig, axs = plt.subplots(nrows=nrow, ncols=ncol,figsize=(12,8))
                e_list = ['WIZ_5','VNSS_1',
                            'FWVZ_1','KRVZ_1', 'PVV_1']
                #nrow = 4
                #ncol = 3
                #fig, axs = plt.subplots(nrows=nrow, ncols=ncol,figsize=(16,10))
                #e_list = ['OGDI_1','OGDI_2','OGDI_3','OGDI_4','OGDI_5','OGDI_6',
                #        'OGDI_7','OGDI_8','OGDI_9','OGDI_10','OGDI_11','OGDI_12']
                #
                col = ['b' for i in range(len(e_list))]#
            if rank == 262: 
                ds = ['zsc2_hfF']           
                nrow = 5
                ncol = 1
                fig, axs = plt.subplots(nrows=nrow, ncols=ncol,figsize=(12,10))
                e_list = ['WIZ_5','WIZ_4','WIZ_3','WIZ_2','WIZ_1']
                #
                col = ['b' for i in range(len(e_list))]#
            # Figure for dsarF change quantiles .4-.6 variance
            if rank == 373:
                ds = ['zsc2_dsarF']
                nrow = 6
                ncol = 1
                fig, axs = plt.subplots(nrows=nrow, ncols=ncol,figsize=(5,8))
                e_list = ['WIZ_5',
                            'FWVZ_2',
                            'PVV_3',
                            'BELO_2',
                            'VNSS_2',
                            'KRVZ_1']
                col = ['b','r','g','m','coral','yellow']#,'purple','plum','coral'] 
                col = ['r' for i in range(len(e_list))]#                       ]
            #
            for i, ax in enumerate(axs.reshape(-1)): 
                #
                # create objects for each station
                if e_list[i] in ['FWVZ_3']: # 'WIZ_3',
                        path_feat_serv = 'C:\\Users\\aar135\\codes_local_disk\\volc_forecast_tl\\features_bkp\\features_server\\'
                        fm_e1 = ForecastModel(window=2., overlap=1., station =  e_list[i][:-2],
                        look_forward=2., data_streams=ds, 
                        feature_dir=path_feat_serv, 
                        savefile_type='pkl') 
                else:
                    try:
                        fm_e1 = ForecastModel(window=2., overlap=1., station = e_list[i][:-2],
                            look_forward=2., data_streams=ds, savefile_type='csv')
                    except:
                        fm_e1 = ForecastModel(window=2., overlap=1., station = e_list[i][:-3],
                            look_forward=2., data_streams=ds, savefile_type='csv')
                # initial and final time of interest for each station
                tf_e1 = fm_e1.data.tes[int(e_list[i][-1:])-1]
                ti_e1 = tf_e1 - look_back #month
                #
                # feature to plot
                ft_nm = pd_rank.loc[rank, 'featNM']
                ft_id = pd_rank.loc[rank, 'featID']
                cc = pd_rank.loc[rank, 'cc']
                ## import features
                # proper name of feature
                ft_nm_aux = ft_nm.replace(" ","__")
                ft_nm_aux = ft_nm_aux.replace("-",'"')
                # adding multiple Axes objects
                ft_e1 = fm_e1.get_features(ti=ti_e1, tf=tf_e1, n_jobs=1, compute_only_features=[ft_nm_aux])
                # extract values to plot 
                ft_e1_t = ft_e1[0].index.values
                ft_e1_v = ft_e1[0].loc[:,ft_nm_aux]
                #
                # import datastream to plot 
                ## ax1 and ax2
                try:
                    ax.plot(ft_e1_t, ft_e1_v, '-', color=col[i], label=erup_dict[e_list[i]]+', VEI: '+erup_vei_dict[e_list[i]])
                except:
                    ax.plot(ft_e1_t, ft_e1_v, '-', color=col[i], label=str(i+1))
                te = fm_e1.data.tes[int(e_list[i][-1:])-1]
                # plot mean and std (1 up)
                if False: # calc from month
                    ax.plot(ft_e1_t, np.mean(ft_e1_v)*np.ones(len(ft_e1_v)), linestyle='--', linewidth=2.0, color=col[i], alpha =0.5)
                    ax.plot(ft_e1_t, (np.mean(ft_e1_v)+np.std(ft_e1_v))*np.ones(len(ft_e1_v)), linestyle='dotted', linewidth=2., color=col[i], alpha =0.7)
                else: # load
                    mean_std_dic =  {'WIZ': [1,10],
                                'FWVZ': [1,10],
                                'KRVZ': [1,10],
                                'BELO': [1,10],
                                'PVV': [1,10],
                                'VNSS': [1,10],
                                'IVGP': [1,10]
                                }
                    pass

                # eruption
                ax.axvline(te, color='k',linestyle='--', linewidth=2, zorder = 1)
                ax.plot([], color='k', linestyle='--', linewidth=2, label = 'eruption')

                if e_list[i] == 'BELO_3':
                    te = fm_e1.data.tes[int(e_list[i][-1:])-2]
                    ax.axvline(te+2*day, color='k', linestyle='--', linewidth=2, zorder = 1)
                    ax.set_ylim([0,20])
                    #ax.legend(loc = 3)
                #
                ax.grid()
                #ax.set_ylabel('feature value')
                #
                if True: # plot data
                    # plot datastream e1
                    dat = fm_e1.data.get_data(ti = ti_e1, tf = tf_e1)
                    nm = ft_nm_aux.split('__')[0]
                    #nm = '_'.join(nm)
                    axb = ax.twinx()  # instantiate a second axes that shares the same x-axis
                    if 'rsam' in nm:
                        _nm = 'nRSAM'
                        ft_nm_aux = ft_nm_aux.replace('rsam','RSAM')
                    if 'mf' in nm:
                        _nm = 'nMF'
                        ft_nm_aux = ft_nm_aux.replace('mf','MF')
                    if 'hf' in nm:
                        _nm = 'nHF'
                        ft_nm_aux = ft_nm_aux.replace('hf','HF')
                    if 'dsar' in nm:
                        _nm = 'nDSAR'
                        #ft_nm_aux = ft_nm_aux.replace('dsarF','DSAR')
                    # plot
                    axb.plot(dat.index.values, dat[nm].values, color='c', linestyle='-', linewidth=1, 
                         alpha = .3) # label = _nm+' data' ,
                    ax.plot([], [], color='c', linestyle='-', linewidth=1, 
                        label = _nm+' data' , alpha = .3)  
                    # 
                    if e_list[i] == 'PVV_3':
                        axb.set_ylim([0,100])
                    if e_list[i] == 'PVV_1':
                        ax.set_ylim([0,.5])
                    # if e_list[i] == 'KRVZ_1':
                    #     ax.set_ylim([0,5.])
                    else:
                        pass
                    #
                    if 'dsar' in nm:
                        pass
                        #axb.set_ylabel('DSAR')
                    #axb.legend(loc = 3)
                    #axb.set_yticks([])
                #
                ax.set_xticks([ft_e1[0].index[-1] - 7*day*i for i in range(int(lb/7)+1)])#[dat.index.values[0],dat.index.values[-1]])#, ]np.arange(0, len(x)+1, 5))
                #
                if False: #plot mean and std
                    if e_list[i][:-2] == 'WIZ':
                        # mean
                        ax.axhline(y=1, color='b', linestyle='--',linewidth=2, zorder = 1)
                        # std
                        ax.axhline(y=10, color='b', linestyle=':',linewidth=2, zorder = 1)
                #
                if False: # ffm 
                    ax2 = ax.twinx() 
                    #v_plot = data[data_stream].loc[inds]
                    inv_rsam = fm_e1.data.get_data(ti=ti_e1, tf=tf_e1)['rsamF']#.loc[ft_e1_t]
                    inv_rsam = 1./inv_rsam
                    ax2.plot(ft_e1_t, inv_rsam, 'o', color= 'k', label='1/rsam')
                #
                ax.legend(loc = 2, framealpha=1.)
                ax.set_yticks([])
                axb.set_yticks([])
        if False: # plot feature for 2 months (explore longer trends)
            lb = 60
            look_back = lb*day
            #
            ds = ['log_zsc2_rsamF']
            e_list = ['WIZ_5','WIZ_4','FWVZ_2','KRVZ_1']#,
            # 'FWVZ_2',
            # 'KRVZ_1']
            nrow = len(e_list)
            ncol = 1
            fig, axs = plt.subplots(nrows=nrow, ncols=ncol,figsize=(5,8))
            col = ['b','r','g','m','coral','yellow']#,'purple','plum','coral'] 
            col = ['r' for i in range(len(e_list))]#   
                            #
            for i, ax in enumerate(axs.reshape(-1)): 
                #
                # create objects for each station
                #if e_list[i] in ['WIZ_3','FWVZ_3']:
                path_feat_serv = 'C:\\Users\\aar135\\codes_local_disk\\volc_forecast_tl\\features_bkp\\features_server\\'
                fm_e1 = ForecastModel(window=2., overlap=1., station = e_list[i][:-2],
                    look_forward=2., data_streams=ds, 
                    feature_dir=path_feat_serv, 
                    savefile_type='pkl') 
                #else:
                    #fm_e1 = ForecastModel(window=2., overlap=1., station = e_list[i][:-2],
                    #    look_forward=2., data_streams=ds, savefile_type='csv')
                # initial and final time of interest for each station
                tf_e1 = fm_e1.data.tes[int(e_list[i][-1:])-1]
                ti_e1 = tf_e1 - look_back #month
                #
                # feature to plot
                ft_nm = pd_rank.loc[rank, 'featNM']
                ft_id = pd_rank.loc[rank, 'featID']
                cc = pd_rank.loc[rank, 'cc']
                ## import features
                # proper name of feature
                ft_nm_aux = ft_nm.replace(" ","__")
                ft_nm_aux = ft_nm_aux.replace("-",'"')
                # adding multiple Axes objects
                ft_e1 = fm_e1.get_features(ti=ti_e1, tf=tf_e1, n_jobs=1, compute_only_features=[ft_nm_aux])
                # extract values to plot 
                ft_e1_t = ft_e1[0].index.values
                ft_e1_v = ft_e1[0].loc[:,ft_nm_aux]
                #
                # import datastream to plot 
                ## ax1 and ax2
                ax.plot(ft_e1_t, ft_e1_v, '-', color=col[i], label=erup_dict[e_list[i]]+', vei: '+erup_vei_dict[e_list[i]])
                te = fm_e1.data.tes[int(e_list[i][-1:])-1]
                ax.axvline(te, color='k',linestyle='--', linewidth=2, zorder = 1)
                #ax.plot([], color='k', linestyle='--', linewidth=2, label = 'eruption')
                ax.legend(loc = 2)
                #
                if e_list[i] == 'BELO_3':
                    te = fm_e1.data.tes[int(e_list[i][-1:])-2]
                    ax.axvline(te+2*day, color='k', linestyle='--', linewidth=2, zorder = 1)
                    ax.set_ylim([0,20])
                    #ax.legend(loc = 3)
                #

                ax.grid()
        # title
        title = False
        if rank == 373:
            title = True
            tl_nm = 'DSAR change quantiles (.6-.4) variance'
        if rank == 713:
            title = True
            tl_nm = 'DSAR median'

        if title:
            fig.suptitle('Feature: '+ tl_nm, ha='center')
        else:
            fig.suptitle('Feature: '+ ft_nm_aux, ha='center')

        plt.tight_layout()
        plt.show()

    # plot diferent feature for same eruption 
    if False: 
        # import file as pandas
        path = '..'+os.sep+'features'+os.sep+'correlations'+os.sep+'corr_0_rank_erup_cc.csv'
        try:
            pd_rank = pd.read_csv(path, index_col=0)
        except: # convert txt to csv
            read_file = pd.read_csv ('..'+os.sep+'features'+os.sep+'correlations'+os.sep+'corr_0_rank_erup_cc.txt')
            read_file.to_csv(path, index=None)
            #
            pd_rank = pd.read_csv(path, index_col=0)
            del read_file
        # locate index in rank for cc cut value of 0.5
        idx, _ = find_nearest(pd_rank.loc[:, 'cc'].values, 0.5)
        # for plotting: import features of eruptions per rank
        import numpy as np
        ranks = np.arange(0,idx,1)
        #
        lb = 30#30
        look_back = lb*day
        # eruption
        #erup = 'WIZ_5'#'KRVZ_1'#'WIZ_5'#FWVZ_3, VNSS_1, KRVZ_1, BELO_2, PVV_1
        # features
        rks = [262, 713, 373]
        #rks = [713]
        def _plt_erup_mult_feat(erup, rks):
            #
            nrow = 1
            ncol = 1
            fig, ax = plt.subplots(nrows=nrow, ncols=ncol,figsize=(24,8))
            #
            col = ['r','b','g']
            alpha = [.7, 1., 1.]
            thick_line = [1., 3., 3.]
            for i, rank in enumerate(rks):
                # Figure for dsarF median
                if rank == 713:
                    ds = ['zsc2_dsarF']
                # dsarF change quantiles .4-.6 variance
                if rank == 373:
                    ds = ['zsc2_dsarF']
                if rank == 262:
                    ds = ['zsc2_hfF']
                if False:
                    path_feat_serv = 'C:\\Users\\aar135\\codes_local_disk\\volc_forecast_tl\\features_bkp\\features_server\\'
                    fm_e1 = ForecastModel(window=2., overlap=1., station =  erup[:-2],
                        look_forward=2., data_streams=ds, 
                        feature_dir=path_feat_serv, 
                        savefile_type='pkl') 
                else:
                    fm_e1 = ForecastModel(window=2., overlap=1., station = erup[:-2],
                        look_forward=2., data_streams=ds, savefile_type='csv')
                # initial and final time of interest for each station
                tf_e1 = fm_e1.data.tes[int(erup[-1:])-1]
                ti_e1 = tf_e1 - look_back #month
                #
                # feature to plot
                ft_nm = pd_rank.loc[rank, 'featNM']
                ft_id = pd_rank.loc[rank, 'featID']
                cc = pd_rank.loc[rank, 'cc']
                ## import features
                # proper name of feature
                ft_nm_aux = ft_nm.replace(" ","__")
                ft_nm_aux = ft_nm_aux.replace("-",'"')
                # adding multiple Axes objects
                ft_e1 = fm_e1.get_features(ti=ti_e1, tf=tf_e1, n_jobs=1, compute_only_features=[ft_nm_aux])
                # extract values to plot 
                ft_e1_t = ft_e1[0].index.values
                ft_e1_v = ft_e1[0].loc[:,ft_nm_aux]
                #
                # import datastream to plot 
                ## ax1 and ax2
                v_plot = (ft_e1_v-np.min(ft_e1_v))/np.max((ft_e1_v-np.min(ft_e1_v)))
                if False:
                    if rank == 262:
                        v_plot = v_plot*40
                        v_plot = v_plot - np.mean(v_plot) +.5
                #
                if rank == 373:
                    ft_nm_aux = 'DSAR change quantiles (.6-.4) variance'
                if rank == 713:
                    ft_nm_aux = 'DSAR median'
                if rank == 262:
                    ft_nm_aux = 'HF Fourier coefficient 38'
                #
                ax.plot(ft_e1_t, v_plot, '-', color=col[i], alpha = alpha[i],label='Feature: '+ ft_nm_aux)
                te = fm_e1.data.tes[int(erup[-1:])-1]
                ax.axvline(te, color='k',linestyle='--', linewidth=2, zorder = 1)
                #ax.plot([], color='k', linestyle='--', linewidth=2, label = 'eruption')
                ax.legend(loc = 2)
                if rank == 262:
                    ax.legend(loc = 3)
                #    ax.set_ylim([,.6])#  = ['zsc2_hfF']
                #
                if erup == 'BELO_3':
                    te = fm_e1.data.tes[int(e_list[i][-1:])-2]
                    ax.axvline(te+2*day, color='k', linestyle='--', linewidth=2, zorder = 1)
                    ax.set_ylim([0,20])
                    #ax.legend(loc = 3)
                    #
            #ax.set_ylim([-.0,1.3])
            ax.grid()
            #ax.set_ylabel('feature value')
            ax.set_xticks([ft_e1[0].index[-1] - 7*day*i for i in range(int(lb/7)+1)])#[dat.index.values[0],dat.index.values[-1]])#, ]np.arange(0, len(x)+1, 5))
            #ax.set_xticks([ft_e1[0].index[-1] - 2*day*i for i in range(int(lb/2)+1)])#[dat.index.values[0],dat.index.values[-1]])#, ]np.arange(0, len(x)+1, 5))
            
            fig.suptitle(erup_dict[erup]+', vei: '+erup_vei_dict[erup])#'Feature: '+ ft_nm_aux, ha='center')
            #plt.tight_layout()
            #plt.show()
            path = '..'+os.sep+'features'+os.sep+'correlations'+os.sep+'comb_feat_analysis'+os.sep
            ft_id = str(rks[0])+'_'+str(rks[1])+'_'+str(rks[2])
            plt.savefig(path+erup+'_'+ft_id+'.png')
            plt.show()
            plt.close()
        #
        erup_list = ['WIZ_5']
        for erup in erup_list:
            _plt_erup_mult_feat(erup, rks)
    
    # Statistical significance of pre_eruptive feature
    if True:
        print('Statistical significance of pre_eruptive feature')
        import numpy as np
        ## Selecting the feature to correlate
        # characteristic feature to correlate
        id713 = True # dsar median
        id027 = False
        id373 = False # dsar ch qt var
        id262 = False # hf fft coef 38
        if id713: 
            if True:
                sta = 'WIZ'
                erup = 4 # fifth 2019
                ds = ['zsc2_dsarF']
                ft = ['zsc2_dsarF__median']
            else:
                sta = 'VNSS'
                erup = 0 # fifth 2019
                ds = ['zsc2_dsarF']
                ft = ['zsc2_dsarF__median']
        if id027:
            sta = 'WIZ'
            erup = 3 # forth 2016
            ds = ['zsc2_mfF']
            ft = ['zsc2_mfF__fft_coefficient__coeff_74__attr_"abs"'] 
            ft = ['zsc2_mfF__fft_coefficient__coeff_74__attr_-abs-']
        if id373:
            if False:
                sta = 'FWVZ'
                erup = 1 # 
                ds = ['zsc2_dsarF']
                ft = ['zsc2_dsarF__change_quantiles__f_agg_"var"__isabs_False__qh_0.6__ql_0.4']
            else: 
                sta = 'WIZ'
                erup = 4 # 
                ds = ['zsc2_dsarF']
                ft = ['zsc2_dsarF__change_quantiles__f_agg_"var"__isabs_False__qh_0.6__ql_0.4']         
        if id262:
            sta = 'WIZ'
            erup = 4 # fifth
            ds = ['zsc2_hfF']
            ft = ['zsc2_hfF__fft_coefficient__coeff_38__attr_"real"']         
        ##
        if False:
            fm_e1 = ForecastModel(window=2., overlap=1., station = sta,
                look_forward=2., data_streams=ds, savefile_type='csv')
        else:
            path_feat_serv = 'C:\\Users\\aar135\\codes_local_disk\\volc_forecast_tl\\features_bkp\\features_server\\'
            fm_e1 = ForecastModel(window=2., overlap=1., station = sta,
                look_forward=2., data_streams=ds, 
                feature_dir=path_feat_serv, 
                savefile_type='pkl') 
        # initial and final time of interest for each station
        tf_e1 = fm_e1.data.tes[erup]
        ti_e1 = tf_e1 - 30*day #month
        # extract feature values 
        ft_e1 = fm_e1.get_features(ti=ti_e1, tf=tf_e1, n_jobs=1, compute_only_features=[ft])
        # extract values to correlate 
        ft_e1_t = ft_e1[0].index.values
        ft_e1_v = ft_e1[0].loc[:,ft].values
        ft_e1_v = [ft_e1_v[i][0] for i in range(len(ft_e1_v))]
        ##
        # period to explore 
        sta_exp = 'FWVZ' # station to explore
        #
        if sta_exp == 'WIZ':
            endtime = datetimeify("2021-06-30")
            years_back = 10 # WIZ 10; VNSS 7 ; FWVZ 14
            look_back = years_back*365 + 3 # days, years back from endtime (day by day)
        if sta_exp == 'FWVZ':
            endtime = datetimeify("2021-12-31")#("2020-12-31")
            years_back = 12 # WIZ 10; VNSS 7 ; FWVZ 14
            look_back = years_back*365 # days, years back from endtime (day by day) 
        if sta_exp == 'KRVZ':
            endtime = datetimeify("2020-12-31")#("2020-12-31")
            years_back = 15 # WIZ 10; VNSS 7 ; FWVZ 14
            look_back = years_back*365 # days, years back from endtime (day by day) 
        if sta_exp == 'VNSS':
            endtime = datetimeify("2019-12-31")# 
            years_back = 3 # WIZ 10; VNSS 7 ; FWVZ 14
            look_back = years_back*365 - 181 # days, years back from endtime (day by day) 
        if sta_exp == 'IVGP':
            endtime = datetimeify("2021-10-05")# 
            years_back = 2 # WIZ 10; VNSS 7 ; FWVZ 14
            look_back = years_back*365 + 35# days, years back from endtime (day by day)
            #look_back = 10
        if sta_exp == 'PVV':
            endtime = datetimeify("2016-06-30")# 
            years_back = 2 # WIZ 10; VNSS 7 ; FWVZ 14
            look_back = years_back*365 + 120 # days, years back from endtime (day by day)         
        # vector of days
        _vec_days = [endtime - day*i for i in range(look_back)]
        a = _vec_days[-1]
        #
        if sta_exp == 'VNSS': # add another period for VNSSS
            #
            endtime = datetimeify("2019-03-01")
            years_back = 2 # WIZ 10; VNSS 7
            look_back = years_back*365 - 180 # years back from endtime (day by day) (+3 for WIZ)
            vec_days_1 = [endtime - day*i for i in range(look_back)]
            #
            #endtime = datetimeify("2015-05-01")
            years_back = 3 # WIZ 10; VNSS 7
            look_back = years_back*365 - 30 # years back from endtime (day by day) (+3 for WIZ)
            vec_days_2 = [endtime - day*i for i in range(look_back)]
            #
            vec_days = vec_days_1 #+ vec_days_2
        #
        path = '..'+os.sep+'features'+os.sep+'correlations'+os.sep+'stat_sign_per_hig_corr_feat'+os.sep
        #
        # filter days already calculated
        # load the file 
        vec_days = []
        if os.path.isfile(path+sta+'_'+str(erup+1)+'_'+ft[0]+'_over_'+sta_exp+'.csv'):
            fl_old = pd.read_csv(path+sta+'_'+str(erup+1)+'_'+ft[0]+'_over_'+sta_exp+'.csv', index_col=0)
            fl_old.index = pd.to_datetime(fl_old.index)
            for d in _vec_days:
                if not d in fl_old.index:
                    vec_days.append(d)
        else:
            vec_days = _vec_days
        #
        if True: # run the correlations 
            #
            if vec_days:
                fl_nm = path+sta+'_'+str(erup+1)+'_'+ft[0]+'_over_'+sta_exp+'.csv'
                fl_nm = fl_nm.replace('"', "-")
                with open(fl_nm, 'w', newline='') as csvfile:
                    spamwriter = csv.writer(csvfile, delimiter=',')
                    spamwriter.writerow(['endtime','cc'])
                    # write existing values
                    if 'fl_old' in locals():
                        for k, d in enumerate(_vec_days):
                            if d in fl_old.index:
                                spamwriter.writerow([_vec_days[k], str(np.round(fl_old.loc[fl_old.index[k]][0],3))])
                    # loop over period
                    count = 0
                    count2 = 0
                    #
                    if True: # run in serial 
                        # timming
                        import numpy as np
                        #tic=timeit.default_timer()
                        print('Running correlation: '+sta+'_'+str(erup+1)+'_'+ft[0]+'_over_'+sta_exp+
                            '\n from '+str(vec_days[-1])+' to '+str(vec_days[0])) 
                        print('Running time: '+str(np.round(((len(vec_days)*10)/3600),2))+' hours') # 10 seconds per iteration
                        #
                        for d in vec_days:
                            # get aux feature
                            server_feat_bkp = True # ['VNSS', 'FWVZ', 'IVGP','KRVZ']:
                            if server_feat_bkp:
                                path_feat_serv = 'C:\\Users\\aar135\\codes_local_disk\\volc_forecast_tl\\features_bkp\\features_server\\'
                                fm_aux = ForecastModel(window=2., overlap=1., station = sta_exp,
                                    look_forward=2., data_streams=ds, 
                                    feature_dir=path_feat_serv, 
                                    savefile_type='pkl') 
                            else:
                                fm_aux = ForecastModel(window=2., overlap=1., station = sta_exp,
                                    look_forward=2., data_streams=ds, savefile_type='pkl')
                            # initial and final time of interest for each station
                            tf_aux = d
                            ti_aux = tf_aux - 30*day #month
                            # extract feature values for aux period and stack with characteristic one
                            ft_aux = fm_aux.get_features(ti=ti_aux, tf=tf_aux, n_jobs=1, compute_only_features=[ft])
                            ft_aux_t = ft_aux[0].index.values
                            ft_aux_v = ft_aux[0].loc[:,ft].values
                            ft_aux_v = [ft_aux_v[i][0] for i in range(len(ft_aux_v))]            
                            data = [ft_e1_v, ft_aux_v]
                            arr_aux =  np.vstack((data)).T

                            # correlate characteristic feature for section 
                            # create aux pandas obj for correlation
                            df = pd.DataFrame(data=arr_aux)#, index=fm_aux.fM.index, columns=lab, dtype=None, copy=None)
                            df_corr = df.corr(method='pearson')

                            # save cc value in pd with time (end of period)
                            spamwriter.writerow([d,str(np.round(df_corr.iloc[0,1],3))])
                            count += 1
                            if count % 30 == 0:
                                count2 +=1
                                print('Months: '+str(count2)+'/'+str(int(len(vec_days)/30)))
                        #
                        # end timming
                        #toc=timeit.default_timer()
                        #print(toc - tic) # print elapsed time in seconds
                        #print((toc - tic)/count)
                    else: # run in parallel (not working)
                        n_jobs = 1 # number of cores
                        if False:
                            print('Parallel')
                            print('Number of features: '+str(len(ftns)))
                            print('Time when run:')
                            print(datetime.now())
                            print('Estimated run time (hours):')
                            print(len(ftns)/n_jobs * (231/3600))
                        #
                        ps = []
                        for k, d in enumerate(vec_days[-4:-1]):
                            p = k, ft_e1_v, sta_exp, d, ft, ds
                            ps.append(p)
                        # run parallel
                        p = Pool(n_jobs)
                        p.map(_calc_corr, ps)
                        # 
                        for k, d in enumerate(vec_days[-4:-1]):
                            _cc = np.loadtxt(str(d))
                            print(d)
                            print(_cc)
                            asdf
                            spamwriter.writerow([d,str(np.round(_cc,3))])
                            os.remove(str(d))
        
        if False: # plot histogram for one record per feature
            # plot histogram of cc values 
            # import csv as dataframe
            fl_nm = path+sta+'_'+str(erup+1)+'_'+ft[0]+'_over_'+sta_exp+'.csv'
            fl_nm = fl_nm.replace('"', "-")
            df_aux = pd.read_csv(fl_nm, index_col=0)
            df_aux.index = pd.to_datetime(df_aux.index)
            #df_aux = df_aux.abs()        
            # plot histogram 
            dat = df_aux.iloc[:,0].values
            n_bins = int(np.sqrt(len(dat)))
            fig, ax1 = plt.subplots(nrows = 1, ncols = 1, figsize = (6,4))
            ax1.hist(dat, n_bins, histtype='bar', color = 'c', edgecolor='#E6E6E6')#, label = 'rsam')
            ax1.set_xlabel('cc')#, fontsize=textsize)
            ax1.set_ylabel('frequency')#, fontsize=textsize)
            ax1.grid(True, which='both', linewidth=0.1)
            #ax1.set_xlim(xlim)
            #ax1.set_ylim(ylim)
            #ax1.set_title(erup_dict[sta+'_'+str(erup+1)]+' feature: '+ft[0])#), fontsize = textsize)
            ax1.set_title('Feature '+ft[0]+'\n in '+erup_dict[sta+'_'+str(erup+1)]+' eruption\n over '
                +sta_code[sta_exp]+' record ('+str(years_back)+' years)')
            # plot vertical line for values at previous eruptions 
            # 95% percentil
            per = 90
            ax1.axvline(np.percentile(dat, per, axis=0), color='k', linestyle='-', linewidth=2, zorder = 1, label = 'Percentile '+str(per)) 
            per = 50
            ax1.axvline(np.percentile(dat, per, axis=0), color='r', linestyle='-', linewidth=2, zorder = 1, label = 'Percentile '+str(per)) 
        
            #tf_e1 = fm_e1.data.tes[erup]
            aux = [.5,.6,.7,.8,.9]
            col = ['r','b','y','m','grey','g','purple','plum','coral']
            if id713: 
                if sta_exp == 'WIZ':
                    ax1.axvline(0.69, color=col[5], linestyle='--', linewidth=2, zorder = 1, label = erup_dict['WIZ_1']) 
                    ax1.axvline(0.71, color=col[1], linestyle='--', linewidth=2, zorder = 1, label = erup_dict['WIZ_2'])
                    ax1.axvline(0.1, color=col[4], linestyle='--', linewidth=2, zorder = 1, label = erup_dict['WIZ_3']) 
                    ax1.axvline(0.73, color=col[3], linestyle='--', linewidth=2, zorder = 1, label = erup_dict['WIZ_4'])
                    #
                    # ax1.axvline(0.73, color=col[2], linestyle='--', linewidth=2, zorder = 1, label = erup_dict['VNSS_1']) 
                    # ax1.axvline(0.68, color=col[5], linestyle='--', linewidth=2, zorder = 1, label = erup_dict['FWVZ_1']) 
                    # ax1.axvline(0.31, color=col[6], linestyle='--', linewidth=2, zorder = 1, label = erup_dict['KRVZ_1']) 
                    # ax1.axvline(0.21, color=col[7], linestyle='--', linewidth=2, zorder = 1, label = erup_dict['PVV_3']) 
                if sta_exp == 'FWVZ':
                    ax1.axvline(0.71, color=col[0], linestyle='--', linewidth=2, zorder = 1, label = erup_dict['FWVZ_1']) 
                    ax1.axvline(0.38, color=col[1], linestyle='--', linewidth=2, zorder = 1, label = erup_dict['FWVZ_2']) 
                    ax1.axvline(0.43, color=col[5], linestyle='--', linewidth=2, zorder = 1, label = erup_dict['FWVZ_3']) 
                if sta_exp == 'VNSS':
                    ax1.axvline(0.75, color=col[0], linestyle='--', linewidth=2, zorder = 1, label = erup_dict['VNSS_1']) 
                    ax1.axvline(0.40, color=col[1], linestyle='--', linewidth=2, zorder = 1, label = erup_dict['VNSS_2']) 
                if sta_exp == 'IVGP':
                    pass
                    #ax1.axvline(0.69, color=col[5], linestyle='--', linewidth=2, zorder = 1, label = erup_dict['WIZ_1']) 
                    #ax1.axvline(0.71, color=col[1], linestyle='--', linewidth=2, zorder = 1, label = erup_dict['WIZ_2'])
                    #ax1.axvline(0.1, color=col[4], linestyle='--', linewidth=2, zorder = 1, label = erup_dict['WIZ_3']) 
                    #ax1.axvline(0.73, color=col[3], linestyle='--', linewidth=2, zorder = 1, label = erup_dict['WIZ_4'])
            if id027: 
                pass
            if id373:
                if sta_exp == 'WIZ':
                    ax1.axvline(0.18, color=col[4], linestyle='--', linewidth=2, zorder = 1, label = erup_dict['WIZ_1']) 
                    ax1.axvline(0.24, color=col[1], linestyle='--', linewidth=2, zorder = 1, label = erup_dict['WIZ_2']) 
                    ax1.axvline(0.11, color=col[2], linestyle='--', linewidth=2, zorder = 1, label = erup_dict['WIZ_3'])
                    ax1.axvline(0.30, color=col[3], linestyle='--', linewidth=2, zorder = 1, label = erup_dict['WIZ_4'])  
                    ax1.axvline(0.77, color=col[0], linestyle='--', linewidth=2, zorder = 1, label = erup_dict['WIZ_5']) 
            if id262: 
                if sta_exp == 'WIZ':
                    ax1.axvline(0.03, color=col[5], linestyle='--', linewidth=2, zorder = 1, label = erup_dict['WIZ_1']) 
                    ax1.axvline(0.00, color=col[1], linestyle='--', linewidth=2, zorder = 1, label = erup_dict['WIZ_2'])
                    ax1.axvline(0.18, color=col[4], linestyle='--', linewidth=2, zorder = 1, label = erup_dict['WIZ_3']) 
                    ax1.axvline(0.01, color=col[3], linestyle='--', linewidth=2, zorder = 1, label = erup_dict['WIZ_4'])
                    #
            ax1.legend(loc = 1)
            #
            if False: # highligth values near the eruption 
                dat2 = []
                # loop index pd
                for index, row in df_aux.iterrows():
                    add = False
                    # loop over eruptions 
                    for etime in fm_e1.data.tes:
                        # check if index is close to eruption
                        if (index < etime and index > etime-month/3):
                            add = True
                    # add to list
                    if add:
                        dat2.append(row['cc'])
                #         
                n_bins = int(np.sqrt(len(dat2)))
                ax1.hist(dat2, n_bins, histtype='bar', color = 'grey', edgecolor='grey', alpha = 0.7)#, label = 'rsam')
            plt.show()

        if False: # plot histogram for every record per feature 

            # signature
            sta = 'WIZ'
            erup = 4
            ds = ['zsc2_dsarF']
            ft = ['zsc2_dsarF__median']
            # plot histogram of cc values 
            rec_list = ['WIZ', 'FWVZ', 'VNSS']#, 'IVGP']
            sta_exp_l = sta_code[rec_list[0]] + ", "+sta_code[rec_list[1]] + ", "+sta_code[rec_list[2]]
            #
            dat_accu = np.asarray([])
            for i, sta_exp in enumerate(rec_list): 
                # import csv as dataframe
                fl_nm = path+sta+'_'+str(erup+1)+'_'+ft[0]+'_over_'+sta_exp+'.csv'
                fl_nm = fl_nm.replace('"', "-")
                df_aux = pd.read_csv(fl_nm, index_col=0)
                df_aux.index = pd.to_datetime(df_aux.index)
                #df_aux = df_aux.abs()        
                # plot histogram 
                dat = df_aux.iloc[:,0].values
                dat_accu = np.concatenate((dat_accu, dat), axis=0)
            #
            n_bins = int(np.sqrt(len(dat_accu)))
            fig, ax1 = plt.subplots(nrows = 1, ncols = 1, figsize = (6,4))
            ax1.hist(dat_accu, n_bins, histtype='bar', color = 'c', edgecolor='#E6E6E6')#, label = 'rsam')
            ax1.set_xlabel('cc')#, fontsize=textsize)
            ax1.set_ylabel('frequency')#, fontsize=textsize)
            ax1.grid(True, which='both', linewidth=0.1)
            #ax1.set_xlim(xlim)
            #ax1.set_ylim(ylim)
            #ax1.set_title(erup_dict[sta+'_'+str(erup+1)]+' feature: '+ft[0])#), fontsize = textsize)
            ax1.set_title('Feature '+ft[0]+'\n in '+erup_dict[sta+'_'+str(erup+1)]+' eruption\n over '
                +sta_exp_l+' records')
            # plot vertical line for values at previous eruptions 
            # 95% percentil
            per = 90
            ax1.axvline(np.percentile(dat_accu, per, axis=0), color='k', linestyle='-', linewidth=2, zorder = 1, label = 'Percentile '+str(per)) 
            per = 50
            ax1.axvline(np.percentile(dat_accu, per, axis=0), color='r', linestyle='-', linewidth=2, zorder = 1, label = 'Percentile '+str(per)) 
        
            #tf_e1 = fm_e1.data.tes[erup]
            aux = [.5,.6,.7,.8,.9]
            col = ['r','b','y','m','grey','g','purple','plum','coral']
            if ft == ['zsc2_dsarF__median']: 
                    ax1.axvline(0.69, color=col[5], linestyle='--', linewidth=2, zorder = 1, label = erup_dict['WIZ_1']) 
                    ax1.axvline(0.71, color=col[1], linestyle='--', linewidth=2, zorder = 1, label = erup_dict['WIZ_2'])
                    ax1.axvline(0.1, color=col[4], linestyle='--', linewidth=2, zorder = 1, label = erup_dict['WIZ_3']) 
                    ax1.axvline(0.73, color=col[3], linestyle='--', linewidth=2, zorder = 1, label = erup_dict['WIZ_4'])
                    ax1.axvline(0.71, color=col[0], linestyle='dotted', linewidth=2, zorder = 1, label = erup_dict['FWVZ_1']) 
                    ax1.axvline(0.38, color=col[1], linestyle='dotted', linewidth=2, zorder = 1, label = erup_dict['FWVZ_2']) 
                    ax1.axvline(0.43, color=col[5], linestyle='dotted', linewidth=2, zorder = 1, label = erup_dict['FWVZ_3']) 
                    ax1.axvline(0.75, color=col[0], linestyle='dashdot', linewidth=2, zorder = 1, label = erup_dict['VNSS_1']) 
                    ax1.axvline(0.40, color=col[1], linestyle='dashdot', linewidth=2, zorder = 1, label = erup_dict['VNSS_2']) 
            # if id027: 
            #     pass
            # if id373:
            #     if sta_exp == 'WIZ':
            #         ax1.axvline(0.18, color=col[4], linestyle='--', linewidth=2, zorder = 1, label = erup_dict['WIZ_1']) 
            #         ax1.axvline(0.24, color=col[1], linestyle='--', linewidth=2, zorder = 1, label = erup_dict['WIZ_2']) 
            #         ax1.axvline(0.11, color=col[2], linestyle='--', linewidth=2, zorder = 1, label = erup_dict['WIZ_3'])
            #         ax1.axvline(0.30, color=col[3], linestyle='--', linewidth=2, zorder = 1, label = erup_dict['WIZ_4'])  
            #         ax1.axvline(0.77, color=col[0], linestyle='--', linewidth=2, zorder = 1, label = erup_dict['WIZ_5']) 

            ax1.legend(loc = 2)
            ax1.set_xlim([-1.1,1.])
            #    
            plt.show()

        if False: # plot stacked histograms one on top of the other ones
            # signature
            if id713: # dsar median
                sta = 'WIZ'#'WIZ'
                erup = 4
                ds = ['zsc2_dsarF']
                ft = ['zsc2_dsarF__median']
                #rec_list = ['IVGP', 'VNSS', 'WIZ', 'FWVZ']
                #sta_exp_l = sta_code[rec_list[0]] + ", "+sta_code[rec_list[1]] + ", "+sta_code[rec_list[2]]+ ", "+sta_code[rec_list[3]]
                rec_list = ['VNSS','KRVZ','WIZ', 'FWVZ']
                sta_exp_l = sta_code[rec_list[0]] + ", "+sta_code[rec_list[1]] + ", "+sta_code[rec_list[2]]+ ", "+sta_code[rec_list[3]]
                colors = ['palegreen', 'grey', 'lightblue', 'salmon']
            if id373: # dsar ch qt var
                sta = 'FWVZ'
                erup = 1 # 
                ds = ['zsc2_dsarF']
                ft = ['zsc2_dsarF__change_quantiles__f_agg_"var"__isabs_False__qh_0.6__ql_0.4']
                rec_list = ['VNSS', 'WIZ', 'FWVZ'] #VNSS
                sta_exp_l = sta_code[rec_list[0]] + ", "+sta_code[rec_list[1]]
                colors = ['palegreen', 'lightblue', 'salmon']
                #
                sta = 'WIZ'
                erup = 4 # 
                ds = ['zsc2_dsarF']
                ft = ['zsc2_dsarF__change_quantiles__f_agg_"var"__isabs_False__qh_0.6__ql_0.4']
                rec_list = ['VNSS','KRVZ','WIZ', 'FWVZ']
                sta_exp_l = sta_code[rec_list[0]] + ", "+sta_code[rec_list[1]] + ", "+sta_code[rec_list[2]]+ ", "+sta_code[rec_list[3]]
                colors = ['palegreen', 'grey', 'lightblue', 'salmon']
                #
            if id262: # hfT fuorier coef 38
                sta = 'WIZ'
                erup = 4 # 
                ds = ['zsc2_dsarF']
                ft = ['zsc2_hfF__fft_coefficient__coeff_38__attr_"real"']
                rec_list = ['WIZ', 'FWVZ'] #VNSS
                sta_exp_l = sta_code[rec_list[0]] + ", "+sta_code[rec_list[1]]
                colors = ['lightblue', 'salmon']#['palegreen', 'lightblue', 'salmon']
                rec_list = ['VNSS','KRVZ','WIZ', 'FWVZ']
                sta_exp_l = sta_code[rec_list[0]] + ", "+sta_code[rec_list[1]]+ ", "+sta_code[rec_list[2]]
                colors = ['palegreen', 'grey', 'lightblue', 'salmon']#, 'lightblue', 'salmon']

            # plot histogram of cc values 
            #
            dat_accu_t = np.asarray([])
            dat_accu = []
            len_rec = []
            for i, sta_exp in enumerate(rec_list): 
                # if id373:
                #     if rec_list[i] == 'VNSS':
                #         sta = 'WIZ'#'WIZ'
                #         erup = 4
                #         ft = ['zsc2_dsarF__median']
                #     else:
                #         pass

                # import csv as dataframe
                fl_nm = path+sta+'_'+str(erup+1)+'_'+ft[0]+'_over_'+sta_exp+'.csv'
                fl_nm = fl_nm.replace('"', "-")
                df_aux = pd.read_csv(fl_nm, index_col=0)
                df_aux.index = pd.to_datetime(df_aux.index)
                #df_aux = df_aux.abs()        
                # plot histogram 
                dat = df_aux.iloc[:,0].values
                dat_accu.append(dat) # = np.concatenate((dat_accu, dat), axis=0)
                dat_accu_t = np.concatenate((dat_accu_t, dat), axis=0)
                len_rec.append(len(dat))
                #
                if  id373 and rec_list[i] == 'VNSS':
                    sta = 'FWVZ'
                    erup = 1 # 
                    ds = ['zsc2_dsarF']
                    ft = ['zsc2_dsarF__change_quantiles__f_agg_"var"__isabs_False__qh_0.6__ql_0.4']
                    #
                    sta = 'WIZ'
                    erup = 4 # 
                    ds = ['zsc2_dsarF']
                    ft = ['zsc2_dsarF__change_quantiles__f_agg_"var"__isabs_False__qh_0.6__ql_0.4']
            if False: # using pandas
                # create the dataframe; enumerate is used to make column names
                df = pd.concat([pd.DataFrame(a, columns=[f'x{i}']) for i, a in enumerate(dat_accu, 1)], axis=1)
                # plot the data
                df.plot.hist(stacked=True, bins=30, density=True, figsize=(10, 6), grid=True)
                plt.show()
            if True: # using hist
                #
                #colors = ['m', 'palegreen', 'lightblue', 'salmon']
                label = [sta_code[rec_list[i]]+' record ('+str(round(len_rec[i]/365,1))+' years)'  for i in range(len(rec_list))]
                try:
                    #n_bins = int(np.sqrt(len(dat_accu[1]+dat_accu[2])))
                    n_bins = int(np.sqrt(len(dat_accu[-1])))
                except:
                    n_bins = int(np.sqrt(len(dat_accu[1])))

                fig, ax1 = plt.subplots(nrows = 1, ncols = 1, figsize = (8,6))
                ax1.hist(dat_accu, n_bins, density=True, histtype='bar', stacked=True,  color=colors[:len(rec_list)], label=label, edgecolor='#E6E6E6')
                ax1.set_xlabel('cc')#, fontsize=textsize)
                ax1.set_ylabel('frequency')#, fontsize=textsize)
                ax1.grid(True, which='both', linewidth=0.1)
                #
                #ax1.set_xlim(xlim)
                #ax1.set_ylim(ylim)
                #ax1.set_title(erup_dict[sta+'_'+str(erup+1)]+' feature: '+ft[0])#), fontsize = textsize)
                ax1.set_title('Feature '+ft[0]+'\n in '+erup_dict[sta+'_'+str(erup+1)]+' eruption over '
                    +sta_exp_l+' records')
                # plot vertical line for values at previous eruptions 
                # 95% percentil
                per = 95
                #ax1.axvline(np.percentile(dat_accu_t, per, axis=0), color='k', linestyle='--', linewidth=4, zorder = 1, label = 'Percentile '+str(per)) 
                per = 90
                ax1.axvline(np.percentile(dat_accu_t, per, axis=0), color='k', linestyle='-', linewidth=4, zorder = 1, label = 'Percentile '+str(per)) 
                per = 70
                ax1.axvline(np.percentile(dat_accu_t, per, axis=0), color='gray', linestyle='-', linewidth=4, zorder = 1, label = 'Percentile '+str(per)) 
                per = 50
                ax1.axvline(np.percentile(dat_accu_t, per, axis=0), color='silver', linestyle='-', linewidth=4, zorder = 1, label = 'Percentile '+str(per)) 

                #tf_e1 = fm_e1.data.tes[erup]
                aux = [.5,.6,.7,.8,.9]
                col = ['r','b','y','m','grey','g','purple','plum','coral']
                if ft == ['zsc2_dsarF__median']: 
                    if False:
                        ax1.axvline(0.69, color=col[1], linestyle='--', linewidth=2, zorder = 1, label = erup_dict['WIZ_1']) 
                        ax1.axvline(0.72, color=col[1], linestyle='dotted', linewidth=2, zorder = 1, label = erup_dict['WIZ_2'])
                        ax1.axvline(0.1, color=col[1], linestyle='dashdot', linewidth=2, zorder = 1, label = erup_dict['WIZ_3']) 
                        ax1.axvline(0.73, color=col[1], linestyle='-', linewidth=2, zorder = 1, label = erup_dict['WIZ_4'])
                        #
                        ax1.axvline(0.71, color=col[0], linestyle='-', linewidth=2, zorder = 1, label = erup_dict['FWVZ_1']) 
                        ax1.axvline(0.38, color=col[0], linestyle='--', linewidth=2, zorder = 1, label = erup_dict['FWVZ_2']) 
                        ax1.axvline(0.43, color=col[0], linestyle='dashdot', linewidth=2, zorder = 1, label = erup_dict['FWVZ_3']) 
                        #
                        ax1.axvline(0.75, color=col[5], linestyle='-', linewidth=2, zorder = 1, label = erup_dict['VNSS_1']) 
                        ax1.axvline(0.40, color=col[5], linestyle='--', linewidth=2, zorder = 1, label = erup_dict['VNSS_2']) 
                        # extras
                        #ax1.axvline(0.48, color=col[2], linestyle='-', linewidth=2, zorder = 1, label = erup_dict['KRVZ_2']) 
                        #ax1.axvline(0.61, color=col[2], linestyle='--', linewidth=2, zorder = 1, label = erup_dict['BELO_2']) 
                    if True: # *
                        # Whakaari
                        ax1.plot(0.65, 1.25, '*', color=col[1])#, linewidth=2, zorder = 1, label = erup_dict['WIZ_1'])
                        ax1.plot(0.69, 1.25, '*', color=col[1])#, linewidth=2)#, zorder = 1, label = erup_dict['WIZ_2'])
                        #ax1.plot(0.1, 1., '*', color=col[1])#, linewidth=2)#, zorder = 1, label = erup_dict['WIZ_3']) 
                        ax1.plot(0.73, 1.25, '*', color=col[1])#, linewidth=2)#, zorder = 1, label = erup_dict['WIZ_3']) 
                        ax1.plot(0.77, 1.25, '*', color=col[1])#, linewidth=2)#, zorder = 1, label = erup_dict['WIZ_4'])
                        ax1.plot( [],[], '*', color=col[1], label= 'Whakaari eruptions')#, linewidth=2)#, zorder = 1, label = erup_dict['WIZ_5'])
                        ax1.text(0.65-.015, 1.25+.1, erup_dict['WIZ_1'][-4:], rotation=90, va='center', color=col[1])
                        ax1.text(0.69-.015, 1.25+.1, erup_dict['WIZ_2'][-5:], rotation=90, va='center', color=col[1])
                        #ax1.text(0.1-.015, 1.+.1, erup_dict['WIZ_3'][-5:], rotation=90, va='center', color=col[1])
                        ax1.text(0.73-.015, 1.25+.1, erup_dict['WIZ_3'][-5:], rotation=90, va='center', color=col[1])
                        ax1.text(0.77-.015, 1.25+.1, erup_dict['WIZ_4'][-4:], rotation=90, va='center', color=col[1])

                        # Ruapehu
                        ax1.plot(0.71, 1., '*', color=col[0])#, linewidth=2)#, zorder = 1, label = erup_dict['WIZ_4'])
                        ax1.plot(0.38, 1., '*', color=col[0])#, linewidth=2)#, zorder = 1, label = erup_dict['WIZ_5'])
                        ax1.plot(0.43, 1., '*', color=col[0])#, linewidth=2)#, zorder = 1, label = erup_dict['WIZ_5'])
                        ax1.plot([],[], '*', color=col[0], label= 'Ruapehu eruptions')#, linewidth=2)#, zorder = 1, label = erup_dict['WIZ_5'])
                        ax1.text(0.71-.015, 1.+.1, erup_dict['FWVZ_1'][-4:], rotation=90, va='center', color=col[0])
                        ax1.text(0.38-.015, 1.+.1, erup_dict['FWVZ_2'][-4:], rotation=90, va='center', color=col[0])
                        ax1.text(0.43-.015, 1.+.1, erup_dict['FWVZ_3'][-4:], rotation=90, va='center', color=col[0])

                        # Veniaminof
                        ax1.plot(0.75, .75, '*', color=col[5])#, linewidth=2)#, zorder = 1, label = erup_dict['WIZ_4'])
                        ax1.plot(0.40, .75, '*', color=col[5])#, linewidth=2)#, zorder = 1, label = erup_dict['WIZ_5'])
                        ax1.plot([],[], '*', color=col[5], label= 'Veniaminof eruptions')#, linewidth=2)#, zorder = 1, label = erup_dict['WIZ_5'])
                        ax1.text(0.75-.015, .75+.1, erup_dict['VNSS_1'][-4:], rotation=90, va='center', color=col[5])
                        ax1.text(0.40-.015, .75+.1, erup_dict['VNSS_2'][-4:], rotation=90, va='center', color=col[5])

                        # Tongariro
                        ax1.plot(0.31, .5, '*', color=col[4])#, linewidth=2)#, zorder = 1, label = erup_dict['WIZ_4'])
                        ax1.plot(0.47, .5, '*', color=col[4])#, linewidth=2)#, zorder = 1, label = erup_dict['WIZ_5'])
                        ax1.plot([],[], '*', color=col[4], label= 'Tongariro eruptions')#, linewidth=2)#, zorder = 1, label = erup_dict['WIZ_5'])
                        ax1.text(0.31-.015, .5+.1, erup_dict['KRVZ_1'][-5:], rotation=90, va='center', color=col[4])
                        ax1.text(0.47-.015, .5+.1, erup_dict['KRVZ_2'][-5:], rotation=90, va='center', color=col[4])

                if ft == ['zsc2_dsarF__change_quantiles__f_agg_"var"__isabs_False__qh_0.6__ql_0.4']:
                    if False: # lines
                        ax1.axvline(0.18, color=col[1], linestyle='--', linewidth=2, zorder = 1, label = erup_dict['WIZ_1']) 
                        ax1.axvline(0.24, color=col[1], linestyle='dotted', linewidth=2, zorder = 1, label = erup_dict['WIZ_2'])
                        ax1.axvline(0.11, color=col[1], linestyle='dashdot', linewidth=2, zorder = 1, label = erup_dict['WIZ_3']) 
                        ax1.axvline(0.30, color=col[1], linestyle='-', linewidth=2, zorder = 1, label = erup_dict['WIZ_4'])
                        ax1.axvline(0.77, color=col[1], linestyle='dashed', linewidth=2, zorder = 1, label = erup_dict['WIZ_5'])
                        #
                        ax1.axvline(0.04, color=col[0], linestyle='-', linewidth=2, zorder = 1, label = erup_dict['FWVZ_1']) 
                        #ax1.axvline(0.38, color=col[0], linestyle='--', linewidth=2, zorder = 1, label = erup_dict['FWVZ_2']) 
                        ax1.axvline(0.13, color=col[0], linestyle='dashdot', linewidth=2, zorder = 1, label = erup_dict['FWVZ_3']) 
                        #
                        ax1.axvline(0.04, color=col[5], linestyle='-', linewidth=2, zorder = 1, label = erup_dict['VNSS_1']) 
                        ax1.axvline(0.33, color=col[5], linestyle='--', linewidth=2, zorder = 1, label = erup_dict['VNSS_2'])
                        # extrax
                        ax1.axvline(0.54, color=col[2], linestyle='-', linewidth=2, zorder = 1, label = erup_dict['PVV_3']) 
                        ax1.axvline(0.46, color=col[2], linestyle='--', linewidth=2, zorder = 1, label = erup_dict['BELO_2'])
                    if True: # *
                        # Whakaari
                        ax1.plot(0.18, 2.5, '*', color=col[1])#, linewidth=2, zorder = 1, label = erup_dict['WIZ_1'])
                        ax1.plot(0.24, 2.5, '*', color=col[1])#, linewidth=2)#, zorder = 1, label = erup_dict['WIZ_2'])
                        #ax1.plot(0.11, 2.5, '*', color=col[1])#, linewidth=2)#, zorder = 1, label = erup_dict['WIZ_3']) 
                        ax1.plot(0.42, 2.5, '*', color=col[1])#, linewidth=2)#, zorder = 1, label = erup_dict['WIZ_3']) 
                        ax1.plot(0.32, 2.5, '*', color=col[1])#, linewidth=2)#, zorder = 1, label = erup_dict['WIZ_4'])
                        #ax1.plot(0.77, 2.5, '*', color=col[1])#, linewidth=2)#, zorder = 1, label = erup_dict['WIZ_5'])
                        ax1.plot( [],[], '*', color=col[1], label= 'Whakaari eruptions')#, linewidth=2)#, zorder = 1, label = erup_dict['WIZ_5'])
                        ax1.text(0.18-.015, 2.5+.17, erup_dict['WIZ_1'][-4:], rotation=90, va='center', color=col[1])
                        ax1.text(0.24-.015, 2.5+.17, erup_dict['WIZ_2'][-5:], rotation=90, va='center', color=col[1])
                        ax1.text(0.42-.015, 2.5+.17, erup_dict['WIZ_3'][-5:], rotation=90, va='center', color=col[1])
                        ax1.text(0.32-.015, 2.5+.17, erup_dict['WIZ_4'][-4:], rotation=90, va='center', color=col[1])
                        #ax1.text(0.77-.015, 2.5+.25, erup_dict['WIZ_5'][-4:], rotation=90, va='center', color=col[1])

                        # Ruapehu
                        ax1.plot(0.02, 2.0, '*', color=col[0])#, linewidth=2)#, zorder = 1, label = erup_dict['WIZ_4'])
                        ax1.plot(0.77, 2.0, '*', color=col[0])#, linewidth=2)#, zorder = 1, label = erup_dict['WIZ_4'])
                        ax1.plot(0.07, 2.0, '*', color=col[0])#, linewidth=2)#, zorder = 1, label = erup_dict['WIZ_5'])
                        ax1.plot([],[], '*', color=col[0], label= 'Ruapehu eruptions')#, linewidth=2)#, zorder = 1, label = erup_dict['WIZ_5'])
                        ax1.text(0.02-.015, 2.0+.15, erup_dict['FWVZ_1'][-4:], rotation=90, va='center', color=col[0])
                        ax1.text(0.77-.015, 2.0+.15, erup_dict['FWVZ_2'][-4:], rotation=90, va='center', color=col[0])
                        ax1.text(0.07-.015, 2.0+.15, erup_dict['FWVZ_3'][-4:], rotation=90, va='center', color=col[0])

                        # Veniaminof
                        ax1.plot(0.18, 1.5, '*', color=col[5])#, linewidth=2)#, zorder = 1, label = erup_dict['WIZ_4'])
                        ax1.plot(0.28, 1.5, '*', color=col[5])#, linewidth=2)#, zorder = 1, label = erup_dict['WIZ_5'])
                        ax1.plot([],[], '*', color=col[5], label= 'Veniaminof eruptions')#, linewidth=2)#, zorder = 1, label = erup_dict['WIZ_5'])
                        ax1.text(0.18-.015, 1.5+.15, erup_dict['VNSS_1'][-4:], rotation=90, va='center', color=col[5])
                        ax1.text(0.28-.015, 1.5+.15, erup_dict['VNSS_2'][-4:], rotation=90, va='center', color=col[5])

                        # Tongariro
                        ax1.plot(0.12, 1., '*', color=col[4])#, linewidth=2)#, zorder = 1, label = erup_dict['WIZ_4'])
                        ax1.plot(0.17, 1., '*', color=col[4])#, linewidth=2)#, zorder = 1, label = erup_dict['WIZ_5'])
                        ax1.plot([],[], '*', color=col[4], label= 'Tongariro eruptions')#, linewidth=2)#, zorder = 1, label = erup_dict['WIZ_5'])
                        ax1.text(0.12-.015, 1.+.17, erup_dict['KRVZ_1'][-5:], rotation=90, va='center', color=col[4])
                        ax1.text(0.17-.015, 1.+.17, erup_dict['KRVZ_2'][-5:], rotation=90, va='center', color=col[4])

                        # Pavlof
                        # ax1.axvline(0.54, color=col[2], linestyle='-', linewidth=2, zorder = 1, label = erup_dict['PVV_3']) 
                        # ax1.axvline(0.46, color=col[2], linestyle='--', linewidth=2, zorder = 1, label = erup_dict['BELO_2'])
                        # ax1.plot(0.07, 1.75, '*', color=col[3])#, linewidth=2)#, zorder = 1, label = erup_dict['WIZ_4'])
                        # ax1.plot(0.11, 1.75, '*', color=col[3])#, linewidth=2)#, zorder = 1, label = erup_dict['WIZ_5'])
                        # ax1.plot(0.56, 1.75, '*', color=col[3])#, linewidth=2)#, zorder = 1, label = erup_dict['WIZ_5'])
                        # ax1.plot( [],[], '*', color=col[3], label= 'Pavlof eruptions')#, linewidth=2)#, zorder = 1, label = erup_dict['WIZ_5'])
                        
                        # # Bezymiany
                        # ax1.plot(0.05, 1.5, '*', color=col[2])#, linewidth=2)#, zorder = 1, label = erup_dict['WIZ_4'])
                        # ax1.plot(0.46, 1.5, '*', color=col[2])#, linewidth=2)#, zorder = 1, label = erup_dict['WIZ_5'])
                        # ax1.plot(0.13, 1.5, '*', color=col[2])#, linewidth=2)#, zorder = 1, label = erup_dict['WIZ_4'])
                        # ax1.plot( [],[], '*', color=col[2], label= 'Bezymiany eruptions')#, linewidth=2)#, zorder = 1, label = erup_dict['WIZ_5'])

                        # Tongariro
                        #ax1.plot(0.13, 2.75, '*', color=col[7])#, linewidth=2)#, zorder = 1, label = erup_dict['WIZ_4'])
                        #ax1.plot(0.14, 2.75, '*', color=col[7])#, linewidth=2)#, zorder = 1, label = erup_dict['WIZ_5'])
                        #ax1.plot( [],[], '*', color=col[7], label= 'Tongariro eruptions')#, linewidth=2)#, zorder = 1, label = erup_dict['WIZ_5'])

                if id262:#ft == ['zsc2_dsarF__change_quantiles__f_agg_"var"__isabs_False__qh_0.6__ql_0.4']:
                    if True:
                        # Whakaari
                        ax1.plot(0.06, 9., '*', color=col[1])#, linewidth=2, zorder = 1, label = erup_dict['WIZ_1'])
                        ax1.plot(0.00, 9., '*', color=col[1])#, linewidth=2)#, zorder = 1, label = erup_dict['WIZ_2'])
                        #ax1.plot(0.1, 1., '*', color=col[1])#, linewidth=2)#, zorder = 1, label = erup_dict['WIZ_3']) 
                        ax1.plot(0.18, 9., '*', color=col[1])#, linewidth=2)#, zorder = 1, label = erup_dict['WIZ_3']) 
                        ax1.plot(0.03, 9., '*', color=col[1])#, linewidth=2)#, zorder = 1, label = erup_dict['WIZ_4'])
                        ax1.plot( [],[], '*', color=col[1], label= 'Whakaari eruptions')#, linewidth=2)#, zorder = 1, label = erup_dict['WIZ_5'])
                        ax1.text(0.06-.01, 9.+.75, erup_dict['WIZ_1'][-4:], rotation=90, va='center', color=col[1])
                        ax1.text(0.00-.01, 9.+.75, erup_dict['WIZ_2'][-5:], rotation=90, va='center', color=col[1])
                        #ax1.text(0.1-.015, 1.+.1, erup_dict['WIZ_3'][-5:], rotation=90, va='center', color=col[1])
                        ax1.text(0.18-.01, 9.+.75, erup_dict['WIZ_3'][-5:], rotation=90, va='center', color=col[1])
                        ax1.text(0.03-.01, 9.+.75, erup_dict['WIZ_4'][-4:], rotation=90, va='center', color=col[1])
                        
                        # Ruapehu
                        ax1.plot(0.04, 7.5, '*', color=col[0])#, linewidth=2)#, zorder = 1, label = erup_dict['WIZ_4'])
                        ax1.plot(0.01, 7.5, '*', color=col[0])#, linewidth=2)#, zorder = 1, label = erup_dict['WIZ_5'])
                        ax1.plot(0.24, 7.5, '*', color=col[0])#, linewidth=2)#, zorder = 1, label = erup_dict['WIZ_5'])
                        ax1.plot([],[], '*', color=col[0], label= 'Ruapehu eruptions')#, linewidth=2)#, zorder = 1, label = erup_dict['WIZ_5'])
                        ax1.text(0.04-.01, 7.5+.75, erup_dict['FWVZ_1'][-4:], rotation=90, va='center', color=col[0])
                        ax1.text(0.01-.01, 7.5+.75, erup_dict['FWVZ_2'][-4:], rotation=90, va='center', color=col[0])
                        ax1.text(0.24-.01, 7.5+.75, erup_dict['FWVZ_3'][-4:], rotation=90, va='center', color=col[0])

                        # Veniaminof
                        ax1.plot(0.1, 6, '*', color=col[5])#, linewidth=2)#, zorder = 1, label = erup_dict['WIZ_4'])
                        ax1.plot(0.04, 6, '*', color=col[5])#, linewidth=2)#, zorder = 1, label = erup_dict['WIZ_5'])
                        ax1.plot([],[], '*', color=col[5], label= 'Veniaminof eruptions')#, linewidth=2)#, zorder = 1, label = erup_dict['WIZ_5'])
                        ax1.text(0.1-.01, 6+.75, erup_dict['VNSS_1'][-4:], rotation=90, va='center', color=col[5])
                        ax1.text(0.04-.01, 6+.75, erup_dict['VNSS_2'][-4:], rotation=90, va='center', color=col[5])

                        # Tongariro
                        ax1.plot(0.04, 4.5, '*', color=col[4])#, linewidth=2)#, zorder = 1, label = erup_dict['WIZ_4'])
                        ax1.plot(0.14, 4.5, '*', color=col[4])#, linewidth=2)#, zorder = 1, label = erup_dict['WIZ_5'])
                        ax1.plot([],[], '*', color=col[4], label= 'Tongariro eruptions')#, linewidth=2)#, zorder = 1, label = erup_dict['WIZ_5'])
                        ax1.text(0.04-.01, 4.5+.75, erup_dict['KRVZ_1'][-5:], rotation=90, va='center', color=col[4])
                        ax1.text(0.14-.01, 4.5+.75, erup_dict['KRVZ_2'][-5:], rotation=90, va='center', color=col[4])

                if ft == ['zsc2_dsarF__change_quantiles__f_agg_"var"__isabs_False__qh_0.6__ql_0.4']:
                    ax1.set_xlim([-1.,1.])
                if id262:#ft == ['zsc2_dsarF__change_quantiles__f_agg_"var"__isabs_False__qh_0.6__ql_0.4']:
                    ax1.set_xlim([-.5,.5])
                if id373:#ft == ['zsc2_dsarF__change_quantiles__f_agg_"var"__isabs_False__qh_0.6__ql_0.4']:
                    pass
                    ax1.set_xlim([-1.2,1.])
                    ax1.set_xlim([-.7,1.])
                if id713:#ft == ['zsc2_dsarF__change_quantiles__f_agg_"var"__isabs_False__qh_0.6__ql_0.4']:
                    pass
                    ax1.set_xlim([-1.3,1.])

                #
                ax1.set_yticklabels([])
                #ax1.legend(loc=2)    
                plt.show()

        if False: # calculate p-value for each eruption in the record
            # plot histogram of cc values 
            # import csv as dataframe
            fl_nm = path+sta+'_'+str(erup+1)+'_'+ft[0]+'_over_'+sta_exp+'.csv'
            fl_nm = fl_nm.replace('"', "-")
            df_aux = pd.read_csv(fl_nm, index_col=0)
            df_aux.index = pd.to_datetime(df_aux.index)
            if sta == 'WIZ':
                if id713: 
                    if sta_exp == 'WIZ':
                        erup_vals = [0.65,0.68,0.71,0.74]
                    if sta_exp == 'FWVZ':
                        erup_vals = [0.71,0.38,0.43]
                    if sta_exp == 'VNSS':
                        erup_vals = [0.75,0.40]
                    if sta_exp == 'KRVZ':
                        erup_vals = [0.31,0.47]
                if id373: 
                    if sta_exp == 'WIZ':
                        erup_vals = [0.18,0.24,0.43,0.30]
                    if sta_exp == 'FWVZ':
                        erup_vals = [0.71,0.38,0.43]
                        erup_vals = [0.02,0.77,0.07]
                    if sta_exp == 'VNSS':
                        pass
                        erup_vals = [0.18,0.28]
                    if sta_exp == 'KRVZ':
                        erup_vals = [0.1,0.17]
            if sta == 'FWVZ':
                if id713:
                    pass 

                if id373: 
                    if sta_exp == 'WIZ':
                        erup_vals = [0.18,0.24,0.43,0.30,.77]
                    if sta_exp == 'FWVZ':
                        erup_vals = [0.04,0.13]
                    if sta_exp == 'VNSS':
                        erup_vals = [0.04,0.33]
            if sta == 'VNSS':
                if id713: 
                    if sta_exp == 'WIZ':
                        erup_vals = [0.74,0.48,0.34,0.56,0.71]
                    if sta_exp == 'FWVZ':
                        erup_vals = [0.67,0.07,0.67]
                    if sta_exp == 'KRVZ':
                        erup_vals = [0.05,0.27]
                if id373: 
                    pass       

            # 
            cc_vals = df_aux[:].loc[:,'cc'].values
            int_tot = np.sum(np.abs(cc_vals))
            p_vals = [] 

            if False: # using integral approach
                for i, e in enumerate(erup_vals):
                    _count = 0
                    _int_ls = []
                    # collect vals
                    for cc in cc_vals:
                        if cc >= e:
                            _count+=1
                            _int_ls.append(cc)
                    # calc int
                    _int = np.sum(np.abs(_int_ls))
                    p_vals.append(_count/len(cc_vals))
                print(p_vals)
                asdf
            if True: # using Kolmogorov-Smirnov test
                from scipy import stats
                #
                pval = stats.kstest(erup_vals, cc_vals).pvalue
                asd

        if False: # calc feat mean and std for the whole record 
            #
            t0 = "2012-01-01" #'2006-01-01' 
            t1 = "2020-12-30"#'2021-07-31'
            sta = 'WIZ'
            # eruption
            #erup = 'WIZ_5'#'KRVZ_1'#'WIZ_5'#FWVZ_3, VNSS_1, KRVZ_1, BELO_2, PVV_1
            # features
            #fts = ['zsc2_hfF__fft_coefficient__coeff_38__attr_"real"',
            #        'zsc2_dsarF__change_quantiles__f_agg_"var"__isabs_False__qh_0.6__ql_0.4',
            #        'zsc2_dsarF__median']
            fts = ['zsc2_dsarF__median', 'zsc2_dsarF__change_quantiles__f_agg_"var"__isabs_False__qh_0.6__ql_0.4']
            #
            def _calc_mean_std(sta,t0, t1, fts):
                #
                means = []
                stds = [] 
                for i, ft in enumerate(fts):
                    if 'zsc2_dsarF' in ft:
                        ds = ['zsc2_dsarF']
                    if 'zsc2_rsamF' in ft:
                        ds = ['zsc2_rsamF']
                    if 'zsc2_mfF' in ft:
                        ds = ['zsc2_mfF']
                    if 'zsc2_hfF' in ft:
                        ds = ['zsc2_hfF']
                    if True:
                        path_feat_serv = 'C:\\Users\\aar135\\codes_local_disk\\volc_forecast_tl\\features_bkp\\features_server\\'
                        fm_e1 = ForecastModel(window=2., overlap=1., station =  sta,
                            look_forward=2., data_streams=ds, 
                            feature_dir=path_feat_serv, 
                            savefile_type='pkl') 
                    else:
                        fm_e1 = ForecastModel(window=2., overlap=1., station = sta,
                            look_forward=2., data_streams=ds, savefile_type='csv')
                    #
                    ft = ft.replace("-",'"')
                    # adding multiple Axes objects
                    ft_e1 = fm_e1.get_features(ti=t0, tf=t1, n_jobs=1, compute_only_features=[ft])
                    # extract values to plot 
                    ft_e1_t = ft_e1[0].index.values
                    ft_e1_v = ft_e1[0].loc[:,ft]
                    #
                    means.append(np.mean(ft_e1_v))
                    stds.append(np.std(ft_e1_v))
                #
                return means, stds
            #
            means, stds = _calc_mean_std(sta,t0, t1, fts)
            asdf

    # plot cc evolution in time
    if False:  
        # load pd of correlation 
        # name of precursor file 
        path = '..'+os.sep+'features'+os.sep+'correlations'+os.sep+'stat_sign_per_hig_corr_feat'+os.sep
        pre_fl_nm = path + 'WIZ_5_zsc2_dsarF__median_over_WIZ.csv'
        #
        df_aux = pd.read_csv(pre_fl_nm, index_col=0)
        df_aux.index = pd.to_datetime(df_aux.index)
        #df_aux.index = [datetimeify(ind) for ind in df_aux.index]#pd.to_datetime(df_aux.index)
        np = len(df_aux.index)
        #
        # calculate number of times cc crossing threshold
        if False:
            count_70 = 0 
            count_90 = 0
            count_95 = 0
            count_min_erup_cc = 0
            for ind in df_aux.index:
                a = df_aux.loc[ind,'cc']
                if a > 0.7: 
                    count_95 += 1
                if a > 0.6: 
                    count_90 += 1
                if a > 0.25: 
                    count_70 += 1
                if a > 0.67: 
                    count_min_erup_cc += 1
            asdf
        # calculate the proportion of correlation coefficients higher than a value c_c that fall one month either side of an eruption
        if True:
            # path eruptions
            path = '..'+os.sep+'data'+os.sep+'WIZ_eruptive_periods.txt'
            with open(path,'r') as fp:
                tes = [datetimeify(ln.rstrip()) for ln in fp.readlines()]
            # list of dates
            ls_dates = []
            for e in tes:
                #e = e-30*day
                for i in range(30):#60):
                    #_e = e+i*day
                    _e = e-15*day +i*day
                    ls_dates.append([_e.year,_e.month,_e.day])
            #
            count_70 = 0 
            count_90 = 0
            count_95 = 0
            count_min_erup_cc = 0
            for ind in df_aux.index:
                _idx = [ind.year,ind.month,ind.day]
                if _idx in ls_dates:
                    a = df_aux.loc[ind,'cc']
                    if a > 0.7: 
                        count_95 += 1
                    if a > 0.6: 
                        count_90 += 1
                    if a > 0.25: 
                        count_70 += 1
                    if a > 0.67: 
                        count_min_erup_cc += 1
            asdf
        # probablity of eruption
        if False: 
            # path eruptions
            def prob_cc_obs(cc, threshold = 0.7):
                '''
                function that calculate of observing a correlation coeficient (cc) during the month before the eruption.
                
                input:
                    cc: observed correlation coeficient 
                    threshold: 0.7(~95%) or 0.6(~90%) or 0.25(~70%)
                output:
                    probability [0,1]    
                '''
                if threshold == 0.7:
                    if cc > threshold: 
                        n_cc = 38 # total number of values with cc>threshold with +- a month from eruptions
                        n_tot = 261 # total number of values with cc>threshold (whole record)
                        n_cc_post = 14 # total number of values with cc>threshold with + a month from eruptions
                if threshold == 0.6:
                    if cc > threshold: 
                        n_cc = 69
                        n_tot = 423
                        n_cc_post = 32
                if threshold == 0.25:
                    if cc > threshold: 
                        n_cc = 104
                        n_tot = 1115
                        n_cc_post = 49
                if not cc:
                    return 0
                else:
                    return n_cc/(n_tot-n_cc_post) # n_cc_obs / n_cc - n_cc_post 
            
            a = prob_cc_obs(0.75, threshold = 0.7)*100
            asd

    # function for forcaster
    if False:
        ## function that received a month of a certain feature and correlated with precursor (1 month)
        # import and write precursor
        # select precursors (one)
        id713 = True
        id373 = False         
        if id713: 
            sta = 'WIZ'
            erup = 4 # fifth 2019
            ds = ['zsc2_dsarF']
            ft = ['zsc2_dsarF__median']
        if id373:
            sta = 'FWVZ'
            erup = 1 # 
            ds = ['zsc2_dsarF']
            ft = ['zsc2_dsarF__change_quantiles__f_agg_"var"__isabs_False__qh_0.6__ql_0.4']         
        # import and write precursor
        if False:
            fm_e1 = ForecastModel(window=2., overlap=1., station = sta,
                look_forward=2., data_streams=ds, savefile_type='csv')
            # initial and final time of interest for each station
            tf_e1 = fm_e1.data.tes[erup]
            ti_e1 = tf_e1 - 30*day #month
            # extract feature values 
            ft_e1 = fm_e1.get_features(ti=ti_e1, tf=tf_e1, n_jobs=1, compute_only_features=[ft])
            # extract values to correlate 
            ft_e1_t = ft_e1[0].index.values
            ft_e1_v = ft_e1[0].loc[:,ft].values
            ft_e1_v = [ft_e1_v[i][0] for i in range(len(ft_e1_v))] 
            path = '..'+os.sep+'features'+os.sep+'correlations'+os.sep+'forecasting_from_single_feature'+os.sep
            if sta == 'FWVZ':
                ft[0] = ft[0].replace('"','-')
            with open(path+'prec'+'_'+sta+'_'+str(erup+1)+'_'+ft[0]+'.csv','w', newline='') as csvfile:
                spamwriter = csv.writer(csvfile, delimiter=',', quoting=csv.QUOTE_MINIMAL)
                spamwriter.writerow(['time', 'feat'])
                for i,val in enumerate(ft_e1_v):
                    spamwriter.writerow([ft_e1_t[i], str(val)])
            #
        # name of precursor file 
        path = '..'+os.sep+'features'+os.sep+'correlations'+os.sep+'forecasting_from_single_feature'+os.sep
        pre_fl_nm = path + 'prec'+'_'+sta+'_'+str(erup+1)+'_'+ft[0]+'.csv'
        if sta == 'FWVZ':
            ft[0] = ft[0].replace('"','-')
        # define funtion that correlates
        if id713: 
            prec_nm = 'WIZ_5_zsc2_dsarF_median'
            prec_code = 1
        if id373:
            prec_nm = 'FWVZ_3_zsc2_dsarF_ch_qt_var'
            prec_code = 2
        #
        def corr_with_precursor(ts_ft, prec_code = None, path_prec = None):
            '''
            Correlation between known precursor (prec_code) and inputed time serie feature values (30 days).
            Function calculate a 'Pearson' correlation coeficient between the two 30 days time series values.

            input:
            ts_ft: feature values for 30 days (before eruption) euqlly space every 10 min.
                type: numpy vector 
                ts_ft required length: 4320
            prec_code: precursor code
                1: WIZ_5_zsc2_dsarF_median
                2: FWVZ_3_zsc2_dsarF_ch_qt_var
                type: integer

            output:
            cc: correlation coeficient 
                type: float

            Default:
                prec_code: 1 ('WIZ_5_zsc2_dsarF_median')
                path_prec: '..'+os.sep+'features'+os.sep+'correlations'+os.sep+'forecasting_from_single_feature'+os.sep

            Notes:
            -  Requiered libraries: Numpy (as np), Pandas (as pd), os
            -  Times are irrelevant. Correlation only uses feature values. 
            '''
            if not prec_code:
                prec_code = 1
            # import precursor 
            if not path_prec:
                path =  '..'+os.sep+'features'+os.sep+'correlations'+os.sep+'forecasting_from_single_feature'+os.sep
            if prec_code == 1:
                path = path+'prec_WIZ_5_zsc2_dsarF__median.csv'
            if prec_code == 2: 
                path = path+'prec_FWVZ_2_zsc2_dsarF__change_quantiles__f_agg_-var-__isabs_False__qh_0.6__ql_0.4.csv'
            # import precursor 
            prec_ts_ft = np.loadtxt(path,skiprows=1,delimiter=',', usecols=1)
            # stack data
            data = [ts_ft, prec_ts_ft]
            arr_aux =  np.vstack((data)).T
            # perform correlation 
            df = pd.DataFrame(data=arr_aux)#, index=fm_aux.fM.index, columns=lab, dtype=None, copy=None)
            df_corr = df.corr(method='pearson')
            cc = np.round(df_corr.iloc[0,1],3)
            # 
            return cc
        
        ## test function 
        # load eruption data
        path_feat = '..'+os.sep+'features'+os.sep
        fm_e1 = ForecastModel(window=2., overlap=1., station = 'VNSS',
            look_forward=2., data_streams=ds,  feature_dir=path_feat, savefile_type='csv')
        # initial and final time of interest for each station
        tf_e1 = fm_e1.data.tes[0]
        ti_e1 = tf_e1 - 30*day #month
        # extract feature values 
        if sta == 'FWVZ':
            ft[0] = ft[0].replace('-','"')
        ft_e1 = fm_e1.get_features(ti=ti_e1, tf=tf_e1, n_jobs=1, compute_only_features=[ft])
        # extract values to correlate 
        ft_e1_t = ft_e1[0].index.values
        ft_e1_v = ft_e1[0].loc[:,ft].values
        ft_e1_v = [ft_e1_v[i][0] for i in range(len(ft_e1_v))]
        ## run fuction
        cc = corr_with_precursor(ft_e1_v, prec_code = prec_code, path_prec = None)

    # eruption probabilities
    if False:
        '''
        Using Bayes equation to estimated eruption probability
        (1) P(e| cc>th') = P(cc>th'|e) * P(e) / P(cc>th')
        (2) P(e| cc>th' & cc>th' for more than x days) = P(cc>th' & cc>th' for more than x days|e) * P(e) / P(cc>th' & cc>th' for more than x days)
        '''
        import numpy as np
        # load pd of correlation 
        # name of precursor file 
        path = '..'+os.sep+'features'+os.sep+'correlations'+os.sep+'stat_sign_per_hig_corr_feat'+os.sep
        pre_fl_nm = path + 'WIZ_5_zsc2_dsarF__median_over_WIZ.csv'
        df_aux = pd.read_csv(pre_fl_nm, index_col=0)
        df_aux.index = pd.to_datetime(df_aux.index)
        N = len(df_aux.index)
        # 
        # path eruptions
        path = '..'+os.sep+'data'+os.sep+'WIZ_eruptive_periods.txt'
        with open(path,'r') as fp:
            tes = [datetimeify(ln.rstrip()) for ln in fp.readlines()]
        # list of dates
        ls_dates = []
        range_days_erup = 20 # range of days that are consider as eruption
        for j, e in enumerate(tes):
            #e = e-30*day
            for i in range(range_days_erup):#60):
                _e = e-int(range_days_erup/2)*day +i*day # half days back and forth
                #_e = e-int(range_days_erup/4)*day +i*day # 1/4 back and 3/4 for back days back and forth
                ls_dates.append([_e.year,_e.month,_e.day])

        # th'
        cc_e = np.zeros((len(tes),))
        _cc = 0
        range_days_erup = 6
        for j, e in enumerate(tes):
            #e = e-30*day
            for i in range(range_days_erup):#60):
                _e = e-int(range_days_erup/2)*day +i*day
                # max cc value during period
                _cc = df_aux.loc[datetimeify(str(_e.year)+'-'+str(_e.month)+'-'+str(_e.day))].values[0]
                if cc_e[j] < _cc:
                    cc_e[j] = _cc
        th =  min(cc_e)
        #th =  min([df_aux.loc[datetimeify(str(_e.year)+'-'+str(_e.month)+'-'+str(_e.day))].values[0] for _e in tes])
        
        # count time that cc passes th' during eruption, and overall
        count_cc_th_erup = 0
        count_cc_th_overall = 0
        for ind in df_aux.index:
            #
            _idx = [ind.year,ind.month,ind.day]
            if _idx in ls_dates:
                if df_aux.loc[ind].values[0] > th:  
                    count_cc_th_erup += 1
            #
            if df_aux.loc[ind].values[0] > th: 
                count_cc_th_overall += 1
        ## (1) P(e| cc>th') = P(cc>th'|e) * P(e) / P(cc>th')
        if False:
            # calc P(cc>th'|e)
            p_lkl = count_cc_th_erup / len(ls_dates)
            p_e = len(ls_dates) / len(df_aux.index)
            p_cc_th = count_cc_th_overall / len(df_aux.index) 
            #
            p_e_th = p_lkl * p_e / p_cc_th

        ## (2) P(e| cc>th' & cc>th' for more than x days) = P(cc>th' & cc>th' for more than x days|e) * P(e) / P(cc>th' & cc>th' for more than x days)
        if True:
           
            # calc cc long during eruptions
            _cc_long_erup = np.zeros((len(tes),))
            range_days_erup = 20
            for j, e in enumerate(tes):
                _count = 0
                #e = e-30*day
                for i in range(range_days_erup):#60):
                    _e = e-int(range_days_erup/2)*day +i*day
                    # max cc value during period
                    _cc = df_aux.loc[datetimeify(str(_e.year)+'-'+str(_e.month)+'-'+str(_e.day))].values[0]
                    if th < _cc:
                        _count = _count + 1
                    #
                _cc_long_erup[j] = _count
            # calc cc long during non eruptive periods 
            _cc_long_non_erup = []
            _in = False
            _count = 0
            for ind in df_aux.index:
                #
                _idx = [ind.year,ind.month,ind.day]
                if _idx not in ls_dates:
                    if df_aux.loc[ind].values[0] > th:  
                        if _in: 
                            _count = _count + 1
                        else:
                            _count = 1
                            _in = True
                    else:
                        if _count > 0:
                            _cc_long_non_erup.append(_count)
                        _count = 0
                        _in = False
            
            ##
            def prob_th_tday(_cc_long_erup, _cc_long_non_erup, t = 3):
                #calc times that cc prolong for more than 't' days
                def _p_lkl(_cc_long_erup, _cc_long_non_erup, t = 3):
                    _c1 = 0 
                    for v in _cc_long_erup:
                        if v > t:
                            _c1 = _c1+1
                    _c2 = 0 
                    for v in _cc_long_non_erup:
                        if v > t:
                            _c2 =_c2 +1
                    if t == 5:
                        pass
                    return _c1, _c2
                _c1, _c2 = _p_lkl(_cc_long_erup, _cc_long_non_erup, t = t)

                # P(cc>th' & cc>th' for more than t days|e)
                p_lkl = _c1 / len(ls_dates)
                p_lkl = _c1 / (_c1+_c2)
                p_lkl = _c1 / len(_cc_long_erup)

                #P(e)
                p_e = len(ls_dates) / len(df_aux.index)

                #P(cc>th' & cc>th' for more than x days)
                p_cc_th = (_c1+_c2) / len(df_aux.index)#(len(_cc_long_non_erup)+len(_cc_long_erup))#len(df_aux.index)
                p_cc_th = _c1 / len(_cc_long_erup) * (_c1+_c2) / (len(_cc_long_non_erup)+len(_cc_long_erup))

                #
                p_e_th = p_lkl * p_e / p_cc_th
                #
                if t == 9:
                    pass
                return p_e_th
            
            _ = np.sort(_cc_long_erup)
            days = np.arange(int(_[-1]))
            probs = []
            for i, d in enumerate(days): 
                p_e_th = prob_th_tday(_cc_long_erup, _cc_long_non_erup, t = d)
                probs.append(p_e_th)
            # log fit (https://stackoverflow.com/questions/3433486/how-to-do-exponential-and-logarithmic-curve-fitting-in-python-i-found-only-poly)
            a = np.polyfit(days[:-2], np.log(probs[:-2]), 1)
            y_fit = np.exp(a[1]) * np.exp(a[0] * days)
            
            # plot 
            fig, ax1 = plt.subplots(nrows = 1, ncols = 1, figsize = (8,6))
            ax1.plot(days, probs, 'b*')
            ax1.plot(days, y_fit, '-', color = 'gray', label = 'exp fit')
            ax1.set_xlabel('cc>th longer than x days [d]')#, fontsize=textsize)
            ax1.set_ylabel('eruption probability')#, fontsize=textsize)
            ax1.grid(True, which='both', linewidth=0.1)
            #
            #ax1.set_title('P(e| cc>th for more than x days) = P(cc>th for more than x days|e) * P(e) / P( cc>th for more than x days)')
            ax1.set_title('Eruption probability given a CC higher than a threshold ('+str(round(th,2))+') \nprolong for more than x days')
            ax1.legend(loc = 2)
            ax1.grid(True, color='gray', linestyle='-', linewidth=.5)
            ax1.set_ylim([0,1])
            plt.show()
            asdf

def _calc_corr(p):
    'auxiliary funtion for parallelization inside function corr_ana_feat()'
    k, ft_e1_v, sta_exp, d, ft, ds = p 
    if sta_exp in ['VNSS', 'FWVZ']:
        path_feat_serv = 'C:\\Users\\aar135\\codes_local_disk\\volc_forecast_tl\\features_bkp\\features_server\\'
        fm_aux = ForecastModel(window=2., overlap=1., station = sta_exp,
            look_forward=2., data_streams=ds, 
            feature_dir=path_feat_serv, 
            savefile_type='pkl') 
    else:
        fm_aux = ForecastModel(window=2., overlap=1., station = sta_exp,
            look_forward=2., data_streams=ds, savefile_type='pkl')
    # initial and final time of interest for each station
    tf_aux = d
    ti_aux = tf_aux - 30*day #month
    # extract feature values for aux period and stack with characteristic one
    ft_aux = fm_aux.get_features(ti=ti_aux, tf=tf_aux, n_jobs=1, compute_only_features=[ft])
    ft_aux_t = ft_aux[0].index.values
    ft_aux_v = ft_aux[0].loc[:,ft].values
    ft_aux_v = [ft_aux_v[i][0] for i in range(len(ft_aux_v))]            
    data = [ft_e1_v, ft_aux_v]
    arr_aux =  np.vstack((data)).T

    # correlate characteristic feature for section 
    # create aux pandas obj for correlation
    df = pd.DataFrame(data=arr_aux)#, index=fm_aux.fM.index, columns=lab, dtype=None, copy=None)
    df_corr = df.corr(method='pearson')
    # save cc value in pd with time (end of period)

    with open(str(d), 'w', newline='') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=',')
        spamwriter.writerow([str(np.round(df_corr.iloc[0,1],3))])

def plot_corr_feat_dendrogram():
    ''' Plot comparison matrix.
    Notes
    -----
    Implementation from https://stackoverflow.com/questions/2982929/plotting-
    results-of-hierarchical-clustering-ontop-of-a-matrix-of-data-in-python
    '''
    def read():
        path_file = '..'+os.sep+'features'+os.sep+'correlations'+os.sep+'corr_0_sum_cum.csv'
        with open(path_file,'r') as fp:
            h = fp.readline()
            h = h.split(',')[1:]
        D = np.genfromtxt(path_file, skip_header=1, delimiter=',', dtype=float)[:,1:]
        return h,D

    hds,D2 = read()
    D = 1.-D2-np.eye(D2.shape[0])

    # process distance matrix
    allclientlist = hds
    n = len(allclientlist)
    condensedD = squareform(D)
    Y1 = sch.linkage(condensedD, method='centroid')
    Y2 = sch.linkage(condensedD, method='single')
    
    # figure and axis set up
    fig = plt.figure(figsize=(12,12))#8.27,11.69 ))
    ax1 = fig.add_axes([0.10,0.1+0.6*0.29,0.11,0.6*0.71])
    ax1b = fig.add_axes([0.205,0.1+0.6*0.29,0.09,0.6*0.71])
    ax2 = fig.add_axes([0.3,0.705 +0.09*0.71,0.6,0.19*0.71])
    ax2b = fig.add_axes([0.3,0.705,0.6,0.09*0.71])
    axmatrix = fig.add_axes([0.3,0.1+0.6*0.29,0.6,0.6*0.71])

    # compute appropriate font size
    ax_inches = 0.6*8.27     # axis dimension in inches
    font_inches = ax_inches/n   # font size in inches
    fs = int(1.2*font_inches*72)  # font size in points
    fs = np.min([fs, 5])+4
    
    # plotting
    Z1 = sch.dendrogram(Y1, orientation='left', ax=ax1)
    ax1.set_xlim([1.05,0])
    Z2 = sch.dendrogram(Y2, ax=ax2)
    ax2.set_ylim([0,1.05])
    idx1 = Z1['leaves']
    idx2 = Z2['leaves']
    
    # shuffle matrix for clustering
    D = D[idx1,:]
    D = D[:,idx2]
    vmax = 1.-D[D.nonzero()].min()
    D = 1.-D
    im = axmatrix.matshow(D, aspect='auto', origin='lower', cmap=plt.cm.YlGnBu, vmin=D.min(), vmax=vmax)
    for i in range(D.shape[0]):
        for j in range(D.shape[1]):
            axmatrix.text(i,j,'{:3.2f}'.format(D[i,j]), ha='center', va='center', color='y', size=6)
    
    dn = 1

    # upkeep
    for ax in [ax1, ax2, axmatrix]:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim(auto=False)
        ax.set_ylim(auto=False)
    
    for axb,ax in zip([ax1b, ax2b],[ax1,ax2]):
        axb.axis('off')
        axb.set_xlim(ax.get_xlim())
        axb.set_ylim(ax.get_ylim())

    # grid lines and client names
    dy = ax1b.get_ylim()[1]/n
    x0 = ax1b.get_xlim()[1]
    for i,idx in enumerate(Z1['leaves']):
        name = hds[idx]
        col = 'k'
        ax1b.text(0.9*x0, (i+0.5)*dy, name, size=fs, color = col, ha='right', va='center')
        if i % dn == 0:
            ax1b.axhline(i*dy, color = 'k', linestyle='-', linewidth=0.25)
            ax1.axhline(i*dy, color = 'k', linestyle='-', linewidth=0.25)

    dx = ax2b.get_xlim()[1]/n
    y0 = ax2b.get_ylim()[0]
    for i,idx in enumerate(Z2['leaves']):
        name = hds[idx]
        col = 'k'
        ax2b.text((i+0.5)*dx, 0.95*y0, name, size=fs, color = col, ha='center', 
            va='bottom', rotation=-90.)
        if i % dn == 0:
            ax2b.axvline(i*dx, color = 'k', linestyle='-', linewidth=0.25)
            ax2.axvline(i*dx, color = 'k', linestyle='-', linewidth=0.25)

    for i in range(n):
        if i % dn == 0:
            axmatrix.axvline(i-0.5, color = 'k', linestyle='-', linewidth=0.25)
            axmatrix.axhline(i-0.5, color = 'k', linestyle='-', linewidth=0.25)

    plt.tight_layout()
    fig.savefig('..'+os.sep+'features'+os.sep+'correlations'+os.sep+'corr_0_dendogram.png', dpi = 100)
    plt.close(fig)

def download_geonet_data():
    # download geonet lake temperature and level data 
    if True:
        ## download temperature data. Save in data folder
        # Download data for Ruapehu crater lake from GEONET(siteID: WI201)
        if True: 
            ruapehu = True
            whakaari = False   
            import json # package to read json code
            import requests # package to get data from an API
            import datetime # package to deal with time
            # Define the longitude and latitude of the point of interest for Ruapehu
            if ruapehu:
                point_long = 175.565
                point_lat = -39.281
                box_size = 0.1
            if whakaari:
                point_long = 177.182
                point_lat = -37.521
                box_size = 0.1
            #
            long_max = str(point_long + box_size) 
            long_min = str(point_long - box_size)
            lat_max  = str(point_lat - box_size)
            lat_min  = str(point_lat + box_size)
            #
            poly = ("POLYGON((" +long_max + " " +lat_max
                    +","+long_max+" "+lat_min
                    +","+long_min+" "+lat_min
                    +","+long_min+" "+lat_max
                    +","+long_max+" "+lat_max +"))")
            # set url
            base_url = "https://fits.geonet.org.nz/"
            endpoint = "site"
            url = base_url + endpoint
            #
            parameters ={'within':poly}
            # get site data
            sites = requests.get(url, params=parameters)
            # read the json file 
            data = sites.json()['features'] 
            # Initialize this data frame
            dat = pd.DataFrame() #empty dataframe
            # Add the site data to this data frame
            for i, val in enumerate(data):
                geometry = val['geometry']
                lon = geometry['coordinates'][0]
                lat = geometry['coordinates'][1]
                properties = val['properties']
                siteID = properties['siteID']
                height = properties['height']
                name = properties['name']
                #append these to df
                dat = dat.append({'siteID': siteID, 'lon': lon, 'lat': lat, 'height': height, 'name': name}, ignore_index=True)
            # save data
            if ruapehu:
                dat.to_csv('..'+os.sep+'data'+os.sep+"ruapehu_sites_temp.csv") 
            if whakaari:
                dat.to_csv('..'+os.sep+'data'+os.sep+"whakaari_sites_temp.csv") 
            #
            def get_volcano_data(site,typeID):
                """
                This function takes a site ID and type ID as a strings
                and returns a dataframe with all the observation of that type for that site
                """
                #Setup
                base_url = "https://fits.geonet.org.nz/"
                endpoint = "observation"
                url = base_url + endpoint
                
                #Set query parameters
                parameters ={"typeID": typeID, "siteID": site}
                
                #Get data
                request = requests.get(url, params=parameters)
                
                #Unpack data
                data = (request.content)
                
                #If there are data points
                if len(data) > 50:
                    #run volcano_dataframe on it
                    df = volcano_dataframe(data.decode("utf-8"))
                    #print some info on it 
                    print(site,"has", typeID, "data and has", len(df.index), 
                        "data points from ",df['date-time'][1]," to ", df['date-time'][len(df.index)])
                    #retrun it
                    return df
            #
            def volcano_dataframe(data):
                """
                This function turns the string of volcano data received by requests.get
                into a data frame with volcano data correctly formatted.
                """
                # splits data on the new line symbol
                data = data.split("\n") 
                
                # For each data point
                for i in range(0, len(data)):
                    data[i]= data[i].split(",")# splits data ponits on the , symbol
                
                # For each data point 
                for i in range(1, (len(data)-1)):
                    data[i][0] = datetime.datetime.strptime(data[i][0], '%Y-%m-%dT%H:%M:%S.%fZ') #make 1st value into a datetime object
                    data[i][1] = float(data[i][1]) #makes 2nd value into a decimal number
                    data[i][2] = float(data[i][2]) #makes 3rd value into a decimal number
                    
                #make the list into a data frame
                df = pd.DataFrame(data[1:-1],index = range(1, (len(data)-1)), columns=data[0]) #make the list into a data frame
                
                #Return this data frame
                return df
            #
            def get_method(typeID):
                """
                This function takes a type ID as a strings
                and returns all methods used for this type
                """
                
                #Setup
                base_url = "https://fits.geonet.org.nz/"
                endpoint = "method"
                url = base_url + endpoint
                
                #Set query parameters
                parameters ={"typeID": typeID}
                
                #Get data
                request = requests.get(url, params=parameters)
                
                #Unpack data
                data = request.json()['method']
                
                #run make_method_df on data
                df =  make_method_df(data)
                
                return df
            #
            def make_method_df(data):
                """
                This function takes method data as a list
                and returns a dataframe with all the method data.
                https://fits.geonet.org.nz/api-docs/
                
                """
                #Initialize this data frame
                df = pd.DataFrame()
                
                #add data to the data frame
                for i, val in enumerate(data):
                    methodID = val['methodID']
                    name = val['name']
                    description = val['description']
                    reference = val['reference']
                    #append these to df
                    df = df.append({'name': name, 'methodID': methodID, 'description': description, 'reference':reference}, 
                                ignore_index=True)
                
                #Return this data frame
                return df  
            #
            # Set the type to the type ID for temperature
            #typeID = "t" # temperature 
            #typeID = "z" # lake level
            # #typeID = "nve" # number of volcanic-tectonic earthquakes recorded per day (empty)
            # typeID = "ph" # degree of acidity or alkalinity of a sample
            # typeID = "tl" # angle of tilt relative to the horizontal
            # typeID =  "u" # displacement from initial position
            # #typeID =  "u_rf" # displacement from initial position (empty)
            # typeID = "Cl-w" # chloride in water sample
            #typeID = "SO4-w" # Sulphate in water sample
            #typeID = 'rain'
            #typeID = 'SO2-flux-a'
            #typeID = 'CO2-flux-a'
            typeID = 'ap' # "air pressure","unit":"hPa"
            # Get the methods for this type ID
            methods = get_method(typeID)
            # Get all temperature data from all these sites
            #Initialize a list to put the data in later
            t={}
            #loop over each site ID
            for i, site in enumerate(dat["siteID"]):
                #use the get_volcano_data funtion to get the data and save it with the key of the site's ID
                t.update({site:get_volcano_data(site,typeID)})
            # Save as CSV file
            if ruapehu:
                if typeID == "z":
                    siteID = 'RU001A' #RU001A is the data logger level (after 2009), RU001 is a manual level 
                if typeID == "t":
                    siteID = 'RU001'
                if typeID ==  "u" or typeID ==  "u_rf": 
                    siteID = 'VGOB'
                if typeID ==  "u" or typeID ==  "u_rf": 
                    siteID = 'VGOB'
                if typeID ==  "SO4-w": 
                    siteID = 'RU001'
                if typeID ==  "rain": 
                    siteID = 'RU010'
                if typeID ==  "SO2-flux-a": 
                    siteID = 'RU000'
                if typeID ==  "CO2-flux-a": 
                    siteID = 'RU000'
                if typeID ==  "ap": 
                    siteID = 'VGWH'
            #
            if whakaari:
                siteID = 'WI201'
            if typeID == "t":
                t[siteID].to_csv('..'+os.sep+'data'+os.sep+siteID+"_temp_data.csv") 
            if typeID == "z":
                t[siteID].to_csv('..'+os.sep+'data'+os.sep+siteID+"_level_data.csv") 
            if typeID == "nvte":
                t[siteID].to_csv('..'+os.sep+'data'+os.sep+siteID+"_nvte_data.csv") 
            if typeID == "ph":
                t[siteID].to_csv('..'+os.sep+'data'+os.sep+siteID+"_ph_data.csv") 
            if typeID == "tl":
                t[siteID].to_csv('..'+os.sep+'data'+os.sep+siteID+"_tilt_data.csv") 
            if typeID == "u":
                t[siteID].to_csv('..'+os.sep+'data'+os.sep+siteID+"_u_disp_abs_data.csv") 
            if typeID == "u_rf":
                t[siteID].to_csv('..'+os.sep+'data'+os.sep+siteID+"_u_disp_reg_filt_data.csv")
            if typeID == "Cl-w":
                t[siteID].to_csv('..'+os.sep+'data'+os.sep+siteID+"_cl_data.csv")
            if typeID == "SO4-w":
                t[siteID].to_csv('..'+os.sep+'data'+os.sep+siteID+"_so4_data.csv")
            if typeID == "rain":
                t[siteID].to_csv('..'+os.sep+'data'+os.sep+siteID+"_rain_data.csv")
            if typeID == "SO2-flux-a":
                t[siteID].to_csv('..'+os.sep+'data'+os.sep+siteID+"_SO2-flux-a_data.csv")
            if typeID == "CO2-flux-a":
                t[siteID].to_csv('..'+os.sep+'data'+os.sep+siteID+"_CO2-flux-a_data.csv")
            if typeID == "ap":
                t[siteID].to_csv('..'+os.sep+'data'+os.sep+siteID+"_air_pressure_data.csv")
            ##

            # if typeID == "t":  # plot temperature data
            #     start = datetime.datetime(2021, 8, 9)
            #     end = datetime.datetime(2021, 9, 12)

            #     # Trim the data
            #     df = t[siteID].loc[t[siteID]['date-time']<end]
            #     df = df.loc[t[siteID]['date-time']>start]
            #     #
            #     plot2 = df.plot(x='date-time', y= ' t (C)', 
            #         title = 'Temperature')


            # if typeID == "z":  # plot level data
            #     start = datetime.datetime(2009, 4, 15)
            #     end = datetime.datetime(2021, 12, 31)
            #     #
            #     #start = datetime.datetime(2006, 9, 1)
            #     #end = datetime.datetime(2006, 10, 30)
            #     # Trim the data
            #     df = t[siteID].loc[t[siteID]['date-time']<end]
            #     df = df.loc[t[siteID]['date-time']>start]
            #     #
            #     plot2 = df.plot(x='date-time', y= ' z (m)', 
            #         title = 'lake level')

            # plt.grid()
            # plt.show()

        # plot temp data against features for Ruapehu
    
def plot_other_data_geonet():
    from obspy import UTCDateTime
    # convert to UTC 0
    utc_0 = True
    #
    t0 = "2021-08-09"#"2021-08-09"
    t1 = "2021-09-09"#"2021-09-09"
    sta = 'FWVZ'
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
    level = True
    rainfall = True
    ph = False
    u = False
    cl = False
    so4 = False
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
            ax.set_ylim([32,40])
            #ax2b.set_ylabel('temperature C')   
            #ax.legend(loc = 2)         
    # plot lake level data

    if level:
        axb = ax.twinx()
        if sta == 'FWVZ':
            # import temp data
            path = '..'+os.sep+'data'+os.sep+"RU001_level_data.csv"
            path = '..'+os.sep+'data'+os.sep+"RU001A_level_data.csv"
            pd_temp = pd.read_csv(path, index_col=1)
            if utc_0:
                pd_temp.index = [datetimeify(pd_temp.index[i])-6*hour for i in range(len(pd_temp.index))]
            else:
                pd_temp.index = [datetimeify(pd_temp.index[i]) for i in range(len(pd_temp.index))]
            # plot data in axis twin axis
            # Trim the data
            temp_e1_tim = pd_temp[ti_e1: tf_e1].index.values
            #temp_e1_tim=to_nztimezone(temp_e1_tim)
            temp_e1_val = pd_temp[ti_e1: tf_e1].loc[:,' z (m)'].values
            # ax2
            #ax2b = ax2.twinx()
            if mov_avg: # plot moving average
                n=30
                v_plot = temp_e1_val
                axb.plot(temp_e1_tim, v_plot, '-', color='b')
                ax.plot([], [], '-', color='b', label='lake level')
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
                axb.plot(temp_e1_tim, v_plot, '-', color='b', label='lake level')
            #
            axb.set_ylabel('Lake level cm') 

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
    if True:
        te = datetimeify("2021 09 07 22 10 00")#fm_e1.data.tes[int(erup[-1:])-1]
        #te = datetimeify("2009 07 13 06 30 00")#fm_e1.data.tes[int(erup[-1:])-1]
        ax.axvline(te, color='k',linestyle='--', linewidth=2, zorder = 4)
        ax.plot([], color='k', linestyle='--', linewidth=2, label = 'rsam peak')
    #
    ax.legend(loc = 2, framealpha = 1.0)
    if False:
        axb.set_xlim([datetimeify("2021 08 09 00 00 00"),datetimeify("2021 09 10 00 00 00")])
    # 
    ax.grid()
    #ax.set_ylabel('feature value')
    #ax.set_xticks([datetimeify(t_aux) - 7*day*i for i in range(int(30/7)+1)])#[dat.index.values[0],dat.index.values[-1]])#, ]np.arange(0, len(x)+1, 5))
    #
    plt.show()

def plot_interpretation():
    # define station and time
    sta = 'FWVZ'#'FWVZ'#'WIZ'
    erup = 1
    #erup_time = datetimeify(erup_times[sta+'_'+str(erup)])
    erup_time = datetimeify('2022 05 11 00 00 00')
    t0 = erup_time - 60*day#30*day
    t1 = erup_time + 0*day#hour

    if False: # other dates
        #t1 = datetimeify("2010-05-25")
        #t0 = t1 - 25*day
        t1 = datetimeify("2010-05-29") + 1*day
        t0 = t1 - 84*day
        plot_periods =  False
        ffm = False
        server = False
    #
    if sta  == 'WIZ': #True: # WIZ5
        if erup == 5:
            plot_periods = ['2019 11 11 00 00 00','2019 11 23 00 00 00', '2019 12 02 00 00 00', '2019 12 06 12 00 00']
            plot_periods_label = ['magma-geothermal system interaction', 'Pulsating gas flux', 
                                    'Sealing consolidation', 'Pressurization and eruption']
            plot_periods_col = ['gray', 'gray', 'gray', 'gray']
            #
            ffm = True
            server = True
    #
    if False: # WIZ1
        t0 = "2012-07-14"#'2009-06-14'#'2009-06-15'#'2019-11-09'#"2021-03-01"#"2005-11-10"#"2021-08-10"#'2021-09-18'
        t1 = "2012-08-05"#'2009-07-13'#'2009-07-13'#'2019-12-09'#"2021-03-31"#"2005-12-16"#"2021-09-09"#'2021-10-17'
        #
        sta = 'WIZ'
        #
        erup_time = '2012 08 04 16 52 00'
        #
        plot_periods = ['2012 07 20 00 00 00', '2012 07 27 16 00 00', '2012 08 01 00 00 00', '2012 08 02 00 00 00']
        plot_periods_label = ['magma-geothermal system interaction', 'Pulsating gas flux', 
                                'Sealing consolidation', 'Pressurization and eruption']
        plot_periods_col = ['gray', 'gray', 'gray', 'gray']
        #
        ffm = True
        server = True
    #
    if False: # WIZ4
        t0 = "2016-03-29"#'2009-06-14'#'2009-06-15'#'2019-11-09'#"2021-03-01"#"2005-11-10"#"2021-08-10"#'2021-09-18'
        t1 = "2016-04-27"#'2009-07-13'#'2009-07-13'#'2019-12-09'#"2021-03-31"#"2005-12-16"#"2021-09-09"#'2021-10-17'
        #
        sta = 'WIZ'
        #
        erup_time = '2016 04 27 09 37 00'
        #
        plot_periods = False#['2019 11 12 00 00 00','2019 11 23 00 00 00', '2019 12 02 00 00 00', '2019 12 06 12 00 00']
        plot_periods_label = ['magma-geothermal system interaction', 'Pulsating gas flux', 
                                'Sealing consolidation', 'Pressurization and eruption']
        plot_periods_col = ['gray', 'gray', 'gray', 'gray']
        #
        ffm = True
        server = False
    #
    if sta == 'VNSS':
        t0 = "2013-05-14"#'2009-06-14'#'2009-06-15'#'2019-11-09'#"2021-03-01"#"2005-11-10"#"2021-08-10"#'2021-09-18'
        t1 = "2013-06-13"#'2009-07-13'#'2009-07-13'#'2019-12-09'#"2021-03-31"#"2005-12-16"#"2021-09-09"#'2021-10-17'
        #
        sta = 'VNSS'
        #
        erup_time = '2013 06 13 00 00 00'
        #
        plot_periods = False#['2019 11 12 00 00 00','2019 11 23 00 00 00', '2019 12 02 00 00 00', '2019 12 06 12 00 00']
        plot_periods_label = ['magma-geothermal system interaction', 'Pulsating gas flux', 
                                'Sealing consolidation', 'Pressurization and eruption']
        plot_periods_col = ['gray', 'gray', 'gray', 'gray']
        ffm = False
        #
        if erup == 1:
            plot_periods = ['2019 11 11 00 00 00','2019 11 23 00 00 00', '2019 12 02 00 00 00', '2019 12 06 12 00 00']
            plot_periods_label = ['magma-geothermal system interaction', 'Pulsating gas flux', 
                                    'Sealing consolidation', 'Pressurization and eruption']
            plot_periods_col = ['gray', 'gray', 'gray', 'gray']
            #
            ffm = True
            server = False
    #
    if False: # BELO3
        t0 = "2007-10-22"#'2009-06-14'#'2009-06-15'#'2019-11-09'#"2021-03-01"#"2005-11-10"#"2021-08-10"#'2021-09-18'
        t1 = "2007-11-05"#'2009-07-13'#'2009-07-13'#'2019-12-09'#"2021-03-31"#"2005-12-16"#"2021-09-09"#'2021-10-17'
        #
        sta = 'BELO'
        #
        erup_time = '2007 11 05 08 43 00'
        #
        plot_periods = False#['2019 11 12 00 00 00','2019 11 23 00 00 00', '2019 12 02 00 00 00', '2019 12 06 12 00 00']
        plot_periods_label = ['magma-geothermal system interaction', 'Pulsating gas flux', 
                                'Sealing consolidation', 'Pressurization and eruption']
        plot_periods_col = ['gray', 'gray', 'gray', 'gray']
        ffm = False
        server = True
    #
    if sta  == 'KRVZ':
        if erup == 1:
            plot_periods = False#['2019 11 11 00 00 00','2019 11 22 00 00 00', '2019 12 03 00 00 00', '2019 12 06 12 00 00']
            plot_periods_label = ['magma-geothermal system interaction', 'Pulsating gas flux', 
                                    'Sealing consolidation', 'Pressurization and eruption']
            plot_periods_col = ['gray', 'gray', 'gray', 'gray']
            #
            ffm = False
            server = True
        if erup == 2:
            plot_periods = False#['2019 11 11 00 00 00','2019 11 22 00 00 00', '2019 12 03 00 00 00', '2019 12 06 12 00 00']
            plot_periods_label = ['magma-geothermal system interaction', 'Pulsating gas flux', 
                                    'Sealing consolidation', 'Pressurization and eruption']
            plot_periods_col = ['gray', 'gray', 'gray', 'gray']
            #
            ffm = False
            server = True
    #
    if sta  == 'FWVZ':
        if erup == 1:
            plot_periods = [ '2006 09 27 00 00 00', '2006 10 02 12 00 00']
            plot_periods_label = ['Sealing consolidation', 'Pressurization and eruption']
            plot_periods_col = ['gray', 'gray', 'gray', 'gray']
            #
            ffm = True
            server = True
        if erup == 2:
            plot_periods = False#['2019 11 11 00 00 00','2019 11 22 00 00 00', '2019 12 03 00 00 00', '2019 12 06 12 00 00']
            plot_periods_label = ['magma-geothermal system interaction', 'Pulsating gas flux', 
                                    'Sealing consolidation', 'Pressurization and eruption']
            plot_periods_col = ['gray', 'gray', 'gray', 'gray']
            #
            ffm = False
            server = True
        if erup == 3:
            plot_periods = False#['2019 11 11 00 00 00','2019 11 22 00 00 00', '2019 12 03 00 00 00', '2019 12 06 12 00 00']
            plot_periods_label = ['magma-geothermal system interaction', 'Pulsating gas flux', 
                                    'Sealing consolidation', 'Pressurization and eruption']
            plot_periods_col = ['gray', 'gray', 'gray', 'gray']
            #
            ffm = False
            server = True
    #
    if sta  == 'TBTN' or sta == 'T01':
        if erup == 1:
            plot_periods = False#['2019 11 11 00 00 00','2019 11 22 00 00 00', '2019 12 03 00 00 00', '2019 12 06 12 00 00']
            plot_periods_label = ['magma-geothermal system interaction', 'Pulsating gas flux', 
                                    'Sealing consolidation', 'Pressurization and eruption']
            plot_periods_col = ['gray', 'gray', 'gray', 'gray']
            #
            ffm = False
            server = False
        if erup == 2:
            plot_periods = False#['2019 11 11 00 00 00','2019 11 22 00 00 00', '2019 12 03 00 00 00', '2019 12 06 12 00 00']
            plot_periods_label = ['magma-geothermal system interaction', 'Pulsating gas flux', 
                                    'Sealing consolidation', 'Pressurization and eruption']
            plot_periods_col = ['gray', 'gray', 'gray', 'gray']
            #
            ffm = False
            server = False
    #
    if sta  == 'MEA01':
        #if erup == 1:
        plot_periods = False#['2019 11 11 00 00 00','2019 11 22 00 00 00', '2019 12 03 00 00 00', '2019 12 06 12 00 00']
        plot_periods_label = ['magma-geothermal system interaction', 'Pulsating gas flux', 
                                'Sealing consolidation', 'Pressurization and eruption']
        plot_periods_col = ['gray', 'gray', 'gray', 'gray']
    #
    if sta  == 'GOD':
        #if erup == 1:
        plot_periods = False#['2019 11 11 00 00 00','2019 11 22 00 00 00', '2019 12 03 00 00 00', '2019 12 06 12 00 00']
        plot_periods_label = ['magma-geothermal system interaction', 'Pulsating gas flux', 
                                'Sealing consolidation', 'Pressurization and eruption']
        plot_periods_col = ['gray', 'gray', 'gray', 'gray']
        #
        ffm = False
        server = False
    #
    if sta  == 'ONTA' or sta == 'VONK':
        #if erup == 1:
        plot_periods = False#['2019 11 11 00 00 00','2019 11 22 00 00 00', '2019 12 03 00 00 00', '2019 12 06 12 00 00']
        plot_periods_label = ['magma-geothermal system interaction', 'Pulsating gas flux', 
                                'Sealing consolidation', 'Pressurization and eruption']
        plot_periods_col = ['gray', 'gray', 'gray', 'gray']
        #
        ffm = False
        server = False
    #
    if sta  == 'NPT':
        #if erup == 1:
        plot_periods = False#['2019 11 11 00 00 00','2019 11 22 00 00 00', '2019 12 03 00 00 00', '2019 12 06 12 00 00']
        plot_periods_label = ['magma-geothermal system interaction', 'Pulsating gas flux', 
                                'Sealing consolidation', 'Pressurization and eruption']
        plot_periods_col = ['gray', 'gray', 'gray', 'gray']
        #
        ffm = False
        server = False
    #
    if sta  == 'REF':
        #if erup == 1:
        plot_periods = False#['2019 11 11 00 00 00','2019 11 22 00 00 00', '2019 12 03 00 00 00', '2019 12 06 12 00 00']
        plot_periods_label = ['magma-geothermal system interaction', 'Pulsating gas flux', 
                                'Sealing consolidation', 'Pressurization and eruption']
        plot_periods_col = ['gray', 'gray', 'gray', 'gray']
        #
        ffm = False
        server = False
    #
    if sta  == 'POS' or sta  == 'DAM' or 'COP':
        #if erup == 1:
        plot_periods = False#['2019 11 11 00 00 00','2019 11 22 00 00 00', '2019 12 03 00 00 00', '2019 12 06 12 00 00']
        plot_periods_label = ['magma-geothermal system interaction', 'Pulsating gas flux', 
                                'Sealing consolidation', 'Pressurization and eruption']
        plot_periods_col = ['gray', 'gray', 'gray', 'gray']
        #
        ffm = False
        server = False
    #
    def _plt_intp(sta, t0, t1):
        # figure
        nrow = 3
        ncol = 1
        fig, (ax1, ax2, ax3) = plt.subplots(nrows=nrow, ncols=ncol,figsize=(12,8))#(14,4))
        #####################################################
        # subplot one: normalize features
        if True:
            # features
            fts = ['zsc2_hfF__fft_coefficient__coeff_38__attr_"real"',
                    'zsc2_dsarF__change_quantiles__f_agg_"var"__isabs_False__qh_0.6__ql_0.4',
                    'zsc2_dsarF__median'
                    ]
            fts = [ 'zsc2_dsarF__median',
                    'zsc2_dsarF__change_quantiles__f_agg_"var"__isabs_False__qh_0.6__ql_0.4'       
                    ]
            col = ['b','g','r']
            alpha = [1., 1., .5]
            thick_line = [8., 3., 1.]
            #
            for i, ft in enumerate(fts):
                if 'zsc2_dsarF' in ft:
                    ds = ['zsc2_dsarF'] 
                if 'zsc2_rsamF' in ft:
                    ds = ['zsc2_rsamF']
                if 'zsc2_mfF' in ft:
                    ds = ['zsc2_mfF']
                if 'zsc2_hfF' in ft:
                    ds = ['zsc2_hfF']
                #
                if server:
                    path_feat_serv = 'C:\\Users\\aar135\\codes_local_disk\\volc_forecast_tl\\features_bkp\\features_server\\'
                    fm_e1 = ForecastModel(window=2., overlap=1., station =  sta,
                        look_forward=2., data_streams=ds, 
                        feature_dir=path_feat_serv, 
                        savefile_type='pkl') 
                else:
                    fm_e1 = ForecastModel(window=2., overlap=1., station = sta,
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
                if i == 0:
                    v_plot = ft_e1_v
                    v_plot_max = np.max(v_plot)
                    pass
                else:
                    v_plot = ft_e1_v
                    v_plot = v_plot/np.max(ft_e1_v) * v_plot_max#np.max(ft_e1_v)
                #
                if ft == 'zsc2_dsarF__change_quantiles__f_agg_"var"__isabs_False__qh_0.6__ql_0.4':
                    ft = 'nDSAR rate variance'#'DSAR change quantiles (.6-.4) variance'
                if ft == 'zsc2_dsarF__median':
                    ft = 'nDSAR median'
                if ft == 'zsc2_hfF__fft_coefficient__coeff_38__attr_"real"':
                    ft = '75-minute nHF harmonic' #'HF Fourier coefficient 38'
                #
                if ft == 'zsc2_mfF__median':
                    ft = 'nMF median'
                if ft == 'zsc2_hfF__median':
                    ft = 'nHF median'
                if ft == 'zsc2_rsamF__median':
                    ft = 'nRSAM median'
                #
                ax1.plot(ft_e1_t, v_plot, '-', color=col[i], alpha = alpha[i],label='Feature: '+ ft)
                #
            #
            if ffm: # ffm 
                ax1b = ax1.twinx() 
                #v_plot = data[data_stream].loc[inds]
                inv_rsam = fm_e1.data.get_data(ti=t0, tf=t1)['rsamF']#.loc[ft_e1_t]
                inv_rsam = 1./inv_rsam
                ax1b.plot(ft_e1_t, inv_rsam, '-', color= 'gray', linewidth=0.5, markersize=0.5, alpha = .7)
                ax1.plot([], [], '-', color= 'gray', markersize=1, label='1/RSAM', alpha = 1.0)
                #ax1b.set_ylim([0,1])
                ax1b.set_yticks([])
                #
                if False:
                    _fig, (_ax1,_ax2) = plt.subplots(nrows=2, ncols=1,figsize=(12,8))#(14,4))
                    inv_rsam = fm_e1.data.get_data(ti=t0, tf=t1)['rsamF']#.loc[ft_e1_t]
                    inv_rsam = 1./inv_rsam
                    _ax1.plot(ft_e1_t, inv_rsam, '-', color= 'gray', linewidth=0.5, markersize=0.5, alpha = 1.)
                    _ax1.plot([], [], '-', color= 'gray', markersize=1, label='1/RSAM', alpha = 1.0)
                    _ax1.set_ylim([0,1])
                    _ax1.set_yticks([])
                    _fig.show()
                    asdf

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
            ax1.set_ylabel('normalized data')
            
            #ax1b.set_yticks([])
            ax1.grid()
            #ax.set_ylabel('feature value')        #ax.set_xticks([ft_e1[0].index[-1]-7*day*i +day for i in range(5)])
            #ax.set_xticks([ft_e1[0].index[-1] - 7*day*i for i in range(int(30/7)+1)])#[dat.index.values[0],dat.index.values[-1]])#, ]np.arange(0, len(x)+1, 5))
        #####################################################
        # subplot two: features median data
        # features
        if True:
            fts2 = ['zsc2_mfF__median', 'zsc2_hfF__median']#, 'zsc2_rsamF__median']
            col = ['g','r','gray']
            alpha = [1., 1., 1.]
            thick_line = [3., 3., 3.]
            for i, ft in enumerate(fts2):
                if 'zsc2_dsarF' in ft:
                    ds = ['zsc2_dsarF']
                if 'zsc2_rsamF' in ft:
                    ds = ['zsc2_rsamF']
                if 'zsc2_mfF' in ft:
                    ds = ['zsc2_mfF']
                if 'zsc2_hfF' in ft:
                    ds = ['zsc2_hfF']
                if server:
                    path_feat_serv = 'C:\\Users\\aar135\\codes_local_disk\\volc_forecast_tl\\features_bkp\\features_server\\'
                    fm_e2 = ForecastModel(window=2., overlap=1., station =  sta,
                        look_forward=2., data_streams=ds, 
                        feature_dir=path_feat_serv, 
                        savefile_type='pkl') 
                else:
                    fm_e2 = ForecastModel(window=2., overlap=1., station = sta,
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
                if ft == 'zsc2_mfF__median':
                    ft = 'nMF median'
                if ft == 'zsc2_hfF__median':
                    ft = 'nHF median'
                if ft == 'zsc2_rsamF__median':
                    ft = 'nRSAM median'
                #
                ax2.plot(fm_e2_t, v_plot, '-', color=col[i], alpha = alpha[i],label='Feature: '+ ft)
            #
            if erup_time: # plot vertical lines
                te = datetimeify(erup_time)#fm_e1.data.tes[int(erup[-1:])-1]
                ax2.axvline(te, color='k',linestyle='--', linewidth=2, zorder = 4)
                ax2.plot([], color='k', linestyle='--', linewidth=2, label = 'eruption')

            if plot_periods:
                for k, t in enumerate(plot_periods):
                    te = datetimeify(t)#fm_e1.data.tes[int(erup[-1:])-1]
                    ax2.axvline(te, color=plot_periods_col[k], linestyle='-', linewidth=20, alpha = 0.2, zorder = 4)
                    #ax3.plot([], color='k', linestyle='--', linewidth=2, label = 'event')
            #
            ax2.set_ylabel('normalized data')
            ax2.legend(loc = 2)
            te = datetimeify(erup_time)#fm_e1.data.tes[int(erup[-1:])-1]
            #ax2.set_xticks([te - 3*day*i for i in range(int(30/3)+1)])#[dat.index.values[0],dat.index.values[-1]])#, ]np.arange(0, len(x)+1, 5))
            #ax2.set_ylim([1,2])
            #ax2.set_xticks([ft_e1[0].index[-1] - 7*day*i for i in range(int(30/7))])#[dat.index.values[0],dat.index.values[-1]])#, ]np.arange(0, len(x)+1, 5))
            #ax.set_yticks([])
            ax2.grid()      
        #####################################################
        # subplot three: data
        # features
        if True:
            #
            td = TremorData(station = sta)
            #td.update(ti=t0, tf=t1)
            data_streams = ['hf', 'mf', 'rsam']#, 'dsarF']
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
            ax3.set_xlim(_range)
            ax3.legend(loc = 2)
            ax3.grid()
            if log:
                ax3.set_ylabel(' ')
            else:
                ax3.set_ylabel('\u03BC m/s')
            #ax3.set_xlabel('Time [month-day hour]')
            ax3.title.set_text('Station '+td.station+' ('+sta_code[td.station]+'): Tremor data')
            #
            if erup_time: # plot vertical lines
                te = datetimeify(erup_time)#fm_e1.data.tes[int(erup[-1:])-1]
                ax3.axvline(te, color='k',linestyle='--', linewidth=2, zorder = 4)
                ax3.plot([], color='k', linestyle='--', linewidth=2, label = 'event')
            
            if plot_periods:
                for k, t in enumerate(plot_periods):
                    te = datetimeify(t)#fm_e1.data.tes[int(erup[-1:])-1]
                    ax3.axvline(te, color=plot_periods_col[k], linestyle='-', linewidth=20, alpha = 0.2, zorder = 4)
                    #ax3.plot([], color='k', linestyle='--', linewidth=2, label = 'event')
            #
            te = datetimeify(erup_time)#fm_e1.data.tes[int(erup[-1:])-1]
            #ax3.set_xticks([te - 3*day*i for i in range(int(30/3)+1)])#[dat.index.values[0],dat.index.values[-1]])#, ]np.arange(0, len(x)+1, 5))
            #ax3.set_ylim([1e9,1e13])
            ax3.set_yscale('log')
        #
        ax1.set_xlim([t0,t1])
        ax2.set_xlim([t0,t1])
        ax3.set_xlim([t0,t1])
        #
        ax1.grid(color='gray', linestyle='-', linewidth=.3, alpha = 0.5)
        ax2.grid(color='gray', linestyle='-', linewidth=.3, alpha = 0.5)
        ax3.grid(color='gray', linestyle='-', linewidth=.3, alpha = 0.5)
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

def plot_interpretation_media():
    '''
    one axis showing DSAR median and RSAM
    '''
    # define station and time
    sta = 'WIZ'#'FWVZ'#'WIZ'
    erup = 5
    erup_time = datetimeify(erup_times[sta+'_'+str(erup)])
    #erup_time = datetimeify('2021 09 09 00 00 00')
    t0 = erup_time - 30*day#30*day
    t1 = erup_time + 1*day#hour
    #
    if sta  == 'WIZ': #True: # WIZ5
        if erup == 5:
            plot_periods = ['2019 11 11 00 00 00','2019 11 23 00 00 00', '2019 12 02 00 00 00', '2019 12 06 12 00 00']
            plot_periods_label = ['magma-geothermal system interaction', 'Pulsating gas flux', 
                                    'Sealing consolidation', 'Pressurization and eruption']
            plot_periods_col = ['gray', 'gray', 'gray', 'gray']
            #
            ffm = False
            server = True
    #
    def _plt_intp(sta, t0, t1):
        # figure
        nrow = 1
        ncol = 1
        fig, ax1 = plt.subplots(nrows=nrow, ncols=ncol,figsize=(8,3))#(14,4))
        #####################################################
        # subplot one: normalize features
        if True:
            # features
            fts = [ 'zsc2_dsarF__median']
            col = ['b','g','r']
            alpha = [1., 1., .5]
            thick_line = [8., 3., 1.]
            #
            for i, ft in enumerate(fts):
                if 'zsc2_dsarF' in ft:
                    ds = ['zsc2_dsarF'] 
                if 'zsc2_rsamF' in ft:
                    ds = ['zsc2_rsamF']
                if 'zsc2_mfF' in ft:
                    ds = ['zsc2_mfF']
                if 'zsc2_hfF' in ft:
                    ds = ['zsc2_hfF']
                #
                if server:
                    path_feat_serv = 'C:\\Users\\aar135\\codes_local_disk\\volc_forecast_tl\\features_bkp\\features_server\\'
                    fm_e1 = ForecastModel(window=2., overlap=1., station =  sta,
                        look_forward=2., data_streams=ds, 
                        feature_dir=path_feat_serv, 
                        savefile_type='pkl') 
                else:
                    fm_e1 = ForecastModel(window=2., overlap=1., station = sta,
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
                if i == 0:
                    v_plot = ft_e1_v
                    v_plot_max = np.max(v_plot)
                    pass
                else:
                    v_plot = ft_e1_v
                    v_plot = v_plot/np.max(ft_e1_v) * v_plot_max#np.max(ft_e1_v)
                #
                if ft == 'zsc2_dsarF__change_quantiles__f_agg_"var"__isabs_False__qh_0.6__ql_0.4':
                    ft = 'nDSAR rate variance'#'DSAR change quantiles (.6-.4) variance'
                if ft == 'zsc2_dsarF__median':
                    ft = 'nDSAR median'
                if ft == 'zsc2_hfF__fft_coefficient__coeff_38__attr_"real"':
                    ft = '75-minute nHF harmonic' #'HF Fourier coefficient 38'
                #
                if ft == 'zsc2_mfF__median':
                    ft = 'nMF median'
                if ft == 'zsc2_hfF__median':
                    ft = 'nHF median'
                if ft == 'zsc2_rsamF__median':
                    ft = 'nRSAM median'
                #
                ax1.plot(ft_e1_t, v_plot, '-', color=col[i], alpha = alpha[i],label='Feature: '+ ft)
                #
            #
            if plot_periods:
                for k, t in enumerate(plot_periods):
                    te = datetimeify(t)#fm_e1.data.tes[int(erup[-1:])-1]
                    ax1.axvline(te, color=plot_periods_col[k], linestyle='-', linewidth=20, alpha = 0.2, zorder = 4)
                    #ax3.plot([], color='k', linestyle='--', linewidth=2, label = 'event')


            #
            
            #ax1.set_ylim([0,1])
            #
            te = datetimeify(erup_time)#fm_e1.data.tes[int(erup[-1:])-1]
            #ax1.set_xticks([te - 3*day*i for i in range(int(30/3)+1)])#[dat.index.values[0],dat.index.values[-1]])#, ]np.arange(0, len(x)+1, 5))
            #ax1.set_xticks([ft_e1[0].index[-1] - 7*day*i for i in range(int(30/7)+1)])#[dat.index.values[0],dat.index.values[-1]])#, ]np.arange(0, len(x)+1, 5))
            #ax1.set_yticks([])
            ax1.set_ylabel('DSAR')
            
            #ax1b.set_yticks([])
            ax1.grid()
            #ax.set_ylabel('feature value')        #ax.set_xticks([ft_e1[0].index[-1]-7*day*i +day for i in range(5)])
            #ax.set_xticks([ft_e1[0].index[-1] - 7*day*i for i in range(int(30/7)+1)])#[dat.index.values[0],dat.index.values[-1]])#, ]np.arange(0, len(x)+1, 5))
        #############################################
        # subplot: data
        # features
        if True:
            ax1b = ax1.twinx()
            #
            td = TremorData(station = sta)
            #td.update(ti=t0, tf=t1)
            data_streams = ['rsam']#, 'mf', 'rsam']#, 'dsarF']
            label = ['RSAM']#,'MF','RSAM','DSAR']
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
            cols = ['k','g','gray','m',[0.5,0.5,0.5],[0.75,0.75,0.75]]
            inds = (data.index>=datetimeify(_range[0]))&(data.index<=datetimeify(_range[1]))
            i=0
            for data_stream, col in zip(data_streams,cols):
                v_plot = data[data_stream].loc[inds]
                if log:
                    v_plot = np.log10(v_plot)
                if norm:
                    v_plot = (v_plot-np.min(v_plot))/np.max((v_plot-np.min(v_plot)))
                if label:
                    if col_def:
                        ax1b.plot(data.index[inds], v_plot, '-', color=col_def, linewidth=0.5, alpha = 1.0)
                    else:
                        ax1b.plot(data.index[inds], v_plot, '-', color=col, linewidth=0.5, alpha = 1.0)
                else:
                    ax1b.plot(data.index[inds], v_plot, '-', color=col, label=data_stream, linewidth=0.5, alpha = 1.0)
                i+=1
            for te in td.tes:
                if [te>=datetimeify(_range[0]) and te<=datetimeify(_range[1])]:
                    pass
                    #ax.axvline(te, color='k', linestyle='--', linewidth=2, zorder = 0)
            #
            
            #ax.plot([], color='k', linestyle='--', linewidth=2, label = 'eruption')
            ax1b.set_xlim(_range)
            ax1b.legend(loc = 2)
            ax1b.grid()
            if log:
                ax1b.set_ylabel(' ')
            else:
                ax1b.set_ylabel('RSAM [\u03BC m/s]')
            #ax3.set_xlabel('Time [month-day hour]')
            #ax3.title.set_text('Station '+td.station+' ('+sta_code[td.station]+'): Tremor data')
            ax1b.set_ylim([1e2,1e4])
            ax1b.set_yscale('log')
        #
        ax1.plot([], [], '-', color=col, alpha = 1.0, label = label[0])
        #
        if False:#erup_time: # plot vertical lines
            te = datetimeify(erup_time)#fm_e1.data.tes[int(erup[-1:])-1]
            ax1.axvline(te, color='r',linestyle='--', linewidth=4, zorder = 6)
            ax1b.axvline(te, color='r',linestyle='--', linewidth=4, zorder = 6)
            ax1.plot([], color='r', linestyle='--', linewidth=4, label = 'eruption')
        #
        if False: # 
            te = datetimeify('2019-11-18 12:00:00')#fm_e1.data.tes[int(erup[-1:])-1]
            ax1.axvline(te, color='m',linestyle='--', linewidth=4, zorder = 6)
            ax1b.axvline(te, color='m',linestyle='--', linewidth=4, zorder = 6)
            ax1.plot([], color='m', linestyle='--', linewidth=4, label = 'VAL: 1 > 2')
        #
        ax1.set_xticks([t1 - 5*day*i for i in range(int(30/5)+1)])#[dat.index.values[0],dat.index.values[-1]])#, ]np.arange(0, len(x)+1, 5))
        #ax1.legend(loc = 2)
        ax1.set_xlim([t0,t1])
        #
        ax1.grid(color='gray', linestyle='-', linewidth=.3, alpha = 0.5)
        #####################################################
        #fig.suptitle(sta_code[sta]+': '+str(t0)+' to '+str(t1))#'Feature: '+ ft_nm_aux, ha='center')
        fig.suptitle('Whakaari 2019 eruption timeline')#'Feature: '+ ft_nm_aux, ha='center')
        plt.tight_layout()
        path = '..'+os.sep+'features'+os.sep+'correlations'+os.sep+'comb_feat_analysis'+os.sep
        #plt.savefig(path+erup+'_'+ft_id+'.png')
        plt.show()
        plt.close()
    ##
    _plt_intp(sta,t0, t1)
    #     

def ext_pval_anal():
    '''
    Extended statistical significance analysis 
    '''
    if False: # Whakaarri (one volcano, 5 eruption) 
        # extract (one month time serie feature) from in-eruptions and out-eruptions (% of the total of each).
        # Then, calculate the p-value 
        if True: # example for whakaari
            # Populations:  in-eruption (2 weeks back and 1 week forward) and out-eruption (rest, non-eruptive periods)
            # Sample: N 'one month time serie feature'
            # Ns: percentage (%) of the total 
            # ST: number of times that samples are extracted from the populations 
            # pv_s: P-value per sample 
            
            ## (1) construct populations 
            # for Whakaari 
                    # period to explore 
            sta_exp = 'WIZ' # station to explore
            # feature to explore
            ds = ['zsc2_dsarF'] 
            #feat = 'median'
            feat = 'rate_var' # ch qt var .4-.6
            #
            ind_samp = True
            # 
            print('Stat Anal one station: '+sta_exp)
            print('Feature: '+ds[0]+' '+feat)
            if sta_exp == 'WIZ':
                endtime = datetimeify("2021-06-30")
                years_back = 10 # WIZ 10; VNSS 7 ; FWVZ 14
                look_back = years_back*365 + 3 # days, years back from endtime (day by day)
            if sta_exp == 'FWVZ':
                endtime = datetimeify("2020-12-31")#("2020-12-31")
                years_back = 14 # WIZ 10; VNSS 7 ; FWVZ 14
                look_back = years_back*365 # days, years back from endtime (day by day) 
            # vector of days
            _vec_days = [endtime - day*i for i in range(look_back)]
            #
            if ind_samp:
                win_d_length = 30 #
                _vec_days = [endtime - day*win_d_length*i for i in range(int(look_back/win_d_length))]
            #
            ## Populations: pop_in_erup and pop_out_erup

            if False:
                fm = ForecastModel(window=2., overlap=1., station = sta_exp,
                    look_forward=2., data_streams=ds, savefile_type='csv')
            else:
                path_feat_serv = 'C:\\Users\\aar135\\codes_local_disk\\volc_forecast_tl\\features_bkp\\features_server\\'
                fm = ForecastModel(window=2., overlap=1., station = sta_exp,
                    look_forward=2., data_streams=ds, 
                    feature_dir=path_feat_serv, 
                    savefile_type='pkl')
            # erup times
            #tes = fm.data.tes[:]
            pop_in = []
            for e in fm.data.tes[:]:
                #e = datetime.date(e.year, e.month, e.day)
                _e = e.replace(hour=00, minute=00)
                n_days_before = 7
                _vdays = [_e - n_days_before*day + i*day for i in range(n_days_before)] # 1 weeks before  
                pop_in =  pop_in + _vdays
            #
            pop_out = []
            # 
            pop_rej = []
            for e in fm.data.tes[:]:
                #e = datetime.date(e.year, e.month, e.day)
                _e = e.replace(hour=00, minute=00)
                n_days_before = 60
                _vdays = [_e - n_days_before*day + i*day for i in range(n_days_before+30)] # 2 weeks before and 1 week after   
                pop_rej =  pop_rej + _vdays

            # construct out of eruption population
            for d in _vec_days: 
                if d not in pop_in:
                    if True: #d.year != 2013:
                        if d not in pop_rej:
                            pop_out.append(d)

            # loop over ST: number of times that samples are extracted from the populations 
            pv_samp_in = []
            pv_samp_out = []
            cc_samp_in  = []
            cc_samp_out  = []
            ST = 10#1#10
            Ns = 10#10
            print('Samples: '+str(Ns))
            for _st in range(ST):
                print('Iteration: '+str(_st+1)+'/'+str(ST))
                ## (2) construct sample (N random from populations)
                #p = 10 # percentague of the total population 
                #Ns = 5
                import random
                samp_in = random.sample(pop_in,Ns)
                if False: # use eruption times
                    samp_in = [te for te in fm.data.tes[::-1]]
                samp_out = random.sample(pop_out,Ns)

                ## (3) convolute sample over the record and calulate p-value for the sample 
                # each data in sample need to be convoluted over the whole record
                _pv_samp_in = [] 
                _cc_samp_in = [] 
                for l, te in enumerate(samp_in):
                    dt = ds[0]#'zsc2_dsarF'
                    # rolling median and signature length window
                    N, M = [2,30]
                    # time
                    j = fm.data.df.index
                    # construct signature
                    df = fm.data.df[(j>(te-(M+N)*day))&(j<te)]
                    #archtype = df[dt].rolling(N*24*6).median()[N*24*6:]
                    if feat == 'median':
                        archtype = df[dt].rolling(N*24*6).median()[N*24*6:]
                    if feat == 'rate_var':
                        archtype = df[dt].rolling(N*24*6).apply(chqv)[N*24*6:]

                    # convolve over the data
                    df = fm.data.df[:]#[(j<te)]
                    if feat == 'median':
                        test = df[dt].rolling(N*24*6).median()[N*24*6:]
                    if feat == 'rate_var':
                        test = df[dt].rolling(N*24*6).apply(chqv)[N*24*6:]
                    out = test.rolling(archtype.shape[0]).apply(partial(conv, (archtype-archtype.mean())/archtype.std()))
                    #
                    cc_te = []
                    _samp_in = [samp_in[k] for k in range(len(samp_in)) if k != l]

                    for _te in _samp_in:
                        cc_te.append(out[out.index.get_loc(_te, method='nearest')])
                    cc_te = np.array(cc_te)

                    ## (4) calulate p-value for the sample 
                    # 2-sample Kolmogorov Smirnov test for difference in underlying distributions
                    from scipy.stats import kstest
                    a = out.iloc[archtype.shape[0]::24*6].values
                    pv = kstest(cc_te, out.iloc[archtype.shape[0]::24*6].values).pvalue
                    #
                    _pv_samp_in.append(pv)
                    [_cc_samp_in.append(cc_te[i]) for i in range(len(cc_te))]
                #
                pv_samp_in = pv_samp_in + _pv_samp_in
                cc_samp_in = cc_samp_in + _cc_samp_in
                #
                del dt, N, M, df, archtype, test, out, a, pv, cc_te
                
                # out
                _pv_samp_out = [] 
                _cc_samp_out = [] 
                for l, te in enumerate(samp_out):
                    dt = ds[0]#'zsc2_dsarF'

                    # rolling median and signature length window
                    N, M = [2,30]
                    # time
                    j = fm.data.df.index
                    # construct signature
                    df = fm.data.df[(j>(te-(M+N)*day))&(j<te)]
                    archtype = df[dt].rolling(N*24*6).median()[N*24*6:]
                    # convolve over the data
                    df = fm.data.df[:]#[(j<te)]
                    test = df[dt].rolling(N*24*6).median()[N*24*6:]
                    out = test.rolling(archtype.shape[0]).apply(partial(conv, (archtype-archtype.mean())/archtype.std()))
                    #
                    cc_te = []
                    _samp_out = [samp_out[k] for k in range(len(samp_out)) if k != l]
                    for _te in _samp_out:
                        cc_te.append(out[out.index.get_loc(_te, method='nearest')])
                    cc_te = np.array(cc_te)

                    ## (4) calulate p-value for the sample 
                    # 2-sample Kolmogorov Smirnov test for difference in underlying distributions
                    from scipy.stats import kstest
                    a = out.iloc[archtype.shape[0]::24*6].values
                    pv = kstest(cc_te, out.iloc[archtype.shape[0]::24*6].values).pvalue
                    #
                    _pv_samp_out.append(pv)
                    [_cc_samp_out.append(cc_te[i]) for i in range(len(cc_te))]
                    #
                # 
                pv_samp_out = pv_samp_out + _pv_samp_out
                cc_samp_out = cc_samp_out + _cc_samp_out

            # write p-values from samples
            path = '..'+os.sep+'features'+os.sep+'correlations'+os.sep+'stat_sign_per_hig_corr_feat'+os.sep
            _nm = 'pv_in_out_samp_pool_'+sta_exp+'_'+ds[0]+'_'+feat+'_Ns'+str(Ns)+'_'+str(ST)+'ite'
            with open(path+_nm+'.txt', 'w') as f:
                for k in range(len(pv_samp_in)):
                    f.write(str(k+1)+'\t'+str(pv_samp_in[k])+'\t'+str(pv_samp_out[k])+'\n')
            # write cc from samples
            path = '..'+os.sep+'features'+os.sep+'correlations'+os.sep+'stat_sign_per_hig_corr_feat'+os.sep
            _nm = 'cc_in_out_samp_pool_'+sta_exp+'_'+ds[0]+'_'+feat+'_Ns'+str(Ns)+'_'+str(ST)+'ite'
            with open(path+_nm+'.txt', 'w') as f:
                for k in range(len(cc_samp_in)):
                    f.write(str(k+1)+'\t'+str(cc_samp_in[k])+'\t'+str(cc_samp_out[k])+'\n')

        if True: # read out put and plot

            # read file
            #path = '..'+os.sep+'features'+os.sep+'correlations'+os.sep+'stat_sign_per_hig_corr_feat'+os.sep+'pv_samp_in_out.txt'
            if ind_samp:
                path = '..'+os.sep+'features'+os.sep+'correlations'+os.sep+'stat_sign_per_hig_corr_feat'+os.sep+'pv_in_out_samp_pool_WIZ_zsc2_dsarF_rate_var_Ns10_10ite_IndSamp.txt'
            else:
                path = '..'+os.sep+'features'+os.sep+'correlations'+os.sep+'stat_sign_per_hig_corr_feat'+os.sep+'pv_in_out_samp_pool_WIZ_zsc2_dsarF_rate_var_Ns10_10ite.txt'
            #
            fl  = np.genfromtxt(path, delimiter="\t")
            pv_samp_in = [fl[i][1] for i in range(len(fl))]
            pv_samp_out = [fl[i][2] for i in range(len(fl))]
            # correct p-values
            if False: # FDR correction
                import statsmodels
                # in
                pv_samp_in_cor = statsmodels.stats.multitest.fdrcorrection(pv_samp_in, alpha=0.05, method='n', is_sorted=False)
                pv_samp_in = pv_samp_in_cor[1][:]
                # out
                pv_samp_out_cor = statsmodels.stats.multitest.fdrcorrection(pv_samp_out, alpha=0.05, method='n', is_sorted=False)
                pv_samp_out = pv_samp_out_cor[1][:]
            
            count_in = 0
            for pv in pv_samp_in:
                if pv < 0.05:
                    count_in+=1
            print('pv in < 0.05: '+str(100*count_in/len(pv_samp_in))+' %')
            #
            count = 0
            for pv in pv_samp_out:
                if pv < 0.05:
                    count+=1
            print('pv out < 0.05: '+str(100*count/len(pv_samp_out))+' %')

            ## (5) construct p-val histogram of both populations
            fig, ax = plt.subplots()
            a_heights, a_bins = np.histogram(pv_samp_in)
            b_heights, b_bins = np.histogram(pv_samp_out)
            width = (a_bins[1] - a_bins[0])/1
            ax.bar(a_bins[:-1], a_heights, width=width, facecolor='cornflowerblue', label = 'in eruption')
            ax.bar(b_bins[:-1]+width, b_heights, width=width, facecolor='seagreen', alpha = 0.6, label = 'out eruption')
            ax.set_xlabel('p-value')
            ax.set_ylabel('frequency')
            #ax.set_xlim([0,0.1])
            #ax.set_xscale('log')
            ax.axvline(x=0.05, color = 'k', label = '0.05 threshold')
            # place a text box in upper left in axes coords
            textstr = '\n'.join((r'pv in < 0.05:  '+str(100*count_in/len(pv_samp_in))+' %',
                    r'pv out < 0.05: '+str(100*count/len(pv_samp_out))+' %'))
            props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
            ax.text(0.2, 0.95, textstr, transform=ax.transAxes, fontsize=12,
                verticalalignment='top', bbox=props)
            ax.legend()
            plt.show()  
            plt.savefig(path+'pv_samp_in_out.png')    

            # plot histogram of cc
            # read file
            path = '..'+os.sep+'features'+os.sep+'correlations'+os.sep+'stat_sign_per_hig_corr_feat'+os.sep+'cc_samp_in_out.txt'
            fl  = np.genfromtxt(path, delimiter="\t")
            cc_samp_in = [fl[i][1] for i in range(len(fl))]
            cc_samp_out = [fl[i][2] for i in range(len(fl))]

            fig, ax = plt.subplots()
            a_heights, a_bins = np.histogram(cc_samp_in)
            b_heights, b_bins = np.histogram(cc_samp_out)
            width = (a_bins[1] - a_bins[0])/2
            ax.bar(a_bins[:-1], a_heights, width=width, facecolor='cornflowerblue', label = 'in eruption')
            ax.bar(b_bins[:-1]+width, b_heights, width=width, facecolor='seagreen', alpha = 0.6, label = 'out eruption')
            ax.set_xlabel('cc-value')
            ax.set_ylabel('frequency')
            #ax.set_xlim([0,0.1])
            #ax.set_xscale('log')
            #ax.axvline(x=0.05, color = 'k', label = '0.05 threshold')
            # place a text box in upper left in axes coords
            #textstr = '\n'.join((r'pv in < 0.05:  '+str(100*count_in/len(pv_samp_in))+' %',
            #        r'pv out < 0.05: '+str(100*count/len(pv_samp_out))+' %'))
            #props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
            #ax.text(0.2, 0.95, textstr, transform=ax.transAxes, fontsize=12,
            #    verticalalignment='top', bbox=props)
            ax.legend()
            plt.show()  
            plt.savefig(path+'cc_samp_in_out.png')  
            asdf

    if True: # six volcano, 18 eruption
        # Populations:  in-eruption (1 weeks back and 0 week forward) and out-eruption (rest, non-eruptive periods)
        # Sample: N (one month time serie feature) 
        # Ns: Numeber of samples (default 10) 
        # ST: iterarions 
        # pv: P-value per sample
        stas = ['WIZ','FWVZ','KRVZ']#['WIZ','FWVZ','KRVZ','PVV','VNSS','BELO']
        ds = ['zsc2_dsarF'] 
        #feat = 'median'
        feat = 'rate_var' # ch qt var .4-.6
        print('Stations: ')
        print(stas)
        print('Feature: '+ds[0]+' '+feat)
        ## sampling from pool of eruptions per itearation (instead of using all eruptions available)
        import random
        smp_pool_erup = True
        if smp_pool_erup: # 
            print('Sampling from pool of eruptions per iteration')
            n_erup_samp = 8 
            print('N of samples (per iterarion): '+str(n_erup_samp))
            ST = 40 # iterations
            out_sample =  True
            balance = True
            n_balance = 2 # eruptions per volcano
            jitter = True # eruption reference time for archetype randomly sampled
            ind_samp = False # cc independent from each other (pop out one sample every 30 days)
            if balance:
                print('with balance: equal eruptions per volcano ('+str(n_balance)+')')
        ##
        print('Iterations: '+str(ST))
        ##
        if not smp_pool_erup:
            ST = 1 # iterations
            #print('Iterations: '+str(ST))
            out_sample =  True
        #
        import time
        start_time = time.time()
        #
        if False: # run analysis
            # (1) construct populations: in eruption, out eruption  
            # loop over eruptions, and add elements to the population 
            # (2) contruct archetypes, and 'ds' time series for each volcano
            pop_in = []
            pop_out = []
            tests = []
            archetypes = []
            archetypes_out = []
            
            # construct archetypes sets (lists)
            for sta in stas:
                #
                if sta == 'WIZ':
                    endtime = datetimeify("2021-06-30")
                    years_back = 10 # WIZ 10; VNSS 7 ; FWVZ 14
                    look_back = years_back*365 + 3 # days, years back from endtime (day by day)
                if sta == 'FWVZ':
                    endtime = datetimeify("2020-12-31")#("2020-12-31")
                    years_back = 14 # WIZ 10; VNSS 7 ; FWVZ 14
                    look_back = years_back*365 # days, years back from endtime (day by day) 
                if sta == 'KRVZ':
                    endtime = datetimeify("2020-12-31")#("2020-12-31")
                    years_back = 15 # WIZ 10; VNSS 7 ; FWVZ 14
                    look_back = years_back*365 # days, years back from endtime (day by day) 
                if sta == 'VNSS':
                    endtime = datetimeify("2019-12-31")# 
                    years_back = 3 # WIZ 10; VNSS 7 ; FWVZ 14
                    look_back = years_back*365 - 181 # days, years back from endtime (day by day) 
                if sta == 'PVV':
                    endtime = datetimeify("2016-06-30")# 
                    years_back = 2 # WIZ 10; VNSS 7 ; FWVZ 14
                    look_back = years_back*365 + 120 # days, years back from endtime (day by day) 
                if sta == 'BELO':
                    endtime = datetimeify("2008-05-07")# 
                    years_back = 0.4 # WIZ 10; VNSS 7 ; FWVZ 14
                    look_back = int(years_back*365) # days, years back from endtime (day by day) 
                            
                # vector of days
                _vec_days = [endtime - day*i for i in range(look_back)]
                if False:
                    fm = ForecastModel(window=2., overlap=1., station = sta,
                    look_forward=2., data_streams=ds, savefile_type='csv')
                else:
                    path_feat_serv = 'C:\\Users\\aar135\\codes_local_disk\\volc_forecast_tl\\features_bkp\\features_server\\'
                    fm = ForecastModel(window=2., overlap=1., station = sta,
                        look_forward=2., data_streams=ds, 
                        feature_dir=path_feat_serv, 
                        savefile_type='pkl')
                
                # erup times
                #tes = fm.data.tes[:]
                _pop_in = []
                if False:
                    for e in fm.data.tes[:]:
                        e = datetime.date(e.year, e.month, e.day)
                    _e = e.replace(hour=00, minute=00)
                    n_days_before = 7
                    _vdays = [_e - n_days_before*day + i*day for i in range(n_days_before+2)] # 1 weeks before  
                    _pop_in =  _pop_in + _vdays
                if True: # pop in of just eruptions 
                    for e in fm.data.tes[:]:
                        #e = datetime.date(e.year, e.month, e.day)
                        _e = e.replace(hour=00, minute=00)
                        _pop_in.append(e)
                    if balance:
                        _pop_in = random.sample(_pop_in, n_balance)
                                        
                #
                _pop_out = []
                # 
                pop_rej = []
                for e in fm.data.tes[:]:
                    #e = datetime.date(e.year, e.month, e.day)
                    _e = e.replace(hour=00, minute=00)
                    n_days_before = 60
                    _vdays = [_e - n_days_before*day + i*day for i in range(n_days_before+30)] # 2 weeks before and 1 week after   
                    pop_rej =  pop_rej + _vdays

                # construct out of eruption population
                for d in _vec_days: 
                    if d not in pop_in:
                        if d.year != 2013:
                            if d not in pop_rej:
                                _pop_out.append(d)
                #if balance:
                #    _pop_out = random.sample(_pop_out, n_balance)

                # add to general population 
                pop_in = pop_in + _pop_in
                pop_out = pop_out + _pop_out

                #########################
                ## archetypes
                #
                N, M = [2,30]
                ## in sample
                # time
                count = 0
                j = fm.data.df.index
                for te in fm.data.tes[:]:
                    if jitter:
                        # random integer from 0 to 9
                        te = te - random.randint(0, 5)*day
                    if smp_pool_erup:
                        dt = ds[0]#'zsc2_dsarF'
                        # construct signature
                        df = fm.data.df[(j>(te-(M+N)*day))&(j<te)]
                        #_archtype = df[dt].rolling(N*24*6).median()[N*24*6:]
                        #archetypes.append([count, dt, _archtype, sta, te, stas])
                        #count += 1
                        #archtype = df[dt].rolling(N*24*6).median()[N*24*6:]
                        if feat == 'median':
                            _archtype = df[dt].rolling(N*24*6).median()[N*24*6:]
                        if feat == 'rate_var':
                            _archtype = df[dt].rolling(N*24*6).apply(chqv)[N*24*6:]
                        #
                        archetypes.append([count, dt, feat, _archtype, sta, te, stas, ind_samp])
                        count += 1
                    else:
                        if ST > 1:
                            for ite in range(ST):
                                r = random.randint(0,4)
                                te = te-r*day + 2*day 
                                dt = ds[0]#'zsc2_dsarF'
                                # construct signature
                                df = fm.data.df[(j>(te-(M+N)*day))&(j<te)]
                                _archtype = df[dt].rolling(N*24*6).median()[N*24*6:]
                                archetypes.append(_archtype)
                        else: 
                            dt = ds[0]#'zsc2_dsarF'
                            # construct signature
                            df = fm.data.df[(j>(te-(M+N)*day))&(j<te)]
                            _archtype = df[dt].rolling(N*24*6).median()[N*24*6:]
                            archetypes.append(_archtype)
                #
                if out_sample:
                    ## out sample
                    # time
                    #samp_out = _pop_out#random.sample(_pop_out)#,len(archetypes))
                    samp_out = random.sample(_pop_out, len(fm.data.tes[:]))

                    #if balance:
                    #    samp_out = random.sample(samp_out, n_balance)
                    j = fm.data.df.index
                    for te in samp_out:
                        #te = random.sample(samp_out, 1)[0]
                        if smp_pool_erup:
                            dt = ds[0]#'zsc2_dsarF'
                            # construct signature
                            df = fm.data.df[(j>(te-(M+N)*day))&(j<te)]
                            #_archtype = df[dt].rolling(N*24*6).median()[N*24*6:]
                            #archetypes_out.append([count, dt, _archtype, sta, te, stas])
                            #count += 1
                            if feat == 'median':
                                _archtype = df[dt].rolling(N*24*6).median()[N*24*6:]
                            if feat == 'rate_var':
                                _archtype = df[dt].rolling(N*24*6).apply(chqv)[N*24*6:]
                            #
                            archetypes_out.append([count, dt, feat, _archtype, sta, te, stas, ind_samp])
                            count += 1
                        else:
                            if ST > 1:
                                for ite in range(ST):
                                    r = random.randint(0,5)
                                    te = te-r*day + 2*day 
                                    dt = ds[0]#'zsc2_dsarF'
                                    # construct signature
                                    df = fm.data.df[(j>(te-(M+N)*day))&(j<te)]
                                    _archtype = df[dt].rolling(N*24*6).median()[N*24*6:]
                                    archetypes.append(_archtype)
                            else: 
                                dt = ds[0]#'zsc2_dsarF'
                                # construct signature
                                df = fm.data.df[(j>(te-(M+N)*day))&(j<te)]
                                _archtype = df[dt].rolling(N*24*6).median()[N*24*6:]
                                archetypes_out.append(_archtype)
            
            if smp_pool_erup:
                #samp_in = random.sample(pop_in,n_erup_samp)
                archetypes_in_set = []
                archetypes_out_set = []
                for ite in range(ST):
                    archetypes_in_set.append([ite, random.sample(archetypes, n_erup_samp)])
                    archetypes_out_set.append([ite, random.sample(archetypes_out, n_erup_samp)])
                
                #
                del archetypes, archetypes_out, samp_out, _pop_out, _pop_in, pop_in, pop_out
                
                ## RUN CONVOLUTION
                if True:
                    out = out_sample
                    # delete previous temperal results 
                    if True:
                        path = '..'+os.sep+'features'+os.sep+'correlations'+os.sep+'stat_sign_per_hig_corr_feat'+os.sep+'_temp'+os.sep
                        _fls = glob.glob(path+"*_in.txt")
                        for _fl in _fls:
                            os.remove(_fl)
                        _fls = glob.glob(path+"*_out.txt")
                        for _fl in _fls:
                            os.remove(_fl)
                    # 
                    ## run convolution in parallel 
                    if True: # in sample
                        print('Running archetypes sets (in sample) in parallel')
                        n_jobs = 4 # number of cores
                        if out:
                            print('Estimated time: '+str(round(110*n_erup_samp*ST/3600 *2,2))+' hr')
                        else:
                            print('Estimated time: '+str(round(110*n_erup_samp*ST/3600,2))+' hr')
                        #
                        ## for testing
                        #conv_set_arch(archetypes_in_set[0])
                        #asdf
                        ##
                        p = Pool(n_jobs)
                        p.map(conv_set_arch, archetypes_in_set)
                        # rename output
                        path = '..'+os.sep+'features'+os.sep+'correlations'+os.sep+'stat_sign_per_hig_corr_feat'+os.sep+'_temp'+os.sep
                        _fls = glob.glob(path+"*_pv.txt")
                        for _fl in _fls:
                            os.rename(_fl, _fl[:-4]+'_in.txt')
                        _fls = glob.glob(path+"*_cc.txt")
                        for _fl in _fls:
                            os.rename(_fl, _fl[:-4]+'_in.txt')
                        #
                        print("--- %s seconds ---" % (time.time() - start_time))
                    
                    if True: # out sample
                        print('Running archetypes sets (out sample) in parallel')
                        print('Estimated time: '+str(round(110*n_erup_samp*ST/3600,2))+' hr')
                        n_jobs = 4 # number of cores
                        p = Pool(n_jobs)
                        p.map(conv_set_arch, archetypes_out_set)
                        # rename output
                        path = '..'+os.sep+'features'+os.sep+'correlations'+os.sep+'stat_sign_per_hig_corr_feat'+os.sep+'_temp'+os.sep
                        _fls = glob.glob(path+"*_pv.txt")
                        for _fl in _fls:
                            os.rename(_fl, _fl[:-4]+'_out.txt')
                        _fls = glob.glob(path+"*_cc.txt")
                        for _fl in _fls:
                            os.rename(_fl, _fl[:-4]+'_out.txt')
                        #
                        print("--- %s seconds ---" % (time.time() - start_time))
                #
                if True: # read results and write a new combined output
                    # read results and write a new combined output
                    out = True
                    ## PV vals
                    path = '..'+os.sep+'features'+os.sep+'correlations'+os.sep+'stat_sign_per_hig_corr_feat'+os.sep+'_temp'+os.sep
                    _fls = glob.glob(path+"*pv_in.txt")
                    pv_samp_join_in = []
                    for _fl in _fls:
                        fl  = np.genfromtxt(_fl, delimiter="\t")
                        [pv_samp_join_in.append(fl[i][1]) for i in range(len(fl))]

                    if out:
                        _fls = glob.glob(path+"*pv_out.txt")
                        pv_samp_join_out = []
                        for _fl in _fls:
                            fl  = np.genfromtxt(_fl, delimiter="\t")
                            [pv_samp_join_out.append(fl[i][1]) for i in range(len(fl))]

                    # rewrite into a new files (both in and out samples)
                    _sta = ''
                    for i in range(len(stas)):
                        _sta = _sta + '_'+stas[i]  
                    if out: 
                        if ind_samp:
                            _nm = 'pv_in_out_samp_pool_'+ds[0]+'_'+feat+'_'+str(n_erup_samp)+'e_'+str(ST)+'ite_'+'indsamp'#'_WIZ_FWZ_KRZ_VNSS_BELO_PVV'#'_WIZ_FWZ_KRZ'            # write p-values from samples                        
                        else:
                            _nm = 'pv_in_out_samp_pool_'+ds[0]+'_'+feat+'_'+str(n_erup_samp)+'e_'+str(ST)+'ite'#'_WIZ_FWZ_KRZ_VNSS_BELO_PVV'#'_WIZ_FWZ_KRZ'            # write p-values from samples
                    else:
                        if ind_samp:
                            _nm = 'pv_in_samp_pool_'+ds[0]+'_'+feat+'_'+str(n_erup_samp)+'e_'+str(ST)+'ite_'+'indsamp'
                        else:
                            _nm = 'pv_in_samp_pool_'+ds[0]+'_'+feat+'_'+str(n_erup_samp)+'e_'+str(ST)+'ite'#
                    path = '..'+os.sep+'features'+os.sep+'correlations'+os.sep+'stat_sign_per_hig_corr_feat'+os.sep
                    with open(path+_nm+_sta+'.txt', 'w') as f:
                        for k in range(len(pv_samp_join_in)):
                            if out:
                                f.write(str(k+1)+'\t'+str(pv_samp_join_in[k])+'\t'+str(pv_samp_join_out[k])+'\n')
                            else:
                                f.write(str(k+1)+'\t'+str(pv_samp_join_in[k])+'\n')

                    ## CC vals
                    # read results and write a new combined output
                    path = '..'+os.sep+'features'+os.sep+'correlations'+os.sep+'stat_sign_per_hig_corr_feat'+os.sep+'_temp'+os.sep
                    _fls = glob.glob(path+"*cc_in.txt")
                    pv_samp_join_in = []
                    for _fl in _fls:
                        fl  = np.genfromtxt(_fl, delimiter="\t")
                        [pv_samp_join_in.append(fl[i][1]) for i in range(len(fl))]

                    if out:    
                        _fls = glob.glob(path+"*cc_out.txt")
                        pv_samp_join_out = []
                        for _fl in _fls:
                            fl  = np.genfromtxt(_fl, delimiter="\t")
                            [pv_samp_join_out.append(fl[i][1]) for i in range(len(fl))]

                    # rewrite into a new files (both in and out samples)
                    _sta = ''
                    for i in range(len(stas)):
                        _sta = _sta + '_'+stas[i]  
                    if out: 
                        if ind_samp:
                            _nm = 'cc_in_out_samp_pool_'+ds[0]+'_'+feat+'_'+str(n_erup_samp)+'e_'+str(ST)+'ite_'+'indsamp'
                        else:
                            _nm = 'cc_in_out_samp_pool_'+ds[0]+'_'+feat+'_'+str(n_erup_samp)+'e_'+str(ST)+'ite'
                    else:
                        if ind_samp:
                            _nm = 'cc_in_samp_pool_'+ds[0]+'_'+feat+'_'+str(n_erup_samp)+'e_'+str(ST)+'ite_'+'indsamp'
                        else:
                            _nm = 'cc_in_samp_pool_'+ds[0]+'_'+feat+'_'+str(n_erup_samp)+'e_'+str(ST)+'ite'#
                    path = '..'+os.sep+'features'+os.sep+'correlations'+os.sep+'stat_sign_per_hig_corr_feat'+os.sep
                    with open(path+_nm+_sta+'.txt', 'w') as f:
                        for k in range(len(pv_samp_join_out)):
                            if out: 
                                f.write(str(k+1)+'\t'+str(pv_samp_join_in[k])+'\t'+str(pv_samp_join_out[k])+'\n')
                            else:
                                f.write(str(k+1)+'\t'+str(pv_samp_join_in[k])+'\n')

                    print("--- %s seconds ---" % (time.time() - start_time))
                    ts = time.time() - start_time  

            else: # standard (run all archetypes)
                if out_sample:
                    print('Archetypes to run: '+str(len(archetypes)+len(archetypes_out)))
                else:
                    print('Archetypes to run: '+str(len(archetypes)))
                ## in sample: iterate over the samples (archetypes) to perform the convolution
                Ns = len(archetypes)#8#5#10
                _pv_samp_in = [] 
                _cc_samp_in = [] 
                # whole record to convolute with 
                pv_samp_in = []
                cc_samp_in  = []

                #for l, te in enumerate(samp_in):
                for l in range(Ns):
                    print('Running in sample: '+str(l+1)+'/'+str(Ns))
                    # 
                    archtype = archetypes[l]
                    cc_te = []
                    cc_non_te = []
                    #
                    # loop over volcanoes and extract cc and pvals
                    for sta in stas:
                        if False:
                            fm = ForecastModel(window=2., overlap=1., station = sta,
                            look_forward=2., data_streams=ds, savefile_type='csv')
                        else:
                            path_feat_serv = 'C:\\Users\\aar135\\codes_local_disk\\volc_forecast_tl\\features_bkp\\features_server\\'
                            fm = ForecastModel(window=2., overlap=1., station = sta,
                                look_forward=2., data_streams=ds, 
                                feature_dir=path_feat_serv, 
                                savefile_type='pkl')
                        #
                        dt = ds[0]#'zsc2_dsarF'
                        # rolling median and signature length window
                        N, M = [2,30]
                        # time
                        j = fm.data.df.index
                        # construct signature
                        df = fm.data.df[(j>(te-(M+N)*day))&(j<te)]
                        # convolve over the station data
                        df = fm.data.df[:]#[(j<te)]
                        test = df[dt].rolling(N*24*6).median()[N*24*6:]
                        out = test.rolling(archtype.shape[0]).apply(partial(conv, (archtype-archtype.mean())/archtype.std()))
                        
                        # cc in eruption times
                        _cc_te = []
                        #_samp_in = [samp_in[k] for k in range(len(samp_in)) if k != l]
                        for _te in fm.data.tes[:]:
                            _cc_te.append(out[out.index.get_loc(_te, method='nearest')])
                        _cc_te = np.array(_cc_te)
                        #
                        cc_te =  np.concatenate((cc_te, _cc_te), axis=0)
                        #
                        # save non-eruptive cc values 
                        _pop_out = []
                        # 
                        pop_rej = []
                        for e in fm.data.tes[:]:
                            #e = datetime.date(e.year, e.month, e.day)
                            _e = e.replace(hour=00, minute=00)
                            n_days_before = 60
                            _vdays = [_e - n_days_before*day + i*day for i in range(n_days_before+30)] # 2 weeks before and 1 week after   
                            pop_rej =  pop_rej + _vdays

                        # construct out of eruption population
                        for d in _vec_days: 
                            if d not in pop_in:
                                if True: #d.year != 2013:
                                    if d not in pop_rej:
                                        _pop_out.append(d)
                        # cc in eruption times
                        _cc_non_te = []
                        #_samp_in = [samp_in[k] for k in range(len(samp_in)) if k != l]
                        for _te in _pop_out:
                            _cc_non_te.append(out[out.index.get_loc(_te, method='nearest')])
                        _cc_non_te = np.array(_cc_non_te)
                        #
                        cc_non_te =  np.concatenate((cc_non_te, _cc_non_te), axis=0)
                    
                    # correct cc_te by removing the value of the archetypr with itself 
                    cc_te = cc_te[cc_te < 0.95]
                    _cc_non_te = _cc_non_te[_cc_non_te < 0.9]
                    ## (4) calulate p-value for the sample 
                    # 2-sample Kolmogorov Smirnov test for difference in underlying distributions
                    from scipy.stats import kstest
                    #a = out.iloc[archtype.shape[0]::24*6].values
                    #pv = kstest(cc_te, out.iloc[archtype.shape[0]::24*6].values).pvalue
                    pv = kstest(cc_te, _cc_non_te).pvalue
                    #
                    _pv_samp_in.append(pv)
                    [_cc_samp_in.append(cc_te[i]) for i in range(len(cc_te))]
                    #             
                #
                pv_samp_in = pv_samp_in + _pv_samp_in
                cc_samp_in = cc_samp_in + _cc_samp_in
                
                if out_sample:
                    ## out sample: convolute with random archetypes
                    Ns = len(archetypes_out)#8#5#10
                    #
                    _pv_samp_out = [] 
                    _cc_samp_out = [] 
                    # whole record to convolute with 
                    #
                    pv_samp_out = []
                    cc_samp_out  = []

                    #for l, te in enumerate(samp_in):
                    for l in range(Ns):
                        print('Running out sample: '+str(l+1)+'/'+str(Ns))
                        # 
                        archtype = archetypes_out[l]
                        cc_te = []
                        cc_non_te = []
                        #
                        # loop over volcanoes and extract cc and pvals
                        for sta in stas:
                            if False:
                                fm = ForecastModel(window=2., overlap=1., station = sta,
                                look_forward=2., data_streams=ds, savefile_type='csv')
                            else:
                                path_feat_serv = 'C:\\Users\\aar135\\codes_local_disk\\volc_forecast_tl\\features_bkp\\features_server\\'
                                fm = ForecastModel(window=2., overlap=1., station = sta,
                                    look_forward=2., data_streams=ds, 
                                    feature_dir=path_feat_serv, 
                                    savefile_type='pkl')
                            #
                            dt = ds[0]#'zsc2_dsarF'
                            # rolling median and signature length window
                            N, M = [2,30]
                            # time
                            j = fm.data.df.index
                            # construct signature
                            df = fm.data.df[(j>(te-(M+N)*day))&(j<te)]
                            # convolve over the station data
                            df = fm.data.df[:]#[(j<te)]
                            test = df[dt].rolling(N*24*6).median()[N*24*6:]
                            out = test.rolling(archtype.shape[0]).apply(partial(conv, (archtype-archtype.mean())/archtype.std()))
                            
                            # cc in eruption times
                            _cc_te = []
                            #_samp_in = [samp_in[k] for k in range(len(samp_in)) if k != l]
                            for _te in fm.data.tes[:]:
                                _cc_te.append(out[out.index.get_loc(_te, method='nearest')])
                            _cc_te = np.array(_cc_te)
                            #
                            cc_te =  np.concatenate((cc_te, _cc_te), axis=0)
                            #
                            # save non-eruptive cc values 
                            _pop_out = []
                            # 
                            pop_rej = []
                            for e in fm.data.tes[:]:
                                #e = datetime.date(e.year, e.month, e.day)
                                _e = e.replace(hour=00, minute=00)
                                n_days_before = 60
                                _vdays = [_e - n_days_before*day + i*day for i in range(n_days_before+30)] # 2 weeks before and 1 week after   
                                pop_rej =  pop_rej + _vdays

                            # construct out of eruption population
                            for d in _vec_days: 
                                if d not in pop_in:
                                    if True: #d.year != 2013:
                                        if d not in pop_rej:
                                            _pop_out.append(d)
                            # cc in eruption times
                            _cc_non_te = []
                            #_samp_in = [samp_in[k] for k in range(len(samp_in)) if k != l]
                            for _te in _pop_out:
                                _cc_non_te.append(out[out.index.get_loc(_te, method='nearest')])
                            _cc_non_te = np.array(_cc_non_te)
                            #
                            cc_non_te =  np.concatenate((cc_non_te, _cc_non_te), axis=0)
                        
                        # correct cc_te by removing the value of the archetypr with itself 
                        cc_te = cc_te[cc_te < 0.95]
                        _cc_non_te = _cc_non_te[_cc_non_te < 0.9]
                        ## (4) calulate p-value for the sample 
                        # 2-sample Kolmogorov Smirnov test for difference in underlying distributions
                        from scipy.stats import kstest
                        #a = out.iloc[archtype.shape[0]::24*6].values
                        #pv = kstest(cc_te, out.iloc[archtype.shape[0]::24*6].values).pvalue
                        pv = kstest(cc_te, _cc_non_te).pvalue
                        #
                        _pv_samp_out.append(pv)
                        [_cc_samp_out.append(cc_te[i]) for i in range(len(cc_te))]
                        #             
                    #
                    pv_samp_out = pv_samp_out + _pv_samp_out
                    cc_samp_out = cc_samp_out + _cc_samp_out

                #
                print("--- %s seconds ---" % (time.time() - start_time))
                ts = time.time() - start_time   

                #
                if out_sample:
                    sta_nm = '_in_out_WIZ_FWVZ_ite_'+str(ST) #'_WIZ_FWZ_KRZ_VNSS_BELO_PVV'#'_WIZ_FWZ_KRZ'            # write p-values from samples
                    path = '..'+os.sep+'features'+os.sep+'correlations'+os.sep+'stat_sign_per_hig_corr_feat'+os.sep
                    with open(path+'pv_samp_in_mult_volc'+sta_nm+'.txt', 'w') as f:
                        for k in range(len(pv_samp_in)):
                            f.write(str(k+1)+'\t'+str(pv_samp_in[k])+'\t'+str(pv_samp_out[k])+'\n')
                    # write cc from samples
                    path = '..'+os.sep+'features'+os.sep+'correlations'+os.sep+'stat_sign_per_hig_corr_feat'+os.sep
                    with open(path+'cc_samp_in_mult_volc'+sta_nm+'.txt', 'w') as f:
                        for k in range(len(cc_samp_in)):
                            f.write(str(k+1)+'\t'+str(cc_samp_in[k])+'\t'+str(cc_samp_out[k])+'\n')
                else:
                    sta_nm = '_WIZ_FWVZ_KRVZ_ite_'+str(ST) #'_WIZ_FWZ_KRZ_VNSS_BELO_PVV'#'_WIZ_FWZ_KRZ'            # write p-values from samples
                    path = '..'+os.sep+'features'+os.sep+'correlations'+os.sep+'stat_sign_per_hig_corr_feat'+os.sep
                    with open(path+'pv_samp_in_mult_volc'+sta_nm+'.txt', 'w') as f:
                        for k in range(len(pv_samp_in)):
                            f.write(str(k+1)+'\t'+str(pv_samp_in[k])+'\n')
                    # write cc from samples
                    path = '..'+os.sep+'features'+os.sep+'correlations'+os.sep+'stat_sign_per_hig_corr_feat'+os.sep
                    with open(path+'cc_samp_in_mult_volc'+sta_nm+'.txt', 'w') as f:
                        for k in range(len(cc_samp_in)):
                            f.write(str(k+1)+'\t'+str(cc_samp_in[k])+'\n')
                
        if True: # plots p-val hist for one feature 
            # read file
            out = True
            path = '..'+os.sep+'features'+os.sep+'correlations'+os.sep+'stat_sign_per_hig_corr_feat'+os.sep
            fl_nm = 'pv_in_out_samp_pool_zsc2_dsarF_median_8e_40ite_WIZ_FWVZ_KRVZ.txt'
            fl  = np.genfromtxt(path+fl_nm, delimiter="\t")
            pv_samp_in = [fl[i][1] for i in range(len(fl))]
            if out:
                pv_samp_out = [fl[i][2] for i in range(len(fl))]
            # correct p-values
            FCR = False
            if FCR: # FDR correction
                import statsmodels
                # in
                pv_samp_in_cor = statsmodels.stats.multitest.fdrcorrection(pv_samp_in, alpha=0.05, method='n', is_sorted=False)
                pv_samp_in_BY = pv_samp_in_cor[1][:]
                pv_samp_in_BY_5p = np.percentile(pv_samp_in_BY,5)
                # out
                if out:
                    pv_samp_out_cor = statsmodels.stats.multitest.fdrcorrection(pv_samp_out, alpha=0.05, method='n', is_sorted=False)
                    pv_samp_out_BY = pv_samp_out_cor[1][:]
                
            count_in = 0
            for pv in pv_samp_in:
                if pv < 0.05:
                    count_in+=1
            print('pv in < 0.05: '+str(100*count_in/len(pv_samp_in))+' %')
            #
            if out:
                count = 0
                for pv in pv_samp_out:
                    if pv < 0.04:
                        count+=1
                print('pv out < 0.05: '+str(100*count/len(pv_samp_out))+' %')

            # calc p_val between the two distibutions 
            if out:
                from scipy.stats import kstest
                pv_dist = kstest(pv_samp_out, pv_samp_in).pvalue

            ## (5) construct p-val histogram of both populations
            fig, ax = plt.subplots()
            a_heights, a_bins = np.histogram(pv_samp_in)
            if out:
                b_heights, b_bins = np.histogram(pv_samp_out)
            width = (a_bins[1] - a_bins[0])/2
            ax.bar(a_bins[:-1], a_heights, width=width, facecolor='cornflowerblue', label = 'in eruption')
            if out:
                ax.bar(b_bins[:-1]+width, b_heights, width=width, facecolor='seagreen', alpha = 0.6, label = 'out eruption')
            ax.set_xlabel('p-value')
            ax.set_ylabel('frequency')
            #ax.set_xlim([0,0.1])
            #ax.set_xscale('log')
            ax.axvline(x=0.05, color = 'k', ls='--', linewidth=1, label = '0.05 threshold')
            if FCR:
                ax.axvline(x=pv_samp_in_BY_5p, color = 'gray', ls='--', linewidth=1, label = 'B.-Y. threshold')
            # place a text box in upper left in axes coords
            textstr = str(r'pv in < 0.05:  '+str(round(100*count_in/len(pv_samp_in),2))+' %')
            if out:
                textstr = '\n'.join((r'pv in < 0.05:  '+str(round(100*count_in/len(pv_samp_in),2))+' %',
                        r'pv out < 0.05: '+str(round(100*count/len(pv_samp_out),2))+' %\n'))
                textstr += r'p value between distributions:  '+"{:.2e}".format(pv_dist)
            props = dict(boxstyle='round', facecolor='white', alpha=0.4)
            ax.text(0.41, .74, textstr, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=props)
            ax.legend()
            plt.show()  
            #plt.savefig(path+'pv_samp_in_multi_volc.png')    

        if False: # plot p-val hist for two features 
            # read file
            out = True
            path = '..'+os.sep+'features'+os.sep+'correlations'+os.sep+'stat_sign_per_hig_corr_feat'+os.sep
            fl_nm_1 = 'pv_in_out_samp_pool_zsc2_dsarF_median_8e_40ite_WIZ_PVV_VNSS_BELO.txt'#'pv_samp_in_out_WIZ_b7_Ns10_Ite10.txt' 
            fl_nm_2 = 'pv_in_out_samp_pool_zsc2_dsarF_rate_var_8e_40ite_WIZ_PVV_VNSS_BELO.txt'#'pv_in_out_samp_pool_WIZ_zsc2_dsarF_rate_var_Ns10_10ite.txt'
            #
            fl_nm_1_feat = r'nDSAR median'
            fl_nm_2_feat = r'nDSAR rate var'
            #
            fl_1  = np.genfromtxt(path+fl_nm_1, delimiter="\t")
            pv_samp_in_1 = [fl_1[i][1] for i in range(len(fl_1))]

            fl_2  = np.genfromtxt(path+fl_nm_2, delimiter="\t")
            pv_samp_in_2 = [fl_2[i][1] for i in range(len(fl_2))]
            if out:
                pv_samp_out_1 = [fl_1[i][2] for i in range(len(fl_1))]
                pv_samp_out_2 = [fl_2[i][2] for i in range(len(fl_2))]
            # 1 
            count_in = 0
            for pv in pv_samp_in_1:
                if pv < 0.05:
                    count_in+=1
            print('pv in < 0.05: '+str(100*count_in/len(pv_samp_in_1))+' %')
            #
            if out:
                count = 0
                for pv in pv_samp_out_1:
                    if pv < 0.04:
                        count+=1
                print('pv out < 0.05: '+str(100*count/len(pv_samp_out_1))+' %')

            # calc p_val between the two distibutions 
            if out:
                from scipy.stats import kstest
                pv_dist = kstest(pv_samp_in_1, pv_samp_out_1).pvalue
            # 2
            count_in = 0
            for pv in pv_samp_in_2:
                if pv < 0.05:
                    count_in+=1
            print('pv in < 0.05: '+str(100*count_in/len(pv_samp_in_2))+' %')
            #
            if out:
                count = 0
                for pv in pv_samp_out_2:
                    if pv < 0.04:
                        count+=1
                print('pv out < 0.05: '+str(100*count/len(pv_samp_out_2))+' %')

            # calc p_val between the two distibutions 
            if out:
                from scipy.stats import kstest
                pv_dist = kstest(pv_samp_in_2, pv_samp_out_2).pvalue

            ## (5) construct p-val histogram of both populations
            fig, ax = plt.subplots()
            a_heights, a_bins = np.histogram(pv_samp_in_1)
            
            # select lists 
            #ax.axvline(x=0.05, color = 'k', ls='--', linewidth=1, label = '0.05 threshold')
            multi = [pv_samp_in_1, pv_samp_out_1, pv_samp_in_2, pv_samp_out_2]
            colors = ['b', 'c', 'r', 'orange']
            labels = [fl_nm_1_feat+': in eruption', fl_nm_1_feat+': out eruption', 
                fl_nm_2_feat+': in eruption', fl_nm_2_feat+': out eruption']
            bins = np.linspace(0, 1, 20)
            ax.hist(multi, bins, color = colors, label=labels)

            ax.set_xlabel('p-value')
            ax.set_ylabel('frequency')
            #ax.set_xlim([0,0.1])
            #ax.set_xscale('log')
            
            plt.legend(loc='upper right')
            plt.show()

def main():
    #import_data()
    #data_Q_assesment()
    #calc_feature_pre_erup()
    #plot_feats()
    #corr_feat_calc()
    #corr_feat_analysis()
    #plot_corr_feat_dendrogram()
    download_geonet_data()
    #plot_other_data_geonet()
    #plot_interpretation()
    #plot_interpretation_media()
    #ext_pval_anal()

if __name__ == "__main__":
    main()

