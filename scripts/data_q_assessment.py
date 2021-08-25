import os, sys
sys.path.insert(0, os.path.abspath('..'))
from whakaari import TremorData, ForecastModel, load_dataframe, datetimeify
from datetime import timedelta, datetime
from matplotlib import pyplot as plt
import numpy as np
import time
from functools import partial
from multiprocessing import Pool
import pandas as pd
import seaborn as sns
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

def import_data():
    if False: # plot raw vel data
        from obspy.clients.fdsn import Client
        client = Client("IRIS")
        from obspy import UTCDateTime
        #t = UTCDateTime("2012-02-27T00:00:00.000")
        starttime = UTCDateTime("2014-01-28")
        endtime = UTCDateTime("2014-01-30")
        inventory = client.get_stations(network="AV", station="SSLW", starttime=starttime, endtime=endtime)
        st = client.get_waveforms(network = "AV", station = "SSLW", location = None, channel = "EHZ", starttime=starttime, endtime=endtime)
        st.plot()  
        asdf

    t0 = "2012-01-01"
    t1 = "2013-07-01"
    td = TremorData(station = 'KRVZ')
    td.update(ti=t0, tf=t1)
    #td.update()

def data_Q_assesment():
    # constants
    month = timedelta(days=365.25/12)
    day = timedelta(days=1)
    station = 'KRVZ'
    # read raw data
    td = TremorData(station = station)
    #t0 = "2007-08-22"
    #t1 = "2007-09-22"
    #td.update(ti=t0, tf=t1)

    # plot data 
    #td.plot( data_streams = ['rsamF'])#(ti=t0, tf=t1)
    t0 = "2012-08-01"
    t1 = "2012-08-10"
    #td.update(ti=t0, tf=t1)
    data_streams = ['rsamF','mfF','hfF']
    td.plot_zoom(data_streams = data_streams, range = [t0,t1])

    # interpolated data 
    #t0 = "2015-01-01"
    #t1 = "2015-02-01"
    td.plot_intp_data(range_dates = None)
    #td.plot_intp_data(range_dates = [t0,t1])

def calc_feature_pre_erup():
    ''' Load data and calculate features in parallel for multiple stations, multiple datastreams, and multiple window sizes.
        Overlap is set to 1.0 (max) 
    '''

    ## data streams
    #ds = ['zsc2_rsamF','zsc2_mfF','zsc2_hfF','zsc2_dsarF','diff_zsc2_rsamF','diff_zsc2_mfF','diff_zsc2_hfF','diff_zsc2_dsarF',
    #    'log_zsc2_rsamF','log_zsc2_mfF','log_zsc2_hfF','log_zsc2_dsarF']
    ds = ['log_zsc2_rsamF', 'zsc2_hfF','zsc2_mfF','zsc2_dsarF']
    ## stations
    ss = ['PVV','VNSS','SSLW','OKWR','REF','BELO','CRPO','VTUN','KRVZ','FWVZ','WIZ','AUR']
    ss = ['PVV','VNSS','KRVZ','FWVZ','WIZ','BELO'] # ,'SSLW'
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
    else: # parallel
        for s in ss:
            for d in ds:
                for lb in lbs:
                    ps.append([lb,s,d])
        n_jobs = 4 # number of cores
        p = Pool(n_jobs)
        p.map(calc_one, ps)

def calc_one(p):
    ''' p = [weeks before eruption, station, datastream] 
    Load HQ data (by calculating features if need) before (p[0] days) every eruption in station given in p[1] for datastreams p[2]. 
    (auxiliary function for parallelization)
    '''
    lb,s,d = p
    #fm = ForecastModel(window=w, overlap=1., station = s,
    #    look_forward=2., data_streams=[d], feature_dir='/media/eruption_forecasting/eruptions/features/', savefile_type='pkl') 
    fm = ForecastModel(window=2., overlap=1., station = s,
        look_forward=2., data_streams=[d], savefile_type='csv')
    a = fm.data.tes
    for etime in fm.data.tes:
        ti = etime - lb*day
        tf = etime 
        fm._load_data(ti, tf, None)

def corr_ana_feat():
    ''' Correlation analysis between features calculated for multiple volcanoes 
        considering 1 month before their eruptions.
        Correlations are performed between multiple eruptions (for several stations)
        for common features derived from multple data streams.
        Correlations are perfomed using pandas Pearson method.
        Results are saved as .csv files and .png in ../features/correlations/
    '''
    ## stations (volcanoes)
    ss = ['WIZ','FWVZ','KRVZ','PVV','VNSS','BELO'] # ,'SSLW'
    ## data streams
    ds = ['log_zsc2_rsamF', 'zsc2_hfF','zsc2_mfF','zsc2_dsarF']
    ## days looking backward from eruptions 
    lbs = 30
    # auxiliary df to extract feature names 
    fm = ForecastModel(window=2., overlap=1., station = 'WIZ',
        look_forward=2., data_streams=ds, savefile_type='csv')
    ti = fm.data.tes[0]
    tf = fm.data.tes[0] + day
    fm.get_features(ti=ti, tf=tf, n_jobs=2, drop_features=[], compute_only_features=[])
    ftns = fm.fM.columns[:] # list of features names
    # directory to be saved
    try:
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
    j, ftn, ss, ds, lbs = p 
    # path to new files 
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

if __name__ == "__main__":
    #import_data()
    #data_Q_assesment()
    #calc_feature_pre_erup()
    corr_ana_feat()
