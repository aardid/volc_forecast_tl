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

#
def import_data():
    '''
    Download tremor data from an specific station. 
    Available stations : 
    'PVV','VNSS','SSLW','OKWR','REF','BELO','CRPO','VTUN','KRVZ','FWVZ','WIZ','AUR'
    '''
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
    '''
    Check section of interpolated data
    '''
    # constants
    month = timedelta(days=365.25/12)
    day = timedelta(days=1)
    station = 'WIZ'
    # read raw data
    td = TremorData(station = station)
    #t0 = "2007-08-22"
    #t1 = "2007-09-22"
    #td.update(ti=t0, tf=t1)

    # plot data 
    #td.plot( data_streams = ['rsamF'])#(ti=t0, tf=t1)
    t0 = "2012-07-05"
    t1 = "2012-08-05"
    #td.update(ti=t0, tf=t1)
    data_streams = ['rsamF']
    td.plot_zoom(data_streams = data_streams, range = [t0,t1])

    # interpolated data 
    #t0 = "2015-01-01"
    #t1 = "2015-02-01"
    #td.plot_intp_data(range_dates = None)
    #td.plot_intp_data(range_dates = [t0,t1])

def calc_feature_pre_erup():
    ''' 
    Load data and calculate features in parallel for multiple stations, multiple datastreams, and multiple window sizes.
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
    if False: # serial
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

def corr_feat_calc():
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

def corr_feat_analysis():
    ' Analysis of feature correlation results'
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
        # import first file
        df_sum = pd.read_csv(file_dir[0], index_col=0)
        # loop over files (starting from the second) (.csv created for each feature correlation)
        count_nan = 0
        for i in range(1,len(file_dir),1):
            # add values from file to cummulative dataframe
            df_aux = pd.read_csv(file_dir[i], index_col=0)
            # sum df that aren't nan
            if df_aux.isnull().values.any():# math.isnan(df_aux.iat[0,0]): 
                #print('DataFrame is empty: '+ file_dir[i])
                count_nan+=1
            else:
                df_sum = abs(df_sum) + abs(df_aux)
        # modified diagonal of df_sum
        #np.fill_diagonal(df_sum.values, 0)
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
        
        del df_aux, 

    # plot feature correlation (from .csv of rank of cc eruptions)
    if False:
        # import file as pandas
        path = '..'+os.sep+'features'+os.sep+'correlations'+os.sep+'corr_0_rank_erup_cc.csv'
        try:
            pd_rank = pd.read_csv(path, index_col=0)
        except: # convert txt to csv
            read_file = pd.read_csv ('..'+os.sep+'features'+os.sep+'correlations'+os.sep+'corr_0_rank_erup_cc.txt')
            read_file.to_csv (path, index=None)
            #
            pd_rank = pd.read_csv(path, index_col=0)
            del read_file
        # locate index in rank for cc cut value of 0.5
        idx, _ = find_nearest(pd_rank.loc[:, 'cc'].values, 0.5)
        # for plotting: import features of eruptions per rank
        ranks = np.arange(130,idx,1)
        plot_2_pairs_erups = True # set to true to plot a third correlated eruption
        if plot_2_pairs_erups:
            ranks = np.arange(130, idx, 1)

        for rank in ranks:
            # name of eruptions to plot
            e1 = pd_rank.loc[rank, 'erup1'] # name of eruptions to plot
            e2 = pd_rank.loc[rank, 'erup2']
            # feature to plot
            ft_nm = pd_rank.loc[rank, 'featNM']
            ft_id = pd_rank.loc[rank, 'featID']
            cc = pd_rank.loc[rank, 'cc']
            # look for the third eruption mostly correlated with the e1 or e2 in ft_nm
            try:
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
                # create objects for each station 
                ds = ['log_zsc2_rsamF', 'zsc2_hfF','zsc2_mfF','zsc2_dsarF']
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
                
                # proper name of feature
                ft_nm_aux = ft_nm.replace(" ","__")
                ft_nm_aux = ft_nm_aux.replace("-",'"')
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

                # plot features
                plt_ds = True # plot datastream 

                # adding multiple Axes objects  
                if not plot_2_pairs_erups:
                    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10,6))
                else:
                    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(10,12))

                # ax1
                ax1.plot(ft_e1_t, ft_e1_v, '-', color='b', label=e1)
                te = fm_e1.data.tes[int(e1[-1:])-1]
                ax1.axvline(te, color='k', linestyle='--', linewidth=2, zorder = 0)
                ax1.plot([], color='k', linestyle='--', linewidth=2, label = 'eruption')
                ax1.legend()
                ax1.grid()
                ax1.set_ylabel('feature value')
                ax1.set_xlabel('time')
                #plt.xticks(rotation=45)
                
                # ax2
                ax2.plot(ft_e2_t, ft_e2_v, '-', color='r', label=e2)
                te = fm_e2.data.tes[int(e2[-1:])-1]
                ax2.axvline(te, color='k', linestyle='--', linewidth=2, zorder = 0)
                ax2.plot([], color='k', linestyle='--', linewidth=2, label = 'eruption')
                ax2.legend()
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
                    ax1b.legend(loc = 3)
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
                    ax2b.legend(loc = 3)
                    ax2b.set_yticks([])

                if plot_2_pairs_erups:

                    # ax3
                    ax3.plot(ft_e1a_t, ft_e1a_v, '-', color='b', label=e1a)
                    te = fm_e1a.data.tes[int(e1a[-1:])-1]
                    ax3.axvline(te, color='k', linestyle='--', linewidth=2, zorder = 0)
                    ax3.plot([], color='k', linestyle='--', linewidth=2, label = 'eruption')
                    ax3.legend()
                    ax3.grid()
                    ax3.set_ylabel('feature value')
                    ax3.set_xlabel('time')
                    #plt.xticks(rotation=45)
                    
                    # ax4
                    ax4.plot(ft_e2a_t, ft_e2a_v, '-', color='r', label=e2a)
                    te = fm_e2a.data.tes[int(e2a[-1:])-1]
                    ax4.axvline(te, color='k', linestyle='--', linewidth=2, zorder = 0)
                    ax4.plot([], color='k', linestyle='--', linewidth=2, label = 'eruption')
                    ax4.legend()
                    ax4.grid()
                    ax4.set_ylabel('feature value')
                    ax4.set_xlabel('time')
                    #plt.xticks(rotation=45)

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
                        ax3b.legend(loc = 3)
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
                        ax4b.legend(loc = 3)
                        ax4b.set_yticks([])

                fig.suptitle('Rank: '+str(rank)+' (cc:'+str(round(cc,2))+')  Eruptions: '+ e1+' and '+ e2
                    +'\n Feature: '+ ft_nm_aux+' (id:'+str(ft_id)+')')#, ha='center')
                plt.tight_layout()
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
            except:
                pass

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

def main():
    #import_data()
    #data_Q_assesment()
    calc_feature_pre_erup()
    #corr_feat_calc()
    #corr_feat_analysis()
    #plot_corr_feat_dendrogram()

if __name__ == "__main__":
    main()

