
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 09:12:13 2019

@author: mkaist
"""


import numpy as np
import scipy.signal as signal
import math
from preprocessor import butter_filter

MINRRI     = 110    #autocorr window size for detecting period (samples)
STEP_LEN   = 5      #step length used to segment signal (sec)
AC_LEN     = [5,7]  #autocorr window length (sec)
RRILIM     = 300    #lim for finding period, first two autocorr peaks (samples)
DRRILIM    = 10     #lim for RRI differences (samples)
AMPLIM     = 0.4    #lim for finding period, first two autocorr peaks
PERIOD_FRACTION = 0.18
FS = 200

# TODO overall

def analyze_corr(corr_period, rr_interval):
    '''documenting here
    '''
    corr_period[corr_period >= 1] = 1
    regularity_index = np.mean(corr_period)*100
    #print('regularity_index =', regularity_index)
    if np.mean(corr_period) < PERIOD_FRACTION: afib = 1
    else: afib = 0 
    return afib, regularity_index, 60/rr_interval*FS # hr calc fixed by Mikko

def detect_period(data, fs=FS):
    data = data.copy()
    
    data = butter_filter(data)
         
    for i in range (0, len(AC_LEN)):
    
        ax_segmented,\
        ay_segmented,\
        az_segmented,\
        gx_segmented,\
        gy_segmented,\
        gz_segmented = segment_all_axis(data, STEP_LEN, AC_LEN[i], FS)
     
        ax_ac = correlate_segments(ax_segmented)
        ay_ac = correlate_segments(ay_segmented)
        az_ac = correlate_segments(az_segmented)
        gx_ac = correlate_segments(gx_segmented)
        gy_ac = correlate_segments(gy_segmented)
        gz_ac = correlate_segments(gz_segmented)        
        
        #initialize empty matrix if vars do not exist
        if 'periodfound_all_axis' in locals():
            None
        else:
            periodfound_all_axis = np.empty([len(ax_ac), len(AC_LEN)])
        
        if 'rri_all_axis' in locals():
            None
        else:
            rri_all_axis = np.empty([len(AC_LEN)])
    
        periodfound_all_axis[:,i], rri_all_axis[i] = find_period_all_axis(ax_ac, ay_ac, az_ac, gx_ac, gy_ac, gz_ac, MINRRI, AMPLIM, RRILIM, FS)
     
    return periodfound_all_axis.sum(axis=1), np.mean(rri_all_axis)

def segment_one_axis(signal_in,STEP_LEN,w,FS):
    """ Segments a single 1D vector into segments and returns segments in
    a list        
    Keyword arguments:
    signal_in -- input vector to segment
    STEP_LEN -- distance between segments starting time (sec)
    w -- length of segments (sec) 
    fs -- sample frequency  
    """   
    n_segments = round((signal_in.size/FS)/STEP_LEN)-2 #reasoning for -2 is unclear 
    segments = []     
    #segments signal with STEP_LEN interval with w length window
    for i in range(0, n_segments + 1):
        startpos    = i*STEP_LEN*FS 
        endpos      = startpos + FS*w - 1
        endpos      = min(endpos,signal_in.size)
        segments.append(signal_in[startpos:endpos + 1]) 
    return segments 

def segment_all_axis(data, STEP_LEN, w, FS):
    """Segments 6 axis data and return 6 separate lists each containing
    a segmented signal
        
    Keyword arguments:
    sensor_data -- dictionary containing 6 axis data
    STEP_LEN -- distance between segments starting time (sec)
    w -- length of segments (sec) 
    fs -- sample frequenc
    """
     
    ax = data['ax'][FS:]
    ay = data['ay'][FS:]
    az = data['az'][FS:]
    gx = data['gx'][FS:]
    gy = data['gy'][FS:]
    gz = data['gz'][FS:]    
    
    ax_segmented = segment_one_axis(ax,STEP_LEN,w,FS)
    ay_segmented = segment_one_axis(ay,STEP_LEN,w,FS)
    az_segmented = segment_one_axis(az,STEP_LEN,w,FS)
    gx_segmented = segment_one_axis(gx,STEP_LEN,w,FS)
    gy_segmented = segment_one_axis(gy,STEP_LEN,w,FS)
    gz_segmented = segment_one_axis(gz,STEP_LEN,w,FS)
    return (ax_segmented, ay_segmented, az_segmented,\
            gx_segmented, gy_segmented, gz_segmented)
   
def correlate_segments(signals_in):
    """Autocorrelate each segment and return
    a list with autocorrelated segments
    
    Keyword arguments: 
    signals_in -- a list of signal segments
    """
    n_segments  = len(signals_in) 
    ac = []
    
    #go through all segments and autocorrelate each
    for i in range (0, n_segments): 
        tmp = signal.correlate(signals_in[i], signals_in[i])  
        tmp = tmp[signals_in[i].size-2:] 
        tmp = tmp/max(tmp)
        #plt.plot(tmp)
        ac.append(tmp)
        
    return ac

def find_sidepeaks_one_segment(signal_in, MINRRI):
    """Find side peaks from one 1D vector and 
    returns peak locations and amplitudes
        
    Keyword arguments: 
    signal_in -- input signal from which peaks are detected
    MINRRI -- backward min distance from where a peak is searched        
    """
    all_locs, _ = signal.find_peaks(signal_in)
    all_pks     = signal_in[all_locs]
    
    #peak elinimation rules
    tmp  = []
    locs = [all_locs[0]]
    pks  = [all_pks[0]]
    for i in range(1, all_locs.size): 
        startpos = max(1,all_locs[i] - MINRRI)
        tmp      = max(signal_in[startpos:all_locs[i]])
        if all_pks[i] >= max(all_pks[i:]) and\
        all_pks[i] >= tmp and\
        all_locs[i] > locs[-1] + MINRRI:             
        #if all conditions are true, append the peaks and location
           locs.append(all_locs[i])
           pks.append(all_pks[i])
           
    return locs, pks

def find_period_segments(signals_in, MINRRI, AMPLIM, RRILIM, DRRILIM):
    """Find periodicity of a segmented signal
    
    Keyword arguments: 
    signals_in -- a segmented signal
    MINRRI -- passed onward 
    AMPLIM -- required ratio between first two found peaks
    RRILIM -- max distance between found peaks
    DRRILIM -- limit of interval differences       
    """
    n_segments  = len(signals_in)  #segments is a list
    periodfound = np.zeros(n_segments)
    rri         = np.zeros(n_segments)
    
    for i in range (0, n_segments): 
        #find all local maximas
        locs_tmp, pks_tmp = find_sidepeaks_one_segment(signals_in[i], MINRRI)

        #period finding rules
        max_locs_diff = math.inf
        max_rri       = math.inf        
        if len(locs_tmp) > 3 and (pks_tmp[1] / pks_tmp[0]) > AMPLIM:
            locs_diff     = np.diff(locs_tmp[0:3])
            max_rri       = locs_tmp[1] - locs_tmp[0]
            max_locs_diff = max(abs(np.diff(locs_diff)))
            
        if max_locs_diff < DRRILIM and max_rri < RRILIM:
            periodfound[i] = 1
            rri[i] = max_rri
        else:
            periodfound[i] = 0   
            rri[i] = 0
          
    return periodfound, rri

def find_period_all_axis(ax, ay, az, gx, gy, gz, MINRRI, AMPLIM, RRILIM, fs):
    """Find period from 6 axis data. Returns a numpy vector
    containing 1 (periodfound) or 0 (period not found) for each
    segment in each 6 axis. Returns a numpy vector containing the average 
    over az, gy, gz lag between 1st and 2nd peaks
    
    Keyword arguments:
    ax, ay, az, gx ,gy, gz -- segmented and autocorrelated inputs
    MINRRI -- passed onward 
    AMPLIM -- passed onward 
    RRILIM -- passed onward 
    fs -- passed onward     
    """
    periodfound_ax, rri_ax = find_period_segments(ax, MINRRI, AMPLIM, RRILIM, DRRILIM)
    periodfound_ay, rri_ay = find_period_segments(ay, MINRRI, AMPLIM, RRILIM, DRRILIM)
    periodfound_az, rri_az = find_period_segments(az, MINRRI, AMPLIM, RRILIM, DRRILIM)
    periodfound_gx, rri_gx = find_period_segments(gx, MINRRI, AMPLIM, RRILIM, DRRILIM)
    periodfound_gy, rri_gy = find_period_segments(gy, MINRRI, AMPLIM, RRILIM, DRRILIM)
    periodfound_gz, rri_gz = find_period_segments(gz, MINRRI, AMPLIM, RRILIM, DRRILIM)
    
    P = np.vstack((periodfound_ax, periodfound_ay, periodfound_az,
                   periodfound_gx, periodfound_gy, periodfound_gz))
    
    #periodfound vector size 1 x n where n is number of segments for a 
    #spesific autocorrelation window length
    periodfound_all_axis = P.sum(axis=0)
    periodfound_all_axis[periodfound_all_axis >= 1] = 1
    
    # mean rri averaged over all segments and all axis where period was found
    # with one correlation window  
    R   = np.vstack((rri_az, rri_ay, rri_az, rri_gx, rri_gy, rri_gz)) #take mean lag from az, gy, gz autocorrelations
    R   = R[np.where(R > 0) ]
     
    if len(R) > 0:
        rri_all_axis = R.mean(axis=0)    
    else: 
        rri_all_axis = math.nan
        
    return periodfound_all_axis, rri_all_axis
    

if __name__ == '__main__':

    from test_helpers import load_data_mat
    import pandas as pd
    import os

    #load data
    if not 'run_correlator' in locals(): 
        run_correlator = True
        #fullpath = os.path.join(os.getcwd(), 'test_data', 'orig_mat') 
        fullpath = os.path.join(os.getcwd(), 'test_data', 'AFIB_or_SR') 
        _, test_data = load_data_mat(fullpath)   
    
    
    #loop a parameter
    #vec = [100, 110, 120, 130] best 110 with 8/8
    #vec = [3,5,7] best 5 with 8/8
    #vec = [250, 300, 350] best 300 with 8/8
    #vec = [5, 10, 15] best 10 with 9/7 (should be 8/8)
    #vec = [0.3, 0.4, 0.5] best 0.4 witn 8/8
    #vec = [0.1, 0.2, 0.3] 8/8, 4/11, 1/18 FN/FP
    vec = [0.2]
    for PERIOD_FRACTION in vec:
    
        #run algo
        afibs, regularity_indexes, hrs = [], [], []    
        keys = [key for key in test_data]
        keys.sort()
        printcounter = 0
        for key in keys:
            printcounter += 1
            if printcounter%10 == 0: print(key)
            data = test_data[key]
            corr_period, rr_interval = detect_period(data)    
            afib, regularity_index, hr = analyze_corr(corr_period, rr_interval)
            afibs.append(afib)
            regularity_indexes.append(regularity_index)
            hrs.append(hr)
        
        #load labels
        xlsfile = 'MODE_AF_results.xlsx'
        fullpath = os.path.join(os.getcwd(), 'test_data', xlsfile)
        df = pd.read_excel(fullpath)
        labels = df[['AF binary']]
        labels.index += 1
        labels = labels[:300].values.tolist()
        labels = [item for sublist in labels for item in sublist]
        
        #analyze results
        TP, TN, FP, FN = 0, 0, 0, 0
        for i in range(len(labels)):
            if labels[i] == afibs[i] == 1:
                TP += 1
            elif labels[i] == afibs[i] == 0:
                TN += 1
            elif afibs[i] == 0 and labels[i] == 1:
                FN += 1
            elif afibs[i] == 1 and labels[i] == 0:
                FP += 1
            
        assert(TP+TN+FN+FP == len(labels))
        print(PERIOD_FRACTION)
        print('TP =', TP)
        print('TN =', TN)
        print('FN =', FN)
        print('FP =', FP)
    
    

