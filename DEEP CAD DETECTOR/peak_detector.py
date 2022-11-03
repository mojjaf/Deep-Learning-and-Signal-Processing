# -*- coding: utf-8 -*-
"""
Created on Thu May  2 13:04:31 2019

@author: mkaist
"""
import numpy as np
import scipy.signal as signal
from preprocessor import butter_filter
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from test_helpers import plot_data
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from scipy.stats import pearsonr
#from test_helpers1 import plot_lorenz

from test_helpers import load_data_mat
import os
#import copy

FS=800

def detect_peaks(data, fs=FS):
    data = data.copy()
    data = butter_filter(data, lf=10, hf=70, order=4) #pca needs 10 to 70
    data = enhance_envelope(data)
    signal_ = PCA_from_data(data)  
    signal_ = butter_filter(signal_, lf=1, hf=50)
    locs = find_peaks_ampd(signal_, FS)
    #locs=ampdFast(signal_, 1, LSMlimit = 1)
    """
    plt.figure(3)
    plt.plot(signal_)
    plt.plot(locs, signal_[locs], "x")
    """
    return signal_, locs

def analyze_peaks(signal_in, locs):
    hr = 60/np.median(np.diff(locs))*FS
    _, _, index, _ = get_quality_peaks(signal_in, locs, plot_ensemble = 0)
    
    peak_intervals_diff = np.diff(np.diff(locs))
    interval_dist = np.sqrt((peak_intervals_diff[:-1])**2+(peak_intervals_diff[1:])**2)
    tp_tmp1 = sum((interval_dist<55)*1) #for lower HR
    tp_tmp2 = sum((interval_dist<45)*1) #for higher HR
    tp1 = tp_tmp1/len(peak_intervals_diff)
    tp2 = tp_tmp2/len(peak_intervals_diff)
   
    if (tp1 >= 0.9 and 40 < hr < 70) or (tp2 >= 0.9 and 45 < hr < 85):
        afib = 0
    else:
        afib = 1
    
    return afib, index, hr


def PCA_from_data(data):
    D = np.stack((data['ax'], data['ay'], data['az'], 
                  data['gx'], data['gy']))
    
    D = StandardScaler().fit_transform(D.T)   
    pca = PCA(n_components=5)
    return pca.fit_transform(D)[:,0]
   
def normalize_signal(signal_in, low=1, high=99):
    """ Normalises signal roughly between 0 and 1. 
    Normalization based on 5% and 95% percentiles. 
    THIS HAS BEEN CHANGED
        
    Keyword arguments:
    signal_in -- input vector 
    low -- percentile used to adjust lower signal bound
    high -- percentile used to adjust upper signal bound
    
    Returns:
    signal_in -- normalized signal_in
    """  
    signal_in = signal_in - np.mean(signal_in)
    signal_in = signal_in / (np.std(signal_in) * 3) 
    
    return signal_in

def find_peaks_ampd(y, fs):
    """Reference: https://github.com/ig248/pyampd/blob/master/pyampd/ampd.py
    Find peaks from quasi-periodic signal
    
    Parameters
    ----------
    y : ndarray
        1-D array on which to find peaks
    
    Returns
    -------
    locs: ndarray
        Array of peak indices found in `y`
    """
    
    
    L = int(np.ceil(fs*2)) #possible need to scale this length
    n = len(y)
    
    M = np.zeros([L, n], dtype=bool) 
    
    for k in range(1, L):    
        M[k - 1, k:n - k] = (
                (y[k:n-k] > y[0:n-2*k]) &\
                (y[k:n-k] > y[2*k:n])
        )
    
   # Find scale with most maxima
    G = np.sum(M, axis=1)
    l_scale = np.argmax(G)

    # find peaks that persist on all scales up to l
    locs_logical = np.min(M[0:l_scale, :], axis=0)
    locs = np.flatnonzero(locs_logical)
       
    return locs 
  
def ampd(sigInput, LSMlimit = 1):
	"""Find the peaks in the signal with the AMPD algorithm.
	
		Original implementation by Felix Scholkmann et al. in
		"An Efficient Algorithm for Automatic Peak Detection in 
		Noisy Periodic and Quasi-Periodic Signals", Algorithms 2012,
		 5, 588-603
		Parameters
		----------
		sigInput: ndarray
			The 1D signal given as input to the algorithm
		lsmLimit: float
			Wavelet transform limit as a ratio of full signal length.
			Valid values: 0-1, the LSM array will no longer be calculated after this point
			  which results in the inability to find peaks at a scale larger than this factor.
			  For example a value of .5 will be unable to find peaks that are of period 
			  1/2 * signal length, a default value of 1 will search all LSM sizes.
		Returns
		-------
		pks: ndarray
			The ordered array of peaks found in sigInput
	"""

	# Create preprocessing linear fit	
	sigTime = np.arange(0, len(sigInput))
	
	# Detrend
	dtrSignal = (sigInput - np.polyval(np.polyfit(sigTime, sigInput, 1), sigTime)).astype(float)
	
	N = len(dtrSignal)
	L = int(np.ceil(N*LSMlimit / 2.0)) - 1
	
	# Generate random matrix
	LSM = np.ones([L,N], dtype='uint8')
	
	# Local minima extraction
	for k in range(1, L):
		LSM[k - 1, np.where((dtrSignal[k:N - k - 1] > dtrSignal[0: N - 2 * k - 1]) & (dtrSignal[k:N - k - 1] > dtrSignal[2 * k: N - 1]))[0]+k] = 0
	
	pks = np.where(np.sum(LSM[0:np.argmin(np.sum(LSM, 1)), :], 0)==0)[0]
	return pks


def ampdFast(sigInput, order, LSMlimit = 1):
	"""A slightly faster version of AMPD which divides the signal in 'order' windows
		Parameters
		----------
		sigInput: ndarray
			The 1D signal given as input to the algorithm
		order: int
			The number of windows in which sigInput is divided
		Returns
		-------
		pks: ndarray
			The ordered array of peaks found in sigInput 
	"""
	# Check if order is valid (perfectly separable)
	if(len(sigInput)%order != 0):
		print("AMPD: Invalid order, decreasing order")
		while(len(sigInput)%order != 0):
			order -= 1
		print("AMPD: Using order " + str(order))

	N = int(len(sigInput) / order / 2)

	# Loop function calls
	for i in range(0, len(sigInput)-N, N):
		print("\t sector: " + str(i) + "|" + str((i+2*N-1)))
		pksTemp = ampd(sigInput[i:(i+2*N-1)], LSMlimit)
		if(i == 0):
			pks = pksTemp
		else:
			pks = np.concatenate((pks, pksTemp+i))
		
	# Keep only unique values
	pks = np.unique(pks)
	
	return pks

def vrange(locs,  fw = 30, bw = 30):
    """Create an array of segments of length w from provided array 
    indices (locs). Note that returned indices are not limited to any 
    spesific end point

    Parameters:
        signal_in (1-D array): array from which segments are extracted
        locs (1-D array): location indices for segments
        fw (integer): forwardd window length in samples
        bw (integer): backward window length in samples

    Returns:
        numpy.ndarray: extracted ranges of size (fw+bw x len(locs))        
    """
    
    fw = int(fw)
    bw = int(bw)
    locs_w1 = locs - bw
    locs_w2 = locs + fw
    
    #Lengths of each range
    l = locs_w2 - locs_w1     
    #concatenated indices for each window
    out = np.repeat(locs_w2 - l.cumsum(), l) + np.arange(l.sum())
       
    return out.reshape(-1,bw + fw).T  

def get_max(signal_in, locs, fw = 30, bw = 30):
    """Get maximum value in segments extracted from signal_in. Vectorized
    implementation.
    
    Parameters
    ----------
    signal_in : 1-D array 
        array from which maxmas are found
    locs : 1-D array
        array defining window locations
    fw : integer
        forwardd window length in samples
    bw : integer 
        backward window length in samples

    Returns
    -------
    locs : 1-D array
        array of maximums found in 'signal_in' based indices in 'locs' 
        
    
    
    #Naive implementation    
    out = []
    for i in range(0, len(locs)):
        start = max(0, locs[i] - 30) 
        stop  = min(locs[-1], locs[i] + 30)        
        tmp   = np.argmax(signal_in[start : stop])
        out.append(tmp + start)    
    
    """
    
    S = vrange(locs, fw, bw)

    #remove colums (signal segments) that overflow signal_in
    S = S[:, S.min(axis=0) >= 0] 
    S = S[:, S.max(axis=0) < len(signal_in)] 
    
    #find max from each segment
    locs = np.argmax(signal_in[S], axis=0)
    
    #refer back signal_in indices
    starts = S[0,:]     
    locs = locs + starts 
   
    return locs

def signal_envelope_triang(series):
    from scipy import signal
    dsrate=1
    nfir=6
    nfir2=8
    bhp=-(np.ones((1,nfir))/nfir)
    bhp[0]=bhp[0]+1
    blp=signal.triang(nfir2)
    final_filterlength=np.round(51/dsrate)
    finalmask=signal.triang(final_filterlength)
    series_filt_hp=signal.lfilter(bhp[0,:],1, series)
    series_filt_lp=signal.lfilter(blp,1, series_filt_hp)
    series_env=signal.lfilter(finalmask,1,np.abs(series_filt_lp))

    
    return series_env

def enhance_envelope(data, window='Triang', w_len = 100, w_std = 5):
    #w_len was 30 changed 14.9.2019
    """Enhance signal envelope. First signal is squared element wise. 
    Secondly a convolution of spesific window shape is applied for smoothing.
  
    Parameters:
        signal_in (1-D array): array 
        window: (str) either 'Gaussian' or 'Triang'
        w_len = length of smoothing window
        w_std = defines Gaussian window shape
        
    Returns:
        numpy.ndarray: extracted ranges of size (fw+bw x len(locs))        
    """
    if window == 'Gaussian':
        w = signal.gaussian(w_len, w_std)
    else:
        w = signal.triang(w_len)  
    
    x = lambda signal_in : np.convolve(w, np.sqrt(signal_in**2), 'same')
    #x = lambda signal_in: signal_envelope_triang(signal_in)
      
    if type(data)==dict: 
        for key in data:    
            data[key] = x(data[key])
        return data
    else: 
        return x(data)
        
def search_back(signal_in, locs):
    #not implemented
    return locs

#TODO understand
def get_quality_peaks(signal_in, locs, fw = 70, bw = 50, plot_ensemble = 0):
    """ Checks signal quality based on found peaks by summing 
    correlation coefficients obtained by correlating waveforms from each peak
    to median waveform. 
        
    Keyword arguments:
    signal_in -- input vector 
    locs -- peak locations (samples)
    fw -- forward length of a window
    bw -- backward length of a window
    fs -- sample frequency
    check_max -- find nearest max of each peak (0 = disable, 1 = enable)
    plot_ensemble -- plot all peak waveforms (0 = disable, 1 = enable)
    
    Returns:
    waveforms -- ndarray; window length waveforms of locs 
    waveform -- narray; median waveform of waveforms
    xcorrs -- narray; Pearson correlation coefficients of each waveforms
    againt the median waveform
    quality -- mean of xcorrs
    index -- indices of selected xcorrs   
    """       
    
    waveforms = np.zeros((fw + bw, len(locs)))
    indices     = np.empty(len(locs))
    indices[:]  = np.nan
    
    for i in range(len(locs)):
        start = locs[i] - bw
        stop  = locs[i] + fw
        x = signal_in[start : stop]
        if len(x) == bw + fw: #include only full waveforms
            waveforms[:,i] = x
            indices[i] = i
    
    waveform = np.median(waveforms, axis=1)
    epsilon = 1e-15 #zero division hack
            
    xcorrs = np.empty(waveforms.shape[1])
    for i in range(waveforms.shape[1]):
        xcorrs[i] = np.corrcoef(waveform + epsilon, waveforms[:,i] + epsilon)[1,0]
    quality = np.nanmean(xcorrs)   
   
    if plot_ensemble == 1:
        plt.figure()
        plt.plot(waveform, linewidth = 2.0)
        plt.text(0, 1, quality, fontsize = 20)
        for i in range(0, waveforms.shape[1]):
            plt.plot(waveforms[:,i])    
    return waveforms, waveform, quality, indices



if __name__ == '__main__':
    
    fullpath = os.path.join(os.getcwd(), 'test_data', 'MODE_AF_DATA',
                                         'physiological_signals', 
                                         'PHYSICIAN_RECORDED',
                                         'AFIB_or_SR') 
    _, test_data = load_data_mat(fullpath)   

    keys = [key for key in test_data]
    
    dataset1 = copy.deepcopy(test_data)
    dataset2 = copy.deepcopy(test_data)

    data1 = dataset1[keys[0]]
    data2 = dataset2[keys[0]]
    
    plot_data(data1)
    
    signal_, locs = detect_peaks(data1)

    plot_data(data1)
    plot_data(data2)






