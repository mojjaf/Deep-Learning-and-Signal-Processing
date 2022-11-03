import pandas as pd
import numpy as np

FS = 200

# TODO overall

def remove_artifacts(signal_in, window):

    rolled_std = rolling_std(signal_in, window)
    free_indices, free_signal, arti_indices, arti_signal = \
        remove_outliers(rolled_std, multiplier = 2)

    def flattener(to_flatten):
        return np.array([item for sublist in to_flatten for item in sublist])

    return flattener(free_indices), flattener(free_signal),\
           flattener(arti_indices), flattener(arti_signal)

# TODO understand
def remove_outliers(signal_in, low=25, high=75, multiplier=1.5, length=FS*2.5):
    """ Removes artifacts from signal based on outlier amplitudes.
        
    Keyword arguments:
    signal_in -- input vector 
    multiplier -- iqr multiplier, relates to outlier bounds
    low -- lower percentile limit for artifact removal
    high -- upper percentile limit for artifact removal
    length -- min lenght of remaining segments (samples)

    Returns:  
    indice_list -- list, each element contains indices of one segment
    signal_list -- list, each element contains signal values of one segment
    """     
    
    sample_time = np.arange(len(signal_in))
    
    q1 = np.percentile(signal_in, low)
    q3 = np.percentile(signal_in, high)
    iqr = q3 - q1
    
    lower_bound = q1 - (multiplier * iqr) 
    upper_bound = q3 + (multiplier * iqr)
    
    u = sample_time[signal_in < upper_bound]
    l = sample_time[signal_in > lower_bound]
    
    clean = np.intersect1d(u, l) #artifact free indices    
    artif = np.setdiff1d(sample_time, clean)
    
    if clean.size == 0: #only artifacts
        
        free_values = []
        free_indices = []        
        arti_values = [signal_in[artif]]
        arti_indices = [artif]
    
    elif artif.size == 0: #clean
        
        free_values  = [signal_in[clean]]
        free_indices = [clean]        
        arti_values  = []
        arti_indices = []        
    
    else: #artif.size != 0:
    
        edges1 = clean[1:][np.diff(clean) > 1] 
        edges2 = artif[1:][np.diff(artif) > 1] 
                
        first_arti = np.array([artif[0]])    
        final_arti = np.array([artif[-1]])
        
        first_free = np.array([clean[0]])    
        final_free = np.array([clean[-1]])
        
        edges = np.concatenate([first_arti, first_free, edges1, edges2,\
                                final_arti, final_free],axis=0)
        
        edges = np.unique(edges)
        
        free_indices = []; free_values = []
        arti_indices = []; arti_values = []
        
        for i in range(edges.size-1):   
            segment = sample_time[edges[i]:edges[i+1]]       
            if (len(segment) > length) and (segment[0] in clean):
                free_indices.append(segment)
                free_values.append(signal_in[segment])
            else:
                arti_indices.append(segment)
                arti_values.append(signal_in[segment])
                        
    return free_indices, free_values, arti_indices, arti_values 

def rolling_std(signal_in, window):
    
    pd_series = pd.Series(signal_in)
    rolling_std = pd_series.rolling(window, min_periods=1, center=True).std()
    return rolling_std.values 

    
if __name__ == '__main__':
    
    import os
    import matplotlib.pyplot as plt
    from test_helpers import plot_data
    from test_helpers import load_data_json
    
    plt.close('all')
    FS = 200
    fullpath = os.path.join(os.getcwd(), 'test_data', 'HRtest')
    _, test_data = load_data_json(fullpath) 
    keys = [key for key in test_data]
    keys.sort()
    
    
    data = test_data[keys[12]]
    signal_in = data['az']
    #plt.plot(signal_in)
    signal_in = np.convolve(signal_in, np.ones(1000), 'same')
    free_ind, free_sig, art_ind, art_sig = remove_artifacts(signal_in,500)
    try:
        plt.plot(art_ind, signal_in[art_ind])
    except:
        None    
    try:
        plt.plot(free_ind, signal_in[free_ind])
    except:
        None
    print(len(free_ind)/(len(free_ind)+len(art_ind)))
    
    
