
import matplotlib.pyplot as plt
import os, json, base64
import numpy as np
import scipy.io
from array import array
from itertools import groupby
from preprocessor import create_algorithm_input,resample_data
from scipy import signal
from scipy import interpolate


def json_decode(data):

    #first check order of sensor data
    if data[0]['sensordata_set'][0]["sensor_type"] == "GYROSCOPE":
        accDataIndex = 1
        gyrDataIndex = 0  
    else:
        accDataIndex = 0
        gyrDataIndex = 1

    #Get timestamp information
    accTS = array('Q') #unsigned long long, 8 bytes
    accTS.frombytes(base64.b64decode(data[0]['sensordata_set'][accDataIndex]['timestamp']))
    gyrTS = array('Q') #unsigned long long, 8 bytes
    gyrTS.frombytes(base64.b64decode(data[0]['sensordata_set'][gyrDataIndex]['timestamp']))
    
    #Get accelerometer data
    accX = array('f') #unsigned long long, 8 bytes
    accX.frombytes(base64.b64decode(data[0]['sensordata_set'][accDataIndex]['x']))
    accY = array('f') #unsigned long long, 8 bytes
    accY.frombytes(base64.b64decode(data[0]['sensordata_set'][accDataIndex]['y']))
    accZ = array('f') #unsigned long long, 8 bytes
    accZ.frombytes(base64.b64decode(data[0]['sensordata_set'][accDataIndex]['z']))
    accelerometer_data_raw = np.array((accX, accY, accZ, accTS)).T

    #Get gyroscope data
    gyrX = array('f') #unsigned long long, 8 bytes
    gyrX.frombytes(base64.b64decode(data[0]['sensordata_set'][gyrDataIndex]['x']))
    gyrY = array('f') #unsigned long long, 8 bytes
    gyrY.frombytes(base64.b64decode(data[0]['sensordata_set'][gyrDataIndex]['y']))
    gyrZ = array('f') #unsigned long long, 8 bytes
    gyrZ.frombytes(base64.b64decode(data[0]['sensordata_set'][gyrDataIndex]['z']))
    gyroscope_data_raw = np.array((gyrX, gyrY, gyrZ, gyrTS)).T
    
    return  accelerometer_data_raw, gyroscope_data_raw

def load_data_json(fullpath):
    data_raw = {}
    for filename in os.listdir(fullpath):
        with open(os.path.join(fullpath, filename), 'r') as file:
            data = json.load(file)
        data_raw[filename] = json_decode(data)
    
    dataset = {}
    keys = [key for key in data_raw]
    for key in keys: 
        data = data_raw[key]
        dataset[key] = create_algorithm_input(data[0], data[1])
    print('Loaded json' + fullpath)           
    return keys, dataset


def load_data_mat(fullpath):
    data_raw = {}
    for filename in os.listdir(fullpath):
        matfile = scipy.io.loadmat(os.path.join(fullpath,filename))
        data_raw[filename] = (matfile['accdata'], matfile['gyrodata'])    

    dataset = {}
    keys = [key for key in data_raw]
    for key in keys: 
        data = data_raw[key]
        dataset[key] = create_algorithm_input(data[0], data[1])
    print('Loaded mat' + fullpath)
    return keys, dataset

def load_data_whisper(fullpath):
    data_raw = {}
    for filename in os.listdir(fullpath):
        matfile = scipy.io.loadmat(os.path.join(fullpath,filename))
        data_raw[filename] = (matfile['accdata'], matfile['gyrodata'])    

    dataset = {}
    keys = [key for key in data_raw]
    for key in keys: 
        data = data_raw[key]
        a_raw = np.array(data[0])[10:,:]
        g_raw = np.array(data[1])[10:,:] 
        ax = a_raw[:,0]
        ay = a_raw[:,1]
        az = a_raw[:,2]
        gx = g_raw[:,0]
        gy = g_raw[:,1]
        gz = g_raw[:,2]

        assert(len(ax)==len(ay)==len(az)==len(gx)==len(gy)==len(gx))
    
        dataset[key] = { 'ax': ax, 'ay': ay, 'az': az, 'gx': gx, 'gy': gy, 'gz': gz}
    print('Loaded mat' + fullpath)
    return keys, dataset




def plot_data(data, title = 'empty title'):
    #plt.figure()
    fig, axs = plt.subplots(6)
    fig.suptitle(title)
    for i, key in enumerate(data):
        axs[i].plot(data[key])
    

def load_json_from_dict(fullpath):
    fname=os.listdir(fullpath) 
    dict_of_data = {}
    for item in fname:
        with open(os.path.join(fullpath,item), 'r') as data_file:
            dict_of_data[item] = json.load(data_file) 
            del item
                    
        dict_tmp=dict_of_data[fname[0]]
            
        
    user_id_all=[]
    for i in range(len(dict_tmp)):
        user_id_all.append(dict_tmp[i]['user_id'])
        
    my_dict = {i:user_id_all.count(i) for i in user_id_all}
#            
    rep_user=[len(list(group)) for key, group in groupby(user_id_all)]
        
        
    user_id_ext=dict()
    key_ids=dict()
    for key in my_dict.keys():
        for i in range(len(rep_user)):
            string = "_"
            user_id_ext[i]=[string+str(i) for i in range(rep_user[i])]
            key_ids[key]=key
            
    new_key_ids = {i: str(key_ids[k]) for i, k in enumerate(sorted(key_ids.keys()))}
  
    new_id_users=dict()  
    for key,value in user_id_ext.items():
      new_id_users[key]= ["{}{}".format(new_key_ids[key],i) for i in value]
          
    list_of_key_names = [y for x in new_id_users.values() for y in x]
    
    dict_of_MEMS_data = {}
    for i in range(len(dict_tmp)):
        MEMS_dict = dict()
        accTS = array('Q') #unsigned long long, 8 bytes
        accTS.frombytes(base64.b64decode(dict_tmp[i]['accel_time']))
        gyrTS = array('Q') #unsigned long long, 8 bytes
        gyrTS.frombytes(base64.b64decode(dict_tmp[i]['gyro_time']))
        accX = array('f') #unsigned long long, 8 bytes
        accX.frombytes(base64.b64decode(dict_tmp[i]['accel_x']))
        accY = array('f') #unsigned long long, 8 bytes
        accY.frombytes(base64.b64decode(dict_tmp[i]['accel_y']))
        accZ = array('f') #unsigned long long, 8 bytes
        accZ.frombytes(base64.b64decode(dict_tmp[i]['accel_z']))
        accelerometer_data_raw = np.array((accX, accY, accZ, accTS)).T
        gyrX = array('f') #unsigned long long, 8 bytes
        gyrX.frombytes(base64.b64decode(dict_tmp[i]['gyro_x']))
        gyrY = array('f') #unsigned long long, 8 bytes
        gyrY.frombytes(base64.b64decode(dict_tmp[i]['gyro_y']))
        gyrZ = array('f') #unsigned long long, 8 bytes
        gyrZ.frombytes(base64.b64decode(dict_tmp[i]['gyro_z']))
        gyroscope_data_raw = np.array((gyrX, gyrY, gyrZ, gyrTS)).T    
        
        MEMS_dict['acc'] = accelerometer_data_raw
        MEMS_dict['gyro'] = gyroscope_data_raw
        dict_of_MEMS_data[list_of_key_names[i]] = MEMS_dict
#                list_of_key_names = list(dict_of_data.keys()) 
    keys = [key for key in dict_of_MEMS_data]
    dataset=dict()
    for key in keys: 
        data = dict_of_MEMS_data[key]
        dataset[key] = create_algorithm_input(data['acc'], data['gyro'])
    print('Loaded json' + fullpath)           
    return keys, dataset
                    


#remove and put to testbench_test_helpers
if __name__ == '__main__': 
    
    plt.close('all')    
    
    if not 'run_json' in locals(): 
        run_json = True
        fullpath = os.path.join(os.getcwd(), 'test_data', 'HRtest')
        keys_json, dataset_json = load_data_json(fullpath)
        testi = dataset_json[keys_json[0]]
        plt.figure()
        plt.plot(testi['gy'])
    
    if not 'run_mat' in locals():
        run_mat = True
        fullpath = os.path.join(os.getcwd(), 'test_data', 'matfiles')
        keys_mat, dataset_mat = load_data_mat(fullpath)
        testi = dataset_mat[keys_mat[0]]
        plt.figure()
        plt.plot(testi['gy'])



