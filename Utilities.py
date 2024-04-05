# -*- coding: utf-8 -*-
"""
Created on Tue Nov 15 19:13:22 2022

@author: lfl


"""


from tqdm import tqdm
import numpy as np
import glob
import os
import pandas as pd

#%% make_progress_meter
# display progress bar and send slack notification
def make_progress_meter(n_handle, n_total):

    # initialize counter
    n0 = 0

    with tqdm(total = n_total, position = 0, leave = True) as progress_bar:

        while(n_handle.is_processing()):

            n_handle.wait_for_values(1)
            n = n_handle.fetch_all() # retrieve iteration value n
            Δn = n - n0

            if Δn > 0:
                progress_bar.update(Δn)
                n0 = n

        
#%% round_to_clk
def clk(num):
    return round(num/4)

def counter(directory,experiment:str='spec',element:str='rr',extension:str='*.csv'):
    try:
        list_of_files = glob.glob(os.path.join(directory,experiment,element,extension))
        latest_file = max(list_of_files, key=os.path.getctime)
        iteration = int(latest_file[-7:-4].lstrip('0')) + 1
    except:
        iteration = 1

    return iteration

#%% get_dt
def get_dt(target_fs):
    return 1/(7 * target_fs * 1e-9)

#&& round_from_uncertainty
def round_from_uncertainty(num, unc, scale = 1):

    # get number of significant digits
    sigdit = int(np.log10(scale) - np.log10(unc))
    scaled_num = round(num / scale, ndigits = sigdit)
    return scaled_num * scale

def save_datadict_to_fgroup(f, name, datadict):
    subgroup = f.create_group(name, track_order=True)
    for key in datadict.keys():
        if key != 'metadata':
            dset_q = subgroup.create_dataset(key, data=datadict[key], track_order=True)
   
        else:
            for k in datadict['metadata'].keys():
                subgroup.attrs[k] = datadict['metadata'][k]
    print(f'write dataset to {name}')
    
def get_index_for_filename(saveDir, filename,file_format='h5'):
    try:
        list_of_files = glob.glob(f'{saveDir}\\{filename}*.{file_format}')
        latest_file = max(list_of_files, key=os.path.getctime)
        iteration = int(latest_file.split('.')[-2].split('_')[-1]) + 1
    except:
        iteration = 1
    return iteration

def IQ_imbalance(g,phi):
    c = np.cos(phi)
    s = np.sin(phi)
    N = 1 / ((1-g**2)*(2*c**2-1))
    return [float(N * x) for x in [(1-g)*c, (1+g)*s, (1-g)*s, (1+g)*c]]

def convert_V_to_dBm(x):
    '''
    converts from rms voltage to dBm, assuming 50 Ohm. Voltage is in V
    '''
    return 10*np.log10(x**2*1000/50.)
# def save_datadict_to_csv(f, name, datadict)