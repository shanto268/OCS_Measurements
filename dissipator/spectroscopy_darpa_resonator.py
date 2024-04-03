# -*- coding: utf-8 -*-
"""
Created on Fri Apr 28 16:27:20 2023

@author: lfl
"""

"""
resonator spectroscopy
"""
from dissipator import *
import instrument_init as inst
import os
import pandas as pd

device = 'darpa2A'
qb = dissipator(device, device_name=device)
qb.update_value('rr_freq', 6.46e9)
qb.update_value('rr_LO', 6.3e9)
qb.update_value('rr_IF', qb.pars['rr_freq'] -qb.pars['rr_LO'] )
qb.update_value('rr_atten', 30)
qb.update_value('diss_freq', 7.97e9)
qb.update_value('qubit_freq', 2.2681e9)
qb.update_value('qubit_LO', 2.2e9)
qb.update_value('qubit_IF',qb.pars['qubit_freq'] - qb.pars['qubit_LO'] )

amp_q_scaling = 0.9
n_avg = 2000
# save data
today = datetime.today()
sDate =  today.strftime("%Y%m%d")
bCalibrateRo = True
saveDir = f'G:\\Shared drives\\CavityCooling\data\\{device}\\{sDate}\\spectroscopy\\diss_spec'
if not os.path.exists(saveDir):
    Path(saveDir).mkdir(parents=True, exist_ok=True)
filename = f'dissSpec_amp_q_scale={amp_q_scaling}_DA={qb.pars["rr_atten"]}dB_navg={n_avg}_calibrate={str(bCalibrateRo)}'
index = get_index_for_filename(saveDir, filename)

start_flux = -70e-6
stop_flux = 10e-6
step_size = 10e-6
flux_list = np.arange(start_flux, stop_flux+step_size/2, step_size)

rrTableFile = 'G:\\Shared drives\\CavityCooling\\data\\diss09_6024\\20230509\\spectroscopy\\diss_spec\\rr_freq_calibrate.csv'
if not bCalibrateRo:
   df = pd.read_csv(rrTableFile)
   fluxes = np.array(df.loc[:,'flux'])
   rrFreqs = np.array(df.loc[:,'rr_freq'])

with h5py.File(f'{saveDir}\\{filename}_{index}.h5','w') as hf:
    now = datetime.now()
    timestamp = now.strftime("%H:%M:%S")
    g_rr = hf.create_group(f'rrSpec_{timestamp}')
    g_diss = hf.create_group(f'dissSpec_{timestamp}')
    # base flux for gain profile calibration
    if bCalibrateRo:
        base_flux = -10e-6 #56e-6
        inst.set_flux_bias(base_flux, step = 2e-6, lower_bound=-1e-3, upper_bound=1e-3)
        I0, Q0, freqs, job = qb.resonator_spec(f_LO=qb.pars['rr_LO'],atten=qb.pars['rr_atten'],IF_min=39e6,IF_max=53e6,df=0.1e6,n_avg=n_avg,savedata=True,fit=False)
        dataDict = {'metadata': {'flux': base_flux,
                                 'rr_freq': qb.pars['rr_freq'],
                                 'rr_LO': qb.pars['rr_LO'],
                                 'rr_IF': qb.pars['rr_IF'],
                                 'report': str(job.execution_report())},
                    'freqs': freqs,
                    'I': I0,
                    'Q': Q0,}
        save_datadict_to_fgroup(g_rr, f'base flux = {int(base_flux*1e6)} uA', dataDict)
    
    for flux in tqdm(flux_list):
        inst.set_flux_bias(flux, step = 2e-6, lower_bound=-1e-3, upper_bound=1e-3)
        if bCalibrateRo:
            I, Q, freqs, job = qb.resonator_spec(f_LO=qb.pars['rr_LO'],atten=qb.pars['rr_atten'],IF_min=39e6,IF_max=53e6,df=0.1e6,n_avg=n_avg,savedata=True,fit=False)
            fc,fwhm = pf.fit_res(freqs,np.abs((I-I0) + 1j*(Q-Q0)))
            qb.update_value("rr_freq", fc)
            qb.update_value("rr_IF", fc-qb.pars['rr_LO'])
        else:
            # read rr_freq from the look up table
            rr_freq = rrFreqs[np.argmin(abs(fluxes-flux*1e6))] * 1e9 
            qb.update_value('rr_freq', rr_freq)
            qb.update_value('rr_IF', qb.pars['rr_freq'] - qb.pars['rr_LO'])
            I, Q, freqs, job = qb.resonator_spec(f_LO=qb.pars['rr_LO'],atten=qb.pars['rr_atten'],IF_min=39e6,IF_max=53e6,df=0.1e6,n_avg=n_avg,savedata=True,fit=False, fc =rr_freq)
        dataDict = {'metadata': {'flux': flux,
                                 'rr_freq': qb.pars['rr_freq'],
                                 'rr_LO': qb.pars['rr_LO'],
                                 'rr_IF': qb.pars['rr_IF'],
                                 'report': str(job.execution_report())},
                    'freqs': freqs,
                    'I': I,
                    'Q': Q,}
        save_datadict_to_fgroup(g_rr, f'flux = {int(flux*1e6)} uA', dataDict)
        
        I,Q,freqs,job = qb.run_scan(df = 5e6, n_avg = n_avg, element='qubit', chunksize = 500e6,  lo_min = 8e9, lo_max = 11e9, amp_q_scaling = amp_q_scaling, saturation_dur = 20e3, showprogress=True, res_ringdown_time = int(4e3), check_mixers=False)
        dataDict = {'metadata': {'flux': flux,
                                 'rr_freq': qb.pars['rr_freq'],
                                 'rr_LO': qb.pars['rr_LO'],
                                 'rr_IF': qb.pars['rr_IF'],
                                 'report': str(job.execution_report())},
                    'freqs': freqs,
                    'I': I,
                    'Q': Q,}
        save_datadict_to_fgroup(g_diss, f'flux = {int(flux*1e6)} uA', dataDict)