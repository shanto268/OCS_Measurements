"""
Created on Mon Oct 4 2022

@author: Evangelos Vlachos <evlachos@usc.edu>
"""
import csv
import glob
import json
import os
import time
import warnings
from collections import OrderedDict
from datetime import date, datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
# from waveform_tools import *
from qm import LoopbackInterface, SimulationConfig, generate_qua_script
from qm.logger import logger
from qm.qua import *
from qm.QuantumMachinesManager import QuantumMachinesManager
from qualang_tools.analysis.discriminator import two_state_discriminator
from qualang_tools.loops import from_array
from scipy.signal import savgol_filter
from scipy.signal.windows import gaussian

import plot_functions as pf
from config import Configuration
from helper_functions import save_data
from Instruments import instruments
from sequence import *
from Utilities import *

logger.setLevel(level='WARNING')
device = 'ocsqp1'
today = date.today()
sDate =  today.strftime("%Y%m%d")


class qubit():
#%% INITIALIZATION

    #%%% default_pars
    default_pars = {
                    # overall setup parameters
                    'elements' :                    ['qubit', 'rr'],
                    "qubit_LO":                     5e9,
                    "rr_LO":                        6e9,
                    'readout_atten':                25,         # default attenuation in dB on variable attenuator

                    #OPX settings and connections
                    'host':                         None,       # OPX IP address (eventually, import from local file)
                    'port':                         '9510',       # OPX port (eventually, import from local file)
                    'Iout':                         {'qubit': 3, 'rr': 1},          # OPX output (goes to modulation mixer input I)
                    'Qout':                         {'qubit': 4, 'rr': 2},          # OPX output (goes to modulation mixer input Q)
                    'Iin':                          1,          # OPX input (comes from demodulation mixer output I)
                    'Qin':                          2,          # OPX input (comes from demodulation mixer output Q)
                    'AWG_trigger_out' :                     1,          # OPX digital marker output port for triggering AWG
                    'controller':                   'con1',     # OPX controller name

                    # Instrument settings (except OPX)
                    "qubit_LO":                     int(4.48e9),
                    "rr_LO":                        int(6.42e9),
                    "readout_atten":                25,

                    # other measurement setup choices
                    'n_avg':                        100,        # number of averages for zero-deadtime measurement
                    'rr_IF':                        50e6,
                    'qubit_IF':                     50e6,
                    "gauss_len":                    48,
                    "gauss_amp":                    0.45,
                    "amp_r":                        0.45,
                    "readout_pulse_len_in_clk":     500,        # length of readout integration weights in clock cycles
                    "saturation_duration" :         clk(10e3),  # saturation duration for spectroscopy
                    'readout_length':                  2000,  # length of a normal readout in ns,

                    # qubit parameters
                    "qubit_freq":                   int(4.5129e9),
                    "rr_freq":                      int(6.2e9),

                    # calibrated setup parameters
                    "analog_input_offsets":         [0,0],
                    "analog_input_gain":           3,
                    "rr_mixer_offsets":            [0,0],
                    "qubit_mixer_offsets":          [0,0],
                    "rr_mixer_imbalance":          [0,0],
                    "qubit_mixer_imbalance":        [0,0],
                    "tof":                         260,       # time of flight in ns
                    "smearing":                    40,         # smearing in ns
                    # processing choices
                    "n_avg" :                       500,
                    "IQ_rotation":                  -0/180*np.pi,  # phase rotation applied to IQ data
                    "switch_weights" :              False,

                    # calibrated device parameters
                    "resettime" :                   {"qubit": clk(100e3), "rr" : clk(5e3)},
                    'kappa':                        200e3,      # resonator linewidth
                    'readout_freq':                     6.2e9,        # resonator center frequency
                    'Q':                            9000,       # resonator quality factor
                    'Qc':                           9000,       # resonator coupling quality factor

                    # waveform parameters
                    'operations':                   dict(qubit=['const','gauss','arb_op','X180','Y180','X90','Y90'],
                                                         rr=['const', 'readout']),
                    'X180_len':                       160,        # length of pi pulse in clock cycles
                    'X180_amp':                       0.45,       # amplitude of pi pulse
                    'X90_len':                          80,         # length of pi/2 pulse in clock cycles
                    'X90_amp':                          0.45,       # amplitude of pi/2 pulse
                    'gauss_len':                        48,         # length of gaussian pulse in clock cycles
                    'amp_q':                            0.375,       # amplitude of qubit pulse, needs to be less than 0.45 to prevent overflowing 
                    'amp_r':                            0.375,       # amplitude of readout pulse, needs to be less than 0.45
                    'arb_op_len':                       160,        # length of arbitrary operation in clock cycles
                    }

    #%%% __init__
    def __init__(self, qb, initialize_qmm=True):
        # load pars from json, OR create new json file
        self._name = qb
        try:
            print('Loading parameter JSON file')
            with open(f'{qb}_pars.json', 'r') as openfile:
                self.pars = json.load(openfile)

            # compare keys
            default_keys = set(self.default_pars.keys())
            keys = set(self.pars.keys())

            # find all the keys in default_pars that are not in pars, and add them to pars w/ default value
            for k in (default_keys - keys):
                self.add_key(k, self.default_pars[k])

            # find all the keys in pars that are not in default_pars, and remove them from pars
            for k in (keys - default_keys):
                self.remove_key(k)


        except FileNotFoundError:
            print('Parameter file not found; loading parameters from template')
            self.pars = self.default_pars

        self.init_quantum_machine(initialize_qmm)
        self.write_pars()
        self._directory = f'G:\\Shared drives\\Quasiparticles\\OCS_QP\data\\{self._name}'
        self._instruments = instruments()
        self.init_instruments()
        # self.make_config(self.pars)
        self.config_maker = Configuration(self)
        self.config = self.config_maker.make_config()
        # self.make_config(self.pars)




#%% EXPERIMENTS
 #%%% play_pulses
    def play_pulses(self, amplitude = 1, element = 'readout'):
        """
        
        DESCRIPTION:
            Same as in original code. Plays constant pulses at correct IF frequencies on `element`.
        
        ARGS:
            None
            
        KWARGS:
            amp (int, 1): Amplitude scaling factor.
            
            
        RETURNS: 
            qm (?): ?
        """
        with program() as play_pulses:
            with infinite_loop_():
                play("const" * amp(amplitude), element)

        initialized = self.qmm is not None 
        qmm = self.qmm if initialized else QuantumMachinesManager(host=self.pars['host'], port=self.pars['port'])
        qm = qmm.open_qm(self.config)
        job = qm.execute(play_pulses)

        return qm,job

    #%%% punchout
    def punchout(self,
                 df = 0.1e6,
                 IF_min = 10e6,
                 IF_max = 20e6,
                 attenuations = [10,30],
                 atten_step = 0.1,
                 savedata=True):
        """
        Executes punchout measurement for list of resonator frequencies

        Args:
            df (TYPE, optional): DESCRIPTION. Defaults to 0.1e6.
            n_avg (TYPE, optional): DESCRIPTION. Defaults to 500.
            atten_range (TYPE, optional): [min,max] attenuation values.
            f_LO (TYPE, optional): list with all the resonator frequencies in GHz. Defaults to [6e9,7.2e9].
            res_ringdown_time (TYPE, optional): DESCRIPTION. Defaults to int(4e3).

        Returns:
            None.

        """
        iteration = counter(self._directory,self.experiment,element='rr',extension='*.csv')
        data = dict(I=[],Q=[],freqs=[],mag=[],z_data=[])
        data['attenuations'] = attenuations

        for a in tqdm(attenuations):
            print(f'Attenuation = {a} dB')
            self._instruments.set('DA','attenuation',a)
            spec_data,job = self.resonator_spec(f_LO=self.pars['rr_LO'],IF_min=IF_min,IF_max=IF_max,df=df,showprogress=True,savedata=False,fit=False)
            data['freqs'].append(np.around(spec_data['freqs']*1e-9,5))
            data['I'].append(spec_data['I'])
            data['Q'].append(spec_data['Q'])
            data['mag'].append(np.abs(spec_data['I']+1j*spec_data['Q']))
            data['z_data'].append(spec_data['I']+1j*spec_data['Q'])

        self._instruments.set('DA','attenuation',self.pars['readout_atten'])

        if savedata:
            metadata = dict(timestamp = datetime.now(),
                            qubit_pars = self.pars,
                            attenuation_range = attenuation_range,
                            atten_step = atten_step,)
            dataPath =  f'{saveDir}\\{self.experiment}\\rr'
            data = dict(I=I,Q=Q,freqs=freq_arr,mag=mag)
            save_data(dataPath, iteration, metadata, data)


        return data, job

    #def find_dispersive_shift(self, ):
        
    #%%% run_scan
    def run_scan(self,
                 df = 0.1e6,
                 element='resonator',
                 check_mixers=False,
                 chunksize = 200e6,
                 lo_min = 6e9,
                 lo_max = 7e9,
                 on_off=True,
                 amp_q_scaling = 1,
                 saturation_dur = 20e3,
                 showprogress=False,
                 savedata=False):
        """
        Scans a broad range of frequencies in search for qubits/resonators

        Args:
            IF_min (TYPE, optional): DESCRIPTION. Defaults  0.1e6.
            IF_max (TYPE, optional): DESCRIPTION. Defaults to 400e6.
            df (TYPE, optional): DESCRIPTION. Defaults to 0.1e6.
            n_avg (TYPE, optional): DESCRIPTION. Defaults to 500.
            res_ringdown_time (TYPE, optional): DESCRIPTION. Defaults to int(4e3).

        Returns:
            I (TYPE): DESCRIPTION.
            Q (TYPE): DESCRIPTION.
            freq_arr: DESCRIPTION.
            TYPE: DESCRIPTION.

        """
        # today = datetime.today()
        # sDate =  today.strftime("%Y%m%d")
        # saveDir = f'G:\\Shared drives\\CavityCooling\data\\{self.name}\\{sDate}'
        
        # dataPath = f'{saveDir}\\spectroscopy\\{element}_spec'
        # filename = 'data'
        # iteration = get_index_for_filename(dataPath, filename, file_format='csv')
        iteration = counter(self._directory,self.experiment,element=element,extension='*.csv')

        freq_arr = []
        I = []
        Q = []
        reports = ''
        if lo_min != lo_max:
            numchunks = int((lo_max-lo_min)/chunksize) + 1
            lo_list = [i*chunksize+lo_min for i in range(numchunks)]
        else:
            numchunks = 1
            lo_list = [lo_min]

        for f in tqdm(lo_list):
            if element == 'resonator':
                data, job = self.resonator_spec(f_LO=f,IF_min=df,IF_max=chunksize,df=df,showprogress=showprogress,savedata=False)
            elif element == 'qubit':
                data,job = self.qubit_spec(f_LO=f,
                                                        amp_q_scaling=amp_q_scaling,
                                                        check_mixers=check_mixers,
                                                        on_off=on_off,
                                                        saturation_dur=saturation_dur,
                                                        showprogress=showprogress,
                                                        IF_min=df,IF_max=chunksize,df=df,
                                                        savedata=False)
                pf.qubit_spec_plot(data,qb_pars=self.pars,find_peaks=True, amp_q_scaling=amp_q_scaling)
            # elif element == 'fflqb':
            #     dataI, dataQ,freqs, job = self.fflt1_spec(spec='qb',f_LO=f, 
            #                                             IF_min=df,IF_max=chunksize,df=df,
            #                                             n_avg=n_avg, 
            #                                             amp_ffl_scaling=amp_q_scaling, 
            #                                             check_mixers= check_mixers,showprogress=True,
            #                                             savedata=True, flux=flux)
            
            # elif element == 'fflrr':
            #     dataI, dataQ,freqs, job = self.fflt1_spec(spec='rr',f_LO=f, 
            #                                             IF_min=df,IF_max=chunksize,df=df,
            #                                             n_avg=n_avg, 
            #                                             amp_ffl_scaling=amp_q_scaling, 
            #                                             check_mixers= check_mixers,showprogress=True,
            #                                             savedata=True, flux=flux)
                
            # elif element == 'diss':
            #     dataI, dataQ,freqs, job = self.diss_spec(f_LO=f, 
            #                                             IF_min=df,IF_max=chunksize,df=df,
            #                                             n_avg=n_avg, 
            #                                             amp_ffl_scale=amp_q_scaling, 
            #                                             check_mixers= check_mixers,
            #                                             savedata=False)
                
            freq_arr.extend(data['freqs'])
            I.extend(data['I'])
            Q.extend(data['Q'])
            reports += str(job.execution_report()) + '\n'
            
        
        if savedata:  
            dataPath = f'{self._directory}\{self.experiment}\\rr'
            data = {"I": I, "Q": Q, "freqs": freq_arr}
            save_data(dataPath, iteration, self.pars, data)

        data = dict(I=np.array(I),Q=np.array(Q),freqs=np.array(freq_arr))

        return data, job

    #%%% resonator_spec
    def resonator_spec(self, IF_min = 0.1e6,
                       f_LO = 7e9,
                       IF_max = 400e6,
                       df = 0.1e6,
                       showprogress=False,
                       savedata=True,
                       on_off=False,
                       flux = None,
                       **kwargs):
        """


        Args:
            IF_min (TYPE, optional): DESCRIPTION. Defaults to 0.1e6.
            f_LO (TYPE, optional): DESCRIPTION. Defaults to 7e9.
            IF_max (TYPE, optional): DESCRIPTION. Defaults to 400e6.
            df (TYPE, optional): DESCRIPTION. Defaults to 0.1e6.
            atten (TYPE, optional): DESCRIPTION. Defaults to 10.
            n_avg (TYPE, optional): DESCRIPTION. Defaults to 500.
            res_ringdown_time (TYPE, optional): DESCRIPTION. Defaults to int(4e3).
            port_type (TYPE, optional): DESCRIPTION. Defaults to 'notch'.
            fit (TYPE, optional): DESCRIPTION. Defaults to True.
            plot (TYPE, optional): DESCRIPTION. Defaults to True.
            savedata (TYPE, optional): DESCRIPTION. Defaults to True.

        Returns:
            I (TYPE): DESCRIPTION.
            Q (TYPE): DESCRIPTION.
            TYPE: DESCRIPTION.
            TYPE: DESCRIPTION.

        """
        iteration = counter(self._directory,self.experiment,element='rr',extension='*.csv')

        freqs = np.arange(IF_min, IF_max + df/2, df, dtype=int)
        
        self.update_value('rr_LO', value = f_LO)
        seq = sequence(self,'rr_spec',on_off=on_off,n_avg=self.pars['n_avg'], IF_min=IF_min, IF_max=IF_max, df=df,)
        rr_spec = seq.make_resonator_spec_sequence()

        datadict,job = self.get_results(rr_spec,result_names=["I","Q"],progress_key='n',showprogress=showprogress)

        I = np.array(datadict["I"])
        Q = np.array(datadict["Q"])
        z_data = I + 1j*Q
        freq_arr = np.array(freqs + self.pars['rr_LO'])
        data = dict(I=I,Q=Q,z_data=z_data,freqs=freq_arr)
        
        if savedata:
            dataPath = f'{self._directory}\{self.experiment}\\rr'
            data = {"I": I, "Q": Q, "freqs": freq_arr}
            save_data(dataPath, iteration, self.pars, data)

        return data, job



    def resonator_spec_wffl(self, IF_min = 0.1e6,
                       f_LO = 7e9,
                       IF_max = 400e6,
                       df = 0.1e6,
                       atten = 10,
                       n_avg = 500,
                       port_type = 'notch',
                       fit=True,
                       savedata=True,
                       flux = None,
                       simulate=True,
                       **kwargs):
        """
    
    
        Args:
            IF_min (TYPE, optional): DESCRIPTION. Defaults to 0.1e6.
            f_LO (TYPE, optional): DESCRIPTION. Defaults to 7e9.
            IF_max (TYPE, optional): DESCRIPTION. Defaults to 400e6.
            df (TYPE, optional): DESCRIPTION. Defaults to 0.1e6.
            atten (TYPE, optional): DESCRIPTION. Defaults to 10.
            n_avg (TYPE, optional): DESCRIPTION. Defaults to 500.
            res_ringdown_time (TYPE, optional): DESCRIPTION. Defaults to int(4e3).
            port_type (TYPE, optional): DESCRIPTION. Defaults to 'notch'.
            fit (TYPE, optional): DESCRIPTION. Defaults to True.
            plot (TYPE, optional): DESCRIPTION. Defaults to True.
            savedata (TYPE, optional): DESCRIPTION. Defaults to True.
    
        Returns:
            I (TYPE): DESCRIPTION.
            Q (TYPE): DESCRIPTION.
            TYPE: DESCRIPTION.
            TYPE: DESCRIPTION.
    
        """
    
        try:
            list_of_files = glob.glob(r'D:\weak_measurements\spectroscopy\resonator_spec_wffl\*.csv')
            latest_file = max(list_of_files, key=os.path.getctime)
            iteration = int(latest_file[-7:-4].lstrip('0')) + 1
        except:
            iteration = 1
    
        freqs = np.arange(IF_min, IF_max + df/2, df, dtype=int)
        # freqs_list = freqs.tolist()
        # set attenuation and change rr_LO freq
        inst.set_attenuator(attenuation=atten)
    
        self.update_value('rr_LO', value = f_LO)
        inst.set_rr_LO(self.pars['rr_LO'])
            
            
        seq = sequence('spec_wffl',n_avg=n_avg, IF_min=IF_min, IF_max=IF_max, df=df,)
        rr_spec_wffl = seq.make_sequence(self)
        if simulate:
            qmm = QuantumMachinesManager(host=host, port=port)
            job = qmm.simulate(config=self.config, program=rr_spec_wffl, simulate=SimulationConfig(duration=9000))
            job.get_simulated_samples().con1.plot()

    
        datadict,job = self.get_results(rr_spec_wffl,result_names=["I","Q","n"],showprogress=False)
    
        I = np.array(datadict["I"])
        Q = np.array(datadict["Q"])
        freq_arr = np.array(freqs + self.pars['rr_LO'])
    
        if fit:
            fc,fwhm = pf.fit_res(freq_arr,np.abs(I+1j*Q))
            print(f'Resonant Frequency: {fc*1e-9:.5f} GHz\nFWHM = {fwhm*1e-6} MHz\nkappa = {2*np.pi*fwhm*1e-6:.3f} MHz')
        else:
            fc = np.nan
            fwhm= np.nan
            if 'fc' in kwargs.keys():
                fc = kwargs['fc']
        pf.spec_plot(freq_arr,I,Q,attenuation=atten,df=df,iteration=iteration,element='resonator',fwhm=fwhm,fc=fc, flux=flux)
        exp_dict = {'date/time':    datetime.now(),
                   'nAverages': n_avg,
                         'w_LO': self.pars['rr_LO'],
                'wait_period':  self.pars['rr_resettime'],
                }
        if savedata:
            # save data
            dataPath = f'{saveDir}\spectroscopy\\resonator_spec'
            if not os.path.exists(dataPath):
                Path(dataPath).mkdir(parents=True, exist_ok=True)
            with open(f"{dataPath}\data_{iteration:03d}.csv","w") as datafile:
                writer = csv.writer(datafile)
                writer.writerow(exp_dict.keys())
                writer.writerow(exp_dict.values())
                writer.writerow(freqs)
                writer.writerow(I)
                writer.writerow(Q)
    
        return I, Q, freqs+self.pars['rr_LO'], job;
    
    
    
    



    #%%% qubit_spec
    def qubit_spec(self,
                   f_LO = 5e9,
                   IF_min = 0.1e6,          # min IF frequency
                   IF_max = 400e6,          # max IF frequency
                   check_mixers=False,
                   df = 0.1e6,              # IF frequency step
                   amp_q_scaling = 0.1,     # prefactor to scale default "const" qubit tone, amp_q
                   saturation_dur = int(10e3),   # time qubit saturated w/ qubit tone, in ns
                   on_off =  True,          # background subtraction
                   notify = False,
                   showprogress=False,
                   savedata=True,
                   **kwargs):         # create notification on Slack when measurement finishes

        iteration = counter(self._directory,self.experiment,element='qubit',extension='*.csv')

        freq_arr = np.arange(IF_min, IF_max + df/2, df, dtype=int)
        saturation_dur = int(saturation_dur)
        
        self.update_value('qubit_LO',value = f_LO)

        if check_mixers:
            self.opt_lo_leakage(mode='coarse',element='qubit',sa_span=0.5e6,threshold=-30,plot=True)
           #self.update_value('qubit_IF',50e6)
            #self.opt_sideband(mode='coarse',element='qubit',sa_span=0.5e6,threshold=-20,plot=True)
           # self.opt_lo_leakage(mode='coarse',element='rr',sa_span=0.5e6,threshold=-30,plot=True)
           # self.update_value('rr_IF',50e6)
           # self.opt_sideband(mode='coarse',element='rr',sa_span=0.5e6,threshold=-20,plot=True)

        # prog = self.make_sequence(self,on_off=on_off,saturation_dur=saturation_dur,amp_q_scaling=amp_q_scaling, IF_min=IF_min, IF_max=IF_max, df=df,)
        
        seq = sequence(self,name='qubit_spec',on_off=on_off,saturation_dur=saturation_dur,amp_q_scaling=amp_q_scaling, IF_min=IF_min, IF_max=IF_max, df=df,)
        prog = seq.make_qubit_spec_sequence()
        # execute 'QubitSpecProg' using configuration settings in 'config'
        # fetch averaged I and Q values that were saved
        datadict, job = self.get_results(jobtype = prog, result_names = ["I", "Q"], showprogress=showprogress, notify = notify)
        
        # print('qubit power')
        # qb_power = self.get_power(sa,freq=self.pars['qubit_LO']+self.pars['qubit_IF'],reference=0,amp_q = amp_q_scaling, span=1e6,config=True,output=False)
        # print('rr power')
        # rr_power = self.get_power(sa,freq=self.pars['rr_LO']+self.pars['rr_IF'],reference=0,span=1e6,config=True,output=False)

        I = np.array(datadict["I"])
        Q = np.array(datadict["Q"])
        freq_arr = np.array(freq_arr + self.pars['qubit_LO'])
        data = dict(I=I,Q=Q,freqs=freq_arr)

        
        
        # print(f'Qubit Frequency: {fc*1e-9:.5f} GHz\nFWHM = {fwhm*1e-6} MHz\nkappa = {2*np.pi*fwhm*1e-6:.3f} MHz')

        if savedata:
            exp_dict = {
                       'n_avg': self.pars["n_avg"],
                       'amp_r': self.pars["amp_r"],
                       'amp_q': self.pars["amp_q"],
                        'readout_atten': self.pars["readout_atten"],
                         'qubit_LO': self.pars['qubit_LO'],
                    }
            # save data
            saveDir = self._directory
            dataPath = f'{saveDir}\\spectroscopy\qubit_spec'
            if not os.path.exists(dataPath):
                Path(dataPath).mkdir(parents=True, exist_ok=True)
            if 'saveto' in kwargs:
                dataDict = {'metadata': exp_dict,
                            'time': freq_arr,
                            'I': I,
                            'Q': Q,
                            
                    }
                file = kwargs.get('saveto')['file']
                name = kwargs.get('saveto')['name']
                save_datadict_to_fgroup(file,name , dataDict)
            else:
                with open(f"{dataPath}\data_{iteration:03d}.csv","w") as datafile:
                    writer = csv.writer(datafile)
                    writer.writerow(exp_dict.keys())
                    writer.writerow(exp_dict.values())
                    writer.writerow(freq_arr)
                    writer.writerow(I)
                    writer.writerow(Q)
        
        return data, job;

    def fflt1_spec(self,
                   sa = 0,
                   f_LO = 5e9,
                   IF_min = 0.1e6,          # min IF frequency
                   IF_max = 400e6,          # max IF frequency
                   check_mixers=False,
                   df = 0.1e6,              # IF frequency step
                   rr_freq = 6e9,       #resonator frequency
                   amp_r_scaling = 1,
                   amp_q_scaling = 0.1,     # prefactor to scale default "const" qubit tone, amp_q
                   n_avg = 500, # number of averages
                   atten=10, # readout attenuation
                   saturation_dur = int(10e3),   # time qubit saturated w/ qubit tone, in ns
                   resettime = int(40e3),      # wait time between experiments, in ns
                   on_off =  True,          # background subtraction
                   notify = False,
                   showprogress=False,
                   savedata=True,
                   amp_cav_scaling=0.01,
                   amp_ffl_scaling=0.01,
                   flux=0,
                   spec='rr',
                   **kwargs):         # create notification on Slack when measurement finishes

        try:
            list_of_files = glob.glob(f'{saveDir}\spectroscopy\qubit_spec\*.csv')
            latest_file = max(list_of_files, key=os.path.getctime)
            iteration = int(latest_file[-7:-4].lstrip('0')) + 1
        except:
            iteration = 1

        freq_arr = np.arange(IF_min, IF_max + df/2, df, dtype=int)
        saturation_dur = int(saturation_dur)
        inst.set_attenuator(attenuation=self.pars['rr_atten'])


        self.update_value('ffl_LO',value = f_LO)
        inst.set_ffl_LO(f_LO)
        inst.set_rr_LO(self.pars['rr_LO'])

        self.check_mix_cal(sa,check=check_mixers,amp_q = amp_q_scaling, threshold = - 60)
        
        if spec=='qb':
            prog = self.make_sequence(exp='fflqb-spec',on_off=on_off,saturation_dur=saturation_dur,var_arr=freq_arr,n_avg=n_avg,amp_q_scaling=amp_q_scaling,amp_r_scale=amp_r_scaling,amp_cav_scaling=amp_cav_scaling,amp_ffl_scaling=amp_ffl_scaling)
        else:
            prog = self.make_sequence(exp='fflrr-spec',on_off=on_off,saturation_dur=saturation_dur,var_arr=freq_arr,n_avg=n_avg,amp_q_scaling=amp_q_scaling,amp_r_scaling=amp_r_scaling,amp_cav_scaling=amp_cav_scaling,amp_ffl_scaling=amp_ffl_scaling)
        # execute 'QubitSpecProg' using configuration settings in 'config'
        # fetch averaged I and Q values that were saved
        datadict, job = self.get_results(jobtype = prog, result_names = ["I", "Q"], n_total=n_avg,showprogress=showprogress, notify = notify)
        
        # print('qubit power')
        # qb_power = self.get_power(sa,freq=self.pars['qubit_LO']+self.pars['qubit_IF'],reference=0,amp_q = amp_q_scaling, span=1e6,config=True,output=False)
        # print('rr power')
        # rr_power = self.get_power(sa,freq=self.pars['rr_LO']+self.pars['rr_IF'],reference=0,span=1e6,config=True,output=False)

        I = np.array(datadict["I"])
        Q = np.array(datadict["Q"])
        freq_arr = np.array(freq_arr + self.pars['ffl_LO'])

        pf.spec_plot(freq_arr,I,Q,iteration=iteration,element='ffl',rrFreq=self.pars['rr_freq'],find_peaks=True, amp_q_scaling=amp_q_scaling, amp_cav_scaling=amp_cav_scaling, amp_ffl_scaling=amp_ffl_scaling, attenuation=self.pars['ffl_atten'],df=df, flux=flux)
        # print(f'Qubit Frequency: {fc*1e-9:.5f} GHz\nFWHM = {fwhm*1e-6} MHz\nkappa = {2*np.pi*fwhm*1e-6:.3f} MHz')

        if savedata:
            exp_dict = {
                       'n_avg': n_avg,
                        'amp_r_scale': amp_r_scaling,
                        'amp_r_scale': amp_q_scaling ,
                        'rr_atten': inst.get_attenuation(),
                         'qubit_LO': self.pars['qubit_LO'],
                         'ffl_LO':self.pars['ffl_LO']
                    }
            # save data
            dataPath = f'{saveDir}\\spectroscopy\qubit_spec'
            if not os.path.exists(dataPath):
                Path(dataPath).mkdir(parents=True, exist_ok=True)
            if 'saveto' in kwargs:
                dataDict = {'metadata': exp_dict,
                            'time': freq_arr,
                            'I': I,
                            'Q': Q,
                            
                    }
                file = kwargs.get('saveto')['file']
                name = kwargs.get('saveto')['name']
                save_datadict_to_fgroup(file,name , dataDict)
            else:
                with open(f"{dataPath}\data_{iteration:03d}.csv","w") as datafile:
                    writer = csv.writer(datafile)
                    writer.writerow(exp_dict.keys())
                    writer.writerow(exp_dict.values())
                    writer.writerow(freq_arr)
                    writer.writerow(I)
                    writer.writerow(Q)
        return I, Q, freq_arr, job;

    #%%% power_rabi
    def power_rabi(self,sa = 0,
                   a_min = 0.01,    # minimum amp_q scaling
                   a_max = 1,     # maximum amp_q scaling
                   da = 0.005,       # step of amp_q
                   check_mixers = True,
                   pulse = 'pi',
                   n_avg = 2000,    # number of averages
                   fit = True,
                   plot = True,
                   detuning = 0e6):

        amps = np.arange(a_min, a_max + da/2, da)

        inst.set_attenuator(attenuation=self.pars['rr_atten'])

        # self.check_mix_cal(sa, check = check_mixers, threshold = -55)

        prog = self.make_sequence(exp = 'p-rabi', pulse = pulse,
                                  var_arr = amps,
                                  detuning = detuning,
                                  n_avg = n_avg)
        qmm = QuantumMachinesManager(host=host, port=port)
        job = qmm.simulate(config=self.config, program=prog, simulate=SimulationConfig(duration=3000))
        job.get_simulated_samples().con1.plot()
        
        datadict, job = self.get_results(jobtype = prog, result_names = ['I','Q'], showprogress = True, progress_key = 'n', n_total = n_avg)

        I = datadict['I']
        Q = datadict['Q']

        if fit:
            fitted_pars,error = pf.fit_data(amps,np.abs(I+1j*Q),sequence='p-rabi',dt=amps[-1]/len(amps),fitFunc='rabi')
            if plot:
                pf.plot_data(x_vector=amps, y_vector=np.abs(I+1j*Q),sequence='p-rabi',fitted_pars=fitted_pars,
                              qubitDriveFreq=self.pars['qubit_LO']+self.pars['qubit_IF']+detuning,savefig=False,nAverages=n_avg)
                
                
        '''Update pulse amplitude'''
        A = fitted_pars[1] #self.pars['pi_amp'] * fitted_pars[1]

        return amps, I, Q, job, A, fitted_pars


    #%%% single_shot
    def single_shot(self,
                    n_reps = 1000,
                    liveplot = False,
                    numSamples = 1000):

        inst.set_attenuator(attenuation=self.pars['rr_atten'])

        prog = self.make_sequence(exp='ss', n_reps = n_reps)

        datadict, job = self.get_results(jobtype = prog,result_names=['n', 'I','Q','Iexc','Qexc'], showprogress=True, liveplot = liveplot)

        if liveplot == False:
            plot, ax = pf.init_IQ_plot()
            pf.plot_single_shot(datadict,axes=ax)

        return datadict, job, prog

    #%%% make_sequence
    def make_sequence(self,exp='rabi',IFmin=200e6, IFmax=204e6, df=50e3,var_arr=0,detuning=0,n_avg=0,amp_q_scaling=1,amp_r_scale=1,numPeriods=2,nIterations=1,n_reps=100,on_off=True,saturation_dur=int(10e3),pulse='pi', play_init_pi=True, amp_cav_scaling=0.01, amp_ffl_scaling=0.01):

        
        if exp == 'qubit-spec':
            resettime_clk= clk(self.pars['qubit_resettime'])
            with program() as prog:
                n, f, I, Q = self.declare_vars([int, int, fixed, fixed])
                update_frequency("rr", self.pars['rr_IF'])
                I_stream, Q_stream, n_stream = self.declare_streams(stream_num=3)

                if on_off:
                    I_b, Q_b, I_tot,Q_tot = self.declare_vars([fixed,fixed, fixed, fixed])

                # loop over n_avg iterations
                with for_(n, 0, n < n_avg, n + 1):
                    save(n,n_stream)
                    # loop over list of IF frequencies
                    with for_each_(f,var_arr): #with for_(*from_array(f,var_arr)):
                        # update IF frequency going into qubit mixer
                        update_frequency("qubit", f)
                        # measure background
                        if on_off:
                            measure("readout"*amp(amp_r_scaling), "rr", None, *self.res_demod(I_b, Q_b))
                            wait(clk(self.pars['rr_resettime']), "rr")
                            align("rr", "qubit") # wait for operations on resonator to finish before playing qubit pulse
                        # play qubit pulse and measure
                        play("gauss" * amp(amp_q_scaling), "qubit", duration = clk(saturation_dur))
                        #play('pi','qubit')
                        play("const"*amp(amp_cav_scaling), "rr", duration=clk(saturation_dur))
                        play('gaussian'*amp(amp_ffl_scaling), "ffl", duration=clk(saturation_dur))
                        align("qubit", "rr") # wait for operations on resonator to finish before playing qubit pulse
                        measure("readout"*amp(amp_r_scaling), "rr", None, *self.res_demod(I, Q))
                        # subtract background and save to stream
                        if on_off:
                            assign(I_tot, I - I_b)
                            assign(Q_tot, Q - Q_b)
                            save(I_tot, I_stream)
                            save(Q_tot, Q_stream)
                        else:
                            save(I, I_stream)
                            save(Q, Q_stream)
                        # wait some time before continuing to next IF frequency
                        wait(resettime_clk, "rr")

                # average data over iterations and save to stream
                with stream_processing():
                    I_stream.buffer(len(var_arr)).average().save('I')
                    Q_stream.buffer(len(var_arr)).average().save('Q')
                    n_stream.save('n')
        
        if exp == 'fflrr-spec':
            resettime_clk= clk(self.pars['qubit_resettime'])
            with program() as prog:
                n, f, I, Q = self.declare_vars([int, int, fixed, fixed])
                update_frequency("rr", self.pars['rr_IF'])
                I_stream, Q_stream, n_stream = self.declare_streams(stream_num=3)
                update_frequency("qubit", self.pars['qubit_IF'])
                #update_frequency("ffl", self.pars['ffl_IF'])
                # loop over n_avg iterations
                with for_(n, 0, n < n_avg, n + 1):
                    save(n,n_stream)
                    # loop over list of IF frequencies
                    with for_each_(f,var_arr): #with for_(*from_array(f,var_arr)):
                        # update IF frequency going into qubit mixer
                        update_frequency("ffl", f)
                        # measure background
                        #update_frequency("rr", f)
                        play("readout", "rr")
                        align("ffl", "rr")
                        play('gaussian'*amp(amp_ffl_scaling), "ffl", duration=clk(200))
                        align("ffl", "rr")
                        measure("void", "rr", None,*self.res_demod(I,Q))
                        wait(resettime_clk, "rr")
                        save(I, I_stream)
                        save(Q, Q_stream)
                        
                        
    

                # average data over iterations and save to stream
                with stream_processing():
                    I_stream.buffer(len(var_arr)).average().save('I')
                    Q_stream.buffer(len(var_arr)).average().save('Q')
                    n_stream.save('n')
        
        
        if exp == 'fflqb-spec':
            resettime_clk= clk(self.pars['qubit_resettime'])
            with program() as prog:
                n, f, I, Q = self.declare_vars([int, int, fixed, fixed])
                update_frequency("rr", self.pars['rr_IF'])
                I_stream, Q_stream, n_stream = self.declare_streams(stream_num=3)
                update_frequency("qubit", self.pars['qubit_IF'])
                #update_frequency("ffl", self.pars['ffl_IF'])
                # loop over n_avg iterations
                with for_(n, 0, n < n_avg, n + 1):
                    save(n,n_stream)
                    # loop over list of IF frequencies
                    with for_each_(f,var_arr): #with for_(*from_array(f,var_arr)):
                        # update IF frequency going into qubit mixer
                        update_frequency("ffl", f)
                        
                        
                        play('pi','qubit')
                        align('ffl','qubit')
                        wait(clk(60),'ffl')
                        play('gaussian'*amp(amp_ffl_scaling), "ffl", duration=clk(10000))
                        wait(clk(30))
                        align("ffl","rr")
                        measure("readout", "rr", None, *self.res_demod(I, Q))
                        # subtract background and save to stream
                        save(I, I_stream)
                        save(Q, Q_stream)
                        wait(resettime_clk, "qubit")
                        # wait some time before continuing to next IF frequency
                        

                # average data over iterations and save to stream
                with stream_processing():
                    I_stream.buffer(len(var_arr)).average().save('I')
                    Q_stream.buffer(len(var_arr)).average().save('Q')
                    n_stream.save('n')


        if exp == 'rabi':
            resettime_clk= clk(self.pars['qubit_resettime'])
            with program() as prog:

                n, t, I, Q = self.declare_vars([int, int, fixed, fixed])

                I_stream, Q_stream, n_stream = self.declare_streams(stream_num=3)
                update_frequency("rr", self.pars['rr_IF'])
                update_frequency('qubit', (self.pars['qubit_freq']-self.pars['qubit_LO']) + detuning) # sets the IF frequency of the qubit

                with for_(n, 0, n < n_avg, n + 1):
                    save(n, n_stream)
                    with for_each_(t,var_arr):
                        with if_(t==0):
                            measure("readout", "rr", None, *self.res_demod(I, Q))
                            # save(t,t_stream)
                            save(I, I_stream)
                            save(Q, Q_stream)
                            wait(resettime_clk,"qubit")
                        with else_():
                            play("pi" * amp(amp_q_scaling), "qubit", duration=t)
                            align("qubit", "rr")
                            measure("readout", "rr", None, *self.res_demod(I, Q))
                            # save(t,t_stream)
                            save(I, I_stream)
                            save(Q, Q_stream)
                            wait(resettime_clk,"qubit")

                with stream_processing():
                    I_stream.buffer(len(var_arr)).average().save("I")
                    Q_stream.buffer(len(var_arr)).average().save("Q")
                    n_stream.save('n')


        elif exp == 'p-rabi':
            resettime_clk= clk(self.pars['qubit_resettime'])
            with program() as prog:

                a, n, I, Q = self.declare_vars([fixed, int, fixed, fixed])

                I_stream, Q_stream, n_stream = self.declare_streams(stream_num=3)
                update_frequency("rr", self.pars['rr_IF'])
                update_frequency('qubit', (self.pars['qubit_freq']-self.pars['qubit_LO']) + detuning) # sets the IF frequency of the qubit

                with for_(n, 0, n < n_avg, n + 1):
                    save(n, n_stream)
                    with for_each_(a,var_arr):  # Sweep pulse duration
                        play('const'*amp(0.2), "ffl", duration=25)
                        wait(10,"qubit")
                        play(pulse * amp(a), "qubit")
                        #align("qubit", "rr")
                        align("ffl","rr")
                        measure("readout", "rr", None, *self.res_demod(I, Q))
                        save(I, I_stream)
                        save(Q, Q_stream)
                        wait(resettime_clk)
                with stream_processing():
                    I_stream.buffer(len(var_arr)).average().save("I")
                    Q_stream.buffer(len(var_arr)).average().save("Q")
                    n_stream.save("n")

        elif exp == 'ramsey':
            resettime_clk= clk(self.pars['qubit_resettime'])
            with program() as prog:
               update_frequency("rr", self.pars['rr_IF']) 
               update_frequency('qubit', (self.pars['qubit_freq']-self.pars['qubit_LO']) + detuning)

               n, t, I, Q = self.declare_vars([int, int, fixed, fixed])

               I_stream,Q_stream,n_stream = self.declare_streams(stream_num=3)

               with for_(n, 0, n < n_avg, n + 1):
                   save(n, n_stream)
                   with for_each_(t,var_arr):
                        with if_(t==0):
                            play("pi_half", "qubit")
                            play("pi_half", "qubit")
                            align("qubit","rr")
                            measure("readout", "rr", None, *self.res_demod(I, Q))
                            save(I, I_stream)
                            save(Q, Q_stream)
                            wait(resettime_clk, "qubit")
                        with else_():
                            play("pi_half", "qubit")
                            wait(t, "qubit")
                            # frame_rotation_2pi(phi, 'qubit')  # this was in Haimeng's code and was commented out by her,
                            play("pi_half", "qubit")
                            align("qubit","rr")
                            measure("readout", "rr", None, *self.res_demod(I, Q))
                            save(I, I_stream)
                            save(Q, Q_stream)
                            wait(resettime_clk, "qubit")

               with stream_processing():
                   I_stream.buffer(len(var_arr)).average().save("I")
                   Q_stream.buffer(len(var_arr)).average().save("Q")
                   n_stream.save("n")
        elif exp == 'ramsey_chi':
            resettime_clk= clk(self.pars['qubit_resettime'])
            with program() as prog:
               update_frequency("rr", self.pars['rr_IF']) 
               update_frequency('qubit', (self.pars['qubit_freq']-self.pars['qubit_LO']) + detuning)

               n, t, I, Q = self.declare_vars([int, int, fixed, fixed])

               I_stream,Q_stream,n_stream = self.declare_streams(stream_num=3)

               with for_(n, 0, n < n_avg, n + 1):
                   save(n, n_stream)
                   with for_each_(t,var_arr):
                        with if_(t==0):
                            play("pi_half", "qubit")
                            play("pi_half", "qubit")
                            align("qubit","rr")
                            measure("readout", "rr", None, *self.res_demod(I, Q))
                            save(I, I_stream)
                            save(Q, Q_stream)
                            wait(resettime_clk, "qubit")
                        with else_():
                            play("pi_half", "qubit")
                            align('qubit','rr')
                            play("readout"*amp(amp_r_scale), "rr",duration=t)
                            align('rr','qubit')
                            
                            # frame_rotation_2pi(phi, 'qubit')  # this was in Haimeng's code and was commented out by her,
                            play("pi_half", "qubit")
                            align("qubit","rr")
                            measure("readout", "rr", None, *self.res_demod(I, Q))
                            save(I, I_stream)
                            save(Q, Q_stream)
                            wait(resettime_clk, "qubit")

               with stream_processing():
                   I_stream.buffer(len(var_arr)).average().save("I")
                   Q_stream.buffer(len(var_arr)).average().save("Q")
                   n_stream.save("n")

        elif exp == 'echo':
            resettime_clk= clk(self.pars['qubit_resettime'])
            with program() as prog:
                
                n, t, I, Q = self.declare_vars([int, int, fixed, fixed])

                I_stream,Q_stream,n_stream = self.declare_streams(stream_num=3)
                update_frequency("rr", self.pars['rr_IF'])
                update_frequency('qubit', (self.pars['qubit_freq']-self.pars['qubit_LO']) + detuning)

                with for_(n, 0, n < n_avg, n + 1):
                    save(n,n_stream)
                    with for_each_(t,var_arr):
                        with if_(t==0):
                            play("pi_half", "qubit")
                            play("pi", "qubit")
                            play("pi_half", "qubit")
                            align("qubit","rr")
                            measure("readout", "rr", None, *self.res_demod(I, Q))
                            save(I, I_stream)
                            save(Q, Q_stream)
                            wait(resettime_clk, "qubit")
                        with else_():
                            play("pi_half", "qubit")
                            wait(t, "qubit")
                            play("pi", "qubit")
                            wait(t, "qubit")
                            play("pi_half", "qubit")
                            align("qubit","rr")
                            measure("readout", "rr", None, *self.res_demod(I, Q))
                            save(I, I_stream)
                            save(Q, Q_stream)
                            wait(resettime_clk, "qubit")


                with stream_processing():
                    I_stream.buffer(len(var_arr)).average().save("I")
                    Q_stream.buffer(len(var_arr)).average().save("Q")
                    n_stream.save("n")

        elif exp == 'T1':
            resettime_clk= clk(self.pars['qubit_resettime'])
            with program() as prog:
                
                n, t, I, Q = self.declare_vars([int, int, fixed, fixed])

                I_stream, Q_stream, n_stream = self.declare_streams(stream_num=3)
                update_frequency("rr", self.pars['rr_IF'])
                update_frequency('qubit', (self.pars['qubit_freq']-self.pars['qubit_LO']) + detuning)

                with for_(n, 0, n < n_avg, n + 1):
                    save(n, n_stream)
                    with for_each_(t,var_arr):
                        with if_(t==0):
                            play("pi", "qubit")
                            align("qubit", "rr")
                            measure("readout", "rr", None,*self.res_demod(I,Q))
                            wait(resettime_clk, "qubit")
                            save(I, I_stream)
                            save(Q, Q_stream)
                        with else_():
                            play("pi", "qubit")
                            wait(t, 'rr')
                            align("qubit", "rr")
                            measure("readout", "rr", None,*self.res_demod(I,Q))
                            wait(resettime_clk, "qubit")
                            save(I, I_stream)
                            save(Q, Q_stream)

                with stream_processing():
                    I_stream.buffer(len(var_arr)).average().save("I")
                    Q_stream.buffer(len(var_arr)).average().save("Q")
                    n_stream.save('n')
        
        elif exp == 'qubit_temp':
            resettime_clk= clk(1.707*self.pars['qubit_resettime'])
            with program() as prog:

                n, t, I, Q = self.declare_vars([int, int, fixed, fixed])

                I_stream, Q_stream, n_stream = self.declare_streams(stream_num=3)
                update_frequency("rr", self.pars['rr_IF'])
                update_frequency('qubit', (self.pars['qubit_freq']-self.pars['qubit_LO']))
                update_frequency('qubit12', (self.pars['qubit12_IF']))
                ##make sure mixer is calibrated properly.

                with for_(n, 0, n < n_avg, n + 1):
                    save(n, n_stream)
                    with for_each_(t,var_arr):
                        play("pi", "qubit", condition= play_init_pi==True)
                        align("qubit","rr")
                        play("pi" * amp(amp_q_scaling), "qubit12", duration=t)
                        align()
                        play("pi", "qubit")
                        align("qubit","rr")
                        measure("readout", "rr", None, *self.res_demod(I, Q))
                        # save(t,t_stream)
                        save(I, I_stream)
                        save(Q, Q_stream)
                        wait(resettime_clk)
                        
                with stream_processing():
                    I_stream.buffer(len(var_arr)).average().save("I")
                    Q_stream.buffer(len(var_arr)).average().save("Q")
                    n_stream.save('n')
              
                    
        elif exp == 'dissT1':
            resettime_clk= clk(self.pars['diss_resettime'])
            with program() as prog:

                n, t, I, Q = self.declare_vars([int, int, fixed, fixed])

                I_stream, Q_stream, n_stream = self.declare_streams(stream_num=3)

                update_frequency('diss', (self.pars['diss_freq']-self.pars['diss_LO']) + detuning)

                with for_(n, 0, n < n_avg, n + 1):
                    save(n, n_stream)
                    with for_(*from_array(t,var_arr)):
                        with if_(t==0):
                            play("const", "diss", duration = clk(saturation_dur))
                            measure("readout_diss", "rr", None,*self.res_demod_diss(I,Q))
                            wait(resettime_clk, "diss")
                            save(I, I_stream)
                            save(Q, Q_stream)
                        with else_():
                            play("const", "diss", duration = clk(saturation_dur))
                            wait(t, 'rr')
                            align("diss", "rr")
                            measure("readout_diss", "rr", None,*self.res_demod_diss(I,Q))
                            wait(resettime_clk, "diss")
                            save(I, I_stream)
                            save(Q, Q_stream)

                with stream_processing():
                    I_stream.buffer(len(var_arr)).average().save("I")
                    Q_stream.buffer(len(var_arr)).average().save("Q")
                    n_stream.save('n')
                    
        elif exp == 'ringdown':
            resettime_clk= clk(self.pars['rr_resettime'])
            with program() as prog:
                n, t, I, Q = self.declare_vars([int, int, fixed, fixed])

                I_stream, Q_stream, n_stream = self.declare_streams(stream_num=3)
                update_frequency("rr", self.pars['rr_IF'])
                update_frequency('qubit', (self.pars['qubit_freq']-self.pars['qubit_LO']) + detuning)

                with for_(n, 0, n < n_avg, n + 1):
                    save(n, n_stream)
                    with for_each_(t,var_arr):
                        play("readout", "rr")
                        wait(t, 'rr')
                        measure("void", "rr", None,*self.res_demod(I,Q))
                        wait(resettime_clk)
                        save(I, I_stream)
                        save(Q, Q_stream)

                with stream_processing():
                    I_stream.buffer(len(var_arr)).average().save("I")
                    Q_stream.buffer(len(var_arr)).average().save("Q")
                    n_stream.save('n')
                    
        elif exp == 'ss':
            resettime_clk = clk(self.pars['qubit_resettime'])
            with program() as prog:

                i,n,I,Iexc,Q,Qexc = self.declare_vars([int,int,fixed,fixed,fixed,fixed])
                i_st, I_st,Q_st,I_st_exc,Q_st_exc = self.declare_streams(stream_num=5)
                update_frequency("rr", self.pars['rr_IF'])
                update_frequency('qubit', (self.pars['qubit_freq']-self.pars['qubit_LO']))
                with for_(i, 0, i < nIterations, i + 1):
                    with for_(n, 0, n < n_reps, n + 1):
                        # do nothing
                        align("qubit", "rr")
                        measure("readout", "rr", None, *self.res_demod(I, Q))
                        align('qubit','rr')
                        wait(resettime_clk, "qubit")
                        align('qubit','rr')
                        # apply pi-pulse
                        play("pi", "qubit")
                        # play("pi_half" * amp(0.01), "qubit", duration = clk(2e3))
                        # play("gauss"*amp(0.41/0.45), "qubit",duration=round(274*2/4))
                        align("qubit", "rr")
                        measure("readout", "rr", None, *self.res_demod(Iexc, Qexc))
                        wait(resettime_clk, "qubit")
                        save(I, I_st)
                        save(Q, Q_st)
                        save(Iexc, I_st_exc)
                        save(Qexc, Q_st_exc)
                        # save(n,N_st)
                    save(i, i_st)

                with stream_processing():
                    # I_st.save('I')
                    # Q_st.save('Q')
                    # I_st_exc.save('I_exc')
                    # Q_st_exc.save('Q_exc')
                    # i_st.save('i')
                    I_st.buffer(n_reps).save('I')
                    Q_st.buffer(n_reps).save('Q')
                    I_st_exc.buffer(n_reps).save('Iexc')
                    Q_st_exc.buffer(n_reps).save('Qexc')
                    i_st.save('i')
        
        elif exp == 'ss_cav':
            resettime_clk = clk(self.pars['qubit_resettime'])
            with program() as prog:

                i,n,I,Iexc,Q,Qexc = self.declare_vars([int,int,fixed,fixed,fixed,fixed])
                i_st, I_st,Q_st,I_st_exc,Q_st_exc = self.declare_streams(stream_num=5)
                update_frequency("rr", self.pars['rr_IF'])
                update_frequency('qubit', (self.pars['qubit_freq']-self.pars['qubit_LO']))
                with for_(i, 0, i < nIterations, i + 1):
                    with for_(n, 0, n < n_reps, n + 1):
                        # do nothing
                        align("qubit", "rr")
                        play("readout"*amp(0.8), "rr")
                        wait(clk(500))
                        measure("readout", "rr", None, *self.res_demod(I, Q))
                        align('qubit','rr')
                        wait(resettime_clk)
                        
                        play("readout"*amp(0.8), "rr")
                        wait(clk(500))
                        align('qubit','rr')
                        # apply pi-pulse
                        play("pi", "qubit")
                        # play("pi_half" * amp(0.01), "qubit", duration = clk(2e3))
                        # play("gauss"*amp(0.41/0.45), "qubit",duration=round(274*2/4))
                        align("qubit", "rr")
                        measure("readout", "rr", None, *self.res_demod(Iexc, Qexc))
                        wait(resettime_clk, "qubit")
                        save(I, I_st)
                        save(Q, Q_st)
                        save(Iexc, I_st_exc)
                        save(Qexc, Q_st_exc)
                        # save(n,N_st)
                    save(i, i_st)

                with stream_processing():
                    # I_st.save('I')
                    # Q_st.save('Q')
                    # I_st_exc.save('I_exc')
                    # Q_st_exc.save('Q_exc')
                    # i_st.save('i')
                    I_st.buffer(n_reps).save('I')
                    Q_st.buffer(n_reps).save('Q')
                    I_st_exc.buffer(n_reps).save('Iexc')
                    Q_st_exc.buffer(n_reps).save('Qexc')
                    i_st.save('i')
       
        elif exp == 'ss_ffl':
            resettime_clk = clk(self.pars['qubit_resettime'])
            with program() as prog:

                i,n,I,Iexc,Q,Qexc = self.declare_vars([int,int,fixed,fixed,fixed,fixed])
                i_st, I_st,Q_st,I_st_exc,Q_st_exc = self.declare_streams(stream_num=5)
                update_frequency("rr", self.pars['rr_IF'])
                update_frequency('qubit', (self.pars['qubit_freq']-self.pars['qubit_LO']))
                with for_(i, 0, i < nIterations, i + 1):
                    with for_(n, 0, n < n_reps, n + 1):
                        # do nothing
                        align("qubit", "rr")
                        play("readout"*amp(0.8), "rr")
                        align('ffl','rr')
                        play('gaussian'*amp(1.), "ffl", duration=clk(500))
                        align('ffl','rr')
                        measure("readout", "rr", None, *self.res_demod(I, Q))
                        #align('qubit','rr')
                        wait(resettime_clk)
                        
                        play("readout"*amp(0.8), "rr")
                        align('ffl','rr')
                        play('gaussian'*amp(1.), "ffl", duration=clk(500))
                        align('qubit','ffl')
                        # apply pi-pulse
                        play("pi", "qubit")
                        # play("pi_half" * amp(0.01), "qubit", duration = clk(2e3))
                        # play("gauss"*amp(0.41/0.45), "qubit",duration=round(274*2/4))
                        align("qubit", "rr")
                        measure("readout", "rr", None, *self.res_demod(Iexc, Qexc))
                        wait(resettime_clk, "qubit")
                        save(I, I_st)
                        save(Q, Q_st)
                        save(Iexc, I_st_exc)
                        save(Qexc, Q_st_exc)
                        # save(n,N_st)
                    save(i, i_st)

                with stream_processing():
                    # I_st.save('I')
                    # Q_st.save('Q')
                    # I_st_exc.save('I_exc')
                    # Q_st_exc.save('Q_exc')
                    # i_st.save('i')
                    I_st.buffer(n_reps).save('I')
                    Q_st.buffer(n_reps).save('Q')
                    I_st_exc.buffer(n_reps).save('Iexc')
                    Q_st_exc.buffer(n_reps).save('Qexc')
                    i_st.save('i')

        
        elif exp == 'IQblob':
            resettime_clk= clk(self.pars['qubit_resettime'])
            with program() as prog:
                update_frequency("rr", self.pars['rr_IF'])
                update_frequency('qubit', (self.pars['qubit_freq']-self.pars['qubit_LO']) + detuning)

                n,I,Iexc,Q,Qexc = self.declare_vars([int,fixed,fixed,fixed,fixed])
                N_st, I_st,Q_st,I_st_exc,Q_st_exc = self.declare_streams(stream_num=5)

                
                with for_(n, 0, n < n_reps, n + 1):
                    # do nothing
                    wait(resettime_clk)
                    align("qubit", "rr")
                    measure("readout", "rr", None, *self.res_demod(I, Q))
                    save(I, I_st)
                    save(Q, Q_st)
                    
                    wait(resettime_clk)
                    align('qubit','rr')
                    # apply pi-pulse
                    play("pi", "qubit")
                    align("qubit", "rr")
                    measure("readout", "rr", None, *self.res_demod(Iexc, Qexc))
                    wait(resettime_clk)
                    save(Iexc, I_st_exc)
                    save(Qexc, Q_st_exc)
                    save(n, N_st)
             
                    

                with stream_processing():
                    I_st.save_all('I')
                    Q_st.save_all('Q')
                    I_st_exc.save_all('Iexc')
                    Q_st_exc.save_all('Qexc')
                    N_st.save('n')
        
        
        
        elif exp== 'opt_rr_freq':
            resettime_clk= clk(self.pars['qubit_resettime'])
            freqs = np.arange(IFmin, IFmax + df/2, df)
            with program() as prog:

                n = declare(int)
                I = declare(fixed)
                I_st = declare_stream()
                Q = declare(fixed)
                Q_st = declare_stream()
                I_exc = declare(fixed)
                Q_exc = declare(fixed)
                I_st_exc = declare_stream()
                Q_st_exc = declare_stream()
                f = declare(int)
                distance = declare(fixed)
                distance_st = declare_stream()
                
                with for_(n, 0, n < 2000, n + 1):
                    with for_(f, IFmin, f <= IFmax, f + df):
                        update_frequency('rr', f)
                        
                        align("qubit", "rr")
                        measure("readout", "rr", None, *self.res_demod(I, Q))
                        save(I, I_st)
                        save(Q, Q_st)
                        
                        align('qubit','rr')
                        wait(resettime_clk)
                        align('qubit','rr')
                        # apply pi-pulse
                        play("pi", "qubit")
                        align("qubit", "rr")
                        measure("readout", "rr", None, *self.res_demod(I_exc, Q_exc))
                        wait(resettime_clk)
                        save(I_exc, I_st_exc)
                        save(Q_exc, Q_st_exc)
                        assign(distance, Math.abs(I_exc - I))
                        save(distance, distance_st)
                        
                with stream_processing():
                    I_st.save_all('I')
                    Q_st.save_all('Q')
                    I_st_exc.save_all('I_exc')
                    Q_st_exc.save_all('Q_exc')
                    distance_st.buffer(len(freqs)).average().save('distance')

                    
        
        return prog

            #%%% pulse_exp
    

    
    def pulse_exp(self,sa = 0,
                      exp='rabi',
                      check_mixers=False,
                      n_avg = 2000,
                      tmin = 16,         # minimum pulse duration in nanoseconds
                      tmax = 10e3,    # maximum pulse duration in nanoseconds
                      dt = 500,        # step of sweep in nanoseconds
                      amp_q_scaling = 1,
                      fit = True,
                      plot = True,
                      detuning = 0e6, play_init_pi=True, simulate=True, ffl_len=1e3, amp_ffl_scale=0, scrambling_amp=1, with_ffl=True, amp_r_scale=1, flux=0):
        """

        Args:
            exp (str):   What experiment to run. Options are: 'rabi','ramsey','T1','echo'.
            n_avg (TYPE, optional): DESCRIPTION. Defaults to 2000.
            t0 (TYPE, optional): minimum pulse duration in nanoseconds. Defaults to 0.
            tf (TYPE, optional): maximum pulse duration in nanoseconds. Defaults to 10e3.
            dt (TYPE, optional): sequence step size in nanoseconds. Defaults to 500.
            amp_q_scaling (TYPE, optional): DESCRIPTION. Defaults to 1.
            plot (TYPE, optional): Whether to plot and fit the data. Defaults to True.
            detuning (float): detuning from fL0-fIF in Hz.
            resettime (TYPE, optional): waiting time between experiments. Defaults to 400e3.

        Returns:
            times (TYPE): DESCRIPTION.
            I (TYPE): DESCRIPTION.
            Q (TYPE): DESCRIPTION.

        """

        try:
            list_of_files = glob.glob(f'{saveDir}\{exp}\*.csv')
            latest_file = max(list_of_files, key=os.path.getctime)
            iteration = int(latest_file[-7:-4].lstrip('0')) + 1
        except:
            iteration = 1

        tmin = clk(tmin)
        tmax = clk(tmax)
        dt = clk(dt)
        t_arr = np.arange(tmin, tmax + dt/2, dt, dtype = int)

        inst.set_attenuator(attenuation=self.pars['rr_atten'])
        # self.update_value('qubit_LO', self.pars['qubit_LO'])
        # self.update_value('rr_LO', self.pars['rr_LO'])

        # qb_IF = self.pars['qubit_freq']-self.pars['qubit_LO']
        # self.update_value('qubit_IF', (self.pars['qubit_freq']-self.pars['qubit_LO']) + detuning)
        self.check_mix_cal(sa, amp_q = amp_q_scaling,check = check_mixers, threshold = -55)

        if exp == 'qb-reset':
            seq = sequence('qb-reset', n_avg=n_avg, amp_ffl_scale=amp_ffl_scale )
            prog = seq.make_sequence(self, tmin=4*tmin, tmax=4*tmax, dt=4*dt,ffl_len=ffl_len,scrambling_amp=scrambling_amp)
        elif exp=='cavity-reset':
            seq = sequence('cavity-reset', n_avg=n_avg, amp_ffl_scale=amp_ffl_scale , amp_r_scale= amp_r_scale)
            prog = seq.make_sequence(self, tmin=4*tmin, tmax=4*tmax, dt=4*dt,ffl_len=ffl_len,scrambling_amp=scrambling_amp, with_ffl=with_ffl, detuning=detuning, )
        elif exp=='cavity-cooling':
            seq = sequence('cavity-cooling', n_avg=n_avg, amp_ffl_scale=amp_ffl_scale , amp_r_scale= amp_r_scale)
            prog = seq.make_sequence(self, tmin=4*tmin, tmax=4*tmax, dt=4*dt,ffl_len=ffl_len,scrambling_amp=scrambling_amp, with_ffl=with_ffl, detuning=detuning, ) 
        elif exp=='cavity-cooling-ramsey':
            seq = sequence('cavity-cooling-ramsey', n_avg=n_avg, amp_ffl_scale=amp_ffl_scale , amp_r_scale= amp_r_scale)
            prog = seq.make_sequence(self, tmin=4*tmin, tmax=4*tmax, dt=4*dt,ffl_len=ffl_len,scrambling_amp=scrambling_amp, with_ffl=with_ffl, detuning=detuning, ) 
        else:
            prog = self.make_sequence(exp=exp,var_arr=t_arr,detuning=detuning,n_avg=n_avg,amp_q_scaling=amp_q_scaling, play_init_pi=play_init_pi,amp_r_scale=amp_r_scale)
        
        if simulate:
            qmm = QuantumMachinesManager(host=host, port=port)
            job = qmm.simulate(config=self.config, program=prog, simulate=SimulationConfig(duration=10000))
            job.get_simulated_samples().con1.plot()
            
        datadict, job = self.get_results(jobtype = prog, result_names = ["I", "Q"], n_total=n_avg, notify = False)
        
        

        # qb_power = self.get_power(sa,freq=self.pars['qubit_LO']+self.pars['qubit_IF'],reference=0,amp_q = amp_q_scaling, span=1e6,config=True,output=False)

        # t_arr = np.array(datadict["t"])/1e3 # times in microseconds
        if exp == 'echo' or exp=='cavity-reset' or exp=='cavity-cooling':
            t_arr = np.array(t_arr)*4/1e3 * 2
    
        else:
            t_arr = np.array(t_arr)*4/1e3
        I = np.array(datadict["I"])
        Q = np.array(datadict["Q"])
        
        #ydata = np.abs(I+1j*Q)
        ydata=I
        #print(len(ydata))
        #print(len(t_arr))
        if fit:
            fitted_pars, error = pf.fit_data(t_arr,ydata,sequence=exp,dt=t_arr[-1]*1e-6/len(t_arr))
            if plot:
                fig = pf.plot_data(t_arr,ydata,sequence=exp,fitted_pars=fitted_pars,nAverages=n_avg, pi2Width=self.pars['pi_half_len'],
                             qubitDriveFreq=self.pars['qubit_LO']+self.pars['qubit_IF']+detuning,fflDriveFreq=self.pars['ffl_LO']+self.pars['ffl_IF'],iteration=iteration,amp_ffl_scale=amp_ffl_scale, amp=amp_r_scale,flux=flux, error=error, ffl_len=ffl_len, rr_atten=self.pars['rr_atten'], ffl_atten=self.pars['ffl_atten'])
        elif plot:
            fitted_pars = None
            fig = pf.plot_data(t_arr, ydata, sequence = exp)
        else:
            fitted_pars = None

        print(error)


        # if exp == 'rabi':
            # self.update_value('pi_half_len',4*(clk(fitted_pars[1]/4)))

        # self.update_value('qubit_IF',qb_IF)

        exp_dict = {'date/time':     datetime.now(),
                   'nAverages': n_avg,
                         'Tmax': tmax,
                         'dt':   dt,
                         'pi2': self.pars['pi_half_len'],
                         'A_d':     amp_q_scaling,
                         'w_d':  self.pars['qubit_LO']+self.pars['qubit_IF']-detuning,
                         'w_LO': self.pars['qubit_LO'],
                'wait_period':  self.pars['qubit_resettime'],
                'amp_ffl_scale':  amp_ffl_scale,
                'amp_r_scale':  amp_r_scale,
                'flux': flux
                }

        # save data
        dataPath = f'{saveDir}\\{exp}'
        if not os.path.exists(dataPath):
            Path(dataPath).mkdir(parents=True, exist_ok=True)
        with open(f"{dataPath}\data_{iteration:03d}.csv","w") as datafile:
            writer = csv.writer(datafile)
            writer.writerow(exp_dict.keys())
            writer.writerow(exp_dict.values())
            writer.writerow(t_arr)
            writer.writerow(I)
            writer.writerow(Q)
        fig.savefig(f"{dataPath}\data_{iteration:03d}.png")
        
        
        dataDict = {
            'I': I,
            'Q': Q,
            'fitted_pars': fitted_pars,
            'error': error,
            't_arr': t_arr,
            'metadata': {'flux': flux},
                        
            }
        # create labber logfile
        # lStep = [dict(name = 'Time', unit='us', values=t_arr)]
        # lLog = [dict(name = 'Signal',unit='V', vector=True)]
        # f = Labber.createLogFile_ForData(exp, lLog, lStep)
        # data = {'I':I, 'Q':Q }
        # f.addEntry(data)
        return fitted_pars
    
    
    





#%% CALIBRATIONS
  #%%% tof_cal
    def tof_cal(self,update_tof=False):
        qmm = QuantumMachinesManager(host=self.pars['host'], port=self.pars['port'])
        with program() as tof_cal:
            n = declare(int)
            adc_st = declare_stream(adc_trace=True)
            update_frequency('rr',5e6)
            with for_(n, 0, n < self.pars['n_avg'], n + 1):
                reset_phase("rr")
                measure("readout", "rr", adc_st)
                wait(self.pars['tof'], "rr")
            with stream_processing():
                adc_st.input1().average().save("adc1")
                adc_st.input2().average().save("adc2")

        qm = qmm.open_qm(self.config)
        job = qm.execute(tof_cal)
        res_handles = job.result_handles
        res_handles.wait_for_all_values()
        adc1 = res_handles.get("adc1").fetch_all()
        adc2 = res_handles.get("adc2").fetch_all()

        
        offset1 = np.mean(adc1)/4096
        offset2 = np.mean(adc2)/4096
        if update_tof:
            # Filter the data to get the pulse arrival time
            adc1_unbiased = adc1 - offset1
            adc2_unbiased = adc2 - offset2
            signal = savgol_filter(np.abs(adc1_unbiased + 1j * adc2_unbiased), 11, 3)
            # Detect the arrival of the readout signal
            th = (np.mean(signal[:100]) + np.mean(signal[:-100])) / 2
            delay = np.where(signal > th)[0][0]
            delay = np.round(delay / 4) * 4  # Find the closest multiple integer of 4ns
            previous_delay = self.pars['tof']
            self.update_value('tof', value = int(delay+previous_delay))
        else:
            pass
        print(f'Input 1 Offset: {offset1*1e3} mV')
        print(f'Input 2 Offset: {offset2*1e3} mV')
        self.update_value('analog_input_offsets', value = [self.pars['analog_input_offsets'][0] - offset1, self.pars['analog_input_offsets'][1] - offset2])
        
        return adc1, adc2
    
    def init_quantum_machine(self, initialize = True):
       self.qmm = QuantumMachinesManager(host=self.pars['host'], port=self.pars['port']) if initialize else None

#%%% cal_pi_pulse
    def cal_pulse(self, pulse = 'pi', pulse_len_target = 20, starting_amp = 0.44, **rabi_kwargs):

        if pulse == 'pi':
            scaling = 2
        elif pulse == 'pi_half':
            scaling = 4

        print(f"CALIBRATING {pulse} pulse... target time = {pulse_len_target}")

        # start with a clean slate based on input parameters
        self.update_value(f'{pulse}_amp', starting_amp)
        self.update_value(f'{pulse}_len', pulse_len_target)

        # run power rabi and extract fit parameter A, which is the scaling factor
        print("Calibrating fit...")
        amps, I, Q, job, A = self.power_rabi(check_mixers = False, pulse = pulse, fit = True, plot = True, **rabi_kwargs)

        print(f"Fitted scaling factor A/{scaling} = {round(A/scaling, ndigits=4)}")

        # check: if A / 2 < 1, then you've probably found your pi pulse and you're donee
        # if A / 2 >= 1, then you probably need to increrase your pi_half_len_target
        # this code will automatically recalibrate with a new pi_half_len_target based on the estimate of A
        if A / scaling < 1:

            print(f"{pulse} pulse calibration successful, updating config...")

            # update pi_amp
            self.update_value(f'{pulse}_amp', starting_amp * A / scaling)

            # as a check to your pi pulse, run single shot and run power rabi again to see your pi pulse in action
            if pulse == 'pi':
                self.single_shot(nIterations=1)
            amps, I, Q, job, A = self.power_rabi(check_mixers = False, pulse = pulse, fit = True, plot = True, **rabi_kwargs)

        else:

            # your new pi pulse time to try is the old time, scaled by A/2, times 1.1 to overshoot a bit
            # convert to clock cycles then multiply by 4 to ensure a pulse time w/ integer clock cycles
            newtarget = 4 * clk(pulse_len_target * (A/scaling) * 1.1)

            print(f"The {pulse} length target time was too short. Trying again for target time {newtarget} ns.")

            self.cal_pulse(pulse_len_target = newtarget, pulse = pulse, starting_amp = starting_amp, **rabi_kwargs)

    

    #%%% cal_freq
    def cal_freq(self,  min_delta=10e3,
                        check_mixers = False,
                        recalibrate_pulses = True,
                        dt_scaling = 4,
                        **kwargs):

        print("RUNNING initial Ramsey experiment...")
        t_arr, I, Q, job, fitted_pars = self.pulse_exp(exp='ramsey', check_mixers = check_mixers,dt=40,tmin=20,tmax=2e4,detuning=0,**kwargs)

        delta = abs(fitted_pars[1]) * 1e6
        tau = fitted_pars[3] * 1e3
        tmax = 2 * tau

        self.update_value('qubit_freq', self.pars['qubit_freq'] + delta)

        n = 0

        while n < 10:

            n += 1

            print(f'RUNNING Ramsey experiment, delta = {round(delta/1e3, ndigits=6)} kHz')

            if delta < 100e3:
                scaling = dt_scaling * 8
            else:
                scaling = dt_scaling

            dt = round(get_dt(target_fs=delta)/scaling)

            t_arr, I, Q, job, fitted_pars = self.pulse_exp(exp='ramsey', detuning = 0e6, check_mixers = check_mixers,dt=dt,tmin=20,tmax=tmax, **kwargs)
            delta_result = abs(fitted_pars[1]) * 1e6

            print(f'RESULT: delta_result = {round(delta_result/1e3, ndigits=6)} kHz')

            if delta_result < min_delta:
                print(f'SUCCESS: delta_result < min_delta = {round(min_delta/1e3, ndigits=6)} kHz :) ')
                rounded_freq = round_from_uncertainty(self.pars['qubit_freq'], 1e3, scale=1e9) # round frequency to nearest kHz
                self.update_value('qubit_freq', rounded_freq)

                if recalibrate_pulses:
                    self.cal_pulse(pulse = 'pi', pulse_len_target = self.pars['pi_len'], **kwargs)
                    self.cal_pulse(pulse = 'pi_half', pulse_len_target = self.pars['pi_half_len'], **kwargs)

                break
            elif delta_result > delta:
                print(f'WRONG DIRECTION: delta_result > delta')
                self.update_value('qubit_freq', self.pars['qubit_freq'] - 2 * delta)
                delta = delta_result
            else:
                print(f'RIGHT DIRECTION: delta_result < delta')
                self.update_value('qubit_freq', self.pars['qubit_freq'] + delta_result)
                delta = delta_result



        print(f"Didn't converge within min_freq = {minfreq}")



    #%%% cal_freq_new
    def cal_freq_new(self,  min_delta=10e3,
                        check_mixers = False,
                        recalibrate_pulses = True,
                        dt_scaling = 4,
                        num_tries = 10,
                        **kwargs):

        # initial Ramsey experiment to find current detuning
        print("RUNNING initial Ramsey experiment...")
        t_arr, I, Q, job, fitted_pars = self.pulse_exp(exp='ramsey', check_mixers = check_mixers,dt=40,tmin=20,tmax=2e4,detuning=0,**kwargs)

        delta = abs(fitted_pars[1]) * 1e6
        tau = fitted_pars[3] * 1e3
        tmax = 2 * tau

        delta_dict = { delta : {"freq" : self.pars['qubit_freq'], "tried_increase" : False}}
        #


        for n in np.arange(0, num_tries - 1):

            # update value of qubit_freq to be that corresponding to the smallest delta in delta_dict,
            # plus or minus delta
            delta = min(delta_dict.keys())
            old_freq = delta_dict[delta]["freq"]
            tried_increase = delta_dict[delta]["tried_increase"]
            if not tried_increase:
                self.update_value('qubit_freq', old_freq + delta)
                delta_dict[delta]["tried_increase"] = True
            else:
                self.update_value('qubit_freq', old_freq - delta)

            # choose appropriate dt for the frequency we're trying to resolve
            if delta < 100e3:
                scaling = dt_scaling * 8
            else:
                scaling = dt_scaling

            dt = round(get_dt(target_fs=delta)/scaling)

            # run Ramsey experiment and get new delta
            t_arr, I, Q, job, fitted_pars = self.pulse_exp(exp='ramsey', detuning = 0e6, check_mixers = check_mixers,dt=dt,tmin=20,tmax=tmax, **kwargs)
            delta_result = abs(fitted_pars[1]) * 1e6

            if delta_result < min_delta:
                print(f'SUCCESS: delta_result < min_delta = {round(min_delta/1e3, ndigits=6)} kHz :) ')
                rounded_freq = round_from_uncertainty(self.pars['qubit_freq'], 1e2, scale=1e9) # round frequency to nearest kHz
                self.update_value('qubit_freq', rounded_freq)

                if recalibrate_pulses:
                    self.cal_pulse(pulse = 'pi', pulse_len_target = self.pars['pi_len'], **kwargs)
                    self.cal_pulse(pulse = 'pi_half', pulse_len_target = self.pars['pi_half_len'], **kwargs)

                return # if we were successful, we skip over the statement at the end
            # ONLY if we did better than before, add the new delta_result and qubit freq to the dict
            elif delta_result < delta:
                delta_dict[delta_result] = {"freq" : self.pars["qubit_freq"], "tried_increase" : False}

        delta = min(delta_dict.keys())
        freq = delta_dict[delta]["freq"]
        self.update_value('qubit_freq', freq)

        print(f"Didn't converge within min_freq = {minfreq} within {num_tries} tries. The best result was {round(delta/1e3, digits=3)} kHz at qubit freq {round(freq / 1e9, 7)} GHz.")


#%%% calibrate
    def calibrate(self,   pulse_len_target = 48,
                          starting_amp = 0.44,
                          min_delta = 10e3,
                          recalibrate_pulses = True,
                          T1_kwargs     = {"dt": 1e3, "tmin" : 100, "tmax" : 100e3},
                          ramsey_kwargs = {"dt": 200, "tmin" : 20, "tmax" : 40e3},
                          echo_kwargs   = {"dt": 1e3, "tmin" : 50, "tmax" : 120e3},
                          **kwargs
                          ):

        # time of flight
        self.tof_cal() # this can be modified to actually calibrate tof automatically, rather than just display plot

        # pi pulse length / time
        self.cal_pulse(pulse = 'pi', pulse_len_target = pulse_len_target, starting_amp = starting_amp, **kwargs)
        self.cal_pulse(pulse = 'pi_half', pulse_len_target = pulse_len_target, starting_amp = starting_amp, **kwargs)

        # frequency
        self.cal_freq(min_delta = min_delta, recalibrate_pulses = recalibrate_pulses, **kwargs)

        # T1, T2 ramsey, T2 echo
        print("RUNNING longer fit experiments...")
        t_arr, I, Q, job, fitted_pars = self.pulse_exp(exp='T1', check_mixers=True, **T1_kwargs, **kwargs)
        T1 = fitted_pars[1]

        t_arr, I, Q, job, fitted_pars = self.pulse_exp(exp='ramsey', check_mixers=True, **ramsey_kwargs, **kwargs)
        T2_ramsey = fitted_pars[3]

        t_arr, I, Q, job, fitted_pars = self.pulse_exp(exp='echo', check_mixers=True, **echo_kwargs, **kwargs)
        T2_echo = fitted_pars[1]

        print(f"CALIBRATION SUCCESSFUL. Qubit parameters: \nT1 = {T1}, \nT2 ramsey = {T2_ramsey}, \nT2_echo = {T2_echo} ")





#%% HELPERS
#%%% Get Results
#%%%% get_results
    def get_results(self,jobtype, result_names = ["I", "Q"],
                                    progress_key = "n",
                                    showprogress = True,
                                    notify = False,
                                    liveplot = False):
        """

        Args:
            jobtype (QUA program): QUA program to execute.
            result_names (array of strings, optional): The names of the variables we want to stream. Defaults to ["I", "Q", "n"].
            showprogress (TYPE, optional): DESCRIPTION. Defaults to False.
            n_total (TYPE, optional): DESCRIPTION. Defaults to 1000.
            notify (TYPE, optional): DESCRIPTION. Defaults to False.
            liveplot (TYPE, optional): DESCRIPTION. Defaults to False.

        Returns:
            datadict (TYPE): DESCRIPTION.
            TYPE: DESCRIPTION.

        """


        # Open Communication with the Server
        qmm = QuantumMachinesManager(host=self.pars['host'], port=self.pars['port'])

        # execute the job and get result handles
        qm = qmm.open_qm(self.config)

        job = qm.execute(jobtype)
        res_handles = job.result_handles


        # create a dictionary with result_names as keys
        # and their corresponding handles as values
        handles_dict = {}
        for name in result_names:
            handles_dict[name] = res_handles.get(name)

        # make live plot, if desired
        if liveplot:
            plot, ax = pf.init_IQ_plot()
            # wait for first values to come in before continuing to run python code

            I_handle = handles_dict["I"]

            for handle in handles_dict.values():
                handle.wait_for_values(1)
                is_processing = lambda: handle.is_processing()
            i0 = 0
            while(I_handle.is_processing()): # while(is_processing()):
                i = handles_dict["i"].fetch_all() # retrieve iteration value n
                datadict = self.get_data_from_handles(handles_dict)
                pf.plot_single_shot(datadict,axes=ax)

        if showprogress:
            n_handle = res_handles.get(progress_key)
            make_progress_meter(n_handle, self.pars['n_avg'])
#
        res_handles.wait_for_all_values()


        # retrieve all values
        datadict = self.get_data_from_handles(handles_dict)
        # plot_IQ_blobs_plt(datadict)

        # close quantum machine
        qmm.close_all_quantum_machines()

        return datadict, job

    def unpack_data(self,datadict, names = ["I", "Q"]):

        unpacked = map(lambda s: datadict[s], names)
        return list(unpacked)


    def fix_length(*args):

        l = min(map(len, args))
        newargs = map(lambda a: a[0:l], args)
        return list(newargs)

    #%%%% get_data_from_handles
    def get_data_from_handles(self,handles_dict,verbose=0):

        datadict = {}

        for (key, handle) in handles_dict.items():
            # datadict[key] = handle.fetch_all()['value']
            datadict[key] = handle.fetch_all()
            if handle.has_dataloss():
                print(f'Dataloss occured in'+datadict[key])
            elif not handle.has_dataloss and verbose == 1:
                print('No data loss')

        return datadict

#%%% Mixer
    #%%%% config_sa


    #%%%% config_sa
    def config_sa(self, fc=5e9, sa_span=5e6, threshold = -30, bandwidth = 1e3):
        """
        Configures settings of the spectrum analyzer

        Args:
        fc (float): center frequency in Hz
        span (float): span of the spectrum analyzer
        threshold (float): highest power level
        bandwidth (float): resolution bandwidth in Hz

        """
        self._instruments.set('sa','frequency', fc)
        self._instruments.set('sa','span', sa_span)
        self._instruments.set('sa','threshold', threshold)
        self._instruments.set('sa','bandwidth', bandwidth)
    
#%% get_power
    def get_power(self, fc=5e9, threshold=-100, sa_span=1e6, config=False, plot=False):
        """
        Configures SA (optional) and measures power at specified frequency

        Args:
        fc (float): center frequency in Hz
        threshold (float): highest power level
        span (float): frequency span of the spectrum analyzer
        config (bool): whether to configure the spectrum analyzer before measuring spectrum
        plot (bool): whether to plot the spectrum

        Returns
        data (dict): dictionary containing frequency and power data

        """
        # skips configuring the spectrum analyzer. Used only when optimizing mixer
        if config:
            self.config_sa(fc=fc, sa_span=sa_span, threshold=threshold)

        # measure
        data = self._instruments.get('sa','signal',verbose=False)
        signal = data['y']
        N = len(signal)
        df = data['dt']
        f0 = data['t0']
        freq_range = np.arange(f0, f0 + (N-0.5) * df, df)
        # print(len(freq_range),N)
        assert len(freq_range) == N
        
        peak_loc = np.argmax(signal)
        peak_loc_in_freq = freq_range[peak_loc]
        peak_power = signal[peak_loc]

        if plot:
            plt.plot(freq_range * 1e-6, signal, '-')
            plt.xlabel('Frequency [MHz]')
            plt.ylabel('Power [dBm]')
            plt.title(f'P= {peak_power:0.1f} dBm at {peak_loc_in_freq/1e9:.5f} GHz')
            plt.show()
            print(f'{peak_power} dBm at {peak_loc_in_freq/1e9} GHz')

        return peak_power

    #%% opt_leakage
    def opt_lo_leakage(self,  mode, element = 'readout', sa_span = 1e6, threshold = -30, plot = True):
        """
        Minimizes leakage at LO ('lo' option) or at image sideband ('sb' option) by sweeping the relevant parameters

        Args:
            sa (): spectrum analyzer handle.
            mode (str): Coarse of fine stepsize.
            element (str):       Which element to optimize (qubit (qubit) or readout (rr)).
            pars (dict):        dictionary containing experimental values like mixer calibration offsets.
            reference (float): Threshold of spectrum analyzer.
            plot (TYPE, optional): DESCRIPTION. Defaults to False.

        Returns:
            values (float array): optimal calibration values.
            argmin (TYPE): DESCRIPTION.

        """

        # gets frequency values
        freqLO = self.pars[f'{element}_LO']
        if element == 'rr':
            atten = self.pars['readout_atten']
            self.update_value('readout_atten', 0)
        leakage = self.get_power(fc=freqLO, threshold=threshold, sa_span=sa_span, config=True)
        freq = freqLO
        par1, par2 = self.pars[f'{element}_mixer_offsets']
        print(f'LO at {round(freqLO*1e-9,5)} GHz\nCurrent I_offset = {round(par1*1e3,1)} mV, Current Q_offset = {round(par2*1e3,1)} mV')        

        self.config_sa(freq, sa_span=sa_span, threshold=leakage+20) # setup spectrum analyzer for measurement

        # initialize sweep parameters
        if mode == 'coarse':
            span=10.1e-3
            step=1e-3
        elif mode == 'fine':
            span=1e-3
            step=0.1e-3

        par1_arr = np.arange(par1-span/2, par1+span/2, step)
        par2_arr = np.arange(par2-span/2, par2+span/2, step)
        L1 = len(par1_arr)
        L2 = len(par2_arr)
        power_data = np.zeros((L1,L2))

        # qm = self.qmm.open_qm(self._config)
        qm = self.qmm.open_qm(self.config)
        # sweep parameters and get power at every point
        with tqdm(total = L1*L2) as progress_bar:
            for i, par1 in enumerate((par1_arr)):
                for j, par2 in enumerate((par2_arr)):
                    qm.set_output_dc_offset_by_element(element, "I", par1)
                    qm.set_output_dc_offset_by_element(element, "Q", par2)
                    time.sleep(0.01)
                    power_data[i,j] = self.get_power()
                    progress_bar.update(1)

        argmin = np.unravel_index(np.argmin(power_data), power_data.shape)

        # set the parameters to the optimal values and modify the JSON dictionary
        opt_I = par1_arr[argmin[0]]
        opt_Q = par2_arr[argmin[1]]
        qm.set_output_dc_offset_by_element(element, "I", opt_I)
        qm.set_output_dc_offset_by_element(element, "Q", opt_Q)
        self.update_value(f'{element}_mixer_offsets', [opt_I, opt_Q])

        print(f'optimal I_offset = {round(opt_I*1e3,1)} mV, optimal Q_offset = {round(1e3*opt_Q,1)} mV')

        if element == 'rr':
            self.update_value('readout_atten', atten)

        print(f'Power: {np.amin(power_data)} dBm at {freq/1e9} GHz')

        if plot:
            pf.plot_mixer_opt(par1_arr, par2_arr, power_data, cal='LO', element=element, fc=freq)
        else:
            pass
    
    #%% opt_sideband
    def opt_sideband(self,  mode, amplitude=0.2, element = 'readout', sa_span = 1e6, threshold = -30, plot = True):
            """
            Minimizes leakage at LO ('lo' option) or at image sideband ('sb' option) by sweeping the relevant parameters

            Args:
                sa (): spectrum analyzer handle.
                mode (str): Coarse of fine stepsize.
                element (str):       Which element to optimize (qubit (qubit) or readout (rr)).
                pars (dict):        dictionary containing experimental values like mixer calibration offsets.
                reference (float): Threshold of spectrum analyzer.
                plot (TYPE, optional): DESCRIPTION. Defaults to False.

            Returns:
                values (float array): optimal calibration values.
                argmin (TYPE): DESCRIPTION.

            """
            
            qm,job = self.play_pulses(amplitude=amplitude, element=element) # to generate sidebands
            if element == 'rr':
                atten = self.pars['readout_atten']
                self.update_value('readout_atten', 0)
            
            # gets frequency values
            freqLO = self.pars[f'{element}_LO']
            freqIF = self.pars[f'{element}_IF']

            freq = freqLO - freqIF
            leakage = self.get_power(fc=freq, threshold=threshold, sa_span=sa_span, config=True,plot=True)
            centers = self.pars[f'{element}_mixer_imbalance']
            print(f'Sideband at {round((freq)*1e-9,5)} GHz\nCurrent gain = {round(centers[0],4)}, Current phase = {round(centers[1],4)}')

        

            # initialize sweep parameters
        
            if mode == 'coarse':
                span = [0.2,0.5]
                n_steps = [10,10]
                
            elif mode == 'fine':
                span = [0.05,0.1]
                n_steps = [10,10]
                
            
            gain = np.linspace(centers[0]-span[0]/2, centers[0]+span[0]/2, n_steps[0])
            phase = np.linspace(centers[1]-span[1]/2, centers[1]+span[1]/2, n_steps[1])
            L1 = len(gain)
            L2 = len(phase)
            power_data = np.zeros((L1,L2))

            # sweep parameters and get power at every point
            with tqdm(total = L1*L2) as progress_bar:
                for i, amp in enumerate(gain):
                    for j, phi in enumerate(phase):
                        # qm,job = self.play_pulses(amplitude=amplitude, element=element)
                        qm.set_mixer_correction(element, int(freqIF), int(freqLO), IQ_imbalance(amp, phi))
                        # job.set_element_correction(element, IQ_imbalance(amp, phi))
                        time.sleep(0.01)
                        power_data[i,j] = self.get_power()
                        progress_bar.update(1)

            argmin = np.unravel_index(np.argmin(power_data), power_data.shape)
            
            # set the parameters to the optimal values and modify the JSON dictionary

            imbalance = gain[argmin[0]], phase[argmin[1]]
            qm.set_mixer_correction(element, int(freqIF), int(freqLO), IQ_imbalance(*imbalance)) # have to set the mixer correction using this method after the optimization has finished
            self.update_value(f'{element}_mixer_imbalance', imbalance)

            print(f"optimal gain = {round(gain[argmin[0]],4)}, optimal phi = {round(phase[argmin[1]],4)}")

            if element == 'rr':
                self.update_value('readout_atten', atten)

            print(f'Power: {np.amin(power_data)} dBm at {freq/1e9} GHz')
            if plot:
                pf.plot_mixer_opt(gain, phase, power_data, cal='SB', element=element, fc=freq)


#%%% Macros

   

    #%%%% IQ_imbalance
    def IQ_imbalance(self,g, phi):
        c = np.cos(phi)
        s = np.sin(phi)
        N = 1 / ((1-g**2)*(2*c**2-1))
        return [float(N * x) for x in [(1-g)*c, (1+g)*s, (1-g)*s, (1+g)*c]]

    #%%%% delayed_gauss
    def delayed_gauss(self,amp=0.45, length=4, sigma=2):
        gauss_arg = np.linspace(-sigma, sigma, length)
        delay = 16 - length - 4
        if delay < 0:
            return amp * np.exp(-(gauss_arg ** 2) / 2)

        return np.r_[np.zeros(delay), amp * np.exp(-(gauss_arg ** 2) / 2), np.zeros(4)]

    #%%%%
    def init_instruments(self):

        self._instruments.set('readout_LO', 'frequency', self.pars['rr_LO'])
        self._instruments.set('readout_LO', 'output', True)
        self._instruments.set('qubit_LO', 'frequency', self.pars['qubit_LO'])
        self._instruments.set('qubit_LO', 'power', 17)
        self._instruments.set('qubit_LO', 'output', True)
        self._instruments.set('DA', 'attenuation', self.pars['readout_atten'])

    #%%%% update_value
    def update_value(self,key,value):
        if key in self.pars.keys():
            print(f'Updating {key} to {value}')
            self.pars[key] = value

            self.config = self.config_maker.update_configuration(new_pars=self.pars)
            self.write_pars()

            if key == 'qubit_LO':
                self._instruments.set('qubit_LO','frequency',value)
            elif key == 'rr_LO':
                self._instruments.set('readout_LO','frequency',value)
            elif key == 'readout_atten':
                self._instruments.set('DA','attenuation',value)
            elif key == 'qubit_freq':
                self.update_value('qubit_IF',self.pars['qubit_freq']-self.pars['qubit_LO'])            
        else:
            warnings.warn(f"key to update does not exist, creating new key: {key} in qb.pars")
            self.pars[key] = value

    #%%%% write_pars
    def write_pars(self):
        with open(f'{self._name}_pars.json', "w") as outfile:
            json.dump(self.pars, outfile)

    #%%%% remove_key
    def remove_key(self, key):
        print(f'Removing {key} from pars')
        del self.pars[key]
        self.write_pars()

    #%%%% add_key
    def add_key(self, key, value):
        print(f'Adding {key} = {value} to pars')
        self.pars[key] = value
        self.write_pars()

#%% GRAVEYARD

#%%% MACROS
    #%%%% pi_pulse
    # play pi pulse, measure, and wait
    def pi_pulse(self, I, Q, resettime = 400e3):

        align("qubit", "rr")
        play("pi", "qubit")
        align("qubit", "rr")
        measure("readout", "rr", None, *self.res_demod(I, Q))
        wait(clk(resettime), "qubit")
        align("qubit", "rr")

        return I, Q

    #%%%% save_many
    # this needs to be tested...
    # save multiple variables to multiple streams in one line

    # example: instead of writing
    # save(I, I_st)
    # save(Q, Q_st)

    # can write
    # self.save_many([I, I_st], [Q, Q_st])
    def save_many(*pairs):

        for (var, var_stream) in pairs:

            save(var, var_stream)


 #%%% UNIT TESTS
    #%%%% pi pulse
    def test_pi_pulse(self, n_avg = 5000, tolerance = 1e-3):

        with program() as PiTest:

            n, I, Q = self.declare_vars([int, fixed, fixed])
            I_st, Q_st = self.declare_streams(stream_num=2)

            with for_(n, 0, n < n_avg, n + 1):
                wait(clk(100e3),'rr')
                # first, measure without doing pi pulse
                measure("readout", "rr", None, *self.res_demod(I, Q))
                wait(clk(100e3), "rr")
                save(I, I_st)
                save(Q, Q_st)

                # then, measure after doing pi pulse
                I, Q = self.pi_pulse(I, Q)
                save(I, I_st)
                save(Q, Q_st)

            with stream_processing():
                I_st.buffer(2).average().save("I")
                Q_st.buffer(2).average().save("Q")

        # run job
        datadict, _ = self.get_results(jobtype = PiTest,
                                         result_names = ["I", "Q"],
                                         n_total=n_avg,
                                         showprogress=False,
                                         notify = False)
        # process data from job
        I0, I1 = datadict["I"]
        Q0, Q1 = datadict["Q"]
        p0 = I0 + 1j * Q0
        p1 = I1 + 1j * Q1
        mag1 = round(np.abs(p1) * 1e3, ndigits=3)
        mag0 = round(np.abs(p0) * 1e3, ndigits=3)
        dist = round(mag1 - mag0, ndigits=3)

        print(f"Distinguishability: {mag1} mV - {mag0} mV = {dist} mV")

        ### eventually, include code that checks that this matches what
        # we expect from power rabi to within {tolerance}

        return datadict

#%%%% test helpers

    # e.g. operation = "pi"
    def get_pulse(self, operation, element):
        return self.config["elements"][element]["operations"][operation]

    def get_waveform(self, operation, element, quad = "I"):
        pulse = self.get_pulse(operation, element)
        return self.config["pulses"][pulse]["waveforms"][quad]

    def waveform(self, operation, element):
        waveform_name = self.get_waveform(operation, element)
        return self.config["waveforms"][waveform_name]["samples"]


    #%%% IQ_blobs
    """
IQ_blobs.py: Measure the qubit in the ground and excited state to create the IQ blobs.
If the separation and the fidelity are good enough, gives the parameters needed for active reset
"""


# ###################
# # The QUA program #
# ###################

# n_runs = 10000

# cooldown_time = 5 * qubit_T1 // 4

# with program() as IQ_blobs:
#     n = declare(int)
#     I_g = declare(fixed)
#     Q_g = declare(fixed)
#     I_g_st = declare_stream()
#     Q_g_st = declare_stream()
#     I_e = declare(fixed)
#     Q_e = declare(fixed)
#     I_e_st = declare_stream()
#     Q_e_st = declare_stream()

#     with for_(n, 0, n < n_runs, n + 1):
#         measure(
#             "readout",
#             "resonator",
#             None,
#             dual_demod.full("rotated_cos", "out1", "rotated_sin", "out2", I_g),
#             dual_demod.full("rotated_minus_sin", "out1", "rotated_cos", "out2", Q_g),
#         )
#         save(I_g, I_g_st)
#         save(Q_g, Q_g_st)
#         wait(cooldown_time, "resonator")

#         align()  # global align

#         play("pi", "qubit")
#         align("qubit", "resonator")
#         measure(
#             "readout",
#             "resonator",
#             None,
#             dual_demod.full("rotated_cos", "out1", "rotated_sin", "out2", I_e),
#             dual_demod.full("rotated_minus_sin", "out1", "rotated_cos", "out2", Q_e),
#         )
#         save(I_e, I_e_st)
#         save(Q_e, Q_e_st)
#         wait(cooldown_time, "resonator")

#     with stream_processing():
#         I_g_st.save_all("I_g")
#         Q_g_st.save_all("Q_g")
#         I_e_st.save_all("I_e")
#         Q_e_st.save_all("Q_e")

# #####################################
# #  Open Communication with the QOP  #
# #####################################
# qmm = QuantumMachinesManager(qop_ip)

# qm = qmm.open_qm(config)

# job = qm.execute(IQ_blobs)
# res_handles = job.result_handles
# res_handles.wait_for_all_values()
# Ig = res_handles.get("I_g").fetch_all()["value"]
# Qg = res_handles.get("Q_g").fetch_all()["value"]
# Ie = res_handles.get("I_e").fetch_all()["value"]
# Qe = res_handles.get("Q_e").fetch_all()["value"]

# angle, threshold, fidelity, gg, ge, eg, ee = two_state_discriminator(Ig, Qg, Ie, Qe, b_print=True, b_plot=True)

#########################################
# The two_state_discriminator gives us the rotation angle which makes it such that all of the information will be in
# the I axis. This is being done by setting the `rotation_angle` parameter in the configuration.
# See this for more information: https://qm-docs.qualang.io/guides/demod#rotating-the-iq-plane
# Once we do this, we can perform active reset using:
#########################################
#
# # Active reset:
# with if_(I > threshold):
#     play("pi", "qubit")
#
#########################################
#
# # Active reset (faster):
# play("pi", "qubit", condition=I > threshold)
#
#########################################
#
# # Repeat until success active reset
# with while_(I > threshold):
#     play("pi", "qubit")
#     align("qubit", "resonator")
#     measure("readout", "resonator", None,
#                 dual_demod.full("rotated_cos", "out1", "rotated_sin", "out2", I))
#
#########################################
#
# # Repeat until success active reset, up to 3 iterations
# count = declare(int)
# assign(count, 0)
# cont_condition = declare(bool)
# assign(cont_condition, ((I > threshold) & (count < 3)))
# with while_(cont_condition):
#     play("pi", "qubit")
#     align("qubit", "resonator")
#     measure("readout", "resonator", None,
#                 dual_demod.full("rotated_cos", "out1", "rotated_sin", "out2", I))
#     assign(count, count + 1)
#     assign(cont_condition, ((I > threshold) & (count < 3)))
#
