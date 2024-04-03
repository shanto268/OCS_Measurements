# -*- coding: utf-8 -*-
"""
Created on Wed Apr 19 00:12:50 2023

@author: lfl
"""
from qubit import *
import h5py
from datetime import datetime
import os
from instrument_init import init_sa, init_sa_by_serial_number
import timeit

class dissipator(qubit):
    
    def __init__(self, qb,device_name):
        super.__init__(self, qb)
        self.device_name = device_name
        
    def optimize_mixer(self,sa, element='rr', cal='LO', switch='on'):
        ref_H = -10
        ref_L = -50

        self.play_pulses(element, switch=switch)
        if element == 'rr':
            inst.set_attenuator(0)
            inst.get_attenuation()
        if switch == 'off':
            ref = ref_L
        else:
            ref = ref_H
        if cal == 'LO':
            fc = self.pars[f'{element}_LO']
        elif cal == 'SB':
            fc = self.pars[f'{element}_LO'] -  self.pars[f'{element}_IF']
        print(f'Checking {element} {cal}')
        qb_lo_leakage = self.get_power(sa, freq = fc, reference = -20, config = True, amp_q = 1, plot = False)
        #qb_lo_leakage = self.get_power(sa, freq=self.pars[f'{element}_LO'],reference=ref,config=True,plot=True)
        #qb_im_leakage = self.get_power(sa, freq=self.pars[f'{element}_LO']-self.pars[f'{element}_IF'],reference=ref,config=True,plot=True)
        #qb_on_power = self.get_power(sa, freq=self.pars[f'{element}_LO']+self.pars[f'{element}_IF'],reference=ref, config=True,plot=True) # reference should be set ABOVE expected image power
        
        j=0
        while qb_lo_leakage > -75:
                    leak0 = qb_lo_leakage
                    print(f'Minimizing {element} {cal} leakage')
                    if qb_lo_leakage > - 42:
                        self.opt_mixer(sa,cal = cal, mode = 'coarse', amp_q = 1., element = element, reference = ref_H, switch='on', plot=False)
                        ref=ref_H
                    elif qb_lo_leakage < - 42 and qb_lo_leakage > - 58:
                        self.opt_mixer(sa, cal = cal, mode = 'intermediate', amp_q = 1., element = element, reference = ref_H, switch='on',  plot=False)
                        ref=ref_H
                    elif qb_lo_leakage < - 58 and qb_lo_leakage > - 75:
                        self.opt_mixer(sa, cal = cal, mode = 'fine', amp_q = 1., element = element, reference = ref_L, switch='on',  plot=False)
                        ref=ref_L


                    qb_lo_leakage = self.get_power(sa,freq = fc,reference = ref, amp_q = 1., config = True, plot = False)
                    j+=1
                    if j>3:
                        if np.abs(qb_lo_leakage - leak0) < 2:
                            print("Can't optimize mixer further")
                            break
                    if j==4:
                        break
        self.write_pars()
        
    def mixer_powers(self, sa, element='rr', switch='on'):
        ref_H = -20
        ref_L = -45
        self.play_pulses(element, switch=switch)
        qb_lo_leakage = self.get_power(sa, freq=self.pars[f'{element}_LO'],reference=ref_H,config=True,plot=False)
        qb_im_leakage = self.get_power(sa, freq=self.pars[f'{element}_LO']-self.pars[f'{element}_IF'],reference=ref_H,config=True,plot=False)
        qb_on_power = self.get_power(sa, freq=self.pars[f'{element}_LO']+self.pars[f'{element}_IF'],reference=ref_H, config=True,plot=False) # reference should be set ABOVE expected image power
        print(f'{element}_lo_leakage = {qb_lo_leakage} dB')
        return {f'{element}_lo_leakage': qb_lo_leakage, 
                f'{element}_im_leakage': qb_im_leakage,
                f'{element}_on_power': qb_on_power,}
    
    def ffl_spec(self,
                 element = 'ffl',
                 f_LO = 7e9,
                 IF_min = 0.1e6,
                 IF_max = 400e6,
                 df = 0.1e6,
                 amp_ffl_scale = 1,
                 amp_r_scale = 1,
                 n_avg = 2000,
                 savedata = True,
                 check_mixers = False,
                 plot = True):
        # TODO: modify for direct spec
        # save data
        today = datetime.today()
        sDate =  today.strftime("%Y%m%d")
        saveDir = f'G:\\Shared drives\\CavityCooling\data\\{self.device_name}\\{sDate}'
        
        dataPath = f'{saveDir}\\spectroscopy\\{element}_spec'
        filename = 'data'
        iteration = get_index_for_filename(dataPath, filename, file_format='csv')
            
        freqs = np.arange(IF_min, IF_max + df/2, df, dtype=int)
        # freqs_list = freqs.tolist()
        # set attenuation and change rr_LO freq
        inst.set_attenuator(attenuation=self.pars['rr_atten'])
        inst.set_ffl_attenuator(self.pars['ffl_atten'])
        inst.set_rr_LO(self.pars['rr_LO'])
        self.update_value('ffl_LO', f_LO)
        inst.set_ffl_LO(self.pars['ffl_LO'])
        self.update_value('ffl_IF', int(IF_max/2))
        if check_mixers:
            sa = inst.init_sa()
            self.play_pulses(element='ffl', switch='off')
            self.optimize_mixer(sa, element='ffl',cal='LO', switch='off')
            #self.optimize_mixer(sa, element='ffl',cal='SB',switch='off')
            sa_close_device(sa)
        resettime_clk = clk(self.pars['rr_resettime'])
        ### QUA code ###
        with program() as ffl_spec:
    
            n = declare(int)
            I = declare(fixed)
            I_st = declare_stream()
            Q = declare(fixed)
            Q_st = declare_stream()
            f = declare(int)
            n_stream = declare_stream()
            
            I_b, Q_b, I_tot,Q_tot = self.declare_vars([fixed,fixed, fixed, fixed])
            
            update_frequency("rr", self.pars['rr_freq'] - self.pars['rr_LO'])
            with for_(n, 0, n < n_avg, n + 1):
                with for_(f, IF_min, f < IF_max + df/2, f + df):
                    update_frequency("ffl", f)
                    
                    measure("readout", "rr", None, *self.res_demod(I_b, Q_b))
                    wait(resettime_clk, "rr")
                    align("rr", "ffl")
                    play('const'*amp(amp_ffl_scale), "ffl", duration=self.pars['rr_pulse_len_in_clk'])
                    measure("readout", "rr", None,*self.res_demod(I,Q))
                    wait(resettime_clk, "rr")
                    assign(I_tot, I-I_b)
                    assign(Q_tot, Q-Q_b)
                    save(I_tot, I_st)
                    save(Q_tot, Q_st)
    
                save(n,n_stream)
    
            with stream_processing():
                I_st.buffer(len(freqs)).average().save('I')
                Q_st.buffer(len(freqs)).average().save('Q')
                n_stream.save('n')
    
        datadict,job = self.get_results(ffl_spec,result_names=["I","Q","n"],n_total=n_avg,progress_key = "n", showprogress=True)
    
        I = np.array(datadict["I"])
        Q = np.array(datadict["Q"])
        freq_arr = np.array(freqs + self.pars['ffl_LO'])
    
        if plot:
            pf.spec_plot(freq_arr,I,Q,attenuation=self.pars['ffl_atten'],df=df,iteration=iteration,element='ffl')
    
        exp_dict = {'date/time':    datetime.now(),
                   'nAverages': n_avg,
                         'ffl_LO': self.pars['ffl_LO'],
                'wait_period': self.pars['rr_resettime'],
                'report': str(job.execution_report()),
                }
        if savedata:
            # save data
            if not os.path.exists(dataPath):
                Path(dataPath).mkdir(parents=True, exist_ok=True)
            with open(f"{dataPath}\data_{iteration:03d}.csv","w") as datafile:
                writer = csv.writer(datafile)
                writer.writerow(exp_dict.keys())
                writer.writerow(exp_dict.values())
                writer.writerow(freqs)
                writer.writerow(I)
                writer.writerow(Q)
    
        return I, Q, freq_arr, job;
    
    def ffl_spec_sweep_lo(self, 
                          element='ffl',
                          fmin=2.4e9,
                          fmax=3.6e9,
                          df=10e6,
                          n_avg=10000,
                          amp_ffl_scale=1,
                          plot = True,
                          savedata = True,
                          bOptimizerrMixer = True, onoff = True):
        start = timeit.default_timer()
        current = inst.get_ffl_bias()
        if bOptimizerrMixer:
            sa = inst.init_sa()
            self.play_pulses(element='rr')
            self.optimize_mixer(sa, element='rr',cal='LO')
            self.optimize_mixer(sa, element='rr',cal='SB')
            sa_close_device(sa)
        # I, Q, freq, job = self.ffl_spec(f_LO = self.pars['ffl_LO'])
        #I, Q, freqs, job = self.resonator_spec(f_LO=self.pars['rr_LO'],atten=self.pars['rr_atten'],IF_min=192e6,IF_max=200e6,df=50e3,n_avg=3000,savedata=True)
        #fc,fwhm = pf.fit_res(freqs,np.abs(I+1j*Q))
        #self.update_value('rr_freq', fc)
        # I, Q, freq, job = self.run_scan(element='ffl',lo_min = 2.5e9, lo_max=3.5e9, chunksize=400e6, amp_q_scaling=0.9,n_avg = 1000)
        # I, Q, freq, job = self.ffl_spec_sweep_lo(n_avg = 100)

        today = datetime.today()
        sDate =  today.strftime("%Y%m%d")
        saveDir = f'G:\\Shared drives\\CavityCooling\data\\{self.device_name}\\{sDate}'

        dataPath = f'{saveDir}\\spectroscopy\\{element}_spec'
        filename = 'data'
        iteration = get_index_for_filename(dataPath, filename, file_format='csv')
            
        freqs = np.arange(fmin, fmax + df/2, df)
        Idata = np.zeros((len(freqs),))
        Qdata = np.zeros((len(freqs),))
        inst.set_attenuator(attenuation=self.pars['rr_atten'])
        inst.set_ffl_attenuator(self.pars['ffl_atten'])
        inst.set_rr_LO(self.pars['rr_LO'])
        self.update_value('ffl_IF', 0)
        resettime_clk = clk(self.pars['rr_resettime'])

        with program() as ffl_spec:
          
            I = declare(fixed)
            I_st = declare_stream()
            Q = declare(fixed)
            Q_st = declare_stream()
            n = declare(int)
            n_stream = declare_stream()
            update_frequency("ffl", self.pars["ffl_IF"])
            update_frequency("rr", self.pars['rr_freq'] - self.pars['rr_LO'])
            
            with for_(n, 0, n < n_avg, n + 1):
                wait(resettime_clk, "rr")
                align("rr", "ffl")
                play('const'*amp(amp_ffl_scale), "ffl", duration=self.pars['rr_pulse_len_in_clk'])
                measure("readout", "rr", None,*self.res_demod(I,Q))
                save(I, I_st)
                save(Q, Q_st)
                save(n,n_stream)

            with stream_processing():
                I_st.buffer(1).average().save('I')
                Q_st.buffer(1).average().save('Q')
                n_stream.save('n')
                
        
        with program() as ffl_res_spec:
          
            I = declare(fixed)
            I_st = declare_stream()
            Q = declare(fixed)
            Q_st = declare_stream()
            n = declare(int)
            n_stream = declare_stream()
            
            update_frequency("rr", self.pars['rr_freq'] - self.pars['rr_LO'])
            update_frequency("ffl", self.pars["ffl_IF"])
            
            with for_(n, 0, n < n_avg, n + 1):
                wait(resettime_clk, "rr")
                align("rr", "ffl")
                play('const'*amp(amp_ffl_scale), "ffl", duration=self.pars['rr_pulse_len_in_clk'])
                measure("readout", "rr", None,*self.res_demod(I,Q))
                save(I, I_st)
                save(Q, Q_st)
                save(n,n_stream)

            with stream_processing():
                I_st.buffer(1).average().save('I')
                Q_st.buffer(1).average().save('Q')
                n_stream.save('n')
                

        with tqdm(total=len(freqs), position=0, leave=True) as pbar:
            for i,freq in enumerate(freqs):
                if onoff:
                    inst.turn_off_ffl_drive()
                    datadict,job = self.get_results(ffl_spec,result_names=["I","Q","n"],n_total=n_avg,progress_key = "n",showprogress=False)
    
                    I_b = np.array(datadict["I"])[0]
                    Q_b = np.array(datadict["Q"])[0]
                self.update_value('ffl_LO', freq)
                inst.set_ffl_LO(freq, bprint=False)
                datadict,job = self.get_results(ffl_spec,result_names=["I","Q","n"],n_total=n_avg,progress_key = "n",showprogress=False)
                I_tot = np.array(datadict["I"])[0]
                Q_tot = np.array(datadict["Q"])[0]
                
                if onoff:
                    Idata[i] = float(I_tot - I_b)
                    Qdata[i] = float(Q_tot - Q_b)
                
                else:
                    Idata[i] = float(I_tot)
                    Qdata[i] = float(Q_tot)
                
                
        if plot:
            pf.spec_plot(freqs,Idata,Qdata,attenuation=self.pars['ffl_atten'],df=df,iteration=iteration,element='ffl',current=current)
        
        exp_dict = {'date/time':    datetime.now(),
                    'flux': current,
                   'nAverages': n_avg,
                         'ffl_LO': self.pars['ffl_LO'],
                'wait_period': self.pars['rr_resettime'],
                'report': str(job.execution_report()),
                }
        if savedata:
            # save data
            if not os.path.exists(dataPath):
                Path(dataPath).mkdir(parents=True, exist_ok=True)
            with open(f"{dataPath}\data_flux={round(current*1e6)}uA_{iteration:03d}.csv","w") as datafile:
                writer = csv.writer(datafile)
                writer.writerow(exp_dict.keys())
                writer.writerow(exp_dict.values())
                writer.writerow(freqs)
                writer.writerow(Idata)
                writer.writerow(Qdata)
                
        stop = timeit.default_timer()
        print('Time: ', stop - start) 
    
        return Idata, Qdata, freqs, job;
    
    

    #TODO: estimate flux modulation
    
   