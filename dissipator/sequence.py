# -*- coding: utf-8 -*-
"""
Created on Fri Apr 28 13:00:18 2023

@author: lfl
"""
from qm import generate_qua_script
from qm.qua import *
from qm import LoopbackInterface
from qm.QuantumMachinesManager import QuantumMachinesManager
from qm import SimulationConfig
import instrument_init as inst
import numpy as np
# from dissipator import *
from Utilities import clk
host='10.71.0.56'
port='9510'

class sequence():
    
    # def __init__(self, name, n_avg=100, amp_r_scale=1, amp_ffl_scale=1, **kwargs):
    def __init__(self,qb,name,**kwargs):
        self.name = name
        self.seq_pars = qb.seq_pars
        self.qb_pars = qb.pars
        # self.exp_dict = {'n_avg': n_avg,
        #                  'amp_r_scale': amp_r_scale,
        #                  'amp_ffl_scale': amp_ffl_scale}
        for key in kwargs.keys():
            self.seq_pars[key] = kwargs.get(key)
        
        
    # def make_sequence(self, qb, tmin = 16, tmax=1e3, dt=8, scrambling_amp=0., ffl_len=2e3, saturation_dur=2e3, with_ffl='True', with_scram='True', detuning=0):
        

    def make_resonator_spec_sequence(self,saturation_dur=2e3,):
        n_avg = self.seq_pars['n_avg']
        IF_min = self.seq_pars['IF_min']
        IF_max = self.seq_pars['IF_max']
        df = self.seq_pars['df']
        freqs = np.arange(IF_min, IF_max + df/2, df, dtype=int)
        rr_pulse_len_in_clk = self.seq_pars['readout_length']
        res_ringdown_time = clk(self.seq_pars['rr_resettime'])
        ### QUA code ###
        with program() as prog:
            
            n = declare(int)
            I = declare(fixed)
            I_st = declare_stream()
            Q = declare(fixed)
            Q_st = declare_stream()
            f = declare(int)
            n_stream = declare_stream()

            with for_(n, 0, n < n_avg, n + 1):

                # with for_each_(f, freqs_list):
                with for_(f, IF_min, f < IF_max + df/2, f + df):
                    update_frequency("rr", f)
                    wait(res_ringdown_time, "rr")
            
                    measure("readout", "rr", None,*qb.res_demod(I, Q,switch_weights=False))
                    save(I, I_st)
                    save(Q, Q_st)

                save(n,n_stream)

            with stream_processing():
                I_st.buffer(len(freqs)).average().save('I')
                Q_st.buffer(len(freqs)).average().save('Q')
                n_stream.save('n')

        return prog
                    
           
    
    def make_qubit_spec_sequence():
        resettime_clk= clk(qb.pars['qubit_resettime'])
        n_avg = self.exp_dict['n_avg']
        IF_min = self.exp_dict['IF_min']
        IF_max = self.exp_dict['IF_max']
        df = self.exp_dict['df']
        amp_ffl_scale = self.exp_dict['amp_ffl_scale']
        freqs = np.arange(IF_min, IF_max + df/2, df, dtype=int)
        rr_pulse_len_in_clk = qb.pars['qb_resettime']
        with program() as prog:
            update_frequency('rr', qb.pars['rr_IF'])
            update_frequency('qubit', (qb.pars['qubit_freq']-qb.pars['qubit_LO']))
            update_frequency('ffl', (qb.pars['ffl_freq']-qb.pars['ffl_LO'])) 
            n, I, Q = qb.declare_vars([int, fixed, fixed])
            
            I_stream, Q_stream, n_stream = qb.declare_streams(stream_num=3)
            f = declare(int)
            
            with for_(n, 0, n < n_avg, n + 1):
                save(n, n_stream)
                with for_(f, IF_min, f < IF_max + df/2, f + df):
                    update_frequency("fflqc", f)
                    wait(resettime_clk)
                    #align("qubit","rr")
                    play("pi", "qubit")
                    align("fflqc", "rr")
                    play('const'*amp(amp_ffl_scale), "ffl", duration=750)
                    align("fflqc", "rr")
                    measure("readout", "rr", None, *qb.res_demod(I, Q))
                    
                    save(I, I_stream)
                    save(Q, Q_stream)
        
            with stream_processing():
                I_stream.buffer(len(freqs)).average().save('I')
                Q_stream.buffer(len(freqs)).average().save('Q')
                n_stream.save('n')

        return prog
        
            # if self.name == 'ringdown_drive_on':
            # tmin = clk(tmin)
            # tmax = clk(tmax)
            # dt = clk(dt)
            # t_arr = np.arange(tmin, tmax + dt/2, dt, dtype = int)
            # resettime_clk= clk(qb.pars['rr_resettime'])
            # n_avg = self.exp_dict['n_avg']
            # amp_r_scale = self.exp_dict['amp_r_scale']
            # amp_ffl_scale = self.exp_dict['amp_ffl_scale']
            # rr_IF = qb.pars['rr_IF']
            # ffl_IF = qb.pars['ffl_IF']
            # with program() as prog:
            #     update_frequency('rr', rr_IF) 
            #     update_frequency('ffl', ffl_IF) 
            #     n, t, I, Q = qb.declare_vars([int, int, fixed, fixed])
            
            #     I_stream, Q_stream, n_stream = qb.declare_streams(stream_num=3)
                
            #     with for_(n, 0, n < n_avg, n + 1):
            #         save(n, n_stream)
            #         with for_each_(t, t_arr):
            #             with if_(t==0):
            #                 play("readout"*amp(amp_r_scale), "rr")
            #                 measure("void", "rr", None,*qb.res_demod(I,Q))
            #                 wait(resettime_clk, "rr")
            #                 save(I, I_stream)
            #                 save(Q, Q_stream)
            #             with else_():
            #                 play("readout"*amp(amp_r_scale), "rr")
            #                 align("ffl", "rr")
            #                 play('gaussian'*amp(amp_ffl_scale), "ffl", duration=t)
            #                 align("ffl", "rr")
            #                 measure("void", "rr", None,*qb.res_demod(I,Q))
            #                 wait(resettime_clk, "rr")
            #                 save(I, I_stream)
            #                 save(Q, Q_stream)
            
            #     with stream_processing():
            #         I_stream.buffer(len(t_arr)).average().save("I")
            #         Q_stream.buffer(len(t_arr)).average().save("Q")
            #         n_stream.save('n')   
        # elif self.name =='spec_wffl':
        #     resettime_clk= clk(qb.pars['qubit_resettime'])
        #     n_avg = self.exp_dict['n_avg']
        #     IF_min = self.exp_dict['IF_min']
        #     IF_max = self.exp_dict['IF_max']
        #     df = self.exp_dict['df']
        #     amp_ffl_scale = self.exp_dict['amp_ffl_scale']
        #     freqs = np.arange(IF_min, IF_max + df/2, df, dtype=int)
        #     rr_pulse_len_in_clk = qb.pars['rr_pulse_len_in_clk']
        #     with program() as prog:
        #         update_frequency('rr', qb.pars['rr_IF'])
        #         update_frequency('qubit', (qb.pars['qubit_freq']-qb.pars['qubit_LO']))
        #         update_frequency('ffl', (qb.pars['ffl_freq']-qb.pars['ffl_LO'])) 
        #         n, I, Q = qb.declare_vars([int, fixed, fixed])
                
        #         I_stream, Q_stream, n_stream = qb.declare_streams(stream_num=3)
        #         f = declare(int)
                
        #         with for_(n, 0, n < n_avg, n + 1):
        #             save(n, n_stream)
        #             with for_(f, IF_min, f < IF_max + df/2, f + df):
        #                 update_frequency("rr", f)
        #                 wait(resettime_clk, "rr")
        #                 #align("qubit","rr")
        #                 #play("pi", "qubit")
        #                 align("ffl", "rr")
        #                 play('const'*amp(amp_ffl_scale), "ffl", duration=9000)
        #                 align("ffl", "rr")
        #                 measure("readout", "rr", None, *qb.res_demod(I, Q))
                        
        #                 save(I, I_stream)
        #                 save(Q, Q_stream)
            
        #         with stream_processing():
        #             I_stream.buffer(len(freqs)).average().save('I')
        #             Q_stream.buffer(len(freqs)).average().save('Q')
        #             n_stream.save('n')

        # def make_qubit_reset_sequence():
        #     ##start with a scrambling pulse of length saturation_dur and amplitude scrambling amp on the qubit and then play (or dont play) ffl pulse for length ffl_len and then do rabi.
        #     resettime_clk= clk(qb.pars['qubit_resettime'])
        #     n_avg = self.exp_dict['n_avg']
        #     tmin = clk(tmin)
        #     tmax = clk(tmax)
        #     dt = clk(dt)
        #     amp_ffl_scale = self.exp_dict['amp_ffl_scale']
        #     amp_q_scaling=1.
        #     var_arr = np.arange(tmin, tmax + dt/2, dt, dtype = int)
        #     with program() as prog:
        #         # n = declare(int)
        #         # I=declare(fixed)
        #         # I_st = declare_stream()
        #         # Q = declare(fixed)
        #         # Q_st = declare_stream()
        #         # n_st=declare_stream()
        #         n, t, I, Q = qb.declare_vars([int, int, fixed, fixed])

        #         I_st, Q_st, n_st = qb.declare_streams(stream_num=3)

        #         update_frequency('qubit', (qb.pars['qubit_freq']-qb.pars['qubit_LO'])) # sets the IF frequency of the qubit
        #         update_frequency('rr', qb.pars['rr_IF']) 
        #         update_frequency('ffl', qb.pars['ffl_IF']) 

        #         with for_(n, 0, n < n_avg, n + 1):

        #             save(n, n_st)
        #             with for_each_(t,var_arr):
        #                 play("const" * amp(scrambling_amp), "qubit", duration = clk(saturation_dur)) #play scrambling pulse
        #                 align("qubit", "ffl")
        #                 play('gaussian'*amp(amp_ffl_scale), "ffl", duration=clk(ffl_len))
        #                 align("qubit","ffl")
        #                 play("pi" * amp(amp_q_scaling), "qubit", duration=t)
        #                 align()
        #                 measure("readout", "rr", None, *qb.res_demod(I, Q)) 
        #                 save(I, I_st)
        #                 save(Q, Q_st)
        #                 wait(resettime_clk,"qubit")
                            
        #         with stream_processing():
        #             I_st.buffer(len(var_arr)).average().save("I")
        #             Q_st.buffer(len(var_arr)).average().save("Q")
        #             n_st.save('n')
                    
        
        # elif self.name == 'cavity-reset':
        #     tmin = clk(tmin)
        #     tmax = clk(tmax)
        #     dt = clk(dt)
        #     t_arr = np.arange(tmin, tmax + dt/2, dt, dtype = int)
        #     resettime_clk= clk(qb.pars['qubit_resettime'])
        #     n_avg = self.exp_dict['n_avg']
        #     amp_r_scale = self.exp_dict['amp_r_scale']
        #     amp_ffl_scale = self.exp_dict['amp_ffl_scale']
        #     rr_IF = qb.pars['rr_IF']
        #     ffl_IF = qb.pars['ffl_IF']
        #     qb_IF=qb.pars['qubit_IF']+detuning
        #     with program() as prog:
        #         update_frequency('rr', rr_IF) 
        #         update_frequency('ffl', ffl_IF) 
        #         update_frequency('qubit' ,qb_IF)
        #         n, t, I, Q = qb.declare_vars([int, int, fixed, fixed])
        #         I_stream, Q_stream, n_stream = qb.declare_streams(stream_num=3)
        #         with for_(n, 0, n < n_avg, n + 1):
        #             save(n, n_stream)
        #             with for_each_(t, t_arr):
        #                 play("readout"*amp(amp_r_scale), "rr")
        #                 align('ffl','rr')
        #                 #play('gaussian'*amp(amp_ffl_scale), "ffl", duration=clk(ffl_len))
        #                 play('gaussian'*amp(amp_ffl_scale), "ffl", duration=clk(ffl_len))
        #                 wait(clk(80))
        #                 align('ffl','qubit')
        #                 play("pi_half", "qubit")
        #                 with if_(t>0):
        #                 #     #align("ffl","qubit")
        #                     wait(t)
        #                 #     #play('gaussian'*amp(amp_ffl_scale), "ffl", duration=t)
        #                 # #     
        #                 # #     #wait(t+clk(20))
        #                 play("pi", "qubit")
        #                 with if_(t>0):
        #                     wait(t, "qubit")
                        
        #                 play("pi_half", "qubit")
        #                 align("qubit","rr")
        #                 # play("pi","qubit")
        #                 # align("qubit","rr")
        #                 # with if_(t>0):
        #                 #     play('gaussian'*amp(amp_ffl_scale), "ffl", duration=t)
        #                 # align("ffl","rr")
        #                 measure("readout", "rr", None, *qb.res_demod(I, Q))
        #                 save(I, I_stream)
        #                 save(Q, Q_stream)
        #                 wait(resettime_clk, 'rr')
        #         with stream_processing():
        #             I_stream.buffer(len(t_arr)).average().save("I")
        #             Q_stream.buffer(len(t_arr)).average().save("Q")
        #             n_stream.save('n')
        
        
        # elif self.name == 'cavity-cooling':
        #     tmin = clk(tmin)
        #     tmax = clk(tmax)
        #     dt = clk(dt)
        #     t_arr = np.arange(tmin, tmax + dt/2, dt, dtype = int)
        #     resettime_clk= clk(qb.pars['qubit_resettime'])
        #     n_avg = self.exp_dict['n_avg']
        #     amp_r_scale = self.exp_dict['amp_r_scale']
        #     amp_ffl_scale = self.exp_dict['amp_ffl_scale']
        #     rr_IF = qb.pars['rr_IF']
        #     ffl_IF = qb.pars['ffl_IF']
        #     qb_IF=qb.pars['qubit_IF']+detuning
        #     with program() as prog:
        #         update_frequency('rr', rr_IF) 
        #         update_frequency('ffl', ffl_IF) 
        #         update_frequency('qubit' ,qb_IF)
        #         n, t, I, Q= qb.declare_vars([int, int, fixed, fixed])
        #         I_stream, Q_stream, n_stream = qb.declare_streams(stream_num=3)
        #         with for_(n, 0, n < n_avg, n + 1):
        #             save(n, n_stream)
        #             with for_each_(t, t_arr):
        #                 #play("readout"*amp(amp_r_scale), "rr")
        #                 #align('ffl','rr')
        #                 #play('gaussian'*amp(amp_ffl_scale), "ffl", duration=clk(ffl_len))
        #                 #play('gaussian'*amp(amp_ffl_scale), "ffl", duration=clk(ffl_len), condition= with_ffl==True)
        #                 #wait(clk(200))
        #                 #align('ffl','qubit')
        #                 #play('gaussian'*amp(amp_ffl_scale), "ffl", duration=2*t+)
        #                 play('const'*amp(amp_ffl_scale), "ffl", duration=2*t+100)
        #                 wait(40,"qubit")
        #                 play("pi_half", "qubit")
        #                 align('rr','qubit')
        #                 play("readout"*amp(amp_r_scale), "rr", duration=2*t+20)
        #                 with if_(t>0):
        #                     wait(t,"qubit")
        #                 play("pi", "qubit")
        #                 with if_(t>0):
        #                     wait(t,"qubit")
        #                 play("pi_half", "qubit")
        #                 #align("qubit","rr")
        #                 align("ffl","rr")
        #                 measure("readout", "rr", None, *qb.res_demod(I, Q))
        #                 save(I, I_stream)
        #                 save(Q, Q_stream)
        #                 wait(resettime_clk)
        #         with stream_processing():
        #             I_stream.buffer(len(t_arr)).average().save("I")
        #             Q_stream.buffer(len(t_arr)).average().save("Q")
        #             n_stream.save('n')
                    
                    
        # elif self.name == 'cavity-cooling-ramsey':
        #     tmin = clk(tmin)
        #     tmax = clk(tmax)
        #     dt = clk(dt)
        #     t_arr = np.arange(tmin, tmax + dt/2, dt, dtype = int)
        #     resettime_clk= clk(qb.pars['qubit_resettime'])
        #     n_avg = self.exp_dict['n_avg']
        #     amp_r_scale = self.exp_dict['amp_r_scale']
        #     amp_ffl_scale = self.exp_dict['amp_ffl_scale']
        #     rr_IF = qb.pars['rr_IF']
        #     ffl_IF = qb.pars['ffl_IF']
        #     qb_IF=qb.pars['qubit_IF']+detuning
        #     with program() as prog:
        #         update_frequency('rr', rr_IF) 
        #         update_frequency('ffl', ffl_IF) 
        #         update_frequency('qubit' ,qb_IF+detuning)
        #         n, t, I, Q= qb.declare_vars([int, int, fixed, fixed])
        #         I_stream, Q_stream, n_stream = qb.declare_streams(stream_num=3)
        #         with for_(n, 0, n < n_avg, n + 1):
        #             save(n, n_stream)
        #             with for_each_(t, t_arr):
        #                 #play("readout"*amp(amp_r_scale), "rr")
        #                 #align('ffl','rr')
        #                 #play('gaussian'*amp(amp_ffl_scale), "ffl", duration=clk(ffl_len))
        #                 #play('gaussian'*amp(amp_ffl_scale), "ffl", duration=clk(ffl_len), condition= with_ffl==True)
        #                 #wait(clk(200))
        #                 #align('ffl','qubit')
        #                 #play('gaussian'*amp(amp_ffl_scale), "ffl", duration=2*t+)
        #                 play('const'*amp(amp_ffl_scale), "ffl", duration=t+90)
        #                 wait(40,"qubit")
        #                 play("pi_half", "qubit")
        #                 align('rr','qubit')
        #                 play("readout"*amp(amp_r_scale), "rr", duration=t)
        #                 with if_(t>0):
        #                     wait(t,"qubit")
        #                 play("pi_half", "qubit")
        #                 #align("qubit","rr")
        #                 align("ffl","rr")
        #                 measure("readout", "rr", None, *qb.res_demod(I, Q))
        #                 save(I, I_stream)
        #                 save(Q, Q_stream)
        #                 wait(resettime_clk)
        #         with stream_processing():
        #             I_stream.buffer(len(t_arr)).average().save("I")
        #             Q_stream.buffer(len(t_arr)).average().save("Q")
        #             n_stream.save('n')
        

    
                    
        # return prog
        
    
    
    def simulate_sequence(self,qb, duration):
        duration = clk(duration)
        qmm = QuantumMachinesManager(host=host, port=port)
        prog = self.make_sequence(qb)
        job = qmm.simulate(qb.config, prog, SimulationConfig(duration=duration))
        samples = job.get_simulated_samples()
        samples.con1.plot()
        qmm.close_all_quantum_machines()
        return samples
    
def main():
    qb = dissipator('diss08_11a',device_name='diss08_11a')
    # qb.update_value('ffl_freq', 3.07e9)
    # qb.update_value('ffl_IF', 350e6)
    # qb.update_value('ffl_LO', qb.pars['ffl_freq'] - qb.pars['ffl_IF'])
    #qb.update_value('rr_pulse_len_in_clk', 20)
    seq = sequence('qb-reset', IF_min=45e6,IF_max=54e6,df=0.1e6, res_ringdown_time=int(40))
    samples = seq.simulate_sequence(qb, duration=3000)
    #qb.update_value('rr_pulse_len_in_clk', 500)
    
if __name__ == "__main__":
    main()
