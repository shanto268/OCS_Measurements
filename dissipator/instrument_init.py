# -*- coding: utf-8 -*-
"""
Created on Fri Sep 30 11:59:17 2022

Description: A script to initialize and setup instruments other than the OPX
@author: Evangelos Vlachos (evlachos@usc.edu)
"""
# import os
# os.chdir()
import sys
from VISAdrivers.sa_api import *
sys.path.append("D:\Program Files\Keysight\Labber\Script")
sys.path.append(r"C:\Users\lfl\measurements_test")
import Labber
import json
import numpy as np
import time
client = Labber.connectToServer()
# qmm = QuantumMachinesManager()
# qm = qmm.open_qm(config)
qubit_LO_model = 'SignalCore SC5506A Signal Generator'
qubit_LO_name = '10002A08' 
qubit_LO_quantity_name = {'freq': 'RF1 frequency',
                       'power': 'RF1 power level',
                       'output':'RF1 output status'}

rr_LO_model = 'BNC 845 Signal Generator'
rr_LO_name = 'readout LO'
rr_LO_quantity_name = {'freq':'Frequency',
                          'power': 'Power',
                          'output': 'Output'}

fflqc_LO_model = 'SignalCore SC5506A Signal Generator'
fflqc_LO_name = '10002A07'
fflqc_LO_quantity_name = {'freq': 'RF1 frequency',
                        'power': 'RF1 power level',
                       'output':'RF1 output status'}

ffl_LO_model = 'SignalCore SC5511A Signal Generator'
ffl_LO_name = '1000334C'
ffl_LO_quantity_name = {'freq':'Frequency',
                          'power': 'Power',
                          'output': 'Output status'}


diss_LO_model = 'SignalCore SC5506A Signal Generator'
diss_LO_name = '10002A08' 
diss_LO_quantity_name = {'freq':'RF2 frequency',
                          'power': 'RF2 power level',
                          'output': 'RF2 output status'}

flux_source_meter_model = 'Keithley 2400 SourceMeter'
coil_source_meter_name = 'Keithley Victoria'

source_meter_quantity_name = {'current': 'Source current',
                              'rate': 'Source current - Sweep rate',
                              'output': 'Output on'}

FFL_source_meter_model = 'Keithley 2400 SourceMeter'
FFL_source_meter_name = 'Keithley Lisa'


def get_flux_bias():
    csource = client.connectToInstrument(flux_source_meter_model, dict(name=coil_source_meter_name, startup = 'Get config'))
    csource.startInstrument()
    current = csource.getValue(source_meter_quantity_name['current'])
    print(f'flux bias = {round(current * 1e3, 4)} mA')
    return current
    
def set_flux_bias(current, step = 0.5e-6, lower_bound=-1e-3, upper_bound=1e-3): #DVK: Changed the bound to 1 mA to make sure we stay in the mA range
    if current > upper_bound or current<lower_bound:
        raise ValueError('current out of range')
    print(f'Setting flux bias to {round(current*1e3, 4)} mA')
    csource = client.connectToInstrument(FFL_source_meter_model, dict(name=coil_source_meter_name, startup = 'Get config'))
    csource.startInstrument()
    csource.setValue('Source current range', upper_bound)
    start_current = csource.getValue(source_meter_quantity_name['current'])
    if current <= start_current:
        step = -step
    current_list = np.round(np.arange(start_current, current + step/2, step),7)
    for value in current_list[1:]:
        csource.setValue(source_meter_quantity_name['current'], value)
        time.sleep(0.5)

def get_ffl_bias():
    csource = client.connectToInstrument(FFL_source_meter_model, dict(name=FFL_source_meter_name, startup = 'Get config'))
    csource.startInstrument()
    current = csource.getValue(source_meter_quantity_name['current'])
    print(f'FFL bias = {round(current * 1e3, 4)} mA')
    return current

def set_ffl_bias(current, step = 0.5e-6, lower_bound=-100e-6, upper_bound=100e-6):
    if current > upper_bound or current<lower_bound:
        raise ValueError('current out of range')
    print(f'Setting FFL bias to {round(current*1e3, 4)} mA')
    csource = client.connectToInstrument(FFL_source_meter_model, dict(name=FFL_source_meter_name, startup = 'Get config'))
    csource.startInstrument()
    csource.setValue('Source current range', upper_bound)
    start_current = csource.getValue(source_meter_quantity_name['current'])
    if current <= start_current:
        step = -step
    current_list = np.round(np.arange(start_current, current + step/2, step),7)
    for value in current_list[1:]:
        csource.setValue(source_meter_quantity_name['current'], value)
        time.sleep(0.5)
        
def turn_on_ffl_source_meter():
    csource = client.connectToInstrument(FFL_source_meter_model, dict(name=FFL_source_meter_name, startup = 'Get config'))
    csource.startInstrument()
    csource.setValue(source_meter_quantity_name['current'], 0)
    csource.setValue(source_meter_quantity_name['output'], True)
    
def turn_off_ffl_source_meter():
    csource = client.connectToInstrument(FFL_source_meter_model, dict(name=FFL_source_meter_name, startup = 'Get config'))
    csource.startInstrument()
    current = get_ffl_bias()
    if current == 0.0:
        csource.setValue(source_meter_quantity_name['output'], False)
    else:
        raise ValueError('ramp back to zero current before turning off')
    
    
def set_qb_LO(freq):
    print(f'Setting qubit LO to {round(freq*1e-9,5)} GHz')
    # initialize qubit LO
    qLO = client.connectToInstrument(qubit_LO_model, dict(name=qubit_LO_name, startup = 'Get config', interface='USB'))
    qLO.startInstrument()
    qLO.setValue(qubit_LO_quantity_name['freq'], freq)
    qLO.setValue(qubit_LO_quantity_name['power'], 15)
    qLO.setValue(qubit_LO_quantity_name['output'],True)

def get_qb_LO():
    # initialize qubit LO
    qLO = client.connectToInstrument(qubit_LO_model, dict(name=qubit_LO_name, startup = 'Get config', interface='USB'))
    qLO.startInstrument()
    return qLO.getValue(qubit_LO_quantity_name['freq'])

def set_diss_LO(freq):
    print(f'Setting dissipator LO to {round(freq*1e-9,5)} GHz')
    # initialize qubit LO
    dLO = client.connectToInstrument(diss_LO_model, dict(name=diss_LO_name, startup = 'Get config'))
    dLO.startInstrument()
    dLO.setValue(diss_LO_quantity_name['freq'], freq)
    dLO.setValue(diss_LO_quantity_name['power'], 15)
    dLO.setValue(diss_LO_quantity_name['output'],True)

def get_diss_LO():
    # initialize qubit LO
    dLO = client.connectToInstrument(diss_LO_model, dict(name=diss_LO_name, startup = 'Get config'))
    dLO.startInstrument()
    return qLO.getValue(diss_LO_quantity_name['freq'])

def set_rr_LO(freq):
    print(f'Setting readout LO to {round(freq*1e-9,5)} GHz')
    # initialize readout LO
    rrLO = client.connectToInstrument(rr_LO_model, dict(name=rr_LO_name, startup = 'Get config'))
    rrLO.startInstrument()
    rrLO.setValue('Frequency', freq)
    # rrLO.setValue(rr_LO_quantity_name['power'], 15)
    rrLO.setValue(rr_LO_quantity_name['output'],True)

def get_rr_LO():
    # initialize qubit LO
    rrLO = client.connectToInstrument(rr_LO_model, dict(name=rr_LO_name, startup = 'Get config'))
    rrLO.startInstrument()
    return rrLO.getValue('Frequency')

def get_ffl_LO():
    # initialize qubit LO
    fLO = client.connectToInstrument(ffl_LO_model, dict(name=ffl_LO_name, startup = 'Get config'))
    fLO.startInstrument()
    return fLO.getValue(ffl_LO_quantity_name['freq'])

def set_ffl_LO(freq, bprint=True):
    if bprint:
        print(f'Setting ffl LO to {round(freq*1e-9,5)} GHz')
    # initialize ffl LO
    fLO = client.connectToInstrument(ffl_LO_model, dict(name=ffl_LO_name, startup = 'Get config'))
    fLO.startInstrument()
    fLO.setValue(ffl_LO_quantity_name['freq'], freq)
    fLO.setValue(ffl_LO_quantity_name['power'], 15)
    fLO.setValue(ffl_LO_quantity_name['output'],True)
    
    
def get_fflqc_LO():
    # initialize qubit LO
    fLO = client.connectToInstrument(fflqc_LO_model, dict(name=fflqc_LO_name, startup = 'Get config'))
    fLO.startInstrument()
    return fLO.getValue(fflqc_LO_quantity_name['freq'])

def set_fflqc_LO(freq, bprint=True):
    if bprint:
        print(f'Setting fflqc LO to {round(freq*1e-9,5)} GHz')
    # initialize ffl LO
    fLO = client.connectToInstrument(fflqc_LO_model, dict(name=fflqc_LO_name, startup = 'Get config'))
    fLO.startInstrument()
    fLO.setValue(fflqc_LO_quantity_name['freq'], freq)
    fLO.setValue(fflqc_LO_quantity_name['power'], 15)
    fLO.setValue(fflqc_LO_quantity_name['output'],True)

    
def turn_on_ffl_drive():
    fLO = client.connectToInstrument(ffl_LO_model, dict(name=ffl_LO_name, startup = 'Get config'))
    fLO.startInstrument()
    fLO.setValue(ffl_LO_quantity_name['output'],True)
    
def turn_off_ffl_drive():
    fLO = client.connectToInstrument(ffl_LO_model, dict(name=ffl_LO_name, startup = 'Get config'))
    fLO.startInstrument()
    fLO.setValue(ffl_LO_quantity_name['output'],False)

def turn_on_fflqc_drive():
    fLO = client.connectToInstrument(fflqc_LO_model, dict(name=fflqc_LO_name, startup = 'Get config'))
    fLO.startInstrument()
    fLO.setValue(fflqc_LO_quantity_name['output'],True)
    
def turn_off_fflqc_drive():
    fLO = client.connectToInstrument(fflqc_LO_model, dict(name=fflqc_LO_name, startup = 'Get config'))
    fLO.startInstrument()
    fLO.setValue(fflqc_LO_quantity_name['output'],False)

    
def turn_diss_LO_off():
    dLO = client.connectToInstrument(diss_LO_model, dict(name=diss_LO_name, startup = 'Get config'))
    dLO.startInstrument()
    dLO.setValue(diss_LO_quantity_name['output'],False)
    
        
def turn_rr_LO_off():
    rLO = client.connectToInstrument(rr_LO_model, dict(name=rr_LO_name, startup = 'Get config'))
    rLO.startInstrument()
    rLO.setValue(rr_LO_quantity_name['output'],False)
    
def turn_qb_LO_off():
    qLO = client.connectToInstrument(qubit_LO_model, dict(name=qubit_LO_name, startup = 'Get config'))
    qLO.startInstrument()
    qLO.setValue(qubit_LO_quantity_name['output'],False)
    
def set_attenuator(attenuation):
    # initialize digital attenuator
    # attn = client.connectToInstrument('Vaunix Lab Brick Digital Attenuator',dict(name='rr atten',address='24679'))
    attn = client.connectToInstrument('Vaunix Lab Brick Digital Attenuator',dict(name='rr atten',interface='USB',address='24680'))
    attn.startInstrument()
    attn.setValue('Attenuation',attenuation)

def get_attenuation():
    # initialize digital attenuator
    # attn = client.connectToInstrument('Vaunix Lab Brick Digital Attenuator',dict(name='rr atten',address='24679'))
    attn = client.connectToInstrument('Vaunix Lab Brick Digital Attenuator',dict(name='rr atten',interface='USB',address='24680'))
    attn.startInstrument()
    return attn.getValue('Attenuation')

def set_ffl_attenuator(attenuation):
    # initialize digital attenuator
    attn = client.connectToInstrument('Vaunix Lab Brick Digital Attenuator',dict(name='ffl attenuator',address='26776'))
    attn.startInstrument()
    attn.setValue('Attenuation',attenuation)

def get_ffl_attenuation():
    # initialize digital attenuator
    attn = client.connectToInstrument('Vaunix Lab Brick Digital Attenuator',dict(name='ffl attenuator',address='26776'))
    attn.startInstrument()
    return attn.getValue('Attenuation')

def turn_on_twpa_pump():
    pLO = client.connectToInstrument(twpa_LO_model, dict(name=twpa_LO_name, startup = 'Get config'))
    pLO.startInstrument()
    pLO.setValue('Frequency', 7.97114e9)
    pLO.setValue(twpa_LO_quantity_name['power'], 11.5)
    pLO.setValue(twpa_LO_quantity_name['output'],True)
    
def turn_off_twpa_pump():
    pLO = client.connectToInstrument(twpa_LO_model, dict(name=twpa_LO_name, startup = 'Get config'))
    pLO.startInstrument()
    pLO.setValue(twpa_LO_quantity_name['output'],False)
    
def init_sa():
    sa = sa_open_device()["handle"]
    sa_config_acquisition(device = sa, detector = SA_AVERAGE, scale = SA_LOG_SCALE)
    sa_config_gain_atten(sa, SA_AUTO_ATTEN, SA_AUTO_GAIN, True)
    sa_config_sweep_coupling(device = sa, rbw = 1e2, vbw = 1e2, reject=0)
    return sa

def init_sa_by_serial_number(serial_number=20234492):
    sa = sa_open_device_by_serial(serial_number)["handle"]
    sa_config_acquisition(device = sa, detector = SA_AVERAGE, scale = SA_LOG_SCALE)
    sa_config_gain_atten(sa, SA_AUTO_ATTEN, SA_AUTO_GAIN, True)
    sa_config_sweep_coupling(device = sa, rbw = 1e2, vbw = 1e2, reject=0)
    return sa

# initialize Keithley
# SC = client.connectToInstrument('Keithley 2400 SourceMeter',dict(interface='GPIB',address='24'))
# SC.startInstrument()

# try:
#     sa
# except NameError:
#     sa = init_sa()

if __name__ == '__main__':
    # initialize spectrum analyzer
    sa = init_sa()
    # sa2 = init_sa_by_serial_number()
else:
    pass
