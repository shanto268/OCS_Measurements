# -*- coding: utf-8 -*-
"""
Created on Wed Aug 30 2023

Description: A class to interface between Labber instruments and qubit.py.
Uses the Labber Python API; see documentation at https://www.keysight.com/us/en/assets/9018-18184/user-manuals/9018-18184.pdf
@author: Sacha Greenfield (srgreenf@usc.edu)
"""

import sys, os
# from VISAdrivers.sa_api import *
sys.path.append("E:\Program Files\Keysight\Labber\Script")
import Labber
import json

class instruments():

    # default settings for interfacing with instrument server
    default_settings =  {   
                'readout_LO' : 
                    {
                    'instrument' : 'SignalCore SC5511A Signal Generator',
                    'frequency' : 'Frequency',
                    'output' : 'Output status',
                    'power': 'Power',
                    'Labber_kwargs' : {'name' : '10002A06',
                                       'interface' : 'USB',}
                    },

                'qubit_LO' :
                    {
                    'instrument' : 'SignalCore SC5511A Signal Generator',
                    'frequency' : 'Frequency',
                    'output' : 'Output status',
                    'power': 'Power',
                    'Labber_kwargs' : {'name' : '10002F1D',
                                       'interface' : 'USB',}
                    },

                # 'Keithley' : 
                #     {
                #     'instrument' : 'Keithley 2400 SourceMeter',
                #     'current' : 'Source current',
                #      'Labber_kwargs' : {
                #                         'name' : 'Victoria',
                #                         'interface' : 'GPIB',
                #                         'address' : '00'
                #                         }
                #      },

                'DA' :
                    {
                    'instrument' : 'Vaunix Lab Brick Digital Attenuator',
                    'attenuation' : 'Attenuation',
                    'Labber_kwargs' :   {
                                        'name' : 'rr atten',
                                        'interface' : 'usb',
                                        'address' : '26551'
                                        }
                    }, 

                'sa':
                    {
                    'instrument' : 'SignalHound SpectrumAnalyzer',
                    'frequency' : 'Center frequency',
                    'span' : 'Span',
                    'bandwidth': 'Bandwidth',
                    'threshold': 'Input Power Level',
                    'signal': 'Signal',
                    'Labber_kwargs' : {'name' : 'sa',
                                       'interface' : 'usb',
                                       'startup': 'Get config',}
                    },
            
                }


    def __init__(self,  name = 'test', 
                        verbose = True):

        # initialize fields
        self.client = Labber.connectToServer()
        self._name = name
        self._exp_path = f'experiments\{name}'
        self._instruments = {}

        filename = f'{self._exp_path}\{name}_instruments.json'

        # check if instrument settings already exist; if so, load them from file; otherwise create default file
        if os.path.isfile(filename):

            if verbose: print(f'Loading instrument settings from {filename}.')        
            with open(filename, 'r') as jsonfile:
                self.settings = json.load(jsonfile)

            # for inst,default_inst_settings in self.default_settings.items():
            #     if inst not in self.settings:
            #         if verbose: print(f'{inst} not found in instruments settings, setting from default settings.')
            #         self.settings[inst] = default_inst_settings
                
            #     for (k, v) in default_inst_settings.items():
            #         if k not in self.settings[inst]:
            #             self.settings[inst][k] = v

            # # initialize all the instruments
            # for k,v in self.settings.items():
            #     self.add_instrument(k, v, verbose=verbose) 
            
        else:

            if verbose: print(f'Instruments settings {filename} not found. Writing {filename} with defaults:')
            self.settings = self.default_settings
            # self.print()
            
            

            if verbose: print(f'Please manually modify {filename} and try again.')
        
        for inst,default_inst_settings in self.default_settings.items():
                if inst not in self.settings:
                    if verbose: print(f'{inst} not found in instruments settings, setting from default settings.')
                    self.settings[inst] = default_inst_settings
                
                for (k, v) in default_inst_settings.items():
                    if k not in self.settings[inst]:
                        self.settings[inst][k] = v

        # initialize all the instruments
        for k,v in self.settings.items():
            self.add_instrument(k, v, verbose=verbose) 

        # Check if directory exists, otherwise create new directory
        if not os.path.exists(self._exp_path):
            os.makedirs(self._exp_path)
        with open(filename, 'w') as jsonfile:
            json.dump(self.settings, jsonfile)


    # add an instrument based on settings
    def add_instrument(self, instrument_name, instrument_setting,
                        values = {}, 
                        verbose = True):

        # get settings
        Labber_name = instrument_setting['instrument']
        Labber_kwargs = instrument_setting['Labber_kwargs']
        if verbose: print('Initializing ' + instrument_name + ' (' + Labber_name + ').')

        # connect to instrument through Labber and add instrument to dictionary of instruments
        inst = self.client.connectToInstrument(Labber_name, Labber_kwargs) 
        self._instruments[instrument_name] = inst                         

        # start the instrument
        inst.startInstrument()

            # the LOs require turning on the output
            # I'd rather this go after setting the value...
            # if 'output' in instrument_setting:
            #     self.set(instrument_name, 'output', True)


    def set(self, instrument_name, key, value, verbose = True):

        # get the Labber instrument
        inst = self._instruments[instrument_name]

        # translate generic key into Labber specific key (i.e. 'frequency' to 'RF1 frequency')
        Labber_key = self.settings[instrument_name][key]

        # set to the new value
        inst.setValue(Labber_key, value)

        if verbose: print(f'Setting {instrument_name} {key} to {value}.')




    def get(self, instrument_name, key, verbose = True):

        # get the Labber instrument
        inst = self._instruments[instrument_name]

        # translate generic key into Labber-specific key (i.e. 'frequency' to 'RF1 frequency)
        Labber_key = self.settings[instrument_name][key]

        # read the current value from the instrument
        value = inst.getValue(Labber_key)

        if verbose: print(f'{instrument_name} {key} is {value}')
        return value

    
    def print(self):

        print(json.dumps(self.settings, sort_keys=False, indent=4))

    


    # def set_qb_LO(freq):
    #     print(f'Setting qubit LO to {round(freq*1e-9,5)} GHz')
    #     # initialize qubit LO
    #     qLO = client.connectToInstrument('BNC 845 Signal Generator', dict(name='qubit', startup = 'Get config'))
    #     qLO.startInstrument()
    #     qLO.setValue('Frequency', freq)
    #     qLO.setValue('Output',True)

    # def get_qb_LO():
    #     # initialize qubit LO
    #     qLO = client.connectToInstrument('BNC 845 Signal Generator', dict(name='qubit', startup = 'Get config'))
    #     qLO.startInstrument()
    #     return qLO.getValue('Frequency')

    # def set_rr_LO(freq):
    #     print(f'Setting readout LO to {round(freq*1e-9,5)} GHz')
    #     # initialize readout LO
    #     rrLO = client.connectToInstrument('BNC 845 Signal Generator', dict(name='readout', startup = 'Get config'))
    #     rrLO.startInstrument()
    #     rrLO.setValue('Frequency', freq)
    #     rrLO.setValue('Output',True)

    # def get_rr_LO():
    #     # initialize qubit LO
    #     rrLO = client.connectToInstrument('BNC 845 Signal Generator', dict(name='readout', startup = 'Get config'))
    #     rrLO.startInstrument()
    #     return rrLO.getValue('Frequency')

    # def set_attenuator(attenuation, print_message = False):
    #     # initialize digital attenuator
    #     if print_message:
    #         print(f'Setting digital attenuation to {attenuation} dB')
    #     attn = client.connectToInstrument('Vaunix Lab Brick Digital Attenuator',dict(name='Readout',address='26777'))
    #     attn.startInstrument()
    #     attn.setValue('Attenuation',attenuation)

    # def get_attenuation():
    #     # initialize digital attenuator
    #     attn = client.connectToInstrument('Vaunix Lab Brick Digital Attenuator',dict(name='Readout',address='26777'))
    #     attn.startInstrument()

    #     return attn.getValue('Attenuation')

# def init_sa():
#     sa = sa_open_device()["handle"]
#     sa_config_acquisition(device = sa, detector = SA_AVERAGE, scale = SA_LOG_SCALE)
#     sa_config_gain_atten(sa, SA_AUTO_ATTEN, SA_AUTO_GAIN, True)
#     sa_config_sweep_coupling(device = sa, rbw = 1e2, vbw = 1e2, reject=0)

#     return sa

#     return sa
# initialize Keithley
# SC = client.connectToInstrument('Keithley 2400 SourceMeter',dict(interface='GPIB',address='24'))
# SC.startInstrument()

# try:
#     sa
# except NameError:
#     sa = init_sa()

if __name__ == '__main__':
    pass
    # initialize spectrum analyzer
    # sa = init_sa()
