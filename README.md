# Measurement Scripts for the OCS QP Project:

This contains code to run on the Sneezy fridge and computer.

## Hardware Setup:

### Fridge Diagram:

![fridge](fridge.png)

### Room Temperature Electronics Diagram:

![rte](rte.png)

#### Equipment Used:

```json
{ 
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

```

---

## Measurements:

### Measurement Notebooks:

- [Resonator Spectroscopy and Punchout (v0)](resonator_spectroscopy_and_punchout.ipynb)
- [Qubit Spectroscopy](qubit_spectroscopy.ipynb)

### Measurement Checklist:

- [x] Resonator spectroscopy
- [x] Punchout
- [ ] qubit spectroscopy over broad range to detect higher order frequencies (up to w_03)
- [ ] Resonator spectroscopy (high averages, small span) at res freqs
- [ ] Single shot measurement of resonator at res freq for histogram (goal: see two peaks)
- [ ] Ramsey interference measurement to determine the instantaneous n_g
- [ ] Single-shot readout of charge parity
- [ ] Charge parity dynamics timeseries measurement
- [ ] basic qubit characterization (T1,T2R,T2E) [OPTIONAL]

---

## Setup:

1. Clone the repository
2. Checkout the appropriate branch for deployment
3. Create a new environment or add the requirements to your environment (`pip install -r requirements.txt`)