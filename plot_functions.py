# -*- coding: utf-8 -*-
"""
Created on Mon May 24 13:13:39 2021

@author: Evangelos Vlachos <evlachos@usc.edu>
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy as sp
import scipy as scy
# from matplotlib import cm
import csv
import itertools
from scipy.interpolate import interp1d
import scipy.fftpack
import time
import os
from json import loads
# from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
# from matplotlib.colors import LightSource
from types import SimpleNamespace
pi=np.pi
import seaborn as sns; sns.set() # styling
from matplotlib.ticker import FormatStrFormatter
from Utilities import convert_V_to_dBm
from scipy.signal import butter,lfilter,freqz,find_peaks,peak_widths
# import imageio

# set some deafault
# for a complete set of parameters "print(plt.rcParams)"
sns.set_style('ticks')
plt.rcParams['font.family'] =  'Arial'
plt.rcParams['font.size'] = 16
plt.rcParams['figure.dpi'] = 150
plt.rcParams["xtick.direction"] = "in"
plt.rcParams["ytick.direction"] = "in"
plt.rcParams["xtick.major.top"] = True
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['axes.labelsize'] = 16
plt.rcParams['axes.titlesize'] = 16
plt.rcParams["xtick.major.bottom"] = True
plt.rcParams["xtick.top"] = True
plt.rcParams["xtick.bottom"] = True
plt.rcParams["ytick.left"] = True
plt.rcParams["ytick.right"] = True
plt.rcParams['ytick.labelsize'] = 14
plt.rcParams["ytick.major.right"] = True
plt.rcParams["ytick.labelright"] = False
plt.rcParams["ytick.minor.visible"] = False

#%% power_plot
def power_plot(freqs,signal,power,fc):
    plt.plot(freqs*1e-6, signal,'-')
    plt.xlabel('Frequency [MHz]')
    plt.ylabel('Power [dBm]')
    plt.show()

#%% tof_plot
def tof_plot(adc1,adc2,delay=0,offsets=[0,0]):
    plt.figure()
    plt.title('time-of-flight calibration analysis')
    plt.plot(adc1)
    plt.gcf().text(1, 0.15, f"Electrical delay: {delay}\nChannel 1 offset: {offsets[0]*1e3:.1f} mV\nChannel 2 offset: {offsets[1]*1e3:.1f} mV", fontsize=14)
    plt.plot(adc2)
    plt.legend(["adc1", "adc2"])
    plt.show()

#%% spec_plot
def resonator_spec_plot(data,qb_pars,fwhm=0,fc=0,iteration=1,**kwargs):
    freq = data['freqs']*1e-9
    df = (freq[1]-freq[0])*1e9
    I = data['I']*1e3
    Q = data['Q']*1e3
    mag = np.abs(I+1j*Q)
    power=10*np.log10((10**-3)*(mag**2)/50)

    phase = np.unwrap(np.angle(I+1j*Q,deg=True),period=360)
    fig = plt.figure(figsize=(10,8))

# Power data
    ax1 = fig.add_subplot(221)
    ax1.plot(freq, mag, '-o', markersize=3, c='C0')
    ax1.set_xlabel('Frequency (GHz)')
    ax1.set_ylabel('Magnitude (mV)')

    # Phase data
    ax2 = fig.add_subplot(222)
    ax2.plot(freq, phase, '-o', markersize=3, c='C0')
    ax2.set_xlabel('Frequency (GHz)')
    ax2.set_ylabel('Phase (deg)')

    # Additional subplot on the bottom row
    ax3 = fig.add_subplot(212)
    ax3.plot(freq, I, '-o', markersize=3, c='r', label='I')
    ax3.plot(freq,Q , '-o', markersize=3, c='b', label='Q')
    ax3.set_xlabel('Frequency (GHz)')
    ax3.set_ylabel('Voltage (mV)')

    txt = f'$\omega_c$ = {fc*1e-9:.5f} GHz\nFWHM = {fwhm*1e-6:.3f} MHz\n$\kappa$ = {2*np.pi*fwhm*1e-6:.3f} MHz\nReadout attenuation: {qb_pars["readout_atten"]} dB\ndf = {df*1e-3:.1f} kHz'
    
    plt.gcf().text(1, 0.15, txt, fontsize=14)
    # fig.set_title(f'{element} spectroscopy {iteration}')
    plt.tight_layout()

def qubit_spec_plot(data,qb_pars,qb_power=0,rr_power=0,iteration=1,find_peaks=True, amp_q_scaling=1,**kwargs):


    freq = data['freqs']*1e-9
    df = freq[1]-freq[0]
    I = data['I']*1e3
    Q = data['Q']*1e3
    mag = np.abs(I+1j*Q)
    power=convert_V_to_dBm(mag*1e-3)

    phase = np.unwrap(np.angle(I+1j*Q,deg=True),period=360)

    sigma = np.std(mag)
    print(f'Peak threshold at {np.mean(mag)+2*sigma}')
    peaks,_ = scy.signal.find_peaks(mag,height=np.mean(mag)+2*sigma,distance=200,width=3)
    try:
        for i in peaks:
            print(f'Peaks at: {round(freq[i],5)} GHz\n')
    except:
        print('Peaks not found or do not exist.')

    fig = plt.figure(figsize=(8,8))

# Power data
    ax1 = fig.add_subplot(221)
    ax1.plot(freq, mag, '-o', markersize=3, c='C0')
    ax1.set_xlabel('Frequency (GHz)')
    ax1.set_ylabel('Magnitude (mV)')

    # Phase data
    ax2 = fig.add_subplot(222)
    ax2.plot(freq, phase, '-o', markersize=3, c='C0')
    ax2.set_xlabel('Frequency (GHz)')
    ax2.set_ylabel('Phase (deg)')

    # Additional subplot on the bottom row
    ax3 = fig.add_subplot(212)
    ax3.plot(freq, I, '-o', markersize=3, c='r', label='I')
    ax3.plot(freq,Q , '-o', markersize=3, c='b', label='Q')
    ax3.set_xlabel('Frequency (GHz)')
    ax3.set_ylabel('Voltage (mV)')

    # if 'lo_list' in kwargs.keys():
    #     lo_list = kwargs.get('lo_list')
    #     axes = fig.axes
    #     for ax in axes: 
    #         ymin, ymax = ax.get_ylim()
    #         for lo in lo_list:
    #             ax.vlines(x = lo/1e9,ymin=ymin, ymax=ymax, ls='--')

    if len(peaks) == 2:
        txt = '$\omega_{01}$ = %.4f GHz\n$\omega_{02}$/2 = %.4f GHz\n$\\alpha$ = %.1f MHz\n$P_{qb}$ = %.1f dBm\n$P_r$ = %.1f dBm\n$\omega_r$ = %.4f GHz'%(freq[peaks[1]],freq[peaks[0]],(freq[peaks[0]]-freq[peaks[1]])*1e3,qb_power,rr_power,qb_pars['rr_freq']*1e-9)
    elif len(peaks) == 1:
        txt = '$\omega_{01}$ = %.4f GHz\n$P_{qb}$ = %.1f dBm\n$P_r$ = %.1f dBm\n$\omega_r$ = %.4f GHz'%(freq[peaks[0]], qb_power, rr_power, qb_pars['rr_freq']*1e-9)
    else:
        txt = '$P_{qb}$ = %.1f dBm\n$P_r$ = %.1f dBm\n$\omega_r$ = %.4f GHz\n$amp_q$ = %.5f'%(qb_power,rr_power,qb_pars['rr_freq']*1e-9, amp_q_scaling)
    
    plt.gcf().text(1, 0.15, txt, fontsize=14)
    # fig.set_title(f'{element} spectroscopy {iteration}')
    plt.tight_layout()
    plt.show()

#%% init_IQ_plot
def init_IQ_plot():
    '''initialize axes for continuous plotting on the IQ plane'''
    plot = sns.jointplot()
    plot.set_axis_labels('I [mV]', 'Q [mV]')
    plot.ax_marg_x.grid('off')
    plot.ax_marg_y.grid('off')
    plot.fig.tight_layout()
    ax = plt.gca()
    return plot, ax

#%% heatmap_plot
def heatplot(xdata, ydata, data, xlabel = "", ylabel = "", normalize=False, cbar_label = 'log mag', **kwargs):
    fig,ax = plt.subplots(1,1,figsize=(4,3), dpi=300)
    if normalize:
        cbar_label += ' (normalized)'

    df = pd.DataFrame(data, columns = xdata, index = ydata)

    if normalize:
        df = df.apply(lambda x: (x-x.mean())/x.std(), axis = 1)

    cbar_options = {
        'label':    cbar_label,
        'ticks':    np.around(np.linspace(np.amin(data),np.amax(data),5),0),
        'pad':      0.1,
        'values':   np.linspace(np.amin(data),np.amax(data),1000),
        'shrink':   1.2,
        'location': 'right',
    }

    heatmap_opts = {
        'ax':    ax,
        'linewidths':  0,
        # 'xticklabels': np.linspace(min(xdata),max(xdata)+0.5,5),
        # 'yticklabels': np.linspace(min(ydata),max(ydata)+0.5,5),
        'vmin':        np.amin(data),
        'vmax':        np.amax(data)
    }

    hm = sns.heatmap(df, ax=ax,cmap = 'viridis', cbar_kws=cbar_options, **kwargs)
    hm.set_xlabel(xlabel, fontsize=12)
    hm.set_ylabel(ylabel, fontsize=12)
    hm.spines[:].set_visible(True)
    ax.tick_params(direction='out',length=0.01,width=0.5,bottom=True, top=True, left=True, right=True,labeltop=False, labelbottom=True,labelrotation=90,labelsize=8,size=8)
    plt.yticks(rotation=0)
    ax.invert_yaxis()
    plt.tight_layout()
    plt.show()

def punchout_plot(data, xlabel = "", ylabel = "", normalize=False, 
             cbar_label = 'log mag',title='', **kwargs):
    
    freqs = data['freqs']
    I = data['I']
    Q = data['Q']
    z_data = convert_V_to_dBm(np.array(data['mag']))
    phase = np.unwrap(np.angle(data['z_data'],deg=True),period=360)
    attenuations = data['attenuations']
    chi= freqs[0][np.argmin(z_data[0])] - freqs[0][np.argmin(z_data[-1])]

    print(f'Dispersive shift: {0.5*chi/np.pi*1e9:.1f} kHz')

    fig = plt.figure(figsize=(6,5), dpi=300)
    ax1  = fig.add_subplot(6,1,(1,4))
    # if normalize:
    #     cbar_label += ' (normalized)'
    # df = pd.DataFrame(z_data, columns = freqs[0], index = attenuations)
    df = pd.DataFrame(phase, columns = freqs[0], index = attenuations)
    
    if normalize:
        # df = df.apply(lambda x: (x-x.mean())/x.std(), axis = 1)
        df = df.apply(lambda x: (x/x.max()), axis = 1)
    
    cbar_options = {
        'label':    cbar_label,
        # 'ticks':    np.around(np.linspace(np.amin(z_data),np.amax(z_data),5),1),
        'pad':      0.05,
        # 'values':   np.linspace(np.amin(z_data),np.amax(z_data),1000),
        'shrink':   1.1,
        'location': 'top',
    }
    # kwargs = {
    #     'linewidths':  0,
    #     # 'xticklabels': np.linspace(min(xdata),max(xdata)+0.5,5),
    #     # 'yticklabels': np.linspace(min(ydata),max(ydata)+0.5,5),
    #     'vmin':        np.amin(z_data),
    #     'vmax':        np.amax(z_data)
    # }
    
    hm = sns.heatmap(df, ax=ax1,cmap = 'seismic', cbar_kws=cbar_options)
    # hm.set_xlabel(xlabel, fontsize=12)
    hm.set_ylabel(ylabel, fontsize=14)
    hm.spines[:].set_visible(True)
    # ax1.set_title(title,fontsize=12)
    ax1.tick_params(direction='out',length=0.01,width=0.5,bottom=False, top=False, left=True, right=False,labeltop=False, labelbottom=True,labelrotation=90,labelsize=10,size=8)
    plt.yticks(rotation=0)
    
    ax2 = fig.add_subplot(6,1,(5,6))
    # ax2.plot(freqs[0],z_data[0]/max(z_data[0]),'o', markersize = 3, c='b',label=f'$P_r$ = -{attenuations[0]} dB')
    # ax2.plot(freqs[0],z_data[-1]/max(z_data[1]),'o', markersize = 3, c='r',label=f'$P_r$ = -{attenuations[-1]} dB')
    ax2.plot(freqs[0],phase[0],'o', markersize = 3, c='b',label=f'$P_r$ = -{attenuations[0]} dB')
    ax2.plot(freqs[0],phase[-1],'o', markersize = 3, c='r',label=f'$P_r$ = -{attenuations[-1]} dB')
    ax2.legend()
    ax2.set_xlabel(xlabel, fontsize=12)
    ax2.set_ylabel(cbar_label,fontsize=12)
    fc1,_ = fit_res(freqs[0],  I=I[0], Q=Q[0])
    fc2,_ = fit_res(freqs[0], I=I[-1], Q = Q[-1])
    txt = f'$f_1$ = {fc1:.5f} GHz\n$f_2$ = {fc2:.5f} GHz\n$2\chi/2\pi$ = {(fc2-fc1)*1e3:.1f} MHz'
    plt.gcf().text(0.95, 0.15, txt, fontsize=14)
    
    # plt.tight_layout()
        
    # return df


    # return hm;

#%% plot_single_shot
def plot_single_shot(datadict):
    plot, ax = init_IQ_plot()
    datadict = {key: np.array(value, dtype=float) for key,value in datadict.items()}
    datadict = {key: value*1e3 for key,value in datadict.items()} # convert to mV
    datadict = {key: value.tolist() for key,value in datadict.items()} # convert to list

    states = []
    # for key,value in datadict.items():
    #     print(key+':'+str(len(value))+'\n')
    [states.append(r'$|g\rangle$') for i in range(len(datadict['I']))]
    [states.append(r'$|e\rangle$') for i in range(len(datadict['Iexc']))]
    data = {
            'I [mV]':   np.hstack((datadict['I'],datadict['Iexc'])),
            'Q [mV]':   np.hstack((datadict['Q'],datadict['Qexc'])),
            'States':   states
                }
    I = np.array(datadict["I"])
    Q = np.array(datadict["Q"])
    Iexc=np.array(datadict["Iexc"])
    Qexc = np.array(datadict["Qexc"])
    
    y_gr = np.average(np.abs(I+1j*Q))
    phase_gr=np.average(np.arctan(Q/I))
    y_exc= np.average(np.abs(Iexc+1j*Qexc))
    phase_exc=np.average(np.arctan(Qexc/Iexc))
    # print("ground voltage:", y_gr)
    # print("excited volgate:", y_exc )
    # print("ground phase:", phase_gr)
    # print("excited phase:", phase_exc )
    print('contrast_mag',y_gr-y_exc)
    print('rotation',(phase_gr+phase_exc)/2)
    print("contrast_Q",y_gr*(np.abs(np.sin((phase_gr-phase_exc)/2)))+(y_exc*np.abs((np.sin((phase_gr-phase_exc))/2))))
    #print("contrast_I",y_gr*(np.abs(np.cos((phase_gr-phase_exc)/2)))-(y_exc*np.abs((np.cos((phase_gr-phase_exc))/2))))
    
          
    dataF = pd.DataFrame(data=data)
    plot = sns.jointplot(data=dataF, x='I [mV]',y='Q [mV]',hue='States',ax=ax,space=0)
    plt.show()

#%% plot_mixer_opt
def plot_mixer_opt(par1,par2,power_data,cal='LO',element='qubit',fc=5e9):
    par1 = np.around(par1*1e3,1)
    par2 = np.around(par2*1e3,1)

    par1 = par1.tolist()
    par2 = par2.tolist()
    df = pd.DataFrame(data=power_data,index=par1,columns=par2)

    hm = sns.heatmap(df,cbar_kws={'label': "Power [dBm]"})

    if cal == 'LO':
        hm.set_ylabel('I [mV]')
        hm.set_xlabel('Q [mV]')
    elif cal == 'SB':
        hm.set_ylabel('Gain Imbalance[x 1e-3]')
        hm.set_xlabel('Phase Imbalance[x 1e-3]')

    hm.spines[:].set_visible(True)
    hm.tick_params(direction='out',length=0.01,width=0.5,bottom=True, top=False, left=True, right=True,labeltop=False, labelbottom=True,labelrotation=90,labelsize=10,size=10)
    plt.yticks(rotation=0)
    plt.tight_layout()
    if element == 'qubit':
        plt.title(f'Qubit Mixer {cal} Calibration at {round(fc*1e-9,4)} GHz')
    elif element == 'rr':
        plt.title(f'Readout Mixer {cal} Calibration at {round(fc*1e-9,4)} GHz')
    plt.show()


#%% fit_res
def fit_res(f_data,I,Q,res_type='notch'):
    
    z_data = np.abs(I+1j*Q)
    fc = f_data[np.argmin(z_data)]
    if res_type == 'notch':
        z_data = -z_data-min(-z_data)
        idx = np.argwhere(np.diff(np.sign(z_data - 0.5*max(z_data)))).flatten()
        if len(idx) >=2:
            fwhm = f_data[idx[1]] - f_data[idx[0]]
        else:
            fwhm = np.nan
    print(f'Resonant Frequency: {fc*1e-9:.5f} GHz\nFWHM = {fwhm*1e-6} MHz\nkappa = {2*np.pi*fwhm*1e-6:.3f} MHz')
    return fc,fwhm

# def find_peak(f_data,z_data):
    
#     return fc

#%% fit_data
def fit_data(x_vector,y_vector,sequence='rabi',dt=0.01,fitFunc='',verbose=0):

    '''
    fit experimental data

    sequence:          'Rabi','ramsey', 'T1' or 'T2'
    x_vector:           time data
    y_vector:           voltage data
    dt:                 sequence stepsize. Used for extracting the frequency of the data
    '''
    x_vector = x_vector
    y_vector = y_vector*1e3

    amp = (max(y_vector)-min(y_vector))/2
    offset = np.mean(y_vector)



    if sequence == "rabi" or sequence == "qubit_temp" or sequence == 'qb-reset':
        fitFunction = rabi
        period = 1e3/(extract_freq(x_vector*1e3, y_vector, dt,plot=0))
        print('Period Initial Guess: %.1f ns'%(period))
        phase = pi
        x_vector = x_vector*1e3
        lb = [0.1*amp,0.1*period,0,-2*abs(offset)]
        ub = [10*amp,10*period,2*pi,2*abs(offset)]
        p0 = [amp,period,phase,offset]

    elif sequence == "p-rabi":
        fitFunction = rabi
        period = 1/(extract_freq(x_vector, y_vector, dt*1e-6,plot=0))
        print('Amplitude Initial Guess: %.3f'%(period))
        phase = pi
        lb = [0.1*amp,0.1*period,0,-2*abs(offset)]
        ub = [10*amp,10*period,2*pi,2*abs(offset)]
        p0 = [amp,period,phase,offset]

    elif sequence == "ramsey" or sequence == "cavity-cooling-ramsey" or sequence=="ramsey_chi":
        f = extract_freq(x_vector, y_vector,dt,plot=0)
        # print('Initial Guess for Freq:%.4f MHz'%(f))
        if x_vector[-1] > 20:
            tau = 30
        else:
            tau = 2
        phi = 0
        amp = abs(amp)
        # try:
        if fitFunc != 'envelope':
            p0 = [amp,f,phi,tau,offset]
            lb = [0.75*amp,0.1*f,-pi,0.01,-2*abs(offset)]
            ub = [2*amp,2*f,pi,100,2*abs(offset)]
            fitFunction = ramsey
            # fitted_pars, covar = scy.optimize.curve_fit(fitFunction, x_vector, y_vector,p0=p0,method='trf',bounds=[lb,ub],xtol=1e-12,maxfev=20e3)
        elif fitFunc == 'envelope':
            tau = 1
            env = get_envelope(y_vector, dt, distance=100)
            env = env(x_vector) + offset
            # env = get_envelope_LPF(x_vector, y_vector)*1e-3
            p0 = [amp,tau,offset]
            if offset < 0:
                p0 = [amp,tau,offset]
                lb = [0.95*amp,0.1,2*offset]
                ub = [2*amp,15,0.5*offset]
            elif offset >= 0:
                p0 = [amp,tau,offset]
                lb = [0.9*amp,0.1,0.9*offset]
                ub = [1.1*amp,15,1.1*offset]
            fitFunction = decay
            y_vector = env
            # fitted_pars, covar = scy.optimize.curve_fit(decay, x_vector, env,p0=p0,method='trf',bounds=[lb,ub],xtol=1e-12,maxfev=20e3)
    elif sequence == "echo" or sequence=='cavity-reset' or sequence=='cavity-cooling':
        if x_vector[-1] < 10:
            tau = 2
            tau_ub = 20
        else:
            tau = 20
            tau_ub = 300
        amp = y_vector[0] - y_vector[-1]
        p0 = [amp, tau, offset]
        amp_bounds = [0.95 * amp, 1.05 * amp]
        off_bounds = [0.95 * offset, 1.05 * offset]
        lb = [min(amp_bounds), 0.1, min(off_bounds)]
        ub = [max(amp_bounds), tau_ub, max(off_bounds)]
        # if offset < 0:
        #     lb = [0.95*amp,0.1,1.05*offset]
        #     ub = [1.05*amp,tau_ub,0.95*offset]
        # elif offset >= 0:
        #     lb = [0.95*amp,0.1,0.95*offset]
        #     ub = [1.05*amp,tau_ub,1.05*offset]
        fitFunction = decay
        # fitted_pars, covar = scy.optimize.curve_fit(decay, x_vector, y_vector,p0=p0,method='trf',bounds=[lb,ub],xtol=1e-12,maxfev=6000)
    elif sequence == "T1" or "T1diss":
        tau = 2
        amp = y_vector[0] - y_vector[-1]
        offset = y_vector[-1]
        if amp < 0:
            p0 = [amp,tau,offset]
            lb = [10*amp,0.1,-2*abs(offset)]
            ub = [0.5*amp,300,2*abs(offset)]
        elif amp >= 0:
            p0 = [amp,tau,offset]
            lb = [0.5*amp,0.1,-2*abs(offset)]
            ub = [10*amp,300,2*abs(offset)]
        fitFunction = decay
        # fitted_pars, covar = scy.optimize.curve_fit(, x_vector, y_vector,p0=p0,method='trf',bounds=[lb,ub],xtol=1e-12,maxfev=6000)
    
    elif sequence =='ringdown_off' or sequence=='ringdown_on':
        tau = 0.2
        amp = y_vector[0] - y_vector[-1]
        offset = y_vector[-1]
        p0 = [amp,tau,offset]
        fitFunction = decay
    fitted_pars, covar = scy.optimize.curve_fit(fitFunction, x_vector, y_vector,p0=p0,method='dogbox',xtol=1e-12,maxfev=40e3)
    error = np.sqrt(abs(np.diag(covar)))

    if verbose == 1:
        print('-'*100)
        print('Lower Bounds:',np.around(lb,1))
        print('Initial Guess:',np.around(p0,1))
        print('Upper Bounds:',np.around(ub,1))
        print('Best Fit Pars:',np.around(fitted_pars,1))
        print('Error:',np.around(error,1))
        print('-'*100)
    else:
        pass

    return fitted_pars,error


#%% plot_data
def plot_data(x_vector,y_vector,sequence='rabi',qubitDriveFreq=3.8e9,qb_power=1,fflDriveFreq=2e9,
                              pi2Width=32,nAverages=1,
                              integration_length=2e-6,cav_resp_time=5e-6,stepSize=5e-6, iteration = 1,
                              Tmax=5e-6,measPeriod=5e-6,active_reset=False,
                              fitted_pars=np.zeros(7),plot_mode=0,rr_IF=5e6,fitFunc='',savefig=True, amp=1, ffl_atten=0, rr_atten=0, flux=0, amp_ffl_scale=0, error=[0,0,0,0,0], ffl_len=0.):

    # x_vector = x_vector*1e3
    y_vector = y_vector*1e3
    #power=10*np.log10((10**-3)*(mag**2)/50)

    if sequence == "p-rabi":
        fig, ax = plt.subplots()
        ax.plot(x_vector, y_vector, '-o', markersize = 3, c='C0')
        ax.set_ylabel('Digitizer Voltage (mV)')
        ax.set_xlabel('Pulse Amplitude Scaling')
        ax.plot(x_vector,rabi(x_vector, fitted_pars[0], fitted_pars[1], fitted_pars[2],fitted_pars[3]),'r')
        ax.set_title('Power Rabi Measurement %03d'%(iteration))
        textstr = '$\omega_d$ = %.4f GHz\n$\hat{n}$ = %d\n$rr atten$=%.1f db'%((qubitDriveFreq)*1e-9,nAverages, rr_atten)

    elif sequence == "rabi" or sequence == "qubit_temp" or sequence == 'qb-reset':
        fig, ax = plt.subplots()
        ax.plot(x_vector*1e3, y_vector, '-o', markersize = 3, c='C0')
        ax.set_ylabel('Digitizer Voltage (mV)')
        ax.set_xlabel('Pulse Duration (ns)')
        ax.plot(x_vector*1e3,rabi(x_vector*1e3, fitted_pars[0], fitted_pars[1], fitted_pars[2],fitted_pars[3]),'r')
        ax.set_title('Rabi Measurement %03d'%(iteration))
        textstr = '$\omega_d$ = %.4f GHz\n$P_{qb}$ = %.2f dBm\n$T_{\pi/2}$ = %.1f ns\n$\hat{n}$ = %d\n$rr atten$=%.1f db\n $contrast$=%.2f mV\n amp scale=%.4f'%(qubitDriveFreq*1e-9,qb_power,round(fitted_pars[1]/4,1),nAverages, rr_atten, fitted_pars[0], amp)

    elif sequence == "ramsey" or sequence == "cavity-cooling-ramsey" or sequence=="ramsey_chi":

        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        fontSize = 16
        tickSize = 11
        markersize= 4
        linewidth = 2
        ax1.plot(x_vector, y_vector, 'o', markersize = markersize, c='C0')
        ax1.set_ylabel('Digitizer Voltage (mV)',fontsize=fontSize)
        ax1.set_xlabel('Pulse Separation ($\mu$s)')
        for label in (ax1.get_xticklabels() + ax1.get_yticklabels()):
            label.set_fontsize(tickSize)
        if fitFunc == 'envelope':
            ax1.plot(x_vector,decay(x_vector, fitted_pars[0], fitted_pars[1], fitted_pars[2]),'r',linewidth=linewidth)
        else:
            ax1.plot(x_vector,ramsey(x_vector, fitted_pars[0], fitted_pars[1], fitted_pars[2],fitted_pars[3],fitted_pars[4]),'r',linewidth=linewidth)
        ax1.set_title('Ramsey %03d'%(iteration))
        textstr = '$T_{\pi/2}$=%.3f ns\n$\omega_d$ = %.4f GHz\n$\Delta$ = %.2f MHz\n$T_2$ = %.3f $\mu$s\n$\hat{n}$ = %d\n$T_2error$ = (%.2f)us \n $(ffl,rr) atten$=(%.1f, %.1f) db \n$flux$= %.3f mA\n$amp ffl scale$=%.2f\n$amp rr scale$=%.4f \n $\omega_{ffl}$=%.3f GHz'%(pi2Width,qubitDriveFreq*1e-9,fitted_pars[1],fitted_pars[3],nAverages, error[3], ffl_atten ,rr_atten, flux,amp_ffl_scale, amp, fflDriveFreq*1e-9)

    elif sequence == "echo":

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(x_vector, y_vector, '-o', markersize = 3, c='C0')
        ax.set_ylabel('Digitizer Voltage (mV)')
        ax.set_xlabel('Pulse Separation ($\mu$s)')
        ax.plot(x_vector,decay(x_vector, fitted_pars[0], fitted_pars[1], fitted_pars[2]),'r')
        textstr = '$T_{\pi/2}$=%.1f ns\n$\omega_d$ = %.4f GHz\n$A_d$ = %.2f V\n$T_2$=%.2f$\mu$s\n$\hat{n}$ = %d\n$T_2error$ = %.2f us \n$flux$= %.3f mA'%(pi2Width,qubitDriveFreq*1e-9,qb_power,fitted_pars[1],nAverages,error[1], flux)
        ax.set_title('Echo Measurement %03d' %(iteration))

    elif sequence=='cavity-reset':

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(x_vector, y_vector, '-o', markersize = 3, c='C0')
        ax.set_ylabel('Digitizer Voltage (mV)')
        ax.set_xlabel('Pulse Separation ($\mu$s)')
        ax.plot(x_vector,decay(x_vector, fitted_pars[0], fitted_pars[1], fitted_pars[2]),'r')
        textstr = '$T_{\pi/2}$=%.1f ns\n$\omega_d$ = %.4f GHz\n$A_d$ = %.2f V\n$T_2$=%.2f$\mu$s\n$\hat{n}$ = %d\n$T_2error$ = %.2f us \n$flux$= %.3f mA\n$amp ffl scale$=%.2f\n$amp rr scale$=%.2f \n$ffl len$=%d \n $\omega_{ffl}$=%.3f GHz \n$(ffl,rr) atten$=(%.1f, %.1f) db'%(pi2Width,qubitDriveFreq*1e-9,qb_power,fitted_pars[1],nAverages,error[1], flux,amp_ffl_scale, amp ,ffl_len, fflDriveFreq*1e-9, ffl_atten,rr_atten)
        ax.set_title('Echo Measurement %03d' %(iteration))
    
    elif sequence=='cavity-cooling':

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(x_vector, y_vector, '-o', markersize = 3, c='C0')
        ax.set_ylabel('Digitizer Voltage (mV)')
        ax.set_xlabel('Pulse Separation ($\mu$s)')
        ax.plot(x_vector,decay(x_vector, fitted_pars[0], fitted_pars[1], fitted_pars[2]),'r')
        textstr = '$T_{\pi/2}$=%.1f ns\n$\omega_d$ = %.4f GHz\n$A_d$ = %.2f V\n$T_2$=%.2f$\mu$s\n$\hat{n}$ = %d\n$T_2error$ = %.2f us \n$flux$= %.3f mA\n$amp ffl scale$=%.2f\n$amp rr scale$=%.2f \n $\omega_{ffl}$=%.3f GHz \n$(ffl,rr) atten$=(%.1f, %.1f) db'%(pi2Width,qubitDriveFreq*1e-9,qb_power,fitted_pars[1],nAverages,error[1], flux,amp_ffl_scale, amp, fflDriveFreq*1e-9, ffl_atten,rr_atten)
        ax.set_title('Echo Measurement %03d' %(iteration))
   

    elif sequence =='ringdown_off' or sequence=='ringdown_on' :
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(x_vector, y_vector, '-o', markersize = 3, c='C0')
        ax.set_ylabel('Digitizer Voltage (mV)')
        ax.set_xlabel('Delay ($\mu$s)')
        ax.plot(x_vector,decay(x_vector, fitted_pars[0], fitted_pars[1], fitted_pars[2]),'r')
        textstr = '$\omega_d$ = %.4f GHz\n$T_{ringdown}$ = %.4f $\mu$s\n$\hat{n}$ = %d \n$flux$= %.3f mA\n$amp ffl scale$=%.2f \n$T_{ring}error$ = (%.2f)us\n$(ffl,rr) atten$=(%.1f, %.1f) db'%(qubitDriveFreq*1e-9,fitted_pars[1],nAverages, flux, amp_ffl_scale, error[1], ffl_atten,rr_atten)
        ax.set_title('resonator ring down %03d' %(iteration))
        
    elif sequence == "T1" or "dissT1":
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(x_vector, y_vector, '-o', markersize = 3, c='C0')
        ax.set_ylabel('Digitizer Voltage (mV)')
        ax.set_xlabel('Delay ($\mu$s)')
        ax.plot(x_vector,decay(x_vector, fitted_pars[0], fitted_pars[1], fitted_pars[2]),'r')
        textstr = '$T_{\pi/2}$=%.1f ns\n$\omega_d$ = %.4f GHz\n$T_1$ = %.3f $\mu$s\n$\hat{n}$ = %d\n$amp ffl scale$=%.2f \n$(ffl,rr) atten$=(%.1f, %.1f) db \n$flux$= %.3f mA\n$T_1error$ = %.2f us'%(pi2Width,qubitDriveFreq*1e-9,fitted_pars[1],nAverages, amp_ffl_scale, ffl_atten, rr_atten,flux, error[1])
        ax.set_title('T1 Measurement %03d' %(iteration))
        
    plt.gcf().text(0.95, 0.15, textstr, fontsize=14)

    plt.tick_params(axis='both',direction='in',bottom=True, top=True, left=True, right=True,size=8)

    plt.show()

    return fig

def extract_data(sweep,B0,nu,tauk,meas_device='CandleQubit_6',nMeasurements=100,nBackMeasurements=100,fileformat='new',sequence='ramsey',iteration=1):

    # get data
    if fileformat == 'old':
        filename = 'B0_%d_uV_nu_%d_kHz_tau_%d_ns' %(round(B0*1e6),round(nu*1e3),round(tauk*1e3))
        datafile = "E:\\generalized-markovian-noise\\%s\\sweep_data\\ramsey\\%s\\data\\data_%s.csv"%(meas_device,sweep,filename)
        tdata_background = pd.read_csv(datafile,on_bad_lines='skip',skiprows=3,header=None,nrows=1).dropna(axis='columns').to_numpy(np.float64)[0]
        tdata = pd.read_csv(datafile,on_bad_lines='skip',skiprows=5,header=None,nrows=1).dropna(axis='columns').to_numpy(np.float64)[0]
        ydata_background = pd.read_csv(datafile,on_bad_lines='skip',skiprows=7,header=None,nrows=nBackMeasurements).dropna(axis='columns').to_numpy(np.float64)
        ydata = pd.read_csv(datafile,on_bad_lines='skip',skiprows=7+2*(nBackMeasurements+1),header=None,nrows=nMeasurements).dropna(axis='columns').to_numpy(np.float64)
        # get exp parameters
        pars = pd.read_csv(datafile,on_bad_lines='skip',header=[0],nrows=1)
        keys = pars.keys()
        values = pars.values
        dictionary = {"mu":loads(pars.loc[0].at['AC_pars'])[0],
                      "sigma":loads(pars.loc[0].at['AC_pars'])[1],
                      "B0":loads(pars.loc[0].at['RT_pars'])[0],
                      # "nu": loads(pars.loc[0].at['RT_pars'])[1],
                      "tauk":loads(pars.loc[0].at['RT_pars'])[1]}
    elif fileformat == 'new':
        # cols_background = [i for i in range(0,94)]
        # cols = [i for i in range(0,250)]
        filename = 'B0_%d_uV_nu_%d_Hz_tau_%d_ns' %(round(B0*1e6),round(nu*1e3),round(tauk*1e3))
        # filename = 'B0_%d_uV_nu_%d_Hz_tau_%d_ns_%d' %(round(B0*1e6),round(nu*1e3),round(tauk*1e3),iteration)
        datafile = "E:\\generalized-markovian-noise\\%s\\sweep_data\\%s\\%s\\data\\data_%s.csv"%(meas_device,sequence,sweep,filename)
        tdata_background = pd.read_csv(datafile,on_bad_lines='skip',skiprows=3,header=None,nrows=1).to_numpy(np.float64)[0]
        tdata = pd.read_csv(datafile,on_bad_lines='skip',skiprows=5,header=None,nrows=1).to_numpy(np.float64)[0]
        ydata_background = pd.read_csv(datafile,on_bad_lines='skip',skiprows=7,header=None,nrows=nBackMeasurements).to_numpy(np.float64)
        ydata = pd.read_csv(datafile,on_bad_lines='skip',skiprows=7+nBackMeasurements+1,header=None,nrows=nMeasurements).to_numpy(np.float64)
        # get exp parameters
        pars = pd.read_csv(datafile,on_bad_lines='skip',header=[0],nrows=1)
        keys = pars.keys()
        values = pars.values
        dictionary = dict(zip(keys,values[0]))
    #print(dictionary)

    return tdata_background, ydata_background, tdata, ydata, dictionary


def rabi(x, amp,period,phase,offset):
    return amp*np.cos(2*pi*x/period+phase)+offset

def ramsey(x,amp,f,phase,tau,offset):
    return amp*np.cos(2*pi*f*x+phase)*np.exp(-x/tau)+offset

def beats(x,amp,f1,f2,phase1,phase2,tau,offset):
    return amp*np.cos(pi*(f1+f2)*x+phase1)*np.cos(pi*(f2-f1)*x+phase2)*np.exp(-x/tau)+offset

def decay(x,amp,tau,offset):
    return amp*np.exp(-x/tau)+offset

def mod_cos(x,amp,B0,nu,phi1,phi2,tau,offset):
    return amp*np.cos(B0/nu*np.sin(2*np.pi*nu*x+phi1)+phi2)*np.exp(-x/tau)+offset
    # return amp*np.cos(np.cos(nu*x)*f*x)*np.exp(-x/tau)+offset

def mod_dec(x,amp1,f1,phi1,tau1,amp2,phi2,tau2,offset):
    return amp1*np.cos(2*np.pi*f1*x+phi1)*np.exp(-x/tau1)+ amp2*np.sin(2*np.pi*f1*x+phi2)*np.exp(-x/tau2)+offset
    # return amp*np.cos(np.cos(nu*x)*f*x)*np.exp(-x/tau)+offset

def extract_freq(t_vector,y_vector,dt,plot=0):
    N = len(t_vector)
    dt = dt*1e6
    yf = scy.fft.fft(y_vector-np.mean(y_vector))
    xf = scy.fft.fftfreq(N,dt)[:round(N/2)]
    # print(len(xf))
    psd = 2.0/N * np.abs(yf[:round(N/2)])
    # print(len(psd))
    # print(psd)
    index_max = np.argmax(psd)
    if plot == 1:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(xf,psd)
        ax.set_xlabel('Frequency (MHz)')
        ax.set_ylabel('Power')
    # print(index_max)

    return xf[index_max]

def get_envelope(sig,dt, distance):
    # split signal into negative and positive parts
    sig = sig - np.mean(sig)
    u_x = np.where(sig > 0)[0]
    l_x = np.where(sig < 0)[0]
    u_y = sig.copy()
    u_y[l_x] = 0
    l_y = -sig.copy()
    l_y[u_x] = 0

    # find upper and lower peaks
    u_peaks, _ = scipy.signal.find_peaks(u_y, distance=distance)
    l_peaks, _ = scipy.signal.find_peaks(l_y, distance=distance)
    # use peaks and peak values to make envelope
    u_x = u_peaks
    u_y = sig[u_peaks]
    l_x = l_peaks
    l_y = sig[l_peaks]

    # add start and end of signal to allow proper indexing
    end = len(sig)
    u_x = np.concatenate(([0],u_x, [end]))*dt
    u_y = np.concatenate(([sig[0]],u_y, [sig[-1]]))
    l_x = np.concatenate(([0],l_x, [end]))*dt
    l_y = np.concatenate(([min(sig)],l_y, [np.mean(sig)]))
    # create envelope functions
    u = scipy.interpolate.interp1d(u_x, u_y,kind='cubic',fill_value="extrapolate")
    # l = scipy.interpolate.interp1d(l_x, l_y,kind='cubic')
    return u

def get_envelope_LPF(x,sig):

    N = len(sig)
    Tmax = x[-1]
    cutoff = 100e6
    fs = N/Tmax

    env = butter_lowpass_filter(sig, cutoff, fs)
    plt.plot(x,env)
    plt.show()

    return env

def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    sos = butter(order, normal_cutoff, btype='low', output = 'sos', analog=False)
    return sos

def butter_lowpass_filter(data,cutoff,fs):
    sos = butter_lowpass(cutoff, fs, order=5)
    y = sp.signal.sosfilt(sos, data)
    return y

def Volt2dBm(data):

    return 10*np.log10(1e3*data**2/50)

def Watt2dBm(x):
    '''
    converts from units of Watts to dBm
    '''
    return 10.*np.log10(x*1000.)

