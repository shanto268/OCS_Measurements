from VISAdrivers.sa_api import *


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
    return sas