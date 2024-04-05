from qm.qua import dual_demod, declare, declare_stream
import h5py
import os
from pathlib import Path
import json
from datetime import datetime

def datetime_converter(o):
    if isinstance(o, datetime):
        return o.__str__()
#%%%% res_demod
def res_demod(I, Q,switch_weights=False):
    if switch_weights:
        return (dual_demod.full("cos", "out1", "minus_sin", "out2", I),
                dual_demod.full("sin", "out1", "cos", "out2", Q))
    else:
        return (dual_demod.full("cos", "out1", "sin", "out2", I),
                dual_demod.full("minus_sin", "out1", "cos", "out2", Q))
    
def save_data(dataPath, iteration, metadata, data, Tags={}):
    # save data
    print(f"Saving data to {dataPath}")
   
    
    if not os.path.exists(dataPath):
        Path(dataPath).mkdir(parents=True, exist_ok=True)
    with h5py.File(f"{dataPath}/data_{iteration:03d}.hdf5", "w") as datafile:
        metadata_str = json.dumps(metadata, default=datetime_converter)
        datafile.create_dataset("metadata", data=metadata_str)
        for key, value in data.items():
                    datafile.create_dataset(f"data/{key}", data=value)
        tags_str = json.dumps(Tags, default=datetime_converter)
        datafile.create_dataset("Tags", data=tags_str.encode('utf8'))
#%%%% declare_vars
def declare_vars(types):
    return [declare(tp) for tp in types]

#%%%% declare_streams
def declare_streams(stream_num=1):
    return [declare_stream() for num in range(stream_num)]     
