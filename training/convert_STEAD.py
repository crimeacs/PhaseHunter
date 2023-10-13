import pandas as pd
import numpy as np
import h5py
import os
from scipy.signal import resample
from tqdm.auto import tqdm

## Uncomment to download STEAD
# import seisbench
# import seisbench.data as sbd

# seisbench.cache_data_root = 'Seisbench_DATA'

# # Download STEAD dataset
# stead = sbd.STEAD(force=True)

# Load metadata
metadata = pd.read_csv('Seisbench_DATA/stead/metadata.csv')
metadata = metadata[(~metadata.trace_p_arrival_sample.isna())]

# # Add a new 'bucket' column to the metadata
metadata['bucket'] = metadata.trace_name.apply(lambda x: x.split('$')[0])

# Set final columns
metadata['trace_P_final'] = -1
metadata['trace_S_final'] = -1

sampling_rate = 100

# # Open the HDF5 file
with h5py.File('Seisbench_DATA/stead/waveforms.hdf5', 'r') as f:
    # Create the memmap array with the appropriate shape and datatype
    target_samples = 9000
    channels = 3
    memmap_shape = (len(metadata), channels, target_samples)
    memmap_dir = "Seisbench_DATA"  # replace with your desired directory
    memmap_filename = os.path.join(memmap_dir, 'stead.dat')
    dtype = np.float32  # adjust this as needed
    memmap_array = np.memmap(memmap_filename, dtype=dtype, mode='w+', shape=memmap_shape)

    i = 0
    # Iterate through rows in the metadata
    for name, row in tqdm(metadata.iterrows(), total=len(metadata)):
        # Get corresponding data
        dataset_name = row.bucket
        # Parse trace name to get individual index and original samples
        _, index_and_samples = row.trace_name.split('$')
        index, _, original_samples = map(int, index_and_samples.split(',:'))
        data = np.array(f['data'][dataset_name][index])
        
        # Resample, pad or cut, and normalize
        sampling_rate = row.trace_sampling_rate_hz
        if sampling_rate != 100:
            data = resample(data, int(data.shape[-1] * 100 / sampling_rate), axis=-1)

        p_index = row.trace_p_arrival_sample * 100 / sampling_rate
        s_index = row.trace_s_arrival_sample * 100 / sampling_rate
        
        # data = data[0,:]
        
        if len(data.shape) < 2:
            data = data.reshape(1, -1)
            
        if data.shape[-1] > target_samples:
            start_index = int(np.round(p_index, 0))

            if start_index >= 300:
                start_index -= 300
                if np.isnan(s_index) == False:
                    s_index = 300+int(np.round(s_index - p_index,0))
                p_index = 300

            else:
                start_index = 0
            data = data[:, start_index:start_index + target_samples]
            
        # check the shape of data and pad accordingly
        if data.shape[0] < channels:
            # If data has less than 3 channels, expand to 3 channels by padding with zeros
            padding = ((0, channels - data.shape[0]), (0, 0))
            data = np.pad(data, padding)
            
        if data.shape[1] < target_samples:
            # If data has less than 9000 samples, pad to 9000 samples
            padding = ((0, 0), (0, target_samples - data.shape[1]))
            data = np.pad(data, padding)
    
        metadata['trace_P_final'].loc[row.name] = p_index
        metadata['trace_S_final'].loc[row.name] = s_index

        # # normalize
        # data = data.astype('float32')
        # data -= np.mean(data, axis=-1, keepdims=True)
        # data /= np.max(np.abs(data))

        # # Write data to the memmap array
        memmap_array[i] = data.astype(dtype)

        i+=1
        
    # Flush changes to disk
    memmap_array.flush()

metadata.to_csv('Seisbench_DATA/stead_mem.csv')