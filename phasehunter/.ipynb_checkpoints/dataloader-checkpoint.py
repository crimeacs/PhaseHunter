from torch.utils.data import Dataset
from torchvision.transforms import functional as F
import torch
import numpy as np
from scipy import signal
from functools import reduce
from scipy.signal import butter, lfilter, detrend

class Augmentations:
    def __init__(self, padding=120, crop_length=6000, fs=100, lowcut=0.2, highcut=40, order=5):
        self.padding = padding
        self.crop_length = crop_length
        self.fs = fs
        self.lowcut = lowcut
        self.highcut = highcut
        self.order = order

        b, a = self.butter_bandpass(self.lowcut, self.highcut, self.fs, self.order)
        self.filter_b = b
        self.filter_a = a

    def butter_bandpass(self, lowcut, highcut, fs, order=5):
        return butter(order, [lowcut, highcut], fs=fs, btype='band')
    
    def butter_bandpass_filter(self, data, lowcut, highcut, fs, order=5):
        b, a = self.butter_bandpass(lowcut, highcut, fs, order=order)
        y = lfilter(self.filter_b, self.filter_a, data)
        return y

    def rotate_waveform(self, waveform, angle):
        fft_waveform = np.fft.fft(waveform)
        rotate_factor = np.exp(1j * angle)
        rotated_fft_waveform = fft_waveform * rotate_factor
        rotated_waveform = np.fft.ifft(rotated_fft_waveform)
        return rotated_waveform

    def shuffle(self, sample, target_P, target_S, test):
        if target_P - (self.crop_length-self.padding) > self.padding:
            start_indx = int(target_P - torch.randint(low=self.padding, 
                                                           high=(self.crop_length-self.padding), 
                                                           size=(1,)))
            if test == True:
                start_indx = int(first_phase - 2*self.padding)

        elif int(target_P-self.padding) > 0:
            start_indx = int(target_P - torch.randint(low=0, 
                                                        high=(int(target_P-self.padding)), 
                                                        size=(1,)))
            if test == True:
                start_indx = int(target_P - self.padding)
        else:
            start_indx = self.padding
            
        end_indx = start_indx + self.crop_length
        
        if (sample.shape[-1] - end_indx) < 0:
            start_indx += (sample.shape[-1] - end_indx)
            end_indx = start_indx + self.crop_length
            
        new_target_P  = target_P - start_indx
        new_target_S  = target_S - start_indx
        
        return start_indx, end_indx, new_target_P, new_target_S
    
    def cut(self, sample, start_indx, end_indx):
        sample_cropped = sample[:,start_indx:end_indx]
        return sample_cropped
    
    def preprocess(self, sample_cropped):
        # sample_cropped = detrend(sample_cropped)
        sample_cropped = self.butter_bandpass_filter(sample_cropped, lowcut=self.lowcut, highcut=self.highcut, fs=self.fs, order=self.order)
        window = signal.windows.tukey(sample_cropped[-1].shape[0], alpha=0.1)
        sample_cropped = sample_cropped*window
        return sample_cropped

    def add_z_component(self, sample_cropped):
        if len(sample_cropped) < 3:
            zeros = np.zeros((3, sample_cropped.shape[-1]))
            zeros[0] = sample_cropped
            sample_cropped = zeros
        return sample_cropped

    def rotate(self, sample_cropped, test):
        if test == False:
            probability = torch.randint(0,2, size=(1,)).item()
            if probability==1:
                angle = torch.FloatTensor(size=(1,)).uniform_(0.01, 359.9).item()
                sample_cropped = self.rotate_waveform(sample_cropped, angle).real
        return sample_cropped

    def normalize(self, sample_cropped):
        max_val = np.max(np.abs(sample_cropped))
        sample_cropped_norm = sample_cropped/max_val
        return sample_cropped_norm

    def channel_dropout(self, sample_cropped_norm, test):
        if test == False:
            probability = torch.randint(0,2, size=(1,)).item()
            channel = torch.randint(1,3, size=(1,)).item()
            if probability==1:
                sample_cropped_norm[channel,:] = 1e-6
        return sample_cropped_norm

    def apply(self, sample, target_P, target_S, test=False):
        
        start_indx, end_indx, new_target_P, new_target_S = self.shuffle(sample, target_P, target_S, test)

        sample_cropped = self.cut(sample, start_indx, end_indx)
        # sample_cropped = self.preprocess(sample_cropped)
        sample_cropped = self.add_z_component(sample_cropped)
        sample_cropped = self.rotate(sample_cropped, test)
        sample_cropped_norm = self.normalize(sample_cropped)
        sample_cropped_norm = self.channel_dropout(sample_cropped_norm, test)

        new_target_P = new_target_P/self.crop_length
        new_target_S = new_target_S/self.crop_length

        return sample_cropped_norm, new_target_P, new_target_S

class Waveforms_dataset(Dataset):
    def __init__(self, meta, data, test=False, transform=None, augmentations=None):
        # self.data_list = glob(data_path)
        self.meta = meta
        self.data = data
        self.test = test
        self.augmentations = augmentations
    
    def __len__(self):
        return len(self.meta)

    def __getitem__(self, idx):
        meta = self.meta.iloc[idx]
        sample = self.data[meta.name]

        target_P = float(meta.trace_P_final)
        target_S = float(meta.trace_S_final)

        if self.augmentations:
            sample, target_P, target_S = self.augmentations.apply(sample, target_P, target_S, test=self.test)

        # Setting labels to zero if they're not in the valid range or are NaNs
        if (target_P <= 0) or (target_P >= 1) or (np.isnan(target_P)):
            target_P = 0  
        if (target_S <= 0) or (target_S >= 1) or (np.isnan(target_S)):
            target_S = 0

        # If something went wrong
        if np.isnan(sample).any():
            sample = np.zeros((3, self.augmentations.crop_length))
            target_P = 0
            target_S = 0
            
        # Convert to tensor
        sample = torch.tensor(sample, dtype=torch.float)
        target_P = torch.tensor(target_P, dtype=torch.float)
        target_S = torch.tensor(target_S, dtype=torch.float)

        return sample, target_P, target_S
