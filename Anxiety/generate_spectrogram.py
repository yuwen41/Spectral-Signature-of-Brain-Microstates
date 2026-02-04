#download MNE first (pip install mne)

import mne
import os
import numpy as np

from scipy import signal

#load preprocessed data
list = [1,3,5,7,8,9,10,12,13,18,22,23,24,27,31,32,33,34,35,40,43,45,46,50,51,52,54,57,59,60]

for i in list:
  data = []
  for j in range(1,4):
    raw = mne.io.read_raw_eeglab(f'./.../sub{i:02d}/sub{i:02d}_{j:02d}_EC.set', preload=True)#load test-retest EEG dataset after downloading
    f, t, Sxx = signal.spectrogram(raw.get_data(), fs=500,window='hann', nperseg=500, noverlap=500//2, nfft=None, detrend='constant', return_onesided=True, scaling='density', axis=-1, mode='psd')
    idx = (f >= 1) & (f <= 60)
    f_filtered = f[idx]
    Sxx_filtered = Sxx[:,idx, :]
    data.append(Sxx_filtered)
    # plt.figure(figsize=(10, 5))
    # plt.pcolormesh(t, f_filtered,Sxx_filtered[17], cmap='viridis')
    # plt.title('EEG Spectrogram')
  data = np.array(data)
  print(data.shape)
  np.save(f'./.../sub{i:02d}', data)#change to your directory to save the spectrogram data