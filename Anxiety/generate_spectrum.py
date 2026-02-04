#download MNE first (pip install mne)

import mne
import os
import numpy as np

#load preprocessed data
list = [1,3,5,7,8,9,10,12,13,18,22,23,24,27,31,32,33,34,35,40,43,45,46,50,51,52,54,57,59,60]

for i in list:
  data = []
  for j in range(1,4):
    raw = mne.io.read_raw_eeglab(f'./.../sub{i:02d}/sub{i:02d}_{j:02d}_EC.set', preload=True)#load test-retest EEG dataset after downloading
    spectrum = raw.compute_psd(method='welch', fmin=1, fmax=60,n_fft=500, n_overlap=500//2, n_per_seg=500, window='hann')
    # spectrum.plot()
    data.append(spectrum.get_data())
  data = np.array(data)
  print(data.shape)
  np.save(f'./.../sub{i:02d}', data)#change to your directory to save the spectrum data