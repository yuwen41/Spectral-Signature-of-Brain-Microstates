#download MNE first (pip install mne)

import mne
import os
import numpy as np
import matplotlib.pyplot as plt
import numpy as np

from scipy import signal
from mne.io import BaseRaw, RawArray
from mne import create_info


def standardize(raw: BaseRaw):
    """Standardize :class:`~mne.io.Raw` from the lemon dataset.

    This function will interpolate missing channels from the standard setup, then
    reorder channels and finally reference to a common average.

    Parameters
    ----------
    raw : Raw
        Raw data from the lemon dataset.

    Returns
    -------
    raw : Raw
        Standardize raw.

    Notes
    -----
    If you don't want to interpolate missing channels, you can use
    :func:`mne.channels.equalize_channels` instead to have the same electrodes across
    different recordings.
    """
    raw = raw.copy()

    standard_channels = [
        "Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "FC5",
        "FC1", "FC2", "FC6", "T7", "C3", "Cz", "C4", "T8",
        "CP5", "CP1", "CP2", "CP6", "AFz", "P7", "P3", "Pz",
        "P4", "P8", "PO9", "O1", "Oz", "O2", "PO10", "AF7",
        "AF3", "AF4", "AF8", "F5", "F1", "F2", "F6", "FT7",
        "FC3", "FC4", "FT8", "C5", "C1", "C2", "C6", "TP7",
        "CP3", "CPz", "CP4", "TP8", "P5", "P1", "P2", "P6",
        "PO7", "PO3", "POz", "PO4", "PO8",
    ]
 
    missing_channels = list(set(standard_channels) - set(raw.info["ch_names"]))

    if len(missing_channels) != 0:
        # add the missing channels as bads (array of zeros)
        missing_data = np.zeros((len(missing_channels), raw.n_times))
        data = np.vstack([raw.get_data(), missing_data])
        ch_names = raw.info["ch_names"] + missing_channels
        ch_types = raw.get_channel_types() + ["eeg"] * len(missing_channels)
        info = create_info(
            ch_names=ch_names, ch_types=ch_types, sfreq=raw.info["sfreq"]
        )
        raw = RawArray(data=data, info=info)
        raw.info["bads"].extend(missing_channels)

    raw.add_reference_channels("FCz")
    raw.reorder_channels(standard_channels)
    raw.set_montage("standard_1005")
    raw.interpolate_bads()
    raw.set_eeg_reference("average")
    return raw

folder_path = './.../'#load LEMON EEG dataset after downloading
names = os.listdir(folder_path)

for name in names:
  print(name)
  data = []
  file_path = os.path.join(folder_path, name)
  raw = mne.io.read_raw_eeglab(file_path+'/{}_EC.set'.format(name), preload=True)
  raw = standardize(raw)
  raw.pick("eeg")
  for i in range(0, 360, 60):#get the top six samples
    top = i
    bottom = i + 60
    print(top,bottom)
    cropped_raw = raw.copy().crop(tmin=top, tmax=bottom, include_tmax=False)
    f, t, Sxx = signal.spectrogram(cropped_raw.get_data(), fs=250, window='hann', nperseg=250, noverlap=250//2, nfft=None, detrend='constant', return_onesided=True, scaling='density', axis=-1, mode='psd')
    idx = (f >= 1) & (f <= 60)
    f_filtered = f[idx]
    Sxx_filtered = Sxx[:,idx, :]
    data.append(Sxx_filtered)
    # plt.figure(figsize=(10, 5))
    # plt.pcolormesh(t, f_filtered,Sxx_filtered[17], cmap='viridis')
    # plt.title('EEG Spectrogram')
  data = np.array(data)
  print(data.shape)
  np.save(f'./.../{name}', data)#change to your directory to save the spectrogram data