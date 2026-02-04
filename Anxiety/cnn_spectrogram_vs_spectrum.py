import matplotlib as mpl

mpl.rcParams.update({
    'font.family': 'Times New Roman',
    'mathtext.fontset': 'stix',
    'axes.unicode_minus': False
})

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

print(plt.rcParams['font.family'])
print(plt.rcParams.get('font.serif'))

#Anxiety: spectrogram vs spectrum
spectrogram_anxiety = pd.read_csv('./results and electrode names(csv)/spectrogram(cnn).csv')#change to your directory
spectrogram_accuracy = spectrogram_anxiety.iloc[30,:].tolist()#LOOCV accuracy
spectrogram_se = spectrogram_anxiety.iloc[32,:].tolist()#standard error

spectrum_anxiety = pd.read_csv('./results and electrode names(csv)/spectrum(cnn).csv')#change to your directory
spectrum_accuracy = spectrum_anxiety.iloc[30,:].tolist()#LOOCV accuracy
spectrum_se = spectrum_anxiety.iloc[32,:].tolist()#standard error

electrode = pd.read_csv('./results and electrode names(csv)/electrode_names.csv')#change to your directory
electrode = electrode['Channel Names'].tolist()

# calculate mean accuracy along electrodes
mean_spectrogram_accuracy_along_electrodes = sum(spectrogram_accuracy) / len(spectrogram_accuracy)
mean_spectrum_accuracy_along_electrodes = sum(spectrum_accuracy) / len(spectrum_accuracy)
print('mean:',mean_spectrogram_accuracy_along_electrodes,mean_spectrum_accuracy_along_electrodes)

#figure
plt.figure(figsize=(20, 5))

plt.errorbar(electrode, spectrogram_accuracy, yerr=spectrogram_se, label='Mean Accuracy and Standard Error of Spectrogram', fmt='-o', capsize=5, color='orange')
plt.errorbar(electrode, spectrum_accuracy, yerr=spectrum_se, label='Mean Accuracy and Standard Error of Spectrum', fmt='-s', capsize=5, color='red')

plt.axhline(y=mean_spectrogram_accuracy_along_electrodes, color='black', linestyle='dashed', label='Mean Spectrogram Accuracy along electrodes')
plt.axhline(y=mean_spectrum_accuracy_along_electrodes, color='black', linestyle='dotted', label='Mean Spectrum Accuracy along electrodes')

plt.title('CNN: Performance on Spectrogram and Spectrum per electrode', fontsize=20)
plt.xlabel('Electrode', fontsize=16)
plt.ylabel('Mean Accuracy', fontsize=16)

plt.xticks(rotation=90, fontsize=12)
plt.yticks(fontsize=12)

plt.legend(fontsize=10)

plt.show()
plt.savefig('./figure/CNN_spectrogram_vs_spectrum.svg', bbox_inches='tight')#change to your directory