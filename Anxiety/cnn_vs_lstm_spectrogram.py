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

#Anxiety: CNN vs LSTM
cnn_anxiety = pd.read_csv('./results and electrode names(csv)/spectrogram(cnn).csv')#change to your directory
cnn_accuracy = cnn_anxiety.iloc[30,:].tolist()#LOOCV accuracy
cnn_se = cnn_anxiety.iloc[32,:].tolist()#standard error

lstm_anxiety = pd.read_csv('./results and electrode names(csv)/spectrogram(lstm).csv')#change to your directory
lstm_accuracy = lstm_anxiety.iloc[30,:].tolist()#LOOCV accuracy
lstm_se = lstm_anxiety.iloc[32,:].tolist()#standard error

electrode = pd.read_csv('./results and electrode names(csv)/electrode_names.csv')
electrode = electrode['Channel Names'].tolist()

# calculate mean accuracy along electrodes
mean_cnn_accuracy_along_electrodes = sum(cnn_accuracy) / len(cnn_accuracy)
mean_lstm_accuracy_along_electrodes = sum(lstm_accuracy) / len(lstm_accuracy)
print('mean:',mean_cnn_accuracy_along_electrodes,mean_lstm_accuracy_along_electrodes)

#figure
plt.figure(figsize=(20, 5))

plt.errorbar(electrode, cnn_accuracy, yerr=cnn_se, label='CNN: Mean Accuracy and Standard Error of Spectrogram', fmt='-o', capsize=5, color='orange')

plt.errorbar(electrode, lstm_accuracy, yerr=lstm_se, label='LSTM: Mean Accuracy and Standard Error of Spectrogram', fmt='-s', capsize=5, color='blue')

plt.axhline(y=mean_cnn_accuracy_along_electrodes, color='black', linestyle='dashed', label='Mean CNN Accuracy along electrodes')
plt.axhline(y=mean_lstm_accuracy_along_electrodes, color='black', linestyle='dotted', label='Mean LSTM Accuracy along electrodes')

plt.title('CNN v.s. LSTM: Performance on Spectrogram per electrode', fontsize=20)
plt.xlabel('Electrode', fontsize=16)
plt.ylabel('Mean Accuracy', fontsize=16)

plt.xticks(rotation=90, fontsize=12)
plt.yticks(fontsize=12)

plt.legend(fontsize=10)

plt.show()
plt.savefig('./figure/CNN_vs_LSTM_spectrogram.svg', bbox_inches='tight')#change to your directory