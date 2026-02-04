Anxiety

1. test-retest EEG dataset download: 
https://drive.google.com/drive/folders/1Dj78BDzMtpNUOpR5CID2tkUrdDezMMJa?usp=sharing

2.Y_label(csv) folder: contains csv files of normal group and anxiety group

3. best_model folder: contains the information about the CNN model of best electrode

4. results and electrode names(csv) folder: contains 
a) electrode name
b) result of CNN on spectrogram 
c) result of LSTM on spectrogram
d) result of CNN on spectrum

5. 'generate_spectrogram.py' for generating spectrogram

6.'generate_spectrum.py' for generating spectrum

7.'spectrogram(cnn).py' for training CNN model on spectrogram

8.'spectrogram(lstm).py' for training LSTM model on spectrogram

9.'spectrum(cnn).py' for training CNN model on spectrum

10.'cnn_spectrogram_spectrum.py' load the csv files in results and electrode names(csv) folder to plot the results of CNN on spectrogram and spectrum

11.'cnn_lstm_spectrogram.py' load the csv files in results and electrode names(csv) folder to plot the results of CNN and LSTM on spectrogram

12.'hierarchical dendrogram and sorted filter plot.py' load the best model in best_model folder to generate dendrogram and sorted filter plot

