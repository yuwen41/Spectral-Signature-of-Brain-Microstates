from torch.nn import Module, Conv1d, Linear, ReLU, Sequential, CrossEntropyLoss, Dropout
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import KFold
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler

import numpy as np
import pandas as pd
import torch
import os

class CustomDataset(Dataset):
    def __init__(self, x_data, y_data):
        assert len(x_data) == len(y_data), "x_data and y_data must have the same length"
        self.x_data = torch.tensor(x_data, dtype=torch.float32)
        self.y_data = torch.tensor(y_data, dtype=torch.long)

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx):
        return self.x_data[idx], self.y_data[idx]

class MyModel(Module):
    def __init__(self, input_channel):
        super(MyModel, self).__init__()

        self.cnn_layers = Sequential(
            Conv1d(input_channel, 64, kernel_size=1, stride=1, padding=0),
            ReLU(),
            Dropout(0.5),
        )

        self.linear_layers = Sequential(
            Linear(64*60, 4096),
            ReLU(),
            Dropout(0.5),
            Linear(4096, 2048),
            ReLU(),
            Dropout(0.5),
            Linear(2048, 1024),
            ReLU(),
            Dropout(0.5),
            Linear(1024, 512),
            ReLU(),
            Dropout(0.5),
            Linear(512, 256),
            ReLU(),
            Dropout(0.5),
            Linear(256, 128),
            ReLU(),
            Dropout(0.5),
            Linear(128, 64),
            ReLU(),
            Dropout(0.5),
            Linear(64, 2)
        )

    def forward(self, x):
        x = self.cnn_layers(x)
        x = x.view(x.size(0), -1)
        x = self.linear_layers(x)
        return x

#load x and y
folder_path = './.../'#change your directory to load spectrum data
data = os.listdir(folder_path)

non_anxiety = pd.read_csv('./Y_label(csv)/normal.csv')#change to your directory
non_subject = non_anxiety.iloc[:,0].tolist()
anxiety = pd.read_csv('./Y_label(csv)/anxiety.csv')#change to your directory
anxiety_subject = anxiety.iloc[:,0].tolist()

#normal
x_non = []
for name in data:
  file_name = name.replace("sub", "sub-")
  file_name = file_name[:6]
  if file_name in non_subject:
      file_path = os.path.join(folder_path, name)
      x = np.load(file_path)
      x_non.append(x)

#anxiety
x_yes = []
for name in data:
  file_name = name.replace("sub", "sub-")
  file_name = file_name[:6]
  if file_name in anxiety_subject:
      file_path = os.path.join(folder_path, name)
      x = np.load(file_path)
      x_yes.append(x)

X_non = np.array(x_non)
X_yes = np.array(x_yes)
print('shape of normal group and anxiety group:', X_non.shape,X_yes.shape)

results = []
for e in range(61):
  print('electrode:', e+1)
  val_accuracies = []

  #change the shape of normal group from (15,3,61,60) to (15*3,61,60)
  non_x = []
  for i in range(X_non.shape[0]):
    for j in range(X_non[i].shape[0]):
      non_x.append(X_non[i][j])
  non_x = np.array(non_x)
  print('shape of normal group:', non_x.shape)

  #change the shape of anxiety group from (15,3,61,60) to (15*3,61,60)
  yes_x = []
  for i in range(X_yes.shape[0]):
    for j in range(X_yes[i].shape[0]):
      yes_x.append(X_yes[i][j])
  yes_x = np.array(yes_x)
  print('shape of anxiety group:', yes_x.shape)

  #change the shape of concatenated group from (30,3,61,60) to (30*3,61,60)
  x = []
  x_all = np.concatenate((X_non,X_yes))
  for i in range(x_all.shape[0]):
    for j in range(x_all[i].shape[0]):
      x.append(x_all[i][j])
  x = np.array(x)
  print('shape of concatenated group:', x.shape)


  #per electrode
  x = x[:,e,:]
  x_n = non_x[:,e,:]
  x_y = yes_x[:,e,:]

  #standardize x
  scaler = StandardScaler()

  scaler.fit(x)
  scaler_non = scaler.transform(x_n)
  scaler_non = np.array(scaler_non)
  print('shape of normal group after standardized:', scaler_non.shape)

  scaler_yes = scaler.transform(x_y)
  scaler_yes = np.array(scaler_yes)
  print('shape of anxiety group after standardized:', scaler_yes.shape)

  scaler_all = scaler.transform(x)
  scaler_all = np.array(scaler_all)
  print('shape of concatenated group after standardized:', scaler_all.shape)

  #expand dimension of x
  expand_non = np.expand_dims(scaler_non, axis=1)
  expand_yes = np.expand_dims(scaler_yes, axis=1)
  expand_all = np.expand_dims(scaler_all, axis=1)

  # create an empty array
  reshaped_non = np.empty((15, 3, 1, 60))

  for i in range(15):
      for j in range(3):
          reshaped_non[i, j, :, :] = expand_non[i * 3 + j, :, :]

  # create an empty array
  reshaped_yes = np.empty((15, 3, 1, 60))

  for i in range(15):
      for j in range(3):
          reshaped_yes[i, j, :, :] = expand_yes[i * 3 + j, :, :]

  # create an empty array
  reshaped_all = np.empty((30, 3, 1, 60))

  for i in range(30):
      for j in range(3):
          reshaped_all[i, j, :, :] = expand_all[i * 3 + j, :, :]

  print('shape of normal group, anxiety group and concatenated group after reshaped back:', reshaped_non.shape, reshaped_yes.shape, reshaped_all.shape)

  #LOOCV
  kf = KFold(n_splits=30, random_state=1234, shuffle=True)
  for k, (train_index, test_index) in enumerate(kf.split(reshaped_all)):
    print('round of k fold:', k+1)
    print(train_index, test_index)
    y = np.concatenate((np.zeros(15), np.ones(15)))
    x_train, x_val,y_train, y_val = reshaped_all[train_index], reshaped_all[test_index], y[train_index], y[test_index]

    #change the shape of train data from (29,3,1,60) to (29*3,1,60)
    X_train = []
    Y_train = []

    for i in range(x_train.shape[0]):
      for j in range(x_train[i].shape[0]):
        X_train.append(x_train[i][j])
        Y_train.append(y_train[i])

    #change the shape of val data from (1,3,1,60) to (1*3,1,60)
    X_val = []
    Y_val = []

    for i in range(x_val.shape[0]):
      for j in range(x_val[i].shape[0]):
        X_val.append(x_val[i][j])
        Y_val.append(y_val[i])

    X_train = np.array(X_train)
    Y_train = np.array(Y_train)
    X_val = np.array(X_val)
    Y_val = np.array(Y_val)
    print('shape of train data and val data:', X_train.shape,Y_train.shape,X_val.shape,Y_val.shape)

    train_dataset = CustomDataset(X_train, Y_train)
    val_dataset = CustomDataset(X_val, Y_val)

    # defining the model
    input_channel = 1
    model = MyModel(input_channel)

    # defining the optimizer
    optimizer = Adam(model.parameters(), lr=5e-5, weight_decay=1e-6)

    # defining the loss function
    criterion = CrossEntropyLoss()

    #defining batch size
    batch_size = 8

    # initialize a dictionary to store training history
    Config = {
      "train_loss": [],
      "train_acc": [],
      "val_loss": [],
      "val_acc": []
    }

    best_train_acc = 0.0
    best_val_acc = 0.0
    num_epochs = 200
    early_stop = 0

    # checking if GPU is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    #dataloader
    trainDataLoader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
    valDataLoader = DataLoader(val_dataset, batch_size=1)
    #testDataLoader = DataLoader(test_dataset, batch_size=1)

    #train and validation
    for epoch in tqdm(range(num_epochs), desc="Epochs"):
      model.train()
      train_loss = 0
      val_loss = 0
      train_correct = 0
      val_correct = 0

      for (x, y) in trainDataLoader:
        (x, y) = (x.to(device), y.to(device))
        y_output = model(x)
        _, y_pred = torch.max(y_output, dim=1)
        loss = criterion(y_output, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        train_correct += (y_pred == y).type(torch.float).sum().item()

      with torch.no_grad():
        model.eval()
        for (x, y) in valDataLoader:
          (x, y) = (x.to(device), y.to(device))
          y_output = model(x)
          _, y_pred = torch.max(y_output, dim=1)
          loss = criterion(y_output, y)

          val_loss += loss.item()
          val_correct += (y_pred == y).type(torch.float).sum().item()
          # print(val_correct)

      # calculate the average training and validation loss
      avg_train_loss = train_loss / len(trainDataLoader)
      avg_val_loss = val_loss / len(valDataLoader)

      # calculate the training and validation accuracy
      train_accuracy = train_correct / len(train_dataset)
      val_accuracy = val_correct / len(val_dataset)

      # update our training history
      Config["train_loss"].append(avg_train_loss)
      Config["train_acc"].append(train_accuracy)
      Config["val_loss"].append(avg_val_loss)
      Config["val_acc"].append(val_accuracy)

      if train_accuracy >= best_train_acc:
        best_train_acc = train_accuracy

      if val_accuracy >= best_val_acc:
        early_stop = 0
        best_val_acc = val_accuracy
        # torch.save(model.state_dict(), './drive/MyDrive/Colab Notebooks/CNN/best_model(anxiety).pth')
        # print(f"Model saved with val accuracy: {val_accuracy:.4f}")

      else:
        early_stop += 1
        print('early stop:', early_stop)
        if early_stop == 30:
          print("Early stopping!")
          break

      tqdm.write(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, Val Loss: {avg_val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}')

    # Store the validation accuracy for each fold
    val_accuracies.append(best_val_acc)

  # Add the results of the current electrode to the DataFrame
  results.append(val_accuracies)
  print(results)

df = pd.DataFrame(results)
df.to_csv('./.../spectrum(cnn).csv', index=False)#change to your directory

print("Results saved to spectrum(cnn).csv")