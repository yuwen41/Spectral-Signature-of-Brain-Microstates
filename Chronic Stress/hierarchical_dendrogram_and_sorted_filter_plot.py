from torch.nn import Module, Conv1d, Linear, ReLU, Sequential,  Dropout

import numpy as np
import matplotlib.pyplot as plt
import torch

class MyModel(Module):
    def __init__(self, input_channel):
        super(MyModel, self).__init__()

        self.cnn_layers = Sequential(
            Conv1d(input_channel, 64, kernel_size=1, stride=1, padding=0),
            ReLU(),
            Dropout(0.5),
        )

        self.linear_layers = Sequential(
            Linear(64*599, 4096),
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

# Load the model
input_channel = 60
model = MyModel(input_channel)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.load_state_dict(torch.load('./best_model/best_model(stress).pth', map_location=device))#load model of the best electrode
model.to(device)
model.eval()
print(model)

# get the CNN filters
conv1_filters = model.cnn_layers[0].weight.data
print("conv1 filters shape:", conv1_filters.shape)

#normalize
f_min, f_max = conv1_filters.min(), conv1_filters.max()
conv1_filters = (conv1_filters - f_min) / (f_max - f_min)
print("conv1 filters shape:", conv1_filters.shape)


from sklearn.metrics.pairwise import cosine_similarity
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist, squareform
from matplotlib.gridspec import GridSpec

# change CNN filters to one dimension
n_filters = conv1_filters.shape[0]
flattened_filters = conv1_filters.reshape(n_filters, -1)

#consine similarity matrix
cosine_sim_matrix = cosine_similarity(flattened_filters)
print("Cosine similarity matrix shape:", cosine_sim_matrix.shape)
clipped_cosine_sim_matrix = np.clip(cosine_sim_matrix, -1, 1)

distance = 1-clipped_cosine_sim_matrix
np.fill_diagonal(distance, 0)

reconstructed_distance_matrix = squareform(distance)

#consine distance
cosine_distances = pdist(flattened_filters, metric='cosine')

linked = linkage(cosine_distances, method='complete', metric='cosine')

# dendrogram
plt.figure(figsize=(20, 7))
dendrogram(linked,
            orientation='top',
            labels=range(1, n_filters + 1),
            distance_sort='descending',
            show_leaf_counts=True,
            leaf_rotation=0)
plt.title('Hierarchical Clustering Dendrogram', fontsize=18)
plt.xlabel('Filter Index', fontsize=14)
plt.ylabel('Cosine Distance', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.show()
# plt.savefig()#change to your directory

filter_order = dendrogram(linked,orientation='top',labels=range(1, n_filters + 1),distance_sort='descending',show_leaf_counts=True)['ivl']


filter_order = [i - 1 for i in filter_order]

# Number of filters
n_filters = conv1_filters.shape[0]

# Create a figure to hold the subplots
fig_height = 10  # height for each subplot
fig_width = 10 # width for the entire figure
fig = plt.figure(figsize=(fig_width, fig_height))
gs = GridSpec(1, n_filters + 2, width_ratios=[1] * n_filters + [5, 1.5], figure=fig)

# Plot each filter
for i in range(n_filters):
    ax = fig.add_subplot(gs[0, i])
    # Get the filter
    f = conv1_filters[filter_order[i], :, :]
    im = ax.imshow(f, cmap='gray_r', aspect='auto')
    ax.set_xlabel(f'{filter_order[i] + 1}', fontsize=6)
    ax.set_xticks([])
    ax.set_yticks([])

    # Add y-axis labels and title for the first filter
    if i == 0:
        ax.set_yticks(np.arange(0, 60))
        ax.set_yticklabels(np.arange(1, 61), fontsize=6)
        ax.set_ylabel('Frequency (Hz)', fontsize=10, fontweight='bold')

cbar_ax = fig.add_subplot(gs[0, -1])
cbar = plt.colorbar(im, cax=cbar_ax, ticks=np.linspace(0.1, 1.0, 10))
cbar.ax.tick_params(labelsize=8)

# Add x-axis label for the entire figure
fig.text(0.5, 0.07, 'Filter Index', ha='center', fontsize=10, fontweight='bold')

plt.subplots_adjust(wspace=0.1, hspace=0.1)
# Add title for the entire figure
fig.suptitle('Sorted Filter Plot', fontsize=12, y=0.93, fontweight='bold')
plt.show()
# plt.savefig()#change to your directory
