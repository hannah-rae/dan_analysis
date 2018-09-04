import numpy as np
import matplotlib.pyplot as plt
import argparse
from sklearn.decomposition import PCA

import datasets

# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--n_components', type=int, default=3, help='number of principal components to use for PCA')
parser.add_argument('--normalize', action='store_true', help='normalize the data by dividing each bin by total counts')
parser.add_argument('--use_thermal', action='store_true', help='subtract ctn-cetn to show thermal counts')
args = parser.parse_args()

# Prepare the plot
fig, axes = plt.subplots(nrows=3, ncols=2, sharey=True, sharex=True)

# Plot the principal components of full grid data
data, _ = datasets.read_sim_data()
if args.normalize:
    data = datasets.normalize_counts(data)
pca = PCA(n_components=3)
pca.fit(data)
axes[0,0].step(datasets.time_bins_sim, pca.components_[0][:len(datasets.time_bins_sim)], where='post', linewidth=2, label='PC 1')
axes[0,0].step(datasets.time_bins_sim, pca.components_[1][:len(datasets.time_bins_sim)], where='post', linewidth=2, label='PC 2')
axes[0,0].step(datasets.time_bins_sim, pca.components_[2][:len(datasets.time_bins_sim)], where='post', linewidth=2, label='PC 3')
axes[0,0].legend(loc='upper right')
axes[0,0].set_xscale('log')
axes[0,0].set_title('Thermal Principal Components (Full Simulation)')
axes[0,0].set_xlabel('Time (us)')
axes[0,0].set_ylabel('Normalized Counts')
axes[0,1].step(datasets.time_bins_sim, pca.components_[0][-len(datasets.time_bins_sim):], where='post', linewidth=2, label='PC 1')
axes[0,1].step(datasets.time_bins_sim, pca.components_[1][-len(datasets.time_bins_sim):], where='post', linewidth=2, label='PC 2')
axes[0,1].step(datasets.time_bins_sim, pca.components_[2][-len(datasets.time_bins_sim):], where='post', linewidth=2, label='PC 3')
axes[0,1].legend(loc='upper right')
axes[0,1].set_xscale('log')
axes[0,1].set_title('Epithermal Principal Components (Full Simulation)')
axes[0,1].set_xlabel('Time (us)')
axes[0,1].set_ylabel('Normalized Counts')

# Plot the principal components of rover grid data
data, _ = datasets.read_grid_data(use_thermals=False)
if args.normalize:
    data = datasets.normalize_counts(data)
if args.use_thermal:
    X_thermal = np.ndarray(data.shape)
    n_bins = len(datasets.time_bins_dan)-1
    for i in range(X_thermal.shape[0]):
        thermal = data[i, :n_bins] - data[i, n_bins:]
        X_thermal[i] = np.concatenate([thermal, data[i, n_bins:]])
    data = X_thermal
pca = PCA(n_components=3)
pca.fit(data)
axes[1,0].step(datasets.time_bins_dan, pca.components_[0][:len(datasets.time_bins_dan)], where='post', linewidth=2, label='PC 1')
axes[1,0].step(datasets.time_bins_dan, pca.components_[1][:len(datasets.time_bins_dan)], where='post', linewidth=2, label='PC 2')
axes[1,0].step(datasets.time_bins_dan, pca.components_[2][:len(datasets.time_bins_dan)], where='post', linewidth=2, label='PC 3')
axes[1,0].legend(loc='upper right')
axes[1,0].set_xscale('log')
axes[1,0].set_title('Thermal Principal Components (Rover Simulation)')
axes[1,0].set_xlabel('Time (us)')
axes[1,0].set_ylabel('Normalized Counts')
axes[1,1].step(datasets.time_bins_dan, pca.components_[0][-len(datasets.time_bins_dan):], where='post', linewidth=2, label='PC 1')
axes[1,1].step(datasets.time_bins_dan, pca.components_[1][-len(datasets.time_bins_dan):], where='post', linewidth=2, label='PC 2')
axes[1,1].step(datasets.time_bins_dan, pca.components_[2][-len(datasets.time_bins_dan):], where='post', linewidth=2, label='PC 3')
axes[1,1].legend(loc='upper right')
axes[1,1].set_xscale('log')
axes[1,1].set_title('Epithermal Principal Components (Rover Simulation)')
axes[1,1].set_xlabel('Time (us)')
axes[1,1].set_ylabel('Normalized Counts')

# Plot principal components of DAN data
data, _, _ = datasets.read_dan_data(use_thermals=False)
if args.normalize:
    data = datasets.normalize_counts(data)
if args.use_thermal:
    X_thermal = np.ndarray(data.shape)
    n_bins = len(datasets.time_bins_dan)-1
    for i in range(X_thermal.shape[0]):
        thermal = data[i, :n_bins] - data[i, n_bins:]
        X_thermal[i] = np.concatenate([thermal, data[i, n_bins:]])
    data = X_thermal
pca = PCA(n_components=3)
pca.fit(data)
axes[2,0].step(datasets.time_bins_dan, pca.components_[0][:len(datasets.time_bins_dan)], where='post', linewidth=2, label='PC 1')
axes[2,0].step(datasets.time_bins_dan, pca.components_[1][:len(datasets.time_bins_dan)], where='post', linewidth=2, label='PC 2')
axes[2,0].step(datasets.time_bins_dan, pca.components_[2][:len(datasets.time_bins_dan)], where='post', linewidth=2, label='PC 3')
axes[2,0].legend(loc='upper right')
axes[2,0].set_xscale('log')
axes[2,0].set_title('Thermal Principal Components (DAN Actual)')
axes[2,0].set_xlabel('Time (us)')
axes[2,0].set_ylabel('Normalized Counts')
axes[2,1].step(datasets.time_bins_dan, pca.components_[0][-len(datasets.time_bins_dan):], where='post', linewidth=2, label='PC 1')
axes[2,1].step(datasets.time_bins_dan, pca.components_[1][-len(datasets.time_bins_dan):], where='post', linewidth=2, label='PC 2')
axes[2,1].step(datasets.time_bins_dan, pca.components_[2][-len(datasets.time_bins_dan):], where='post', linewidth=2, label='PC 3')
axes[2,1].legend(loc='upper right')
axes[2,1].set_xscale('log')
axes[2,1].set_title('Epithermal Principal Components (DAN Actual)')
axes[2,1].set_xlabel('Time (us)')
axes[2,1].set_ylabel('Normalized Counts')

plt.show()