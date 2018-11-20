import matplotlib
matplotlib.use('TkAgg')

import numpy as np
import argparse
import os.path
import matplotlib.pyplot as plt

from glob import glob
from subprocess import call
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture

import datasets

# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--n_components', type=int, default=3, help='number of principal components to use for PCA')
parser.add_argument('--n_gaussians', type=int, default=2, help='number of gaussians to use for Gaussian Mixture Model in PC space')
parser.add_argument('--plot_sol_hist', action='store_true', help='plot frequency of sol occurrence for DANs in each cluster')
parser.add_argument('--plot_geochem_hist', action='store_true', help='plot histogram of H and ACS values for DANs in each cluster')
# parser.add_argument('--plot_cluster_centers', action='store_true', help='plot DAN at center point of each cluster (as opposed to mean measurement)')
# parser.add_argument('--plot_cluster_means', action='store_true', help='plot mean measurement of each cluster (as opposed to center point)')
parser.add_argument('--use_restricted_bins', action='store_true', help='only run analysis for bins 18-34 (CTN) and 12-17 (CETN)')
args = parser.parse_args()

X, Y, Y_err, X_filenames = datasets.read_dan_data()
time_bins = datasets.time_bins_dan
n_bins = 64
if args.use_restricted_bins:
    X = np.take(X, range(17, 34)+range(n_bins+12, n_bins+17), axis=1)

# Fit PCA model and project data into PC space (and back out)
pca = PCA(n_components=args.n_components)
pca.fit(X)
transformed = pca.transform(X)

# gm = GaussianMixture(n_components=args.n_gaussians).fit(transformed)
# assignments = gm.predict(transformed)

# Plot distributions in PC space
# fig = plt.figure()
# ax1 = fig.add_subplot(111, projection='3d')
# ax1.set_xlabel('PC 1')
# ax1.set_ylabel('PC 2')
# ax1.set_zlabel('PC 3')
# clustered_filenames = []
# clustered_data = []
# clustered_y = []
# for i in range(args.n_gaussians):
#     cluster = transformed[np.where(assignments == i)]
#     cluster_filenames = X_filenames[np.where(assignments == i)]
#     cluster_y = Y[np.where(assignments == i)]
#     # Save these for later use
#     clustered_filenames.append(cluster_filenames)
#     clustered_data.append(np.array(cluster))
#     clustered_y.append(np.array(cluster_y))
#     # if args.save_rhaz_clusters:
#     #     make_rhaz_montages(cluster_filenames, i)
#     ax1.scatter(cluster[:,0], cluster[:,1], cluster[:,2], label='Dist. %d' % (i+1))
# ax1.legend(loc='upper right')

# Color points if they are outside of the standard deviation
mu = np.mean(transformed, axis=0)
std = np.std(transformed, axis=0)

outliers = transformed[np.where((transformed[:,0] > mu[0] + 2*std[0]) & (transformed[:,1] > mu[1] + 2*std[1]))]
inliers = transformed[np.where(np.invert((transformed[:,0] > mu[0] + 2*std[0]) & (transformed[:,1] > mu[1] + 2*std[1])))]
outlier_files = X_filenames[np.where((transformed[:,0] > mu[0] + 2*std[0]) & (transformed[:,1] > mu[1] + 2*std[1]))]
inlier_files = X_filenames[np.where(np.invert((transformed[:,0] > mu[0] + 2*std[0]) & (transformed[:,1] > mu[1] + 2*std[1])))]
print outliers.shape
# Plot distributions in PC space
fig = plt.figure()
ax1 = fig.add_subplot(111, projection='3d')
ax1.set_xlabel('PC 1')
ax1.set_ylabel('PC 2')
ax1.set_zlabel('PC 3')
ax1.scatter(inliers[:,0], inliers[:,1], inliers[:,2], label='Inliers', color='blue', picker=True)
ax1.scatter(outliers[:,0], outliers[:,1], outliers[:,2], label='Outliers', color='red', picker=True)
plt.legend(loc='best')

# Allow user to click on points and print which measurement the point belongs to
def onpick(event):
    ind = event.ind
    print 'if red:', outlier_files[ind]
    print 'if blue:', inlier_files[ind]

fig.canvas.mpl_connect('pick_event', onpick)

fig2, (ax2, ax3) = plt.subplots(nrows=1, ncols=2)
# Get the mean outlier curve
outliers_fspace = pca.inverse_transform(outliers)
outlier_mean = np.mean(outliers_fspace, axis=0)
outlier_std = np.std(outliers_fspace, axis=0)
# Get the mean inlier curve
inliers_fspace = pca.inverse_transform(inliers)
inlier_mean = np.mean(inliers_fspace, axis=0)
inlier_std = np.std(inliers_fspace, axis=0)
time_bins_m = [np.mean([time_bins[t], time_bins[t+1]]) for t in range(len(time_bins)-1)]
if args.use_restricted_bins:
    # Plot outlier mean thermal curve
    ax2.step(time_bins[17:34], outlier_mean[:17], where='post', linewidth=2, label='Outliers', color='red')
    ax2.errorbar(time_bins_m[17:34], outlier_mean[:17], yerr=outlier_std[:17], fmt='None', ecolor='k')
    # Plot outlier mean epithermal curve
    ax3.step(time_bins[12:17], outlier_mean[17:], where='post', linewidth=2, label='Outliers', color='red')
    ax3.errorbar(time_bins_m[12:17], outlier_mean[17:], yerr=outlier_std[17:], fmt='None', ecolor='k')
    # Plot inlier mean thermal curve
    ax2.step(time_bins[17:34], inlier_mean[:17], where='post', linewidth=2, label='Inliers', color='blue')
    ax2.errorbar(time_bins_m[17:34], inlier_mean[:17], yerr=inlier_std[:17], fmt='None', ecolor='k')
    # Plot inlier mean epithermal curve
    ax3.step(time_bins[12:17], inlier_mean[17:], where='post', linewidth=2, label='Inliers', color='blue')
    ax3.errorbar(time_bins_m[12:17], inlier_mean[17:], yerr=inlier_std[17:], fmt='None', ecolor='k')
else:
    ax2.step(time_bins[5:-1], cluster_mean[:64][5:], where='post', linewidth=2, label='Cluster %d' % (i+1))
    ax2.errorbar(time_bins_m[5:], cluster_mean[:64][5:], yerr=cluster_std[:64][5:], fmt='None', ecolor='k')
    ax3.step(time_bins[5:-1], cluster_mean[64:][5:], where='post', linewidth=2, label='Cluster %d' % (i+1))
    ax3.errorbar(time_bins_m[5:], cluster_mean[64:][5:], yerr=cluster_std[64:][5:], fmt='None', ecolor='k')

# Add graph labels
ax2.set_title("Thermal Neutron Die-Away")
ax3.set_title("Epithermal Neutron Die-Away")
ax2.set_xscale('log')
ax3.set_xscale('log')
ax2.set_xlabel('Time bin (us)')
ax3.set_xlabel('Time bin (us)')
ax2.set_ylabel('Counts')
ax3.set_ylabel('Counts')
ax2.legend(loc='upper right')
ax3.legend(loc='upper right')

plt.show()