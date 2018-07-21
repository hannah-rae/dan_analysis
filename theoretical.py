'''
Data from Hardgrove et al. 2011. These do not use a real He-3 detector or a real PNG, 
so the results represent the behavior of neutron die-away curves in the absence of real
detectors and electronics. Effectively, these outputs represent the theoretical behavior 
of thermal and epithermal neutrons after a 14.1 MeV neutron pulse.
'''

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
from sklearn.cluster import KMeans

# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--n_components', type=int, help='number of principal components to use for PCA')
parser.add_argument('--n_clusters', type=int, help='number of clusters to use for K-means clustering of PCs')
parser.add_argument('--plot_cluster_centers', action='store_true', help='plot DAN at center point of each cluster (as opposed to mean measurement)')
parser.add_argument('--plot_cluster_means', action='store_true', help='plot mean measurement of each cluster (as opposed to center point)')
parser.add_argument('--use_all_bins', action='store_true', help='use time bins up to 200000 rather than 100000 us')
parser.add_argument('--plot_components', action='store_true', help='plot the principal components')

args = parser.parse_args()

# Since the DAN time bins only go up to 100000 us, include the option to only model this range
time_bins = [0.00, 1250.00, 2500.00, 3750.00, 5000.00, 6250.00, 7500.00, 8750.00, 10000.00, 11250.00, 
             12500.00, 13750.00, 15000.00, 16250.00, 17500.00, 18750.00, 20000.00, 21250.00, 22500.00, 
             23750.00, 25000.00, 26250.00, 27500.00, 28750.00, 30000.00, 31250.00, 32500.00, 33750.00, 
             35000.00, 36250.00, 37500.00, 38750.00, 40000.00, 41250.00, 42500.00, 43750.00, 45000.00, 
             46250.00, 47500.00, 48750.00, 50000.00, 51250.00, 52500.00, 53750.00, 55000.00, 56250.00, 
             57500.00, 58750.00, 60000.00, 61250.00, 62500.00, 63750.00, 65000.00, 66250.00, 67500.00, 
             68750.00, 70000.00, 71250.00, 72500.00, 73750.00, 75000.00, 76250.00, 77500.00, 78750.00, 
             80000.00, 81250.00, 82500.00, 83750.00, 85000.00, 86250.00, 87500.00, 88750.00, 90000.00, 
             91250.00, 92500.00, 93750.00, 95000.00, 96250.00, 97500.00, 98750.00, 100000.00]

if args.use_all_bins:
    time_bins = [0.00, 1250.00, 2500.00, 3750.00, 5000.00, 6250.00, 7500.00, 8750.00, 10000.00, 11250.00, 
                 12500.00, 13750.00, 15000.00, 16250.00, 17500.00, 18750.00, 20000.00, 21250.00, 22500.00, 
                 23750.00, 25000.00, 26250.00, 27500.00, 28750.00, 30000.00, 31250.00, 32500.00, 33750.00, 
                 35000.00, 36250.00, 37500.00, 38750.00, 40000.00, 41250.00, 42500.00, 43750.00, 45000.00, 
                 46250.00, 47500.00, 48750.00, 50000.00, 51250.00, 52500.00, 53750.00, 55000.00, 56250.00, 
                 57500.00, 58750.00, 60000.00, 61250.00, 62500.00, 63750.00, 65000.00, 66250.00, 67500.00, 
                 68750.00, 70000.00, 71250.00, 72500.00, 73750.00, 75000.00, 76250.00, 77500.00, 78750.00, 
                 80000.00, 81250.00, 82500.00, 83750.00, 85000.00, 86250.00, 87500.00, 88750.00, 90000.00, 
                 91250.00, 92500.00, 93750.00, 95000.00, 96250.00, 97500.00, 98750.00, 100000.00, 101250.00, 
                 102500.00, 103750.00, 105000.00, 106250.00, 107500.00, 108750.00, 110000.00, 111250.00, 
                 112500.00, 113750.00, 115000.00, 116250.00, 117500.00, 118750.00, 120000.00, 121250.00, 
                 122500.00, 123750.00, 125000.00, 126250.00, 127500.00, 128750.00, 130000.00, 131250.00, 
                 132500.00, 133750.00, 135000.00, 136250.00, 137500.00, 138750.00, 140000.00, 141250.00, 
                 142500.00, 143750.00, 145000.00, 146250.00, 147500.00, 148750.00, 150000.00, 151250.00, 
                 152500.00, 153750.00, 155000.00, 156250.00, 157500.00, 158750.00, 160000.00, 161250.00, 
                 162500.00, 163750.00, 165000.00, 166250.00, 167500.00, 168750.00, 170000.00, 171250.00, 
                 172500.00, 173750.00, 175000.00, 176250.00, 177500.00, 178750.00, 180000.00, 181250.00, 
                 182500.00, 183750.00, 185000.00, 186250.00, 187500.00, 188750.00, 190000.00, 191250.00, 
                 192500.00, 193750.00, 195000.00, 196250.00, 197500.00, 198750.00, 200000.00]

time_bins_m = [np.mean([time_bins[t], time_bins[t+1]]) for t in range(len(time_bins)-1)]

data_dir = '/Users/hannahrae/data/dan_theoretical'
n = len(glob(os.path.join(data_dir, '*.o')))
if args.use_all_bins:
    X = np.ndarray((n, 322))
else:
    X = np.ndarray((n, 160))
X_filenames = []
for i, simfile in enumerate(glob(os.path.join(data_dir, '*.o'))):
    correct_userbin = False
    reading_th = None
    reading_epi = None
    counts_th = []
    counts_epi = []
    for line in open(simfile):
        if 'user bin total' in line.rstrip() and prev_line == ' \n':
            correct_userbin = True

        if 'energy bin:   0.00000E+00 to  3.00000E-07' in line.rstrip() and correct_userbin and reading_th == None:
            reading_th = True
            continue
        if reading_th and 'total' in line.rstrip():
            reading_th = False
            correct_userbin = False

        if 'energy bin:   3.00000E-07 to  1.00000E-05' in line.rstrip() and correct_userbin and reading_epi == None:
            reading_epi = True
            continue
        if reading_epi and 'total' in line.rstrip():
            reading_epi = False
            correct_userbin = False

        if reading_th:
            if 'detector' not in line and 'time' not in line:
                counts_th.append(float(line.rstrip().split()[1]))
        elif reading_epi:
            if 'detector' not in line and 'time' not in line:
                counts_epi.append(float(line.rstrip().split()[1]))
        prev_line = line
    if args.use_all_bins:
        X[i] = np.concatenate([np.array(counts_th), np.array(counts_epi)])
    else:
        X[i] = np.concatenate([np.array(counts_th)[:len(time_bins)-1], np.array(counts_epi)[:len(time_bins)-1]])
    X_filenames.append(simfile)

X_filenames = np.array(X_filenames)

# Fit PCA model and project data into PC space (and back out)
pca = PCA(n_components=args.n_components)
pca.fit(X)
transformed = pca.transform(X)
back_proj = pca.inverse_transform(transformed)
# Plot points in PC space
fig = plt.figure()
if args.n_components == 3:
    ax1 = fig.add_subplot(111, projection='3d')
    ax1.set_xlabel('PC 1')
    ax1.set_ylabel('PC 2')
    ax1.set_zlabel('PC 3')
    ax1.scatter(transformed[:,0], transformed[:,1], transformed[:,2], picker=True)
elif args.n_components == 2:
    ax1 = fig.add_subplot(111)
    ax1.set_xlabel('PC 1')
    ax1.set_ylabel('PC 2')
    ax1.scatter(transformed[:,0], transformed[:,1], picker=True)

# Allow user to click on points and print which measurement the point belongs to
def onpick(event):
    ind = event.ind
    print X_filenames[ind[0]]

fig.canvas.mpl_connect('pick_event', onpick)

if args.plot_components:
    fig2, (ax2, ax3) = plt.subplots(nrows=1, ncols=2)
    if args.use_all_bins:
        ax2.step(time_bins, pca.components_[0][:161], where='post', linewidth=2, label='PC 1')
        ax2.step(time_bins, pca.components_[1][:161], where='post', linewidth=2, label='PC 2')
        ax2.step(time_bins, pca.components_[2][:161], where='post', linewidth=2, label='PC 3')

        ax3.step(time_bins, pca.components_[0][161:], where='post', linewidth=2, label='PC 1')
        ax3.step(time_bins, pca.components_[1][161:], where='post', linewidth=2, label='PC 2')
        ax3.step(time_bins, pca.components_[2][161:], where='post', linewidth=2, label='PC 3')
    else:
        ax2.step(time_bins[:-1], pca.components_[0][:80], where='post', linewidth=2, label='PC 1')
        ax2.step(time_bins[:-1], pca.components_[1][:80], where='post', linewidth=2, label='PC 2')
        ax2.step(time_bins[:-1], pca.components_[2][:80], where='post', linewidth=2, label='PC 3')

        ax3.step(time_bins[:-1], pca.components_[0][80:], where='post', linewidth=2, label='PC 1')
        ax3.step(time_bins[:-1], pca.components_[1][80:], where='post', linewidth=2, label='PC 2')
        ax3.step(time_bins[:-1], pca.components_[2][80:], where='post', linewidth=2, label='PC 3')
    
    ax2.legend(loc='upper right')
    ax2.set_xscale('log')
    ax2.set_title('Thermal neutron counts')
    ax2.set_xlabel('Time (us)')
    ax2.set_ylabel('Counts')
    ax3.legend(loc='upper right')
    ax3.set_xscale('log')
    ax3.set_title('Epithermal neutron counts')
    ax3.set_xlabel('Time (us)')
    ax3.set_ylabel('Counts')

# # Determine the number of clusters to use
# # inertia = []
# # for i in range(1, 15):
# #     kmeans = KMeans(n_clusters=i).fit(transformed)
# #     inertia.append(kmeans.inertia_)

# # plt.plot(range(1,15), inertia)
# # plt.xticks(range(1,15))
# # plt.xlabel("Number of clusters")
# # plt.ylabel("Intertia/Distortion")
# # plt.title("How many means for k-means?")
# # plt.show()

# Find clusters in the data in PC space
# kmeans = KMeans(n_clusters=args.n_clusters).fit(transformed)
# assignments = kmeans.predict(transformed)

# # Plot clusters in PC space
# fig = plt.figure()
# ax1 = fig.add_subplot(111, projection='3d')
# ax1.set_xlabel('PC 1')
# ax1.set_ylabel('PC 2')
# ax1.set_zlabel('PC 3')
# clustered_filenames = []
# clustered_data = []
# for i in range(args.n_clusters):
#     cluster = transformed[np.where(assignments == i)]
#     cluster_filenames = X_filenames[np.where(assignments == i)]
#     # Save these for later use
#     clustered_filenames.append(cluster_filenames)
#     clustered_data.append(np.array(cluster))
#     ax1.scatter(cluster[:,0], cluster[:,1], cluster[:,2], label='Cluster %d' % (i+1))
# ax1.legend(loc='upper right')

# # Plot die-away curves for CTN and CETN
# # NOTE: we are currently plotting the "de-noised" curves, i.e. projected back from PC space
# # Need to determine if this is the right thing to do, or we want to plot the original data
# fig2, (ax2, ax3) = plt.subplots(nrows=1, ncols=2)

# if args.plot_cluster_centers:
#     centers = kmeans.cluster_centers_
#     centers_fspace = pca.inverse_transform(centers)
#     # time_bins[0] = 1e-20
#     for i, c in enumerate(centers_fspace):
#         if args.show_early_bins:
#             ax2.step(time_bins[:-1], c[:64], where='post', linewidth=2, label='Cluster %d' % (i+1))
#             ax3.step(time_bins[:-1], c[64:], where='post', linewidth=2, label='Cluster %d' % (i+1))
#         elif args.use_restricted_bins:
#             ax2.step(time_bins[17:33], c[:16], where='post', linewidth=2, label='Cluster %d' % (i+1))
#             ax3.step(time_bins[11:16], c[16:], where='post', linewidth=2, label='Cluster %d' % (i+1))
#         else:
#             ax2.step(time_bins[5:-1], c[:64][5:], where='post', linewidth=2, label='Cluster %d' % (i+1))
#             ax3.step(time_bins[5:-1], c[64:][5:], where='post', linewidth=2, label='Cluster %d' % (i+1))
# elif args.plot_cluster_means:
#     for i, c in enumerate(clustered_data):
#         cluster_fspace = pca.inverse_transform(c)
#         cluster_mean = np.mean(cluster_fspace, axis=0)
#         cluster_std = np.std(cluster_fspace, axis=0)
#         ax2.step(time_bins[5:-1], cluster_mean[:64][5:], where='post', linewidth=2, label='Cluster %d' % (i+1))
#         ax2.errorbar(time_bins_m[5:], cluster_mean[:64][5:], yerr=cluster_std[:64][5:], fmt='None', ecolor='k')
#         ax3.step(time_bins[5:-1], cluster_mean[64:][5:], where='post', linewidth=2, label='Cluster %d' % (i+1))
#         ax3.errorbar(time_bins_m[5:], cluster_mean[64:][5:], yerr=cluster_std[64:][5:], fmt='None', ecolor='k')

# # Add graph labels
# ax2.set_title("CTN")
# ax3.set_title("CETN")
# ax2.set_xscale('log')
# ax3.set_xscale('log')
# ax2.set_xlabel('Time bin (us)')
# ax3.set_xlabel('Time bin (us)')
# ax2.set_ylabel('Counts')
# ax3.set_ylabel('Counts')
# ax2.legend(loc='upper right')
# ax3.legend(loc='upper right')

# if args.plot_sol_hist:
#     fig3, ax4 = plt.subplots(1)
#     for i, cluster in enumerate(clustered_filenames):
#         # extract the sol from all the filenames
#         sols = [int(fname.split('/')[-2][3:]) for fname in cluster]
#         ax4.hist(sols, bins=20, alpha=0.6, label='Cluster %d' % (i+1))
#     ax4.legend(loc='upper right')
#     ax4.set_xlabel('Sol')
#     ax4.set_ylabel('Frequency')
#     ax4.set_title('Frequency of Sol Membership in PC Clusters')

plt.show()
