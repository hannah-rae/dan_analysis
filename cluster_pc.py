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

def normalize_png(count_vector):
    '''For both detectors, bins between 24 us and 75 us (bins 5-9 counting from 1) (bins 4-8
    counting from 0) are selected as reference bins. We calculate the total number of counts
    in these bins. Then we divide the number of counts in every bin by this total. '''
    sum4to8 = float(np.sum(count_vector[4:9]))
    corrected = [np.divide(n, sum4to8) for n in count_vector]
    return corrected

def make_rhaz_montages(fnames, n):
    print fnames.shape
    im_cmd = 'montage '
    for name in fnames:
        images = glob(os.path.join(name, '*.JPG')) + glob(os.path.join(name, '*.jpg'))
        if len(images) > 0:
            # There are probably several RHAZ images. Just take the first one.
            img = images[0]
            im_cmd = im_cmd + '%s ' % img
    im_cmd = im_cmd + '-geometry 400x400+2+2 cluster%d.jpg' % n
    call(im_cmd, shell=True)

# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--n_components', type=int, help='number of principal components to use for PCA')
parser.add_argument('--n_clusters', type=int, help='number of clusters to use for K-means clustering of PCs')
parser.add_argument('--save_rhaz_clusters', action='store_true', help='store montage of RHAZ images matching DANs for each cluster')
parser.add_argument('--plot_sol_hist', action='store_true', help='plot frequency of sol occurrence for DANs in each cluster')
parser.add_argument('--plot_cluster_centers', action='store_true', help='plot DAN at center point of each cluster (as opposed to mean measurement)')
parser.add_argument('--plot_cluster_means', action='store_true', help='plot mean measurement of each cluster (as opposed to center point)')
parser.add_argument('--show_early_bins', action='store_true', help='show data for first 5 time bins in die-away curves')
parser.add_argument('--use_restricted_bins', action='store_true', help='only run analysis for bins 18-33 (CTN) and 12-16 (CETN)')
parser.add_argument('--show_thermal', action='store_true', help='use thermal (CTN-CETN) rather than total (CTN) neutrons in addition to CETN')
parser.add_argument('--plot_components', action='store_true', help='plot the individual principal components')
args = parser.parse_args()

data_dir = '/Users/hannahrae/data/dan_bg_sub'
X = []
X_filenames = []
for sol_dir in glob(os.path.join(data_dir, '*')):
    if int(sol_dir.split('/')[-1][3:]) > 1378:
        continue
    for meas_dir in glob(os.path.join(sol_dir, '*')):
        try:
            counts = np.load(os.path.join(meas_dir, 'bg_dat.npy'))
            ctn_counts = normalize_png(counts[:][0])
            cetn_counts = normalize_png(counts[:][1])
            if args.show_thermal:
                ctn_counts = [ctot-cet for ctot, cet in zip(ctn_counts, cetn_counts)]
            if args.use_restricted_bins:
                ctn_counts = ctn_counts[17:33]
                cetn_counts = cetn_counts[11:16]
            x = ctn_counts + cetn_counts
            X.append(x)
            X_filenames.append(meas_dir)
        except IOError:
            pass

X = np.array(X)
X_filenames = np.array(X_filenames)

# Fit PCA model and project data into PC space (and back out)
pca = PCA(n_components=args.n_components)
pca.fit(X)
transformed = pca.transform(X)
back_proj = pca.inverse_transform(transformed)

# Determine the number of clusters to use
# inertia = []
# for i in range(1, 15):
#     kmeans = KMeans(n_clusters=i).fit(transformed)
#     inertia.append(kmeans.inertia_)

# plt.plot(range(1,15), inertia)
# plt.xticks(range(1,15))
# plt.xlabel("Number of clusters")
# plt.ylabel("Intertia/Distortion")
# plt.title("How many means for k-means?")
# plt.show()

# Find clusters in the data in PC space
kmeans = KMeans(n_clusters=args.n_clusters).fit(transformed)
assignments = kmeans.predict(transformed)

# Plot clusters in PC space
fig = plt.figure()
ax1 = fig.add_subplot(111, projection='3d')
ax1.set_xlabel('PC 1')
ax1.set_ylabel('PC 2')
ax1.set_zlabel('PC 3')
clustered_filenames = []
clustered_data = []
for i in range(args.n_clusters):
    cluster = transformed[np.where(assignments == i)]
    cluster_filenames = X_filenames[np.where(assignments == i)]
    # Save these for later use
    clustered_filenames.append(cluster_filenames)
    clustered_data.append(np.array(cluster))
    if args.save_rhaz_clusters:
        make_rhaz_montages(cluster_filenames, i)
    ax1.scatter(cluster[:,0], cluster[:,1], cluster[:,2], label='Cluster %d' % (i+1))
ax1.legend(loc='upper right')

# Plot die-away curves for CTN and CETN
# NOTE: we are currently plotting the "de-noised" curves, i.e. projected back from PC space
# Need to determine if this is the right thing to do, or we want to plot the original data
fig2, (ax2, ax3) = plt.subplots(nrows=1, ncols=2)
time_bins = [0, 5, 10.625, 16.9375, 24, 31.9375, 40.8125, 50.75, 61.875, 74.375, 88.4375, 104.25, 122, 141.938, 164.312, 189.438, 217.688, 249.438, 285.125, 325.25, 370.375, 421.125, 478.188, 542.375, 614.562, 695.75, 787.062, 889.75, 1005.25, 1135.19, 1281.31, 1445.69, 1630.56, 1838.5, 2072.38, 2335.44, 2631.38, 2964.25, 3338.69, 3759.88, 4233.69, 4766.69, 5366.31, 6040.88, 6799.75, 7653.44, 8611.94, 9692.31, 10907.7, 12274.9, 13813.1, 15543.4, 17490.1, 19680, 22143.6, 24915.2, 28033.2, 31540.9, 35487.1, 39926.6, 44920.9, 50539.4, 56860.3, 63971.2, 100000]
time_bins_m = [np.mean([time_bins[t], time_bins[t+1]]) for t in range(len(time_bins)-1)]
if args.plot_cluster_centers:
    centers = kmeans.cluster_centers_
    centers_fspace = pca.inverse_transform(centers)
    # time_bins[0] = 1e-20
    for i, c in enumerate(centers_fspace):
        if args.show_early_bins:
            ax2.step(time_bins[:-1], c[:64], where='post', linewidth=2, label='Cluster %d' % (i+1))
            ax3.step(time_bins[:-1], c[64:], where='post', linewidth=2, label='Cluster %d' % (i+1))
        elif args.use_restricted_bins:
            ax2.step(time_bins[17:33], c[:16], where='post', linewidth=2, label='Cluster %d' % (i+1))
            ax3.step(time_bins[11:16], c[16:], where='post', linewidth=2, label='Cluster %d' % (i+1))
        else:
            ax2.step(time_bins[5:-1], c[:64][5:], where='post', linewidth=2, label='Cluster %d' % (i+1))
            ax3.step(time_bins[5:-1], c[64:][5:], where='post', linewidth=2, label='Cluster %d' % (i+1))
elif args.plot_cluster_means:
    for i, c in enumerate(clustered_data):
        cluster_fspace = pca.inverse_transform(c)
        cluster_mean = np.mean(cluster_fspace, axis=0)
        cluster_std = np.std(cluster_fspace, axis=0)
        if args.show_early_bins:
            ax2.step(time_bins[:-1], cluster_mean[:64], where='post', linewidth=2, label='Cluster %d' % (i+1))
            ax2.errorbar(time_bins_m, cluster_mean[:64], yerr=cluster_std[:64], fmt='None', ecolor='k')
            ax3.step(time_bins[:-1], cluster_mean[64:], where='post', linewidth=2, label='Cluster %d' % (i+1))
            ax3.errorbar(time_bins_m, cluster_mean[64:], yerr=cluster_std[64:], fmt='None', ecolor='k')
        elif args.use_restricted_bins:
            ax2.step(time_bins[17:33], cluster_mean[:16], where='post', linewidth=2, label='Cluster %d' % (i+1))
            ax2.errorbar(time_bins_m[17:33], cluster_mean[:16], yerr=cluster_std[:16], fmt='None', ecolor='k')
            ax3.step(time_bins[11:16], cluster_mean[16:], where='post', linewidth=2, label='Cluster %d' % (i+1))
            ax3.errorbar(time_bins_m[11:16], cluster_mean[16:], yerr=cluster_std[16:], fmt='None', ecolor='k')
        else:
            ax2.step(time_bins[5:-1], cluster_mean[:64][5:], where='post', linewidth=2, label='Cluster %d' % (i+1))
            ax2.errorbar(time_bins_m[5:], cluster_mean[:64][5:], yerr=cluster_std[:64][5:], fmt='None', ecolor='k')
            ax3.step(time_bins[5:-1], cluster_mean[64:][5:], where='post', linewidth=2, label='Cluster %d' % (i+1))
            ax3.errorbar(time_bins_m[5:], cluster_mean[64:][5:], yerr=cluster_std[64:][5:], fmt='None', ecolor='k')

if args.use_restricted_bins != True:
    # Draw boundaries for bins normally used to study curves
    ax2.axvline(x=time_bins[18-1], color='k', linestyle=':')
    ax2.axvline(x=time_bins[33-1], color='k', linestyle=':')
    ax3.axvline(x=time_bins[12-1], color='k', linestyle=':')
    ax3.axvline(x=time_bins[16-1], color='k', linestyle=':')

# Add graph labels
if args.show_thermal:
    ax2.set_title("Thermal Neutron Die-Away")
else:
    ax2.set_title("Total (Thermal & Epithermal) Neutron Die-Away")
ax3.set_title("Epithermal Neutron Die-Away")
ax2.set_xscale('log')
ax3.set_xscale('log')
ax2.set_xlabel('Time bin (us)')
ax3.set_xlabel('Time bin (us)')
ax2.set_ylabel('Counts')
ax3.set_ylabel('Counts')
ax2.legend(loc='upper right')
ax3.legend(loc='upper right')

if args.plot_sol_hist:
    fig3, ax4 = plt.subplots(1)
    for i, cluster in enumerate(clustered_filenames):
        # extract the sol from all the filenames
        sols = [int(fname.split('/')[-2][3:]) for fname in cluster]
        ax4.hist(sols, bins=20, alpha=0.6, label='Cluster %d' % (i+1))
    ax4.legend(loc='upper right')
    ax4.set_xlabel('Sol')
    ax4.set_ylabel('Frequency')
    ax4.set_title('Frequency of Sol Membership in PC Clusters')

if args.plot_components:
    fig, (ax5, ax6) = plt.subplots(nrows=1, ncols=2)
    ax5.step(time_bins[:-1], pca.components_[0][:64], where='post', linewidth=2, label='PC 1')
    ax5.step(time_bins[:-1], pca.components_[1][:64], where='post', linewidth=2, label='PC 2')
    ax5.step(time_bins[:-1], pca.components_[2][:64], where='post', linewidth=2, label='PC 3')
    ax5.legend(loc='upper right')
    ax5.set_xscale('log')
    ax5.set_title('CTN')
    ax5.set_xlabel('Time (us)')
    ax5.set_ylabel('Normalized Counts')
    ax6.step(time_bins[:-1], pca.components_[0][64:], where='post', linewidth=2, label='PC 1')
    ax6.step(time_bins[:-1], pca.components_[1][64:], where='post', linewidth=2, label='PC 2')
    ax6.step(time_bins[:-1], pca.components_[2][64:], where='post', linewidth=2, label='PC 3')
    ax6.legend(loc='upper right')
    ax6.set_xscale('log')
    ax6.set_title('CETN')
    ax6.set_xlabel('Time (us)')
    ax6.set_ylabel('Normalized Counts')

plt.show()
