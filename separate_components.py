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

# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--n_components', type=int, help='number of principal components to use for PCA')
parser.add_argument('--show_early_bins', action='store_true', help='show data for first 5 time bins in die-away curves')
parser.add_argument('--use_restricted_bins', action='store_true', help='only run analysis for bins 18-33 (CTN) and 12-16 (CETN)')
parser.add_argument('--show_thermal', action='store_true', help='use thermal (CTN-CETN) rather than total (CTN) neutrons in addition to CETN')
args = parser.parse_args()

data_dir = '/Users/hannahrae/data/dan_bg_sub'
time_bins = [0, 5, 10.625, 16.9375, 24, 31.9375, 40.8125, 50.75, 61.875, 74.375, 88.4375, 104.25, 122, 141.938, 164.312, 189.438, 217.688, 249.438, 285.125, 325.25, 370.375, 421.125, 478.188, 542.375, 614.562, 695.75, 787.062, 889.75, 1005.25, 1135.19, 1281.31, 1445.69, 1630.56, 1838.5, 2072.38, 2335.44, 2631.38, 2964.25, 3338.69, 3759.88, 4233.69, 4766.69, 5366.31, 6040.88, 6799.75, 7653.44, 8611.94, 9692.31, 10907.7, 12274.9, 13813.1, 15543.4, 17490.1, 19680, 22143.6, 24915.2, 28033.2, 31540.9, 35487.1, 39926.6, 44920.9, 50539.4, 56860.3, 63971.2, 100000]
X_t = []
X_e = []
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
            X_t.append(ctn_counts)
            X_e.append(cetn_counts)
            X_filenames.append(meas_dir)
        except IOError:
            pass

X_t = np.array(X_t)
X_e = np.array(X_e)
X_filenames = np.array(X_filenames)

# Fit thermal/total PCA model and project data into PC space (and back out)
pca_t = PCA(n_components=args.n_components)
pca_t.fit(X_t)

# Fit epithermal PCA model and project data into PC space (and back out)
pca_e = PCA(n_components=args.n_components)
pca_e.fit(X_e)

fig, (ax5, ax6) = plt.subplots(nrows=1, ncols=2)
ax5.step(time_bins[:-1], pca_t.components_[0], where='post', linewidth=2, label='PC 1')
ax5.step(time_bins[:-1], pca_t.components_[1], where='post', linewidth=2, label='PC 2')
ax5.step(time_bins[:-1], pca_t.components_[2], where='post', linewidth=2, label='PC 3')

ax6.step(time_bins[:-1], pca_e.components_[0], where='post', linewidth=2, label='PC 1')
ax6.step(time_bins[:-1], pca_e.components_[1], where='post', linewidth=2, label='PC 2')
ax6.step(time_bins[:-1], pca_e.components_[2], where='post', linewidth=2, label='PC 3')

ax5.legend(loc='upper right')
ax5.set_xscale('log')
if args.show_thermal:
    ax5.set_title('Thermal neutron counts')
else:
    ax5.set_title('Total neutron counts')
ax5.set_xlabel('Time (us)')
ax5.set_ylabel('Normalized Counts')
ax6.legend(loc='upper right')
ax6.set_xscale('log')
ax6.set_title('Epithermal neutron counts')
ax6.set_xlabel('Time (us)')
ax6.set_ylabel('Normalized Counts')
plt.tight_layout()
plt.show()
