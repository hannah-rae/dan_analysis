import matplotlib as mpl
mpl.use('TkAgg')

import numpy as np
import argparse
import os.path
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from glob import glob
from subprocess import call
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from matplotlib.colors import LinearSegmentedColormap
import datasets

import seaborn as sns
sns.set()
sns.set_style("white")

plt.rc('font', family='Arial', size=10)

# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--n_components', type=int, default=3, help='number of principal components to use for PCA')
parser.add_argument('--dataset', help='which dataset to use: polar, equatorial, or dan')
parser.add_argument('--contour_plot', action='store_true', help='plot contours of PC1 and PC2 vs. H and BNACS')
parser.add_argument('--thermal_only', action='store_true', help='discard epithermal feature vector')
parser.add_argument('--normalize', action='store_true', help='normalize the data before PCA')
# parser.add_argument('--use_restricted_bins', action='store_true', help='only run analysis for bins 18-33 (CTN) and 12-16 (CETN)')
args = parser.parse_args()

# X, Y, err, names = datasets.read_dan_data()
# n_bins = 64
# if args.use_restricted_bins:
#     X = np.take(X, range(17, 34)+range(n_bins+12, n_bins+17), axis=1)

if args.dataset == 'equatorial':
    X, Y = datasets.read_acs_grid_data(shuffle=False)
    if args.thermal_only:
        X = np.take(X, range(64), axis=1)
elif args.dataset == 'polar':
    X, Y = datasets.read_polar_data(shuffle=False)
    if args.thermal_only:
        X = np.take(X, range(161), axis=1)

if args.normalize:
    X = datasets.normalize_counts(X)

# Fit PCA model and project data into PC space
pca = PCA(n_components=args.n_components)
pca.fit(X)
transformed = pca.transform(X)

# Plot clusters in PC space
fig = plt.figure()
ax1 = fig.add_subplot(111, projection='3d')
ax1.set_xlabel('1')
ax1.set_ylabel('2')
ax1.set_zlabel('3')

fig2 = plt.figure()
ax2 = fig2.add_subplot(111, projection='3d')
ax2.set_xlabel('1')
ax2.set_ylabel('2')
ax2.set_zlabel('3')

data = np.asarray([Y[:,0], Y[:,1], transformed[:,0], transformed[:,1], transformed[:,2]]).transpose()
np.savetxt('for_contours.csv', data, delimiter=',')

# Rescale the colors to 0, 1
hcol = Y[:,0]
bcol = Y[:,1]
print np.max(bcol)
print pca.explained_variance_ratio_
print sum(pca.explained_variance_ratio_)
# Define a color map
colors = [(h, 0, b) for h, b in zip(hcol, bcol)]
p = ax1.scatter(transformed[:,0], transformed[:,1], transformed[:,2], c=hcol, cmap='BrBG')
fig.colorbar(p)
p2 = ax2.scatter(transformed[:,0], transformed[:,1], transformed[:,2], c=bcol, cmap='copper_r')
fig2.colorbar(p2)

if args.contour_plot:
    fig, (ax3, ax4) = plt.subplots(nrows=1, ncols=2)
    if args.dataset == 'equatorial':
        h_max = 6.1
        bnacs_max = 1.524
    elif args.dataset == 'polar':
        h_max = 25.1
        bnacs_max = 3.754
    cs = ax3.tricontour(transformed[:,0], transformed[:,1], Y[:,0], levels=np.arange(0, h_max, 0.5), linewidths=0.5, colors='k')
    ax3.clabel(cs, fontsize=8, inline=1, fmt='%1.1f')
    cntr2 = ax3.tricontourf(transformed[:,0], transformed[:,1], Y[:,0], levels=np.arange(0, h_max, 0.1), cmap="BrBG", alpha=0.7)

    cs2 = ax4.tricontour(transformed[:,0], transformed[:,1], Y[:,1], levels=np.arange(0, bnacs_max, 0.15), linewidths=0.5, colors='k')
    ax4.clabel(cs2, fontsize=8, inline=1, fmt='%1.2f')
    cntr3 = ax4.tricontourf(transformed[:,0], transformed[:,1], Y[:,1], levels=np.arange(0, bnacs_max, 0.15), cmap="copper_r", alpha=0.7)
    
    fig.colorbar(cntr2, ax=ax3)
    fig.colorbar(cntr3, ax=ax4)
    
    ax3.set_xlabel('PC 1')
    ax3.set_ylabel('PC 2')

    ax4.set_xlabel('PC 1')
    ax4.set_ylabel('PC 2')

    ax3.set_title('PC1 and PC2 vs. H')
    ax4.set_title('PC1 and PC2 vs. $\Sigma_{abs}$')

    #plt.set_title('tricontour (%d points)' % npts)

    # X_grid, Y_grid = np.meshgrid(transformed[:,0], transformed[:,1])
    # Z_grid = symmetricize(Y[:,0])
    # CS = plt.contour(X_grid, Y_grid, Z_grid,
    #                  colors='k',  # negative contours will be dashed by default
    #                  )
    # plt.clabel(CS, fontsize=9, inline=1)

plt.show()