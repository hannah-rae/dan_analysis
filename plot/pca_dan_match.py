import matplotlib
matplotlib.use('TKAgg')

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from glob import glob
import argparse
import datasets

# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--n_components', type=int, default=3, help='number of principal components to use for PCA')
parser.add_argument('--normalize', action='store_true', help='normalize the data before PCA')
args = parser.parse_args()

X_sebina, Y_sebina = datasets.read_acs_grid_data()
print Y_sebina.shape
X_dan, Y_dan, err_dan, names_dan = datasets.read_dan_data()
print Y_dan.shape

time_bins = datasets.time_bins_dan
n_bins = 64

if args.normalize:
    X_sebina = datasets.normalize_counts(X_sebina)
    X_dan = datasets.normalize_counts(X_dan)

pca = PCA(n_components=args.n_components)
X_t = pca.fit_transform(X_sebina)

# Plot the Sebina grid points in PC space
fig = plt.figure()
ax1 = fig.add_subplot(1, 1, 1, projection='3d')
ax1.set_xlabel('PC 1')
ax1.set_ylabel('PC 2')
ax1.set_zlabel('PC 3')
for x_t, (h, acs) in zip(X_t, Y_sebina):
    
    exists = False
    for [h_dan, acs_dan] in Y_dan:
        if np.array_equal(np.array([h, acs]), np.array([h_dan, acs_dan])):
            ax1.scatter(x_t[0], x_t[1], x_t[2], color='red')
            exists = True
            break
    
    if not exists:
        ax1.scatter(x_t[0], x_t[1], x_t[2], color='k')

plt.show()