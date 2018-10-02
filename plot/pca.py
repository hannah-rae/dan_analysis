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
parser.add_argument('--dataset', help='which dataset to use: rover, full, or dan')
parser.add_argument('--normalize', action='store_true', help='normalize the data before PCA')
args = parser.parse_args()

if args.dataset == 'rover':
    X, Y = datasets.read_grid_data()
elif args.dataset == 'full':
    X, Y = datasets.read_sim_data()
elif args.dataset == 'dan':
    X, Y, err, names = datasets.read_dan_data()
elif args.dataset == 'both_models':
    X_rover, Y_rover = datasets.read_grid_data(limit_2000us=True)
    X_full, Y_full = datasets.read_sim_data(use_dan_bins=True)
    X = np.concatenate([X_full, X_rover])
    Y = np.concatenate([Y_rover, Y_full])

if args.normalize:
    X = datasets.normalize_counts(X)

pca = PCA(n_components=args.n_components)
X_t = pca.fit_transform(X)

# Plot clusters in PC space
if args.n_components >= 3:
    fig = plt.figure()
    ax1 = fig.add_subplot(111, projection='3d')
    ax1.set_xlabel('PC 1')
    ax1.set_ylabel('PC 2')
    ax1.set_zlabel('PC 3')
    ax1.scatter(X_t[:,0], X_t[:,1], X_t[:,2], picker=True)
elif args.n_components == 2:
    fig, ax1 = plt.subplots(1)
    ax1.set_xlabel('PC 1')
    ax1.set_ylabel('PC 2')
    ax1.scatter(X_t[:,0], X_t[:,1], picker=True)

# Allow user to click on points and print which measurement the point belongs to
def onpick(event):
    ind = event.ind
    # print Y_test[ind[0]]
    print Y[ind]

fig.canvas.mpl_connect('pick_event', onpick)

plt.show()