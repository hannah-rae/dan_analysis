import numpy as np 
import matplotlib.pyplot as plt
import argparse 
from sklearn.neighbors import KNeighborsRegressor
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score, accuracy_score

import datasets

# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--n_neighbors', type=int, default=2, help='number of neighbors to evaluate')
parser.add_argument('--n_components', type=int, default=3, help='number of principal components to use for PCA')
parser.add_argument('--use_restricted_bins', action='store_true', help='only use bins 18-34 and 13-17 for thermal and epithermal')
args = parser.parse_args()

# Load the data sets
X, y = datasets.read_acs_grid_data()
dan_X, dan_y, _, names = datasets.read_dan_data()

# Normalize counts to approximately same range
X = datasets.normalize_counts(X)
dan_X = datasets.normalize_counts(dan_X)

if args.use_restricted_bins:
    n_bins = 64
    X = np.take(X, range(17, 34)+range(n_bins+12, n_bins+17), axis=1)
    dan_X = np.take(dan_X, range(17, 34)+range(n_bins+12, n_bins+17), axis=1)

# Project the data into principal subspace of model data
pca = PCA(n_components=args.n_components)
pca.fit(X)
X = pca.transform(X)
dan_X = pca.transform(dan_X)

# Load training data for KNN
neigh = KNeighborsRegressor(n_neighbors=args.n_neighbors, weights='distance', metric='chebyshev')
neigh.fit(X, y) 
# Predict nearest neighbor for DAN points
dan_pred = neigh.predict(dan_X)
# Assess accuracy of predictions
print 'r2 H %f' % r2_score(dan_y[:,0], dan_pred[:,0])
print 'r2 BNACS %f' % r2_score(dan_y[:,1], dan_pred[:,1])

fig, (ax1, ax2) = plt.subplots(2)
ax1.scatter(dan_y[:,0], dan_pred[:,0], label='H')
ax2.scatter(dan_y[:,1], dan_pred[:,1], label='BNACS')
plt.legend(loc='best')

# Find the optimal K?
r2_h = []
r2_bnacs = []
for i in range(1, 500):
    neigh = KNeighborsRegressor(n_neighbors=i, weights='distance', metric='chebyshev')
    neigh.fit(X, y) 
    # Predict nearest neighbor for DAN points
    dan_pred = neigh.predict(dan_X)
    # Assess accuracy of predictions
    r2_h.append(r2_score(dan_y[:,0], dan_pred[:,0]))
    r2_bnacs.append(r2_score(dan_y[:,1], dan_pred[:,1]))

fig2, ax = plt.subplots(1)
ax.plot(range(1, 500), r2_h, label='H')
ax.plot(range(1, 500), r2_bnacs, label='BNACS')
plt.legend(loc='best')

plt.show()
