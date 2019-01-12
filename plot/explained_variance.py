# import matplotlib
# # matplotlib.use('TkAgg')

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

import datasets

# plt.rc('font', family='serif')
# plt.rc('xtick', labelsize='x-small')
# plt.rc('ytick', labelsize='x-small')
plt.rc('font', family='Arial', size=10)


X_d, _, _, _ = datasets.read_dan_data()
X_d = np.take(X_d, range(17, 34)+range(64+12, 64+17), axis=1)
X_s, _ = datasets.read_acs_grid_data()
X_p, _ = datasets.read_polar_data()

variances_d = []
variances_s = []
variances_p = []
for k in range(1, 21):
    pca = PCA(n_components=k)
    pca2 = PCA(n_components=k)
    pca3 = PCA(n_components=k)
    pca.fit(X_d)
    pca2.fit(X_s)
    pca3.fit(X_p)
    print k
    print sum(pca.explained_variance_ratio_)
    print sum(pca2.explained_variance_ratio_)
    print sum(pca3.explained_variance_ratio_)

    variances_d.append(sum(pca.explained_variance_ratio_))
    variances_s.append(sum(pca2.explained_variance_ratio_))
    variances_p.append(sum(pca3.explained_variance_ratio_))

plt.axvline(x=4, color='gray', alpha=0.5)
plt.plot(range(1,21), variances_d, color='k', lw=2, label='DAN Dataset')
plt.plot(range(1,21), variances_s, color='k', linestyle='--', label='Equatorial Dataset')
plt.plot(range(1,21), variances_p, color='k', linestyle=':', label='Polar Dataset')
plt.legend(loc='lower right')
plt.xlabel("Number of Components")
plt.xlim(1)
plt.ylabel("% of Variance Explained")
plt.title("Variance Explained by Principal Components")
plt.xticks(range(1,21))
plt.savefig('/Users/hannahrae/Desktop/explained_variance.pdf')
plt.show()