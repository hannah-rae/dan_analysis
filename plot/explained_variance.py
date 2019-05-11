# import matplotlib
# # matplotlib.use('TkAgg')

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

import datasets

plt.rc('font', family='Arial', size=10)

X_d, _, _, _ = datasets.read_dan_data()
#X_d = datasets.normalize_counts(X_d)
X_d = np.take(X_d, range(17, 34)+range(64+12,64+17), axis=1)

X_m, _ = datasets.read_highweh_grid()
#X_m = datasets.normalize_counts(X_m)


variances_d = []
variances_m = []

for k in range(1, 11):
    pca_d = PCA(n_components=k)
    pca_m = PCA(n_components=k)
    pca_d.fit(X_d)
    pca_m.fit(X_m)

    print k
    print 'DAN'
    print pca_d.explained_variance_ratio_
    print sum(pca_d.explained_variance_ratio_)
    print 'Modeled'
    print pca_m.explained_variance_ratio_
    print sum(pca_m.explained_variance_ratio_)

    variances_d.append(sum(pca_d.explained_variance_ratio_)*100)
    variances_m.append(sum(pca_m.explained_variance_ratio_)*100)

plt.axvline(x=2, color='gray', alpha=0.5)
plt.plot(range(1,11), variances_m, color='k', linestyle='--', label='Modeled Dataset')
plt.plot(range(1,11), variances_d, color='k', label='DAN Dataset')

plt.legend(loc='lower right')
plt.xlabel("Number of Components")
plt.xlim(1,10)
plt.ylabel("Percentage of Total Variance Explained")
plt.title("Percentage of Total Variance Explained by $k$ Principal Components")
plt.xticks(range(1,11))
plt.savefig('/Users/hannahrae/Documents/Grad School/DAN Manuscript/figures/explained_variance.pdf')
plt.show()