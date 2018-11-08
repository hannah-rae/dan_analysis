import matplotlib.pyplot as plt 
import numpy as np
from sklearn.decomposition import PCA
import argparse

import datasets

# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--n_components', type=int, default=3, help='number of principal components to use for PCA')
parser.add_argument('--use_restricted_bins', action='store_true', help='only use bins 18-34 and 13-17 for thermal and epithermal')
parser.add_argument('--score', help='likelihood or mse')
args = parser.parse_args()

X, Y, Y_err, names = datasets.read_dan_data()
n_meas = X.shape[0]

pca = PCA(n_components=args.n_components)
pca.fit(X)

scores = []
for i in range(n_meas):
    if args.score == 'likelihood':
        score = pca.score_samples([X[i,:]])[0]
    elif args.score == 'mse':
        x_recon = pca.inverse_transform([pca.transform([X[i]])])[0][0]
        score = np.mean(np.square(np.subtract(X[i,:], x_recon)))
    scores.append(score)

# Sort by scores
scores, names = zip(*sorted(zip(scores, names)))
scores = np.array(scores)
print scores[:30]
print names[:30]
# anomalous_scores = scores[np.where(scores < 350)]
# anomalous_names = names[np.where(scores < 350)]

plt.hist(scores, bins=60)
plt.xlabel('Log likelihood using PCA for all measurements through 2171')
plt.ylabel('Frequency')
plt.show()