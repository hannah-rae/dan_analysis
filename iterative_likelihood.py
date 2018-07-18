import argparse
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from glob import glob

data_dir = '/Users/hannahrae/data/dan/all'

# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--n_components', type=int, help='number of principal components to use for PCA')
args = parser.parse_args()

n_bins = 128
n_components = args.n_components

x_files = glob(data_dir + '/*.npy')
X = np.ndarray((len(x_files), n_bins))
for i, file in enumerate(x_files):
    x = np.load(file)
    X[i] = x

# This puts the measurements more or less in the order they were taken
# only problem is that sub-sol measurements can be out of order
names = [f.split('/')[-1][:-4] for f in x_files]
X = np.array([x for _, x in sorted(zip(names, X), key=lambda pair: int(pair[0].split('_')[0]))])
names = [y for y, _ in sorted(zip(names, X), key=lambda pair: int(pair[0].split('_')[0]))]

scores = []
for i in range(len(X)):
    if i in range(4):
        scores.append(0)
        continue
    pca = PCA(n_components=args.n_components)
    # Build a model using measurements up to this point
    pca.fit(X[:i])
    # Get the log likelihood that the new sample belongs to the distribution
    # of data in the previous sols
    scores.append(pca.score_samples([X[i]])[0])

fig = plt.figure()
plt.plot(range(len(scores)), scores, picker=True)
plt.xticks(range(len(names)), names, size='small')
plt.xlabel("Measurement")
plt.ylabel("Log Likelihood")
plt.title("Likelihood that sample came from the same distribution as previous samples")

# Allow user to click on points and print which measurement the point belongs to
def onpick(event):
    ind = event.ind
    print names[ind[0]]

fig.canvas.mpl_connect('pick_event', onpick)

plt.show()
