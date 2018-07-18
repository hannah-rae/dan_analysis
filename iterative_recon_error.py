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

def mse(x, y):
    return ((x - y) ** 2).mean(axis=None)

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

errors = []
for i in range(len(X)):
    pca = PCA(n_components=args.n_components)
    # Add this measurement to the model
    pca.fit(X[:i+1])
    # Use this model to transform/reconstruct all measurements
    X_pc = pca.transform(X)
    X_recon = pca.inverse_transform(X_pc)
    avg_error = np.mean([mse(x,x_r) for x, x_r in zip(X, X_recon)])
    errors.append(avg_error)

fig = plt.figure()
plt.plot(range(len(errors)), errors, picker=True)
plt.xticks(range(len(names)), names, size='small')
plt.xlabel("Measurement")
plt.ylabel("Average Reconstruction Error of Samples over Entire Traverse")
plt.title("Average Reconstruction Error of Samples over Entire Traverse using Increasing Number of Measurements")

# Allow user to click on points and print which measurement the point belongs to
def onpick(event):
    ind = event.ind
    print names[ind[0]]

fig.canvas.mpl_connect('pick_event', onpick)

plt.show()
