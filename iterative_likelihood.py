import argparse
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from glob import glob
import datasets

# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--n_components', type=int, default=3, help='number of principal components to use for PCA')
parser.add_argument('--use_restricted_bins', action='store_true', help='only use bins 18-34 and 13-17 for thermal and epithermal')
args = parser.parse_args()

X, Y, Y_err, names = datasets.read_dan_data()
n_meas = X.shape[0]

if args.use_restricted_bins:
    X = np.take(X, range(17, 34)+range(64+12, 64+17), axis=1)

scores = []
for i in range(n_meas):
    if i in range(5):
        scores.append(0)
        continue
    pca = PCA(n_components=args.n_components)
    # Build a model using measurements up to this point
    pca.fit(X[:i,:])
    # Get the log likelihood that the new sample belongs to the distribution
    # of data in the previous sols
    scores.append(pca.score_samples([X[i,:]])[0])

fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1)
ax1.plot(range(1,len(scores)+1), scores, picker=True)
ax1.set_xlabel("Measurement")
ax1.set_ylabel("Log Likelihood")
ax1.set_title("Log Likelihood of Measurement $m_i$ Under Model of $m_1, ..., m_{i-1}$")

ax2.plot(range(1,len(scores)+1), Y[:,0], label='Wt % H (IKI)')
ax2.plot(range(1,len(scores)+1), Y[:,1], label='Wt % Cl (IKI)')
ax2.set_title("H and Cl Abundances")
ax2.set_xlabel("Measurement")
ax2.set_ylabel("Wt %")
ax2.legend(loc='upper left')

# Allow user to click on points and print which measurement the point belongs to
def onpick(event):
    ind = event.ind
    print names[ind[0]]
    print Y[ind[0]]

fig.canvas.mpl_connect('pick_event', onpick)

plt.show()
