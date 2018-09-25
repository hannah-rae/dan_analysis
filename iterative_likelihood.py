import argparse
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from glob import glob
import datasets

# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--n_components', type=int, default=3, help='number of principal components to use for PCA')
args = parser.parse_args()

X, Y, Y_err, names = datasets.read_dan_data()
n_meas = X.shape[0]

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

print scores

fig = plt.figure()
plt.plot(range(1,len(scores)+1), scores, picker=True)
plt.xlabel("Measurement")
plt.ylabel("Log Likelihood")
plt.title("Log Likelihood of Measurement $m_i$ Under Model of $m_1, ..., m_{i-1}$")

# Allow user to click on points and print which measurement the point belongs to
def onpick(event):
    ind = event.ind
    print names[ind[0]]

fig.canvas.mpl_connect('pick_event', onpick)

plt.show()
