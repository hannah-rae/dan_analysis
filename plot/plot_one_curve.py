import matplotlib.pyplot as plt 
import numpy as np
from sklearn.decomposition import PCA
import argparse

import datasets

# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--meas_id', help='identifier for mesaurement to plot, e.g. 428122068EAC03450100288')
parser.add_argument('--plot_mean', action='store_true', help='plot the mean epithermal and thermal die-away curve')
parser.add_argument('--plot_until', action='store_true', help='plot the mean epithermal and thermal die-away curve to date')
parser.add_argument('--plot_recon', action='store_true', help='plot the curve reconstructed from PCs of curves to date')
parser.add_argument('--use_restricted_bins', action='store_true', help='only use bins 18-34 and 13-17 for thermal and epithermal')
args = parser.parse_args()

X, Y, Y_err, names = datasets.read_dan_data()
if args.use_restricted_bins:
    X = np.take(X, range(17, 34)+range(64+12, 64+17), axis=1)
X_avg = np.mean(X, axis=0)
idx = np.where(np.array(names) == args.meas_id)
X_avg_todate = np.mean(X[:idx[0][0]], axis=0)

fig, (ax1, ax2) = plt.subplots(2)

if args.use_restricted_bins:
    n_th_bins = len(range(17, 34))
    ax1.step(datasets.time_bins_dan[17:34], X[idx][0][:n_th_bins], where='post', linewidth=2, label='%s' % args.meas_id)
    ax2.step(datasets.time_bins_dan[12:17], X[idx][0][n_th_bins:], where='post', linewidth=2, label='%s' % args.meas_id)
    if args.plot_mean:
        ax1.step(datasets.time_bins_dan[17:34], X_avg[:n_th_bins], where='post', linewidth=2, label='Mean curve')
        ax2.step(datasets.time_bins_dan[12:17], X_avg[n_th_bins:], where='post', linewidth=2, label='Mean curve')
    if args.plot_until:
        ax1.step(datasets.time_bins_dan[17:34], X_avg_todate[:n_th_bins], where='post', linewidth=2, label='Mean Curve To Date')
        ax2.step(datasets.time_bins_dan[12:17], X_avg_todate[n_th_bins:], where='post', linewidth=2, label='Mean Curve To Date')
    if args.plot_recon:
        pca = PCA(n_components=3)
        pca.fit(X[:idx[0][0]])
        x_recon = pca.inverse_transform([pca.transform([X[idx][0]])])[0][0]
        print x_recon.shape
        ax1.step(datasets.time_bins_dan[17:34], x_recon[:n_th_bins], where='post', linewidth=2, label='Reconstructed')
        ax2.step(datasets.time_bins_dan[12:17], x_recon[n_th_bins:], where='post', linewidth=2, label='Reconstructed')
else:
    ax1.step(datasets.time_bins_dan[:-1], X[idx][0][:64], where='post', linewidth=2, label='%s' % args.meas_id)
    ax2.step(datasets.time_bins_dan[:-1], X[idx][0][64:], where='post', linewidth=2, label='%s' % args.meas_id)
    if args.plot_mean:
        ax1.step(datasets.time_bins_dan[:-1], X_avg[:64], where='post', linewidth=2, label='Mean curve')
        ax2.step(datasets.time_bins_dan[:-1], X_avg[64:], where='post', linewidth=2, label='Mean curve')
    if args.plot_until:
        ax1.step(datasets.time_bins_dan[:-1], X_avg_todate[:64], where='post', linewidth=2, label='Mean Curve To Date')
        ax2.step(datasets.time_bins_dan[:-1], X_avg_todate[64:], where='post', linewidth=2, label='Mean Curve To Date')
    if args.plot_recon:
        pca = PCA(n_components=3)
        pca.fit(X[:idx[0][0]])
        x_recon = pca.inverse_transform([pca.transform([X[idx][0]])])[0][0]
        print x_recon.shape
        ax1.step(datasets.time_bins_dan[:-1], x_recon[:64], where='post', linewidth=2, label='Reconstructed')
        ax2.step(datasets.time_bins_dan[:-1], x_recon[64:], where='post', linewidth=2, label='Reconstructed')

ax1.set_xscale('log')
ax1.legend(loc='lower left')
ax1.set_xlabel('Time (us)')
ax1.set_ylabel('Normalized Counts')
ax1.set_title("Thermal Die-Away Curve")
ax2.set_xscale('log')
ax2.legend(loc='lower left')
ax2.set_xlabel('Time (us)')
ax2.set_ylabel('Normalized Counts')
ax2.set_title("Epithermal Die-Away Curve")

plt.show()