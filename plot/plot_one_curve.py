import matplotlib.pyplot as plt 
import numpy as np
import argparse

import datasets

# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--meas_id', help='identifier for mesaurement to plot, e.g. 428122068EAC03450100288')
parser.add_argument('--plot_mean', action='store_true', help='plot the mean epithermal and thermal die-away curve')
parser.add_argument('--plot_until', action='store_true', help='plot the mean epithermal and thermal die-away curve to date')
args = parser.parse_args()

X, Y, Y_err, names = datasets.read_dan_data()
X_avg = np.mean(X, axis=0)
idx = np.where(np.array(names) == args.meas_id)
X_avg_todate = np.mean(X[:idx[0][0]], axis=0)

fig, (ax1, ax2) = plt.subplots(2)

ax1.step(datasets.time_bins_dan[:-1], X[idx][0][:64], where='post', linewidth=2, label='%s' % args.meas_id)
ax2.step(datasets.time_bins_dan[:-1], X[idx][0][64:], where='post', linewidth=2, label='%s' % args.meas_id)
if args.plot_mean:
    ax1.step(datasets.time_bins_dan[:-1], X_avg[:64], where='post', linewidth=2, label='Mean curve')
    ax2.step(datasets.time_bins_dan[:-1], X_avg[64:], where='post', linewidth=2, label='Mean curve')
if args.plot_until:
    ax1.step(datasets.time_bins_dan[:-1], X_avg_todate[:64], where='post', linewidth=2, label='Mean Curve To Date')
    ax2.step(datasets.time_bins_dan[:-1], X_avg_todate[64:], where='post', linewidth=2, label='Mean Curve To Date')

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