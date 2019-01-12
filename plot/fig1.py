import numpy as np
import matplotlib.pyplot as plt

import datasets

# plt.rc('font', family='serif')
# plt.rc('xtick', labelsize='x-small')
# plt.rc('ytick', labelsize='x-small')
plt.rc('font', family='Arial', size=10)

X, Y, X_error, names = datasets.read_dan_data()

h1 = 'DNB_456787389EAC06680361170_______M1' # 2.6 H, 1.123 ACS
h2 = 'DNB_455739444EAC06560341120_______M1' # 3.3 H, 1.123 ACS

acs1 = 'DNB_456787389EAC06680361170_______M1' # 2.6 H, 1.123 ACS
acs2 = 'DNB_459287442EAC06960391552_______M1' # 2.6 H, 0.93 ACS

# Get the matching data
h1_idx = np.where(names == h1)
h2_idx = np.where(names == h2)
acs1_idx = np.where(names == acs1)
acs2_idx = np.where(names == acs2)

fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)
time_bins_m = [np.mean([datasets.time_bins_dan[t], datasets.time_bins_dan[t+1]]) for t in range(len(datasets.time_bins_dan)-1)]
# Thermal die-away curves
ax1.step(datasets.time_bins_dan[:-1], X[h1_idx][0][:64], where='post', linewidth=2, label='H: 2.6 wt %, $\Sigma_{abs}$: 1.123 b', color='k')
ax1.step(datasets.time_bins_dan[:-1], X[h2_idx][0][:64], where='post', linewidth=2, label='H: 3.3 wt %, $\Sigma_{abs}$: 1.123 b', color='blue')
ax1.step(datasets.time_bins_dan[:-1], X[acs2_idx][0][:64], where='post', linewidth=2, label='H: 2.6 wt %, $\Sigma_{abs}$: 0.93 b', color='purple')

# Epithermal die-away curves
ax2.step(datasets.time_bins_dan[:-1], X[h1_idx][0][64:], where='post', linewidth=2, label='H: 2.6 wt %, $\Sigma_{abs}$: 1.123 b', color='k')
ax2.step(datasets.time_bins_dan[:-1], X[h2_idx][0][64:], where='post', linewidth=2, label='H: 3.3 wt %, $\Sigma_{abs}$: 1.123 b', color='blue')
ax2.step(datasets.time_bins_dan[:-1], X[acs2_idx][0][64:], where='post', linewidth=2, label='H: 2.6 wt %, $\Sigma_{abs}$: 0.93 b', color='purple')


# ax2.step(datasets.time_bins_dan[:-1], X[idx][0][64:], where='post', linewidth=2, label='%s' % args.meas_id)
# if args.plot_mean:
#     ax1.step(datasets.time_bins_dan[:-1], X_avg[:64], where='post', linewidth=2, label='Mean curve')
#     ax2.step(datasets.time_bins_dan[:-1], X_avg[64:], where='post', linewidth=2, label='Mean curve')
# if args.plot_until:
#     ax1.step(datasets.time_bins_dan[:-1], X_avg_todate[:64], where='post', linewidth=2, label='Mean Curve To Date')
#     ax2.step(datasets.time_bins_dan[:-1], X_avg_todate[64:], where='post', linewidth=2, label='Mean Curve To Date')
# if args.plot_recon:
#     pca = PCA(n_components=3)
#     pca.fit(X[:idx[0][0]])
#     x_recon = pca.inverse_transform([pca.transform([X[idx][0]])])[0][0]
#     print x_recon.shape
#     ax1.step(datasets.time_bins_dan[:-1], x_recon[:64], where='post', linewidth=2, label='Reconstructed')
#     ax2.step(datasets.time_bins_dan[:-1], x_recon[64:], where='post', linewidth=2, label='Reconstructed')


# # Set figure parameters
ax1.set_xscale('log')
ax1.legend(loc='best')
ax1.set_xlabel('Time (us)')
ax1.set_ylabel('Thermal Neutron Counts')
ax1.set_title("Effect of H and $\Sigma_{abs}$ on Thermal Die-Away Curves")
ax2.set_xscale('log')
ax2.legend(loc='best')
ax2.set_xlabel('Time (us)')
ax2.set_ylabel('Epithermal Neutron Counts')
ax2.set_title("Effect of H and $\Sigma_{abs}$ on Epithermal Die-Away Curves")

plt.show()