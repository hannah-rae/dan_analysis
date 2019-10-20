import numpy as np
import matplotlib.pyplot as plt

import datasets

plt.rc('font', family='Arial', size=10)

# Load all the DAN data
X, Y, X_error, names = datasets.read_dan_data()

# Specify which DAN measurements we want to plot
h1 = 'DNB_456787389EAC06680361170_______M1' # 2.6 H, 0.0169 BNACS
h2 = 'DNB_455739444EAC06560341120_______M1' # 3.3 H, 0.0169 BNACS

acs1 = 'DNB_456787389EAC06680361170_______M1' # 2.6 H, 0.0169 BNACS
acs2 = 'DNB_459287442EAC06960391552_______M1' # 2.6 H, 0.0143 ACS

# Get the matching data
h1_idx = np.where(names == h1)
h2_idx = np.where(names == h2)
acs1_idx = np.where(names == acs1)
acs2_idx = np.where(names == acs2)

fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(15,5))

# The last time bin is the end of the last bin
time_bins = datasets.time_bins_dan[:-1]

# Avoid error with x-axis log scale
time_bins[0] = 1e-20

# Thermal die-away curves
ax1.step(time_bins, X[h1_idx][0][:64], where='post', linewidth=2, label='WEH: 2.6 wt. %, $\Sigma_{abs}$: 0.0169 cm$^2$/g', color='k')
ax1.step(time_bins, X[h2_idx][0][:64], where='post', linewidth=2, label='WEH: 3.3 wt. %, $\Sigma_{abs}$: 0.0169 cm$^2$/g', color='blue')
ax1.step(time_bins, X[acs2_idx][0][:64], where='post', linewidth=2, label='WEH: 2.6 wt. %, $\Sigma_{abs}$: 0.0143 cm$^2$/g', color='purple')

# Epithermal die-away curves
ax2.step(time_bins, X[h1_idx][0][64:], where='post', linewidth=2, label='WEH: 2.6 wt. %, $\Sigma_{abs}$: 0.0169 cm$^2$/g', color='k')
ax2.step(time_bins, X[h2_idx][0][64:], where='post', linewidth=2, label='WEH: 3.3 wt. %, $\Sigma_{abs}$: 0.0169 cm$^2$/g', color='blue')
ax2.step(time_bins, X[acs2_idx][0][64:], where='post', linewidth=2, label='WEH: 2.6 wt. %, $\Sigma_{abs}$: 0.0143 cm$^2$/g', color='purple')

# Set figure parameters
ax1.set_xscale('log')
# ax1.set_xticks(datasets.time_bins_dan[1:])
ax1.set_xlim(1, datasets.time_bins_dan[-1])
ax1.legend(loc='best')
ax1.set_xlabel('Time ($\mu$s)')
ax1.set_ylabel('Thermal Neutrons (neutrons)')
ax1.set_title("Effect of WEH and $\Sigma_{abs}$ on Thermal Die-Away Curves")

ax2.set_xscale('log')
ax2.set_xlim(1, datasets.time_bins_dan[-1])
# ax2.set_xticks(datasets.time_bins_dan)
ax2.legend(loc='best')
ax2.set_xlabel('Time ($\mu$s)')
ax2.set_ylabel('Epithermal Neutrons (neutrons)')
ax2.set_title("Effect of WEH and $\Sigma_{abs}$ on Epithermal Die-Away Curves")

plt.savefig('/Users/hannahrae/Documents/Grad School/DAN Manuscript/figures/example_curves_horiz.pdf', transparent=True)
plt.show()
