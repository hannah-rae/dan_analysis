import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from decimal import getcontext
import warnings
import copy

plt.rc('font', family='Arial', size=10)

getcontext().prec = 2

drill_samples = {
                 'HSM': [0.0055, 0.0007],
                 'LU': [0.0058, 0.0001],
                 'BK': [0.006, 0],
                 'GB': [0.0098, 0.0002],
                 'OK': [0.0112, 0.0001],
                 'OU': [0.0091, 0],
                 'TP': [0.0089, 8e-5],
                 'SB': [0.0130, 0.0002],
                 'BS': [0.0121, 0.0001],
                 'OB': [0.0103, 0.0001],
                 'GH': [0.0081, 8e-5],
                 'MA': [0.0103, 0.0001],
                 'WJ': [0.0137, 0.0042],
                 'CB': [0.0154, 0.0002],
                 'PDK': [0.0228, 0.0002]
                 }
import operator
print sorted(drill_samples.items(), key=operator.itemgetter(1))

models = [0.0042,
          0.0047,
          0.0052,
          0.0057,
          0.0061,
          0.0066,
          0.0071,
          0.0075,
          0.0084,
          0.0092,
          0.0101,
          0.0109,
          0.0118,
          0.0126,
          0.0135,
          0.0143,
          0.0152,
          0.016,
          0.0169,
          0.0177,
          0.0186,
          0.0194,
          0.0203,
          0.0211,
          0.022,
          0.0228,
          0.0237]

# print(models)
Cl = np.full(7, 0.1)
for i in range(len(models)-len(Cl)):
    Cl = np.append(Cl, (0.15*i+0.1))
Fe = np.zeros(7)
for i in range(7):
    Fe[i] = 1.96*i+0.48
Fe = np.append(Fe, [14.22]*(len(models)-len(Fe)))

fig, ax = plt.subplots()
fig.set_figheight(7)
fig.set_figwidth(6)
plt.title('Bulk Neutron Absorption Cross Section ($\Sigma_{abs}$)\n in Observed vs. Modeled Geochemistries', fontsize=12, fontweight='bold')
plt.ylabel('$\Sigma_{abs}$ (barns)', fontsize=12, fontweight='bold')

xposition = [0.6, 2]
for sample in drill_samples.keys():
    plt.scatter(xposition[0], drill_samples[sample][0], c='k')
    #plt.scatter(xposition[i], yposition, c=color[i], size=8)
    #plt.errorbar(xposition[0], drill_BNACS[i], yerr=drill_err[i], fmt='None', ecolor='k')
    #plt.errorbar(xposition[i], drill_BNACS[i], yerr=drill_err[i], fmt='None', ecolor=color[i], elinewidth=6)
# ax.annotate('Lubango', (xposition[0]+0.05, drill_BNACS[1]-0.01))
# ax.annotate('Passadumkaeg', (xposition[0]+0.05, drill_BNACS[-1]-0.01))

for i in range(len(models)):
    plt.scatter(xposition[1]-.1, models[i], c='darkgray')
    if(Fe[i] < 10):
        plt.text(xposition[1]+0.05, models[i]-0.0001, '(%.2f,   %.2f)' % (Cl[i],Fe[i]), fontsize=10)
    else:
        plt.text(xposition[1]+0.05, models[i]-0.0001, '(%.2f, %.2f)' % (Cl[i],Fe[i]), fontsize=10)

xlabel = ['Observations', 'Models (Cl, Fe [wt %])']
plt.xticks([0.95, 2.15], xlabel)

ax.yaxis.set_minor_locator(ticker.LinearLocator(numticks=29))
xticklabels = plt.getp(plt.gca(),'xticklabels')
yticklabels = plt.getp(plt.gca(),'yticklabels')
plt.setp(xticklabels, fontsize=12, weight='bold')
plt.setp(yticklabels, fontsize=12, weight='bold')
plt.tick_params(axis='x', which='both', bottom=False, top=False)
plt.tick_params(axis='y', which='minor', left=False, right=False)
plt.xlim(0.3, 2.7)
# plt.ylim(0.2, 1.6)
with warnings.catch_warnings():
   warnings.simplefilter("ignore")
plt.tight_layout()
plt.savefig("/Users/hannahrae/Documents/Grad School/DAN Manuscript/figures/sean.pdf", transparent=True)
# plt.close()
plt.show()