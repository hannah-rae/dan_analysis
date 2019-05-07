import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from decimal import getcontext
import warnings
import copy

plt.rc('font', family='Arial', size=10)

getcontext().prec = 2
drill_BNACS = [0.344, 0.368, 0.384, 0.731, 0.830, 0.680, 0.656, 0.902, 0.878, 0.757, 0.555, 0.775, 1.030, 1.100, 1.477]
drill_err = [0, 0.008, 0.004, 0.015, 0.009, 0.006, 0.007, 0.011, 0.008, 0.009, 0.005, 0.009, 0.010, 0.012, 0.011]
drill_labels = ['ChemCam', 'LU', 'BK', 'GB', 'OK', 'OU', 'TP', 'SB', 'BS', 'OB', 'GH', 'MA', 'WJ', 'CB', 'Passadumkaeg']
print sorted(zip(drill_BNACS, drill_labels))
xlabel = ['Observations', 'Models (Cl, Fe [wt %])']
xposition = [0.6,2]
models = np.arange(0.26, 0.59, 0.0465)
models = np.append(models, np.arange(0.634, 1.51, 0.0485))
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
for i in range(len(drill_BNACS)):
    plt.scatter(xposition[0], drill_BNACS[i], c='k')
    #plt.scatter(xposition[i], yposition, c=color[i], size=8)
    #plt.errorbar(xposition[0], drill_BNACS[i], yerr=drill_err[i], fmt='None', ecolor='k')
    #plt.errorbar(xposition[i], drill_BNACS[i], yerr=drill_err[i], fmt='None', ecolor=color[i], elinewidth=6)
# ax.annotate('Lubango', (xposition[0]+0.05, drill_BNACS[1]-0.01))
# ax.annotate('Passadumkaeg', (xposition[0]+0.05, drill_BNACS[-1]-0.01))
for i in range(len(models)):
    plt.scatter(xposition[1]-.1, models[i], c='darkgray')
    if(Fe[i] < 10):
        plt.text(xposition[1]+0.05, models[i]-0.01, '(%.2f,   %.2f)'%(Cl[i],Fe[i]), fontsize=10)
    else:
        plt.text(xposition[1]+0.05, models[i]-0.01, '(%.2f, %.2f)'%(Cl[i],Fe[i]), fontsize=10)
plt.xticks([0.95,2.15], xlabel)
ax.yaxis.set_minor_locator(ticker.LinearLocator(numticks=29))
xticklabels = plt.getp(plt.gca(),'xticklabels')
yticklabels = plt.getp(plt.gca(),'yticklabels')
plt.setp(xticklabels, fontsize=12, weight='bold')
plt.setp(yticklabels, fontsize=12, weight='bold')
plt.tick_params(axis='x', which='both', bottom=False, top=False)
plt.tick_params(axis='y', which='minor', left=False, right=False)
plt.xlim(0.3, 2.7)
plt.ylim(0.2, 1.6)
with warnings.catch_warnings():
   warnings.simplefilter("ignore")
plt.tight_layout()
plt.savefig("/Users/hannahrae/Desktop/sean.pdf", transparent=True)
# plt.close()
plt.show()