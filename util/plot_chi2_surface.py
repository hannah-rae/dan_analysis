import csv
import matplotlib
matplotlib.use('TkAgg')
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.mlab import griddata

import numpy as np
import os.path

from glob import glob

t = 30.578 # critical value threshold for p=0.01

# Go through each sol directory
for soldir in sorted(glob('/Users/hannahrae/data/dan/dan_asufits_hiweh/*')):
    sol = int(soldir.split('/')[-1][3:])
    # Go through each measurement directory
    for mdir in glob(os.path.join(soldir, '*')):
        # Load all the chi-squared values
        chi2_all = []
        weh_all = []
        bnacs_all = []
        with open(os.path.join(mdir, 'gridInfo.csv'), 'rb') as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                # skip the headings row
                if 'Simulation' in row:
                    continue
                simname, chi2, pval, dof = row
                weh_all.append(float(simname.split('_')[0].split('H')[0]))
                bnacs_all.append(float(simname.split('_')[1].split('BNACS')[0]))
                chi2_all.append(float(chi2))
        fig = plt.figure()
        # Prepare the data for the surface plot
        x = np.array(weh_all)
        y = np.array(bnacs_all)
        z = np.array(chi2_all)
        min_ind = np.where(z==np.min(z))
        xi = np.linspace(min(x), max(x))
        yi = np.linspace(min(y), max(y))
        X, Y = np.meshgrid(xi, yi)
        Z = griddata(x, y, z, xi, yi, interp='linear')
        # Plot surface plot
        ax = fig.gca(projection='3d')
        surf = ax.plot_surface(X, Y, Z, cmap=cm.jet, linewidth=0, antialiased=True, vmin=np.min(Z), vmax=np.max(Z))
        fig.colorbar(surf, ax=ax)
        # Mark the best fit model
        ax.scatter(x[min_ind], y[min_ind], z[min_ind], marker='+', color='pink')
        # Draw a plane at the critical value threshold
        T = X*0 + t
        ax.plot_surface(X, Y, T, color='k', alpha=0.5)
        # Get the range of values for models below that threshold
        err_ind = np.where(z <= t)
        err_min_bnacs = sorted(y[err_ind])[0]
        err_max_bnacs = sorted(y[err_ind])[-1]
        err_min_weh = sorted(x[err_ind])[0]
        err_max_weh = sorted(x[err_ind])[-1]
        ax.set_xlabel('WEH (wt. %)')
        ax.set_ylabel('$\Sigma_{abs}$ (b)')
        ax.set_zlabel('$\chi^2$')
        ax.set_zlim(np.min(Z), np.max(Z))
        ax.set_title('Surface Plot of $\chi^2$ Values over Model Grid')
        #print mdir
        #plt.show()
        plt.savefig(os.path.join(mdir, 'chi2_surface_t=%s_WEH_%s_%s-%s_BNACS%s_%s-%s.png' % (str(round(t, 3)), str(round(x[min_ind], 1)), str(round(err_min_weh, 1)), str(round(err_max_weh, 1)), str(round(y[min_ind], 3)), str(round(err_min_bnacs, 3)), str(round(err_max_bnacs, 3)))))
        plt.close()