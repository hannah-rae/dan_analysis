import csv
import matplotlib.pyplot as plt
import numpy as np
import os.path

from glob import glob

# Go through each sol directory
for soldir in sorted(glob('/Users/hannahrae/data/dan/dan_asufits_hiweh/*')):
    sol = int(soldir.split('/')[-1][3:])
    # Go through each measurement directory
    for mdir in glob(os.path.join(soldir, '*')):
        # Load all the chi-squared values
        chi2_all = []
        with open(os.path.join(mdir, 'gridInfo.csv'), 'rb') as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                # skip the headings row
                if 'Simulation' in row:
                    continue
                simname, chi2, pval, dof = row
                chi2_all.append(float(chi2))
        fig, ax = plt.subplots(1)
        ax.hist(chi2_all, bins=50)
        # Plot a vertical line for the min chi-squared value
        # plt.axvline(np.min(chi2_all))
        ax.set_xlabel('$\chi^2$ Value')
        ax.set_ylabel('Frequency')
        ax.set_title('Distribution of $\chi^2$ Values for Model Grid')
        plt.savefig(os.path.join(mdir, 'chi2_dist.png'))
        plt.close()