from glob import glob
import csv
import os.path
import numpy as np 

mnames, sols, bf_wehs, bf_bnacss, unc_wehs, unc_bnacss, chi2s = [], [], [], [], [], [], []

threshold = 't=30.578' # p = 0.01

# Go through each sol directory
for soldir in sorted(glob('/Users/hannahrae/data/dan/dan_asufits_hiweh/*')):
    sol = int(soldir.split('/')[-1][3:])
    if sol > 2080: # we don't have elevations past 2080
        continue
    else:
        sols.append(sol)
    # Go through each measurement directory
    for mdir in glob(os.path.join(soldir, '*')):
        mnames.append(mdir.split('/')[-1])
        # Get the best fit WEH and BNACS
        chi2_desc = glob(os.path.join(mdir, 'chi2_surface_%s_*.png' % threshold))[0].split('/')[-1][:-4]
        bf_wehs.append(chi2_desc.split('_')[4])
        bf_bnacss.append(chi2_desc.split('_')[6][5:]) # includes the word BNACS oops
        # Get the uncertainty range for WEH
        unc_wehs.append(chi2_desc.split('_')[5])
        unc_bnacss.append(chi2_desc.split('_')[7])
        # Get the chi-squared value for best-fit model
        with open(os.path.join(mdir, 'gridInfo_statistics.csv'), 'rb') as csvfile:
            csvreader = csv.reader(csvfile)
            for row in csvreader:
                if 'Minimum' in row[0]:
                    chi2s.append(row[0].split()[3])


# Load the elevation data
elev_dict = {}
with open('/Users/hannahrae/data/dan/dan_places.csv') as csvfile:
    csvreader = csv.reader(csvfile)
    for row in csvreader:
        if row[2] == '170': # ApID for DAN Active
            name = row[-4].split('/')[-1][:-4]
            elev = float(row[17])
            elev_dict[name] = elev

# Get the elevations in the order of the other lists
elevs = []
for name in mnames:
    elevs.append(elev_dict[name])

np.savetxt('/Users/hannahrae/data/dan/dan_sol-weh-bnacs-unc_%s-chi2-elev_hiweh_sol1-2100.csv' % threshold, 
            zip(mnames, sols, elevs, bf_wehs, unc_wehs, bf_bnacss, unc_bnacss, chi2s), 
            fmt='%s', 
            delimiter=',')