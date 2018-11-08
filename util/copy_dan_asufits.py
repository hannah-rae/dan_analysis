import os.path
from glob import glob
from subprocess import call

out_dir = '/Users/danops/Desktop/dan_asufits_hrk'

for soldir in glob('/Users/danops/Desktop/DANOPS/FEI/DAN/sol*'):
    # Make a new directory for this sol
    sol = soldir.split('/')[-1]
    call(['mkdir', os.path.join(out_dir, sol)])
    for mdir in glob(os.path.join(soldir, '*')):
        mname = mdir.split('/')[-1]
        # Make a new directory for this measurement
        call(['mkdir', os.path.join(out_dir, sol, mname)])
        # Copy the background-subtracted data
        call(['cp', os.path.join(mdir, 'bg_dat.npy'), os.path.join(out_dir, sol, mname)])
        # Copy the best-fit model information
        call(['cp', os.path.join(mdir, 'asu1/M1R_homogeneous_APXS_SB_FT_0.1-6.0H_0.48-14.0Fe_0.1-3.0Cl_1.8rho_MASTER/results/gridInfo_statistics.csv'), os.path.join(out_dir, sol, mname)])
        call(['cp', os.path.join(mdir, 'asu1/M1R_homogeneous_APXS_SB_FT_0.1-6.0H_0.48-14.0Fe_0.1-3.0Cl_1.8rho_MASTER/results/gridInfo.csv'), os.path.join(out_dir, sol, mname)])
        call(['cp', os.path.join(mdir, 'asu1/M1R_homogeneous_APXS_SB_FT_0.1-6.0H_0.48-14.0Fe_0.1-3.0Cl_1.8rho_MASTER/results/gridInfo_goodFitList.csv'), os.path.join(out_dir, sol, mname)])
        call('cp %s %s' % (os.path.join(mdir, '*.JPG'), os.path.join(out_dir, sol, mname)), shell=True)
        call('cp %s %s' % (os.path.join(mdir, '*.png'), os.path.join(out_dir, sol, mname)), shell=True)