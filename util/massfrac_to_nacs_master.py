import os.path
import pandas as pd
import argparse
from glob import glob
from subprocess import call
from scipy.constants import N_A

material = 'm7010'
nacs_by_iso = pd.read_csv('/Users/hannahrae/data/dan/BNACS_by_isotope.csv')

# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--sink_dir', help='Path to directory to save renamed MCNP output files')
parser.add_argument('--source_dir', help='Path to MCNP output files to rename')
parser.add_argument('--cl_grid', action='store_true', help='Set if this is a Cl-varying grid')
parser.add_argument('--fe_grid', action='store_true', help='Set if this is an Fe-varying grid')
parser.add_argument('--lawrence_nacs', action='store_true', help='Use Lawrence 2011 computation for NACS')
parser.add_argument('--simple_nacs', action='store_true', help='Use simple weighted sum computation for NACS')
args = parser.parse_args()

def nacs_lawrence2011(acs, mass_fraction, nuclide):
    # Ignore the .80c
    if '.80c' in nuclide:
        nuclide = nuclide[:-4]
    return (acs*N_A*mass_fraction)/(int(nuclide)/100.)

def nacs_simple(acs, mass_fraction):
    return acs*mass_fraction

def compute_nacs(mass_fractions):
    nacs = 0
    for nuclide in mass_fractions.keys():
        # Ignore the .80c
        if '.80c' in nuclide:
            this_acs = nacs_by_iso.loc[nacs_by_iso['nuclide'] == int(nuclide[:-4]), 'ACS'].item()
        else:
            this_acs = nacs_by_iso.loc[nacs_by_iso['nuclide'] == int(nuclide), 'ACS'].item()
        if args.simple_nacs:
            nacs += nacs_simple(this_acs, mass_fractions[nuclide])
        elif args.lawrence_nacs:
            nacs += nacs_lawrence2011(this_acs, mass_fractions[nuclide], nuclide)
        #nacs += this_acs*mass_fractions[nuclide]
    return nacs

# Go through each output file
for f in glob(os.path.join(args.source_dir, '*')):
    # Read in the mass fractions for each isotope for each file
    mf = {}
    reading_m = False
    with open(f) as f_:
        for line in f_:
            if reading_m and 'nlib' in line:
                reading_m = False
            if reading_m:
                _, nuclide, massfrac = line.strip().split()
                massfrac = float(massfrac[1:])
                mf[nuclide] = massfrac
            # The material only appears once in the output file
            if material in line:
                reading_m = True
    # Compute the thermal neutron absorption cross section for this material
    nacs = compute_nacs(mf)
    # Copy the output file to a new file with NACS appended to the filename
    if args.cl_grid:
        # pre-pend 14.22Fe to the start of the filenames (constant Fe)
        # insert ACS after Cl
        name_pre, name_post = f.split('/')[-1].split('Cl')
        new_name = '14.22Fe_%sCl_%sACS%s' % (name_pre, str(round(nacs, 3)), name_post)
    elif args.fe_grid:
        name_pre, name_post = f.split('/')[-1].split('Cl')
        new_name = '%sCl_%sACS%s' % (name_pre, str(round(nacs, 3)), name_post)
    call(['cp', f, os.path.join(args.sink_dir, new_name)])
