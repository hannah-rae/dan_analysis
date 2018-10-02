import os.path
import pandas as pd
from glob import glob
from subprocess import call

material = 'm7010'
nacs_by_iso = pd.read_csv('/Users/hannahrae/data/dan/BNACS_by_isotope.csv')
output_dir = '/Users/hannahrae/data/dan/M1R_homogeneous_APXS_SB_FT_0.1-6.0H_0.48-14.0Fe_1.8rho_complete/mcnp_outputs_nacs'

def compute_nacs(mass_fractions):
    nacs = 0
    for nuclide in mass_fractions.keys():
        # Ignore the .80c
        if '.80c' in nuclide:
            this_acs = nacs_by_iso.loc[nacs_by_iso['nuclide'] == int(nuclide[:-4]), 'ACS'].item()
        else:
            this_acs = nacs_by_iso.loc[nacs_by_iso['nuclide'] == int(nuclide), 'ACS'].item()
        nacs += this_acs*mass_fractions[nuclide]
    return nacs

# Go through each output file
for f in glob(os.path.join('/Users/hannahrae/data/dan/M1R_homogeneous_APXS_SB_FT_0.1-6.0H_0.48-14.0Fe_1.8rho_complete/mcnp_outputs', '*')):
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
    name_pre, name_post = f.split('/')[-1].split('Cl')
    new_name = '%sCl_%sACS%s' % (name_pre, str(round(nacs, 3)), name_post)
    call(['cp', f, os.path.join(output_dir, new_name)])
