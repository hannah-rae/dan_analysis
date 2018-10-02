import errno
from glob import glob
from os import mkdir, path
from subprocess import call

data_dir = '/Users/hannahrae/data/dan_bg_sub'
# This HazCam directory includes all Hazcam images by sol
# The first %s parameter is the sol (fmt solXXXXX, zero-padded)
# The second %s parameter is the 7-digit site/drive at the end of "long name" below
lrhaz_url = '/Users/danops/Desktop/DANOPS/PDS/HazCam/sol%s/R*EDR*F%sRHAZ*.JPG' 
rrhaz_url = '/Users/danops/Desktop/DANOPS/PDS/HazCam/sol%s/R*EDR*F%sRHAZ*.JPG'

save_dir = '/Users/hannahrae/data/dan/rhaz'

all_measurements = glob('/Users/hannahrae/data/dan/all/*.npy')
for meas in all_measurements:
    short_name = meas.split('/')[-1][:-4]
    # Some names are different if there is more than one measurement that sol
    if len(short_name.split('_')) == 2:
        sol = short_name.split('_')[0]
        num = int(short_name[-1])
        long_name = glob('%s/sol%s/*' % (data_dir, sol.zfill(5)))[num-1].split('/')[-1]
        site_drive = long_name[-7:]
    else:
        sol = short_name
        long_name = glob('%s/sol%s/*' % (data_dir, sol.zfill(5)))[0].split('/')[-1]
        site_drive = long_name[-7:]
    
    try:
        mkdir(path.join(save_dir, short_name))
        # call('scp -oProxyJump=hannah@bikini.sese.asu.edu danops@craigs-mac-pro.istb4.dhcp.asu.edu:%s %s' % ('/Users/danops/Desktop/DANOPS/FEI/DAN/sol%s/DNB_%s*/*.JPG' % (sol.zfill(5), long_name), path.join(save_dir, short_name)), shell=True)
        call('scp -oProxyJump=hannah@bikini.sese.asu.edu danops@craigs-mac-pro.istb4.dhcp.asu.edu:%s %s' % (lrhaz_url % (sol.zfill(5), site_drive), path.join(save_dir, short_name)), shell=True)
    except OSError as exc:
        if exc.errno != errno.EEXIST:
            raise
        pass

    #call('scp -oProxyJump=hannah@bikini.sese.asu.edu danops@craigs-mac-pro.istb4.dhcp.asu.edu:%s %s' % ('/Users/danops/Desktop/DANOPS/FEI/DAN/sol%s/DNB_%s*/*.JPG' % (sol.zfill(5), long_name), path.join(save_dir, short_name)), shell=True)
    #call('scp -oProxyJump=hannah@bikini.sese.asu.edu danops@craigs-mac-pro.istb4.dhcp.asu.edu:%s %s' % (lrhaz_url % (sol.zfill(5), site_drive), path.join(save_dir, short_name)), shell=True)
    #call('scp -oProxyJump=hannah@bikini.sese.asu.edu danops@craigs-mac-pro.istb4.dhcp.asu.edu:%s %s' % (rrhaz_url % (sol.zfill(5), site_drive), path.join(save_dir, short_name)), shell=True)