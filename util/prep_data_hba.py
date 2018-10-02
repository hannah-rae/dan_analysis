import numpy as np
from glob import glob

data_dir = '/Users/hannahrae/data/dan_bg_sub'
data_list = '/Users/hannahrae/src/autoencoder/dan/util/sol_h2o_cl.txt'
# train_dir = '/Users/hannahrae/data/dan/train'
# test_dir = '/Users/hannahrae/data/dan/test'
data_txt = '/Users/hannahrae/data/dan_typical_spectra.txt'


def normalize_png(count_vector):
    '''For both detectors, bins between 24 us and 75 us (bins 5-9 counting from 1) (bins 4-8
    counting from 0) are selected as reference bins. We calculate the total number of counts
    in these bins. Then we divide the number of counts in every bin by this total. '''
    sum4to8 = float(np.sum(count_vector[4:9]))
    corrected = [np.divide(n, sum4to8) for n in count_vector]
    return corrected

def get_train_test():
    train = []
    test = []
    with open(data_list, 'r') as f:
        for row in f:
            sol, h2o, cl = row.rstrip().split()
            h2o = float(h2o)
            cl = float(cl)
            # Check if both H2O and Cl values in typical range
            if (h2o > 1.5 and h2o < 3.2) and (cl > 0.55 and cl < 1.3):
                train.append(sol)
            else:
                test.append(sol)
    return train, test

train, test = get_train_test()

train_vectors = np.ndarray((len(train), 128))
for i, meas in enumerate(train):
    if len(meas.split('_')) > 1:
        num = int(meas[-1])
        sol = meas.split('_')[0]
        count_file = glob(data_dir + '/sol' + sol.zfill(5) + '/*')[num-1] + '/bg_dat.npy'
    else:
        sol = meas
        count_file = glob(data_dir + '/sol' + sol.zfill(5) + '/*')[0] + '/bg_dat.npy'
    counts = np.load(count_file)
    # Here we are using all of the bins since we are looking at time series
    ctn_counts = normalize_png(counts[:][0])
    cetn_counts = normalize_png(counts[:][1])
    feature_vec = ctn_counts + cetn_counts

    train_vectors[i] = feature_vec

# Write each measurement as a line in text file
np.savetxt(data_txt, train_vectors)

# for meas in test:
#     if len(meas.split('_')) > 1:
#         num = int(meas[-1])
#         sol = meas.split('_')[0]
#         count_file = glob(data_dir + '/sol' + sol.zfill(5) + '/*')[num-1] + '/bg_dat.npy'
#     else:
#         sol = meas
#         count_file = glob(data_dir + '/sol' + sol.zfill(5) + '/*')[0] + '/bg_dat.npy'
#     counts = np.load(count_file)
#     # Here we are using all of the bins since we are looking at time series
#     ctn_counts = normalize_png(counts[:][0])
#     cetn_counts = normalize_png(counts[:][1])
#     feature_vec = ctn_counts + cetn_counts
#     # Write into train directory
#     np.save(test_dir + '/' + meas + '.npy', feature_vec)