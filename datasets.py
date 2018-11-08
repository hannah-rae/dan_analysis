import numpy as np
import csv
import os.path
from glob import glob


time_bins_sim = [0.00, 1250.00, 2500.00, 3750.00, 5000.00, 6250.00, 7500.00, 8750.00, 10000.00, 11250.00, 
                 12500.00, 13750.00, 15000.00, 16250.00, 17500.00, 18750.00, 20000.00, 21250.00, 22500.00, 
                 23750.00, 25000.00, 26250.00, 27500.00, 28750.00, 30000.00, 31250.00, 32500.00, 33750.00, 
                 35000.00, 36250.00, 37500.00, 38750.00, 40000.00, 41250.00, 42500.00, 43750.00, 45000.00, 
                 46250.00, 47500.00, 48750.00, 50000.00, 51250.00, 52500.00, 53750.00, 55000.00, 56250.00, 
                 57500.00, 58750.00, 60000.00, 61250.00, 62500.00, 63750.00, 65000.00, 66250.00, 67500.00, 
                 68750.00, 70000.00, 71250.00, 72500.00, 73750.00, 75000.00, 76250.00, 77500.00, 78750.00, 
                 80000.00, 81250.00, 82500.00, 83750.00, 85000.00, 86250.00, 87500.00, 88750.00, 90000.00, 
                 91250.00, 92500.00, 93750.00, 95000.00, 96250.00, 97500.00, 98750.00, 100000.00, 101250.00, 
                 102500.00, 103750.00, 105000.00, 106250.00, 107500.00, 108750.00, 110000.00, 111250.00, 
                 112500.00, 113750.00, 115000.00, 116250.00, 117500.00, 118750.00, 120000.00, 121250.00, 
                 122500.00, 123750.00, 125000.00, 126250.00, 127500.00, 128750.00, 130000.00, 131250.00, 
                 132500.00, 133750.00, 135000.00, 136250.00, 137500.00, 138750.00, 140000.00, 141250.00, 
                 142500.00, 143750.00, 145000.00, 146250.00, 147500.00, 148750.00, 150000.00, 151250.00, 
                 152500.00, 153750.00, 155000.00, 156250.00, 157500.00, 158750.00, 160000.00, 161250.00, 
                 162500.00, 163750.00, 165000.00, 166250.00, 167500.00, 168750.00, 170000.00, 171250.00, 
                 172500.00, 173750.00, 175000.00, 176250.00, 177500.00, 178750.00, 180000.00, 181250.00, 
                 182500.00, 183750.00, 185000.00, 186250.00, 187500.00, 188750.00, 190000.00, 191250.00, 
                 192500.00, 193750.00, 195000.00, 196250.00, 197500.00, 198750.00, 200000.00]

time_bins_dan = [0, 5, 10.625, 16.9375, 24, 31.9375, 40.8125, 50.75, 61.875, 74.375, 88.4375, 104.25, 122, 
             141.938, 164.312, 189.438, 217.688, 249.438, 285.125, 325.25, 370.375, 421.125, 478.188, 
             542.375, 614.562, 695.75, 787.062, 889.75, 1005.25, 1135.19, 1281.31, 1445.69, 1630.56, 
             1838.5, 2072.38, 2335.44, 2631.38, 2964.25, 3338.69, 3759.88, 4233.69, 4766.69, 5366.31, 
             6040.88, 6799.75, 7653.44, 8611.94, 9692.31, 10907.7, 12274.9, 13813.1, 15543.4, 17490.1, 
             19680, 22143.6, 24915.2, 28033.2, 31540.9, 35487.1, 39926.6, 44920.9, 50539.4, 56860.3, 63971.2, 100000]


# Convert from shakes to us
time_bins_sim = np.array(time_bins_sim) * 0.01

def normalize_png(count_vector):
    '''For both detectors, bins between 24 us and 75 us (bins 5-9 counting from 1) (bins 4-8
    counting from 0) are selected as reference bins. We calculate the total number of counts
    in these bins. Then we divide the number of counts in every bin by this total. '''
    sum4to8 = float(np.sum(count_vector[4:9]))
    corrected = [np.divide(n, sum4to8) for n in count_vector]
    return corrected

def normalize_counts(count_vectors):
    normalized_counts = np.ndarray(count_vectors.shape)
    for i in range(count_vectors.shape[0]):
        sum_counts = np.sum(count_vectors[i])
        normalized_counts[i] = count_vectors[i] / sum_counts
    return normalized_counts

def model_to_dan_bins(counts):
    counts_th = counts[:counts.shape[0]/2]
    counts_epi = counts[counts.shape[0]/2:]
    counts_th_dan = np.zeros([34])
    counts_epi_dan = np.zeros([34])
    # Linearly interpolate between the simulation bins to get the DAN bin values
    for i in range(len(time_bins_sim)-1):
        # Check if any DAN time bins fall in this range
        for j in range(len(time_bins_dan[:34])):
            if time_bins_dan[j] >= time_bins_sim[i] and time_bins_dan[j] < time_bins_sim[i+1]:
                #x_interp = np.arange(time_bins_sim[i], time_bins_sim[i+1], 0.0001)
                def y(x, x1, x2, y1, y2):
                    b = (x - x1)/(x2 - x1)
                    a = 1 - b
                    return a*y1 + b*y2
                counts_th_dan[j] = y(x=time_bins_dan[j], x1=time_bins_sim[i], x2=time_bins_sim[i+1], y1=counts_th[i], y2=counts_th[i+1])
                counts_epi_dan[j] = y(x=time_bins_dan[j], x1=time_bins_sim[i], x2=time_bins_sim[i+1], y1=counts_epi[i], y2=counts_epi[i+1])

    return np.concatenate([counts_th_dan, counts_epi_dan])

def read_sim_data(shuffle=True, use_dan_bins=False):
    # Read in the data
    data_dir = '/Users/hannahrae/data/dan/dan_theoretical'
    n = len(glob(os.path.join(data_dir, '*.o')))
    if use_dan_bins:
        X = np.ndarray((n, 34*2))
    else:
        X = np.ndarray((n, len(time_bins_sim)*2))
    X_filenames = []
    for i, simfile in enumerate(glob(os.path.join(data_dir, '*.o'))):
        correct_userbin = False
        reading_th = None
        reading_epi = None
        counts_th = []
        counts_epi = []
        for line in open(simfile):
            if 'user bin total' in line.rstrip() and prev_line == ' \n':
                correct_userbin = True

            if 'energy bin:   0.00000E+00 to  3.00000E-07' in line.rstrip() and correct_userbin and reading_th == None:
                reading_th = True
                continue
            if reading_th and 'total' in line.rstrip():
                reading_th = False
                correct_userbin = False

            if 'energy bin:   3.00000E-07 to  1.00000E-05' in line.rstrip() and correct_userbin and reading_epi == None:
                reading_epi = True
                continue
            if reading_epi and 'total' in line.rstrip():
                reading_epi = False
                correct_userbin = False

            if reading_th:
                if 'detector' not in line and 'time' not in line:
                    counts_th.append(float(line.rstrip().split()[1]))
            elif reading_epi:
                if 'detector' not in line and 'time' not in line:
                    counts_epi.append(float(line.rstrip().split()[1]))
            prev_line = line

        if use_dan_bins:
            X[i] = model_to_dan_bins(np.concatenate([np.array(counts_th), np.array(counts_epi)]))
        else:
            X[i] = np.concatenate([np.array(counts_th), np.array(counts_epi)])
        X_filenames.append(simfile)

    X_filenames = np.array(X_filenames)
    Y = np.ndarray((n, 2))
    # Get H and Cl values from filenames
    for idx, f in enumerate(X_filenames):
        Y[idx,0] = int(X_filenames[idx].split('/')[-1].split('_')[0][:-1])/10. # H
        Y[idx,1] = int(X_filenames[idx].split('/')[-1].split('_')[1][:-4])/10. # Cl
        # print "H %f Cl %f" % (Y[idx,0], Y[idx,1])
        # if int(Y[idx,0]) == 5 and int(Y[idx,1]) == 1:
        #     np.savetxt('/Users/hannahrae/data/dan/sim_5H1CL.txt', X[idx])

    if shuffle:
        # Shuffle data and labels at the same time
        combined = list(zip(X, Y))
        np.random.shuffle(combined)
        X[:], Y[:] = zip(*combined)
    
    return X, Y

def read_grid_data(shuffle=True, use_thermals=True, limit_2000us=False):
    X = []
    Y = []
    data_dir = '/Users/hannahrae/data/dan/model_grid_WEH06-01_Cl03-015'
    for mfile in glob(os.path.join(data_dir, '*.npy')):
        cl = mfile.split('/')[-1].split('_')[0].split('C')[0]
        h = mfile.split('/')[-1].split('_')[1].split('H')[0]
        counts = np.load(mfile)
        cetn_counts = counts[:, 0]
        ctn_counts = counts[:, 2]
        if limit_2000us:
            cetn_counts = counts[:, 0][:34]
            ctn_counts = counts[:, 2][:34]
        if use_thermals:
            thermals = ctn_counts - cetn_counts
            feature_vec = np.concatenate([thermals, cetn_counts])
            feature_vec[np.where(feature_vec < 0)] = 0
        else:
            feature_vec = np.concatenate([ctn_counts, cetn_counts])
        X.append(feature_vec)
        Y.append([float(h), float(cl)])
    if shuffle:
        # Shuffle data and labels at the same time
        combined = list(zip(X, Y))
        np.random.shuffle(combined)
        X[:], Y[:] = zip(*combined)

    return np.array(X), np.array(Y)

def read_acs_grid_data(shuffle=True, use_thermals=True, limit_2000us=False):
    X = []
    Y = []
    data_dir = '/Users/hannahrae/data/dan/M1R_homogeneous_APXS_SB_FT_0.1-6.0H_0.48-14.0Fe_0.1-3.0Cl_1.8rho_MASTER'
    for mfile in glob(os.path.join(data_dir, '*.npy')):
        acs = mfile.split('/')[-1].split('_')[2].split('ACS')[0]
        h = mfile.split('/')[-1].split('_')[3].split('H')[0]
        counts = np.load(mfile)
        cetn_counts = counts[:, 0]
        ctn_counts = counts[:, 2]
        if limit_2000us:
            cetn_counts = counts[:, 0][:34]
            ctn_counts = counts[:, 2][:34]
        if use_thermals:
            thermals = ctn_counts - cetn_counts
            feature_vec = np.concatenate([thermals, cetn_counts])
            feature_vec[np.where(feature_vec < 0)] = 0
        else:
            feature_vec = np.concatenate([ctn_counts, cetn_counts])
        X.append(feature_vec)
        Y.append([float(h), float(acs)])
    if shuffle:
        # Shuffle data and labels at the same time
        combined = list(zip(X, Y))
        np.random.shuffle(combined)
        X[:], Y[:] = zip(*combined)

    return np.array(X), np.array(Y)

def read_dan_data(use_thermals=True, limit_2000us=False, label_source='asu'):
    X = []
    Y = []
    Y_error = []
    names = []
    if label_source == 'iki':
        with open('/Users/hannahrae/data/dan/dan_iki_params.csv', 'rU') as csvfile:
            reader = csv.reader(csvfile, dialect=csv.excel_tab)
            for row in reader:
                site, sol, name, h, h_error, cl, cl_error = row[0].split(',')
                # The shape of this data is (4,64) with the rows being:
                # [CTN counts, CETN counts, CTN count error, CETN count error]
                counts = np.load('/Users/hannahrae/data/dan/dan_bg_sub/sol%s/%s/bg_dat.npy' % (sol.zfill(5), name))
                if limit_2000us:
                    ctn_counts = normalize_png(counts[0])[:34]
                    cetn_counts = normalize_png(counts[1])[:34]
                else:
                    ctn_counts = normalize_png(counts[0])
                    cetn_counts = normalize_png(counts[1])
                # Negative values often occur after background correction because 
                # the background correction takes the total background neutrons (all
                # the counts after about 5000 microseconds) and divides those counts 
                # into the respective bin widths in the background bins. Those counts 
                # don't represent the actual background counts in each of those bins 
                # and it's overestimated in some bins. Basically, where we have negative 
                # values, the background is higher than the signal and we have effectively
                # 0 signal, so we can set these to 0.
                ctn_counts = np.array(ctn_counts)
                cetn_counts = np.array(cetn_counts)
                ctn_counts[np.where(ctn_counts < 0)] = 0
                cetn_counts[np.where(cetn_counts < 0)] = 0
                if use_thermals:
                    thermals = ctn_counts - cetn_counts
                    feature_vec = np.concatenate([ctn_counts, cetn_counts])
                    feature_vec[np.where(feature_vec < 0)] = 0
                else:
                    feature_vec = np.concatenate([ctn_counts, cetn_counts])
                X.append(feature_vec)
                Y.append([float(h), float(cl)])
                Y_error.append([float(h_error), float(cl_error)])
                names.append(name)
        return np.array(X), np.array(Y), np.array(Y_error), np.array(names)

    elif label_source == 'asu':
        for soldir in sorted(glob('/Users/hannahrae/data/dan/dan_asu_fits/*')):
            # sol = int(soldir.split('/')[-1][3:])
            # if sol > 1378:
            #     continue
            for mdir in glob(os.path.join(soldir, '*')):
                name = mdir.split('/')[-1]
                counts = np.load(os.path.join(mdir, 'bg_dat.npy'))
                if limit_2000us:
                    ctn_counts = normalize_png(counts[0])[:34]
                    cetn_counts = normalize_png(counts[1])[:34]
                else:
                    ctn_counts = normalize_png(counts[0])
                    cetn_counts = normalize_png(counts[1])
                # Negative values often occur after background correction because 
                # the background correction takes the total background neutrons (all
                # the counts after about 5000 microseconds) and divides those counts 
                # into the respective bin widths in the background bins. Those counts 
                # don't represent the actual background counts in each of those bins 
                # and it's overestimated in some bins. Basically, where we have negative 
                # values, the background is higher than the signal and we have effectively
                # 0 signal, so we can set these to 0.
                ctn_counts = np.array(ctn_counts)
                cetn_counts = np.array(cetn_counts)
                ctn_counts[np.where(ctn_counts < 0)] = 0
                cetn_counts[np.where(cetn_counts < 0)] = 0
                if use_thermals:
                    thermals = ctn_counts - cetn_counts
                    feature_vec = np.concatenate([ctn_counts, cetn_counts])
                    feature_vec[np.where(feature_vec < 0)] = 0
                else:
                    feature_vec = np.concatenate([ctn_counts, cetn_counts])
                X.append(feature_vec)
                # Get the best fit H and ACS
                with open(os.path.join(mdir, 'gridInfo_statistics.csv'), 'rb') as csvfile:
                    csvreader = csv.reader(csvfile)
                    for row in csvreader:
                        if 'Minimum' in row[0]:
                            chi2 = row[0].split()[3]
                            acs = row[0].split()[-1].split('_')[2].split('ACS')[0]
                            h = row[0].split()[-1].split('_')[3].split('H')[0]
                Y.append([float(h), float(acs)])
                # In this case, "error" is the best fit Chi-2 value
                #Y_error.append(float(chi2))
                with open(os.path.join(mdir, 'gridInfo_goodFitList.csv'), 'rb') as csvfile:
                    csvreader = csv.reader(csvfile)
                    Y_error.append(sum(1 for row in csvreader))
                names.append(name)
        return np.array(X), np.array(Y), np.array(Y_error), np.array(names)
    