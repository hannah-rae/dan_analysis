'''
Data from Hardgrove et al. 2011. These do not use a real He-3 detector or a real PNG, 
so the results represent the behavior of neutron die-away curves in the absence of real
detectors and electronics. Effectively, these outputs represent the theoretical behavior 
of thermal and epithermal neutrons after a 14.1 MeV neutron pulse.
'''

import matplotlib
matplotlib.use('TkAgg')

import numpy as np
np.random.seed(1) # for reproducibility
import argparse
import os.path
import matplotlib.pyplot as plt
import csv

from glob import glob
from subprocess import call
from mpl_toolkits.mplot3d import Axes3D

import sklearn
import scipy
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_score

# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--n_components', type=int, help='number of principal components to use for PCA')
parser.add_argument('--testing_percentage', type=int, default=20, help='percent of training data to use for testing')
parser.add_argument('--test_dan', action='store_true', help='use DAN data for testing')
parser.add_argument('--test_split', action='store_true', help='use percentage of training data for testing')
parser.add_argument('--linear', action='store_true', help='perform linear regression to predict H and Cl values')
parser.add_argument('--dnn', action='store_true', help='perform regression using deep neural network to predict H and Cl values')
parser.add_argument('--lasso', action='store_true', help='perform lasso regression to predict H and Cl values')
parser.add_argument('--elasticnet', action='store_true', help='perform ElasticNet regression to predict H and Cl values')
parser.add_argument('--ignore_early_bins', action='store_true', help='ignore the first five time bins (PNG noise)')
parser.add_argument('--use_restricted_bins', action='store_true', help='only use bins 17-34 and 11-17 (counting from 1)')
parser.add_argument('--plot_error_bars', action='store_true', help='plot error bars for H and Cl values (DAN data)')
args = parser.parse_args()

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

def to_dan_bins(counts, ex_num):
    counts_th = counts[:len(time_bins_sim)]
    counts_epi = counts[len(time_bins_sim):]
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

    # # Sanity check: is coarsify doing what we want it to do
    # if ex_num==0:
    #     fig2, axes = plt.subplots(nrows=2, ncols=2)
    #     axes[0,0].set_title('Original Simulation Thermal Die-Away Curve')
    #     axes[0,0].set_xlabel('Time (us) - Simulation bins')
    #     axes[0,0].set_ylabel('Simulated Counts')
    #     axes[0,0].step(time_bins_sim, counts_th, where='post', linewidth=2)
    #     #axes[0,0].plot(bins_sim_interp.flatten(), counts_th_interp.flatten(), 'r:')

    #     axes[0,1].set_title('Original Simulation Epithermal Die-Away Curve')
    #     axes[0,1].set_xlabel('Time (us) - Simulation bins')
    #     axes[0,1].set_ylabel('Simulated Counts')
    #     axes[0,1].step(time_bins_sim, counts_epi, where='post', linewidth=2)
    #     #axes[0,1].plot(bins_sim_interp.flatten(), counts_epi_interp.flatten(), 'r:')

    #     axes[1,0].set_title('Binned Simulation Thermal Die-Away Curve')
    #     axes[1,0].set_xlabel('Time (us) - DAN bins')
    #     axes[1,0].set_ylabel('Simulated Counts')
    #     axes[1,0].step(time_bins_dan[:34], counts_th_dan, where='post', linewidth=2)

    #     axes[1,1].set_title('Binned Simulation Epithermal Die-Away Curve')
    #     axes[1,1].set_xlabel('Time (us) - DAN bins')
    #     axes[1,1].set_ylabel('Simulated Counts')
    #     axes[1,1].step(time_bins_dan[:34], counts_epi_dan, where='post', linewidth=2)
    #     plt.tight_layout()
    #     plt.show()

    return np.concatenate([counts_th_dan, counts_epi_dan])

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
        # Separate the data into thermal and epithermal components
        th_counts = count_vectors[i][:count_vectors.shape[1]/2]
        epi_counts = count_vectors[i][count_vectors.shape[1]/2:]
        # Compute a sum of detector data separately for both detectors
        sum_th = sum(th_counts)
        sum_epi = sum(epi_counts)
        # Divide the counts in each bin by the total number of counts,
        th_counts /= float(sum_th)
        epi_counts  /= float(sum_epi)
        normalized_counts[i] = np.concatenate([th_counts, epi_counts])
    return normalized_counts

def read_dan_data():
    X = []
    Y = []
    Y_error = []
    with open('/Users/hannahrae/data/dan/dan_iki_params.csv', 'rU') as csvfile:
        reader = csv.reader(csvfile, dialect=csv.excel_tab)
        for row in reader:
            site, sol, name, h, h_error, cl, cl_error = row[0].split(',')
            counts = np.load('/Users/hannahrae/data/dan/dan_bg_sub/sol%s/%s/bg_dat.npy' % (sol.zfill(5), name))
            ctn_counts = normalize_png(counts[:][0])[:34]
            cetn_counts = normalize_png(counts[:][1])[:34]
            thermals = [total - epi for total, epi in zip(ctn_counts, cetn_counts)]
            feature_vec = thermals + cetn_counts
            X.append(feature_vec)
            Y.append([float(h), float(cl)])
            Y_error.append([float(h_error), float(cl_error)])
            # if float(h) == 5.19 and float(cl) == 1.00:
            #     np.savetxt('/Users/hannahrae/data/dan/dan_5H1CL.txt', feature_vec)
    return np.array(X), np.array(Y), np.array(Y_error)

# Read in the data
data_dir = '/Users/hannahrae/data/dan/dan_theoretical'
n = len(glob(os.path.join(data_dir, '*.o')))
X = np.ndarray((n, 34*2))
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

    X[i] = to_dan_bins(np.concatenate([np.array(counts_th), np.array(counts_epi)]), i)
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

# Shuffle data and labels at the same time
combined = list(zip(X, Y))
np.random.shuffle(combined)
X[:], Y[:] = zip(*combined)
# Separate train and test sets
if args.test_split:
    n_test = int(n * args.testing_percentage/100.0)
    X_test = X[:n_test]
    Y_test = Y[:n_test]
    X_train = X[n_test:]
    Y_train = Y[n_test:]
    X_train = normalize_counts(X_train)
    X_test = normalize_counts(X_test)
elif args.test_dan:
    X_train = X
    Y_train = Y
    X_test, Y_test, Y_test_error = read_dan_data()
    n_test = X_test.shape[0]
    # DAN bins have some count overlap in the early bins
    # between CTN (total neutrons) and CETN, leading to 
    # negative thermal counts in the early bins
    # Normalize counts to approximately same range
    X_train = normalize_counts(X_train)
    X_test = normalize_counts(X_test)

    if args.ignore_early_bins:
        X_train = np.take(X_train, range(5,34)+range(39,68), axis=1)
        X_test = np.take(X_test, range(5,34)+range(39,68), axis=1)
    if args.use_restricted_bins:
        X_train = np.take(X_train, range(15,34)+range(34+12,34+17), axis=1)
        X_test = np.take(X_test, range(15,34)+range(34+12,34+17), axis=1)
    print X_train.shape
    print X_test.shape
    # if args.use_restricted_bins:
    #     X_train


# Normalize data to zero mean and unit variance
from sklearn import preprocessing
# X_train = preprocessing.scale(X_train)
# X_test = preprocessing.scale(X_test)

# scaler = preprocessing.StandardScaler().fit(X_train)
# X_train = scaler.transform(X_train)
# X_test = scaler.transform(X_test)

# scaler = preprocessing.MinMaxScaler().fit(X_train)
# X_train = scaler.transform(X_train)
# X_test = scaler.transform(X_test)

print 'train min %f' % np.min(X_train)
print 'train max %f' % np.max(X_train)
print 'test min %f' % np.min(X_test)
print 'test max %f' % np.max(X_test)

# Fit PCA model and project data into PC space
pca = PCA(n_components=args.n_components)
pca.fit(X_train)
X_train = pca.transform(X_train)
X_test = pca.transform(X_test)

if args.linear:
    from sklearn.linear_model import LinearRegression
    # Perform Linear Regression with constraint that Yi >=0
    lr = LinearRegression(normalize=True)
    print np.mean(cross_val_score(lr, X_train, np.log(Y_train+1), cv=5, scoring=sklearn.metrics.make_scorer(r2_score)))
    lr.fit(X=X_train, y=np.log(Y_train+1))
    #lr.fit(X=X_train, y=Y_train)
    # if args.test_dan:
    #     pca_test = PCA(n_components=args.n_components)
    #     pca_test.fit(X_test)
    #     X_test_transformed = pca_test.transform(X_test)
    #     Y_pred = np.exp(lr.predict(X=X_test_transformed))-1
    # else:
    #     Y_pred = np.exp(lr.predict(X=pca.transform(X_test)))-1

    Y_pred = np.exp(lr.predict(X=X_test))-1
    #Y_pred = lr.predict(X=X_test)

    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)
    if args.plot_error_bars:
        ax1.errorbar(range(1, n_test+1), Y_test[:, 0], yerr=Y_test_error[:,0], ecolor='r', color='k', label='True value')
    else:
        ax1.plot(range(1, n_test+1), Y_test[:, 0], color='k', label='True value')
    ax1.plot(range(1, n_test+1), Y_pred[:, 0], color='k', linestyle='--', label='Predicted value')

    ax1.legend(loc='upper right')
    ax1.set_title('Linear Regression Predictions for H Values ($R^2$=%f)' % r2_score(Y_test[:,0], Y_pred[:,0]))
    if args.plot_error_bars:
        ax2.errorbar(range(1, n_test+1), Y_test[:, 1], yerr=Y_test_error[:,1], ecolor='r', color='k', label='True value')
    else:
        ax2.plot(range(1, n_test+1), Y_test[:, 1], color='k', label='True value')
    ax2.plot(range(1, n_test+1), Y_pred[:, 1], color='k', linestyle='--', label='Predicted value')
    ax2.set_ylabel('Cl value (wt %)')
    ax2.set_xlabel('Test Example')
    ax2.legend(loc='upper right')
    ax2.set_title('Linear Regression Predictions for Cl Values ($R^2$=%f)' % r2_score(Y_test[:,1], Y_pred[:,1]))

    if args.plot_error_bars:
        # Plot a scatter plot to show correlations
        fig, (ax3, ax4) = plt.subplots(nrows=1, ncols=2)
        ax3.scatter(Y_test[:, 0]+Y_test_error[:,0], Y_pred[:, 0])
        ax3.set_ylabel('Predicted H value (wt %)')
        ax3.set_xlabel('Actual H value (wt %)')
        ax4.scatter(Y_test[:, 1]+Y_test_error[:,1], Y_pred[:, 1])
        ax4.set_ylabel('Predicted Cl value (wt %)')
        ax4.set_xlabel('Actual Cl value (wt %)')

if args.lasso:
    from sklearn.linear_model import Lasso

    lasso = Lasso(alpha=0.1, normalize=True)
    print np.mean(cross_val_score(lasso, X_train, Y_train, cv=5, scoring=sklearn.metrics.make_scorer(r2_score)))

    lasso.fit(X=X_train, y=np.log(1+Y_train))
    Y_pred = np.exp(lasso.predict(X=X_test))-1

    fig, (ax5, ax6) = plt.subplots(nrows=1, ncols=2)
    ax5.plot(range(1, n_test+1), Y_test[:, 0], color='k', label='True value')
    ax5.plot(range(1, n_test+1), Y_pred[:, 0], color='k', linestyle='--', label='Predicted value')
    ax5.set_ylabel('H value (wt %)')
    ax5.set_xlabel('Example')
    ax5.legend(loc='upper right')
    ax5.set_title('Lasso Regression Predictions for H Values (R^2=%f)' % r2_score(Y_test[:,0], Y_pred[:,0]))
    ax6.plot(range(1, n_test+1), Y_test[:, 1], color='k', label='True value')
    ax6.plot(range(1, n_test+1), Y_pred[:, 1], color='k', linestyle='--', label='Predicted value')
    ax6.set_ylabel('Cl value (wt %)')
    ax6.set_xlabel('Test Example')
    ax6.legend(loc='upper right')
    ax6.set_title('Lasso Regression Predictions for Cl Values (R^2=%f)' % r2_score(Y_test[:,1], Y_pred[:,1]))

if args.elasticnet:
    from sklearn.linear_model import ElasticNet

    elasticnet = ElasticNet(alpha=0.01, normalize=True)
    print np.mean(cross_val_score(elasticnet, X_train, Y_train, cv=5, scoring=sklearn.metrics.make_scorer(r2_score)))

    elasticnet.fit(X=X_train, y=np.log(1+Y_train))
    Y_pred = np.exp(elasticnet.predict(X=X_test))-1

    fig, (ax5, ax6) = plt.subplots(nrows=1, ncols=2)
    ax5.plot(range(1, n_test+1), Y_test[:, 0], color='k', label='True value')
    ax5.plot(range(1, n_test+1), Y_pred[:, 0], color='k', linestyle='--', label='Predicted value')
    ax5.set_ylabel('H value (wt %)')
    ax5.set_xlabel('Example')
    ax5.legend(loc='upper right')
    ax5.set_title('Lasso Regression Predictions for H Values (R^2=%f)' % r2_score(Y_test[:,0], Y_pred[:,0]))
    ax6.plot(range(1, n_test+1), Y_test[:, 1], color='k', label='True value')
    ax6.plot(range(1, n_test+1), Y_pred[:, 1], color='k', linestyle='--', label='Predicted value')
    ax6.set_ylabel('Cl value (wt %)')
    ax6.set_xlabel('Test Example')
    ax6.legend(loc='upper right')
    ax6.set_title('Lasso Regression Predictions for Cl Values ($R^2$=%f)' % r2_score(Y_test[:,1], Y_pred[:,1]))


if args.dnn:
    # TODO: we should cross validate this to really assess performance
    # Perform regression with deep neural network
    from keras.models import Sequential
    from keras.layers import Dense
    from keras.callbacks import TensorBoard

    model = Sequential()
    model.add(Dense(4, input_dim=3, activation='relu'))
    model.add(Dense(4, activation='relu'))
    model.add(Dense(2, activation='linear'))
    model.compile(loss='mse', optimizer='sgd')
    model.fit(X_train, 
              np.log(1+Y_train), 
              epochs=400, 
              batch_size=100, 
              callbacks=[TensorBoard(log_dir='/tmp/dnn_regression')],
              validation_data=(X_test, Y_test))

    Y_pred = np.exp(model.predict(X_test))-1

    #r2_score = r2_score(Y_test, Y_pred)

    fig, (ax3, ax4) = plt.subplots(nrows=1, ncols=2)
    ax3.plot(range(1, n_test+1), Y_test[:, 0], color='k', label='True value')
    ax3.plot(range(1, n_test+1), Y_pred[:, 0], color='k', linestyle='--', label='Predicted value')
    ax3.set_ylabel('H value (wt %)')
    ax3.set_xlabel('Example')
    ax3.legend(loc='upper right')
    ax3.set_title('DNN Regression Predictions for H Values (R^2=%f)' % r2_score(Y_test[:,0], Y_pred[:,0]))
    ax4.plot(range(1, n_test+1), Y_test[:, 1], color='k', label='True value')
    ax4.plot(range(1, n_test+1), Y_pred[:, 1], color='k', linestyle='--', label='Predicted value')
    ax4.set_ylabel('Cl value (wt %)')
    ax4.set_xlabel('Test Example')
    ax4.legend(loc='upper right')
    ax4.set_title('DNN Regression Predictions for Cl Values (R^2=%f)' % r2_score(Y_test[:,1], Y_pred[:,1]))


plt.show()
