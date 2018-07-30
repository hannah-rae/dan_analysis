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
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

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
args = parser.parse_args()

# time_bins = [0.00, 1250.00, 2500.00, 3750.00, 5000.00, 6250.00, 7500.00, 8750.00, 10000.00, 11250.00, 
#              12500.00, 13750.00, 15000.00, 16250.00, 17500.00, 18750.00, 20000.00, 21250.00, 22500.00, 
#              23750.00, 25000.00, 26250.00, 27500.00, 28750.00, 30000.00, 31250.00, 32500.00, 33750.00, 
#              35000.00, 36250.00, 37500.00, 38750.00, 40000.00, 41250.00, 42500.00, 43750.00, 45000.00, 
#              46250.00, 47500.00, 48750.00, 50000.00, 51250.00, 52500.00, 53750.00, 55000.00, 56250.00, 
#              57500.00, 58750.00, 60000.00, 61250.00, 62500.00, 63750.00, 65000.00, 66250.00, 67500.00, 
#              68750.00, 70000.00, 71250.00, 72500.00, 73750.00, 75000.00, 76250.00, 77500.00, 78750.00, 
#              80000.00, 81250.00, 82500.00, 83750.00, 85000.00, 86250.00, 87500.00, 88750.00, 90000.00, 
#              91250.00, 92500.00, 93750.00, 95000.00, 96250.00, 97500.00, 98750.00, 100000.00, 101250.00, 
#              102500.00, 103750.00, 105000.00, 106250.00, 107500.00, 108750.00, 110000.00, 111250.00, 
#              112500.00, 113750.00, 115000.00, 116250.00, 117500.00, 118750.00, 120000.00, 121250.00, 
#              122500.00, 123750.00, 125000.00, 126250.00, 127500.00, 128750.00, 130000.00, 131250.00, 
#              132500.00, 133750.00, 135000.00, 136250.00, 137500.00, 138750.00, 140000.00, 141250.00, 
#              142500.00, 143750.00, 145000.00, 146250.00, 147500.00, 148750.00, 150000.00, 151250.00, 
#              152500.00, 153750.00, 155000.00, 156250.00, 157500.00, 158750.00, 160000.00, 161250.00, 
#              162500.00, 163750.00, 165000.00, 166250.00, 167500.00, 168750.00, 170000.00, 171250.00, 
#              172500.00, 173750.00, 175000.00, 176250.00, 177500.00, 178750.00, 180000.00, 181250.00, 
#              182500.00, 183750.00, 185000.00, 186250.00, 187500.00, 188750.00, 190000.00, 191250.00, 
#              192500.00, 193750.00, 195000.00, 196250.00, 197500.00, 198750.00, 200000.00]

# time_bins_m = [np.mean([time_bins[t], time_bins[t+1]]) for t in range(len(time_bins)-1)]

# # Convert from shakes to us
# time_bins = np.array(time_bins) * 0.01
# time_bins_m = np.array(time_bins_m) * 0.01

def read_dan_data():
    X = []
    Y = []
    with open('/Users/hannahrae/data/dan_iki_params.csv', 'rb') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            site, sol, name, h, cl = row
            X.append(np.load('/Users/hannahrae/data/dan_bg_sub/%s/%s/bg_dat.npy' % (sol.zfill(5), name)))
            Y.append([float(h), float(cl)])
    return np.array(X), np.array(Y)

# Read in the data
data_dir = '/Users/hannahrae/data/dan_theoretical'
n = len(glob(os.path.join(data_dir, '*.o')))
X = np.ndarray((n, 322))
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

    X[i] = np.concatenate([np.array(counts_th), np.array(counts_epi)])
    X_filenames.append(simfile)

X_filenames = np.array(X_filenames)
Y = np.ndarray((n, 2))
# Get H and Cl values from filenames
for idx, f in enumerate(X_filenames):
    Y[idx,0] = int(X_filenames[idx].split('/')[-1].split('_')[0][:-1])/10. # H
    Y[idx,1] = int(X_filenames[idx].split('/')[-1].split('_')[1][:-4])/10. # Cl

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
elif args.test_dan:
    X_test, Y_test = read_dan_data()

# Fit PCA model and project data into PC space
pca = PCA(n_components=args.n_components)
pca.fit(X) # should we only be fitting on train data? probably?
X_transformed = pca.transform(X_train)

if args.linear:
    from sklearn.linear_model import LinearRegression
    # TODO: we should cross validate this to really assess performance
    # Perform Linear Regression
    lr = LinearRegression(normalize=True)
    print np.mean(cross_val_score(lr, X_transformed, Y_train, cv=5, scoring=sklearn.metrics.make_scorer(r2_score)))

    # Fit linear regression with constraint that Yi >= 0
    lr.fit(X=X_transformed, y=np.log(1 + Y_train))
    Y_pred = np.exp(lr.predict(X=pca.transform(X_test))) - 1

    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)
    ax1.plot(range(1, n_test+1), Y_test[:, 0], color='k', label='True value')
    ax1.plot(range(1, n_test+1), Y_pred[:, 0], color='k', linestyle='--', label='Predicted value')
    ax1.set_ylabel('H value (wt %)')
    ax1.set_xlabel('Test Example')
    ax1.legend(loc='upper right')
    ax1.set_title('Linear Regression Predictions for H Values ($R^2$=%f)' % r2_score(Y_test[:,0], Y_pred[:,0]))
    ax2.plot(range(1, n_test+1), Y_test[:, 1], color='k', label='True value')
    ax2.plot(range(1, n_test+1), Y_pred[:, 1], color='k', linestyle='--', label='Predicted value')
    ax2.set_ylabel('Cl value (wt %)')
    ax2.set_xlabel('Test Example')
    ax2.legend(loc='upper right')
    ax2.set_title('Linear Regression Predictions for Cl Values ($R^2$=%f)' % r2_score(Y_test[:,1], Y_pred[:,1]))

if args.lasso:
    from sklearn.linear_model import Lasso

    lasso = Lasso(alpha=0.1, normalize=True)
    print np.mean(cross_val_score(lasso, X_transformed, Y_train, cv=5, scoring=sklearn.metrics.make_scorer(r2_score)))

    lasso.fit(X=X_transformed, y=np.log(1+Y_train))
    Y_pred = np.exp(lasso.predict(X=pca.transform(X_test)) - 1)

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
    print np.mean(cross_val_score(elasticnet, X_transformed, Y_train, cv=5, scoring=sklearn.metrics.make_scorer(r2_score)))

    elasticnet.fit(X=X_transformed, y=np.log(1+Y_train))
    Y_pred = np.exp(elasticnet.predict(X=pca.transform(X_test)) - 1)

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
    from sklearn.preprocessing import MinMaxScaler

    scalerX, scalerY = MinMaxScaler(), MinMaxScaler()
    scalerX.fit(X_transformed)
    scalerY.fit(Y_train)
    X = scalerX.transform(X_transformed)
    y = scalerY.transform(Y_train)
    Y_test = scalerY.transform(Y_test)

    model = Sequential()
    model.add(Dense(4, input_dim=3, activation='relu'))
    model.add(Dense(4, activation='relu'))
    model.add(Dense(2, activation='linear'))
    model.compile(loss='mse', optimizer='sgd')
    model.fit(X, 
              y, 
              epochs=400, 
              batch_size=10, 
              callbacks=[TensorBoard(log_dir='/tmp/dnn_regression')],
              validation_data=(scalerX.transform(pca.transform(X_test)), scalerY.transform(Y_test)))

    Y_pred = model.predict(scalerX.transform(pca.transform(X_test)))

    r2_score = r2_score(Y_test, Y_pred)

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
