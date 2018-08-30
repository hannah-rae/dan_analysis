'''
Data from Hardgrove et al. 2011. These do not use a real He-3 detector or a real PNG, 
so the results represent the behavior of neutron die-away curves in the absence of real
detectors and electronics. Effectively, these outputs represent the theoretical behavior 
of thermal and epithermal neutrons after a 14.1 MeV neutron pulse.
'''

import numpy as np
import scipy
np.random.seed(1) # for reproducibility

import matplotlib.pyplot as plt
import argparse

from mpl_toolkits.mplot3d import Axes3D

import sklearn
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_score
from sklearn import preprocessing

import datasets

# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--n_components', type=int, help='number of principal components to use for PCA')
parser.add_argument('--testing_percentage', type=int, default=20, help='percent of training data to use for testing')
parser.add_argument('--test_dan', action='store_true', help='use DAN data for testing')
parser.add_argument('--test_sim', action='store_true', help='use percentage of training data for testing')
parser.add_argument('--linear', action='store_true', help='perform linear regression to predict H and Cl values')
parser.add_argument('--dnn', action='store_true', help='perform regression using deep neural network to predict H and Cl values')
parser.add_argument('--lasso', action='store_true', help='perform lasso regression to predict H and Cl values')
parser.add_argument('--elasticnet', action='store_true', help='perform ElasticNet regression to predict H and Cl values')
parser.add_argument('--ignore_early_bins', action='store_true', help='ignore the first five time bins (PNG noise)')
parser.add_argument('--use_restricted_bins', action='store_true', help='only use bins 17-34 and 11-17 (counting from 1)')
parser.add_argument('--plot_error_bars', action='store_true', help='plot error bars for H and Cl values (DAN data)')
args = parser.parse_args()

# Separate train and test sets
if args.test_sim:
    X, Y = datasets.read_sim_data(use_dan_bins=True)
    n_test = int(X.shape[0] * args.testing_percentage/100.0)
    X_test = X[:n_test]
    Y_test = Y[:n_test]
    X_train = X[n_test:]
    Y_train = Y[n_test:]
elif args.test_dan:
    X, Y = datasets.read_sim_data(use_dan_bins=True)
    X_train = X
    Y_train = Y
    X_test, Y_test, Y_test_error = datasets.read_dan_data()
    n_test = X_test.shape[0]

    # Normalize counts to approximately same range
    X_train = datasets.normalize_counts(X_train)
    X_test = datasets.normalize_counts(X_test)

    # DAN bins have some count overlap in the early bins
    # between CTN (total neutrons) and CETN, leading to 
    # negative thermal counts in the early bins
    if args.ignore_early_bins:
        X_train = np.take(X_train, range(5,34)+range(39,68), axis=1)
        X_test = np.take(X_test, range(5,34)+range(39,68), axis=1)
    
    # These bins demonstrate the most dynamic range with respect to changing
    # subsurface geochemistry: 18-34 for CTN and 13-17 for CETN
    if args.use_restricted_bins:
        X_train = np.take(X_train, range(15,34)+range(34+12,34+17), axis=1)
        X_test = np.take(X_test, range(15,34)+range(34+12,34+17), axis=1)
    print X_train.shape
    print X_test.shape

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

    # Plot a scatter plot to show correlations
    fig, (ax3, ax4) = plt.subplots(nrows=1, ncols=2)
    ax3.scatter(Y_test[:, 0], Y_pred[:, 0])
    ax3.set_ylabel('Predicted H value (wt %)')
    ax3.set_xlabel('Actual H value (wt %)')
    ax4.scatter(Y_test[:, 1], Y_pred[:, 1])
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
