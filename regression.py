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
from sklearn.decomposition import PCA, FastICA
from sklearn.cluster import KMeans
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_score
from sklearn import preprocessing

import datasets

# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--n_components', type=int, default=3, help='number of principal components to use for PCA')
parser.add_argument('--testing_percentage', type=int, default=20, help='percent of training data to use for testing')
parser.add_argument('--epsilon', type=float, default=0, help='wiggle room for accuracy')
parser.add_argument('--test_dan', action='store_true', help='use DAN data for testing')
parser.add_argument('--test_sim', action='store_true', help='use percentage of training data for testing')
parser.add_argument('--linear', action='store_true', help='perform linear regression to predict H and Cl values')
parser.add_argument('--dnn', action='store_true', help='perform regression using deep neural network to predict H and Cl values')
parser.add_argument('--lasso', action='store_true', help='perform lasso regression to predict H and Cl values')
parser.add_argument('--elasticnet', action='store_true', help='perform ElasticNet regression to predict H and Cl values')
parser.add_argument('--ignore_early_bins', action='store_true', help='ignore the first five time bins (PNG noise)')
parser.add_argument('--use_restricted_bins', action='store_true', help='only use bins 17-34 and 11-17 (counting from 1)')
parser.add_argument('--plot_error_bars', action='store_true', help='plot error bars for H and Cl values (DAN data)')
parser.add_argument('--plot_intermediate_steps', action='store_true', help='visualize steps through regression process')
parser.add_argument('--pca', action='store_true', help='whether to use PCA to reduce dimensions or not')
parser.add_argument('--model_grid', help='Which model dataset to use: options are rover, full, or both')
args = parser.parse_args()

def get_accuracy(y_pred, y_true, y_pred_err, epsilon):
    correct = 0
    correct_mask = []
    for y_p, y_t, y_e in zip(y_pred, y_true, y_pred_err):
        if y_p <= y_t + y_e + epsilon and y_p >= y_t - y_e - epsilon:
            correct += 1
            correct_mask.append(1)
        else:
            correct_mask.append(0)
    acc = correct / float(y_pred.shape[0])
    return acc, correct_mask

# Separate train and test sets
if args.test_sim:
    X, Y = datasets.read_sim_data()
    n_test = int(X.shape[0] * args.testing_percentage/100.0)
    X_test = X[:n_test]
    Y_test = Y[:n_test]
    X_train = X[n_test:]
    Y_train = Y[n_test:]
elif args.test_dan:
    if args.model_grid == 'full':
        X, Y = datasets.read_sim_data(use_dan_bins=True)
        X_test, Y_test, Y_test_error, test_names = datasets.read_dan_data(limit_2000us=True)
        n_bins = 34
    elif args.model_grid == 'rover':
        X, Y = datasets.read_grid_data()
        X_test, Y_test, Y_test_error, test_names = datasets.read_dan_data(limit_2000us=False)
        n_bins = len(datasets.time_bins_dan)-1
    elif args.model_grid == 'both':
        X_full, Y_full = datasets.read_sim_data(use_dan_bins=True)
        X_rover, Y_rover = datasets.read_grid_data(limit_2000us=True)
        X = np.concatenate([X_full, X_rover])
        Y = np.concatenate([Y_full, Y_rover])
        X_test, Y_test, Y_test_error, test_names = datasets.read_dan_data(limit_2000us=True)
        n_bins = 34
    X_train = X
    Y_train = Y
    n_test = X_test.shape[0]

# Normalize counts to approximately same range
X_train = datasets.normalize_counts(X_train)
X_test = datasets.normalize_counts(X_test)

# DAN bins have some count overlap in the early bins
# between CTN (total neutrons) and CETN, leading to 
# negative thermal counts in the early bins
if args.ignore_early_bins:
    X_train = np.take(X_train, range(5, n_bins)+range(n_bins+5, n_bins*2), axis=1)
    X_test = np.take(X_test, range(5, n_bins)+range(n_bins+5, n_bins*2), axis=1)

# These bins demonstrate the most dynamic range with respect to changing
# subsurface geochemistry: 18-34 for CTN and 13-17 for CETN
if args.use_restricted_bins:
    X_train = np.take(X_train, range(17, 34)+range(n_bins+12, n_bins+17), axis=1)
    X_test = np.take(X_test, range(17, 34)+range(n_bins+12, n_bins+17), axis=1)
print X_train.shape
print X_test.shape

print 'train min %f' % np.min(X_train)
print 'train max %f' % np.max(X_train)
print 'test min %f' % np.min(X_test)
print 'test max %f' % np.max(X_test)

if args.pca:
    # Fit PCA model and project data into PC space
    pca = PCA(n_components=args.n_components)
    pca.fit(X_train)
    X_train = pca.transform(X_train)
    X_test = pca.transform(X_test)

if args.plot_intermediate_steps:
    # Plot train and test set in PC space
    fig = plt.figure()
    ax0 = fig.add_subplot(111, projection='3d')
    ax0.set_xlabel('PC 1')
    ax0.set_ylabel('PC 2')
    ax0.set_zlabel('PC 3')
    ax0.scatter(X_train[:,0], X_train[:,1], X_train[:,2], label='Train')
    ax0.scatter(X_test[:,0], X_test[:,1], X_test[:,2], label='Test')
    ax0.legend(loc='upper right')

if args.plot_intermediate_steps:
    if args.test_dan and args.model_grid == 'rover':
        time_bins = datasets.time_bins_dan
    elif args.test_dan and (args.model_grid == 'full' or args.model_grid == 'both'):
        time_bins = datasets.time_bins_dan[:34]
    elif args.test_sim:
        time_bins = datasets.time_bins_sim
    # Plot principal components of training data
    fig, (ax5, ax6) = plt.subplots(nrows=1, ncols=2)
    if args.use_restricted_bins:
        time_bins_th = np.take(time_bins, range(17,34), axis=0)
    else:
        time_bins_th = time_bins
    ax5.step(time_bins_th, pca.components_[0][:len(time_bins_th)], where='post', linewidth=2, label='PC 1')
    ax5.step(time_bins_th, pca.components_[1][:len(time_bins_th)], where='post', linewidth=2, label='PC 2')
    ax5.step(time_bins_th, pca.components_[2][:len(time_bins_th)], where='post', linewidth=2, label='PC 3')
    ax5.legend(loc='upper right')
    ax5.set_xscale('log')
    ax5.set_title('Thermal Principal Components (Training Data)')
    ax5.set_xlabel('Time (us)')
    ax5.set_ylabel('Normalized Counts')
    if args.use_restricted_bins:
        time_bins_epi = np.take(time_bins, range(12,17), axis=0)
    else:
        time_bins_epi = time_bins
    ax6.step(time_bins_epi, pca.components_[0][-len(time_bins_epi):], where='post', linewidth=2, label='PC 1')
    ax6.step(time_bins_epi, pca.components_[1][-len(time_bins_epi):], where='post', linewidth=2, label='PC 2')
    ax6.step(time_bins_epi, pca.components_[2][-len(time_bins_epi):], where='post', linewidth=2, label='PC 3')
    ax6.legend(loc='upper right')
    ax6.set_xscale('log')
    ax6.set_title('Epithermal Principal Components (Training Data)')
    ax6.set_xlabel('Time (us)')
    ax6.set_ylabel('Normalized Counts')

if args.linear:
    from sklearn.linear_model import LinearRegression
    lr = LinearRegression()
    print np.mean(cross_val_score(lr, X_train, np.log(Y_train+1), cv=5, scoring=sklearn.metrics.make_scorer(r2_score)))
    
    lr.fit(X=X_train, y=np.log(Y_train+1))
    #lr.fit(X=X_train, y=Y_train)

    Y_pred = np.exp(lr.predict(X=X_test))-1
    #Y_pred = lr.predict(X=X_test)

    # Should this be allowed?
    Y_pred[np.where(Y_pred < 0)] = 0

    if args.plot_intermediate_steps:
        fig, rvis = plt.subplots(nrows=args.n_components, ncols=2)
        # Plot the regression lines for each PC
        for pc in range(args.n_components):
            # Plot H
            rvis[pc,0].scatter(X_train[:,pc], Y_train[:,0])
            rvis[pc,0].set_xlabel('Value along PC %d' % (pc+1))
            rvis[pc,0].set_ylabel('H Value (wt %)')
            xs = np.arange(X_train[:,pc].min(), X_train[:,pc].max(), 0.01)
            rvis[pc,0].plot(xs, xs*lr.coef_[0,pc]+lr.intercept_[0], color='r')
            # Plot Cl
            rvis[pc,1].scatter(X_train[:,pc], Y_train[:,1])
            rvis[pc,1].plot(xs, xs*lr.coef_[1,pc]+lr.intercept_[1], color='r')
            rvis[pc,1].set_xlabel('Value along PC %d' % (pc+1))
            rvis[pc,1].set_ylabel('Cl Value (wt %)')

    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)
    if args.plot_error_bars:
        ax1.errorbar(range(1, n_test+1), Y_test[:, 0], yerr=Y_test_error[:,0], ecolor='r', color='k', label='True value')
        ax1.set_title('Linear Regression Predictions for H Values ($R^2$=%f)' % r2_score(Y_test[:,0], Y_pred[:,0]))
    else:
        ax1.plot(range(1, n_test+1), Y_test[:, 0], color='k', label='True value')
        ax1.set_title('Linear Regression Predictions for H Values ($R^2$=%f)' % r2_score(Y_test[:,0], Y_pred[:,0]))
    ax1.plot(range(1, n_test+1), Y_pred[:, 0], color='k', linestyle='--', label='Predicted value')
    ax1.set_ylabel('H value (wt %)')
    ax1.set_xlabel('Test Example')
    ax1.legend(loc='upper right')
    
    if args.plot_error_bars:
        ax2.errorbar(range(1, n_test+1), Y_test[:, 1], yerr=Y_test_error[:,1], ecolor='r', color='k', label='True value')
        ax2.set_title('Linear Regression Predictions for Cl Values ($R^2$=%f)' % r2_score(Y_test[:,1], Y_pred[:,1]))
    else:
        ax2.plot(range(1, n_test+1), Y_test[:, 1], color='k', label='True value')
        ax2.set_title('Linear Regression Predictions for Cl Values ($R^2$=%f)' % r2_score(Y_test[:,1], Y_pred[:,1]))
    ax2.plot(range(1, n_test+1), Y_pred[:, 1], color='k', linestyle='--', label='Predicted value')
    ax2.set_ylabel('Cl value (wt %)')
    ax2.set_xlabel('Test Example')
    ax2.legend(loc='upper right')

    # Print accuracy scores
    if args.plot_error_bars:
        print "H accuracy = %f" % get_accuracy(Y_pred[:,0], Y_test[:,0], Y_test_error[:,0], epsilon=args.epsilon)[0]
        print "Cl accuracy = %f" % get_accuracy(Y_pred[:,1], Y_test[:,1], Y_test_error[:,1], epsilon=args.epsilon)[0]

        # Test how a random regressor would do
        import random
        h_rand = [random.uniform(0.0, 8.0) for i in range(len(Y_test[:,0]))]
        cl_rand = [random.uniform(0.0, 3.0) for i in range(len(Y_test[:,1]))]
        print "H random accuracy = %f" % get_accuracy(np.array(h_rand), Y_test[:,0], Y_test_error[:,0], epsilon=args.epsilon)[0]
        print "Cl random accuracy = %f" % get_accuracy(np.array(cl_rand), Y_test[:,1], Y_test_error[:,1], epsilon=args.epsilon)[0]
        print "H random R^2 = %f" % r2_score(np.array(h_rand), Y_test[:,0])
        print "H random R^2 = %f" % r2_score(np.array(cl_rand), Y_test[:,1])
        # print "Incorrect H:"
        # # mask = get_accuracy(Y_pred[:,0], Y_test[:,0], Y_test_error[:,0])[1]
        # # for i in range(len(mask)):
        # #     if mask[i] == 1:
        # #         print test_names[i]
        # # np.savetxt('mask.txt', mask)
    else:
        print "H accuracy = %f" % get_accuracy(Y_pred[:,0], Y_test[:,0], np.zeros(Y_pred[:,0].shape))[0]
        print "Cl accuracy = %f" % get_accuracy(Y_pred[:,1], Y_test[:,1], np.zeros(Y_pred[:,1].shape))[0]
        print "Incorrect H:"
        mask = get_accuracy(Y_pred[:,0], Y_test[:,0])[1]
        for i in range(len(mask)):
            if mask[i] == 1:
                print test_names[i]


    # Plot a scatter plot to show correlations
    fig, (ax3, ax4) = plt.subplots(nrows=1, ncols=2)
    ax3.scatter(Y_test[:, 0], Y_pred[:, 0], picker=True)
    ax3.set_ylabel('Predicted H value (wt %)')
    ax3.set_xlabel('Actual H value (wt %)')
    ax4.scatter(Y_test[:, 1], Y_pred[:, 1], picker=True)
    ax4.set_ylabel('Predicted Cl value (wt %)')
    ax4.set_xlabel('Actual Cl value (wt %)')
    # Allow user to click on points and print which measurement the point belongs to
    def onpick(event):
        ind = event.ind
        # print Y_test[ind[0]]
        print Y_test[ind]

    fig.canvas.mpl_connect('pick_event', onpick)

    if args.plot_intermediate_steps:
        fig, rvis_test = plt.subplots(nrows=args.n_components, ncols=2)
        # Plot the regression lines for each PC
        for pc in range(args.n_components):
            # Plot H
            rvis_test[pc,0].scatter(X_test[:,pc], Y_test[:,0])
            rvis_test[pc,0].set_xlabel('Value along PC %d' % (pc+1))
            rvis_test[pc,0].set_ylabel('H Value (wt %)')
            xs = np.arange(X_test[:,pc].min(), X_test[:,pc].max(), 0.01)
            rvis_test[pc,0].plot(xs, xs*lr.coef_[0,pc]+lr.intercept_[0], color='r')
            # Plot Cl
            rvis_test[pc,1].scatter(X_test[:,pc], Y_test[:,1])
            rvis_test[pc,1].plot(xs, xs*lr.coef_[1,pc]+lr.intercept_[1], color='r')
            rvis_test[pc,1].set_xlabel('Value along PC %d' % (pc+1))
            rvis_test[pc,1].set_ylabel('Cl Value (wt %)')

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

    elasticnet = ElasticNet(alpha=0.01)
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
    model.compile(loss='mse', optimizer='adam')
    model.fit(X_train, 
              np.log(1+Y_train), 
              epochs=400, 
              batch_size=25, 
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
