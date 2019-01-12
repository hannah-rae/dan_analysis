import matplotlib
# matplotlib.use('TKAgg')

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from glob import glob
import argparse
import datasets

# plt.rc('font', family='Times New Roman')
# plt.rc('xtick', labelsize='x-small')
# plt.rc('ytick', labelsize='x-small')
plt.rc('font', family='Arial', size=10)

# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--n_components', type=int, default=3, help='number of principal components to use for PCA')
parser.add_argument('--dataset', help='which dataset to use: rover, polar, acs, or dan')
parser.add_argument('--normalize', action='store_true', help='normalize the data before PCA')
parser.add_argument('--traverse_manifold', action='store_true', help='show points across the 2D manifold of the first two PCs')
parser.add_argument('--traverse_components', action='store_true', help='show points across each principal axis')
parser.add_argument('--grid_size', type=int, default=5, help='number of points N to reconstruct in NxN grid of manifold')
parser.add_argument('--use_restricted_bins', action='store_true', help='only use bins 17-34 and 11-17 (counting from 1)')
parser.add_argument('--thermal_only', action='store_true', help='discard epithermal feature vector')
args = parser.parse_args()

if args.dataset == 'rover':
    X, Y = datasets.read_grid_data()
    if args.thermal_only:
        X = np.take(X, range(64), axis=1)
    time_bins = datasets.time_bins_dan
    n_bins = 64
elif args.dataset == 'acs':
    X, Y = datasets.read_acs_grid_data()
    if args.thermal_only:
        X = np.take(X, range(64), axis=1)
    time_bins = datasets.time_bins_dan[:-1]
    n_bins = 64
elif args.dataset == 'polar':
    X, Y = datasets.read_polar_data()
    if args.thermal_only:
        X = np.take(X, range(161), axis=1)
    time_bins = datasets.time_bins_sim
    n_bins = 161
elif args.dataset == 'dan':
    X, Y, err, names = datasets.read_dan_data()
    if args.thermal_only:
        X = np.take(X, range(64), axis=1)
    time_bins = datasets.time_bins_dan
    n_bins = 64
elif args.dataset == 'both_models':
    X_rover, Y_rover = datasets.read_grid_data(limit_2000us=True)
    X_full, Y_full = datasets.read_sim_data(use_dan_bins=True)
    X = np.concatenate([X_full, X_rover])
    Y = np.concatenate([Y_rover, Y_full])
    time_bins = datasets.time_bins_dan
    n_bins = 64

if args.normalize:
    X = datasets.normalize_counts(X)

if args.use_restricted_bins and args.thermal_only:
    X = np.take(X, range(17, 34), axis=1)
elif args.use_restricted_bins:
    X = np.take(X, range(17, 34)+range(n_bins+12, n_bins+17), axis=1)

pca = PCA(n_components=args.n_components)
X_t = pca.fit_transform(X)

if args.n_components >= 3:
    fig = plt.figure()
    ax1 = fig.add_subplot(1, 3, 1, projection='3d')
    ax2 = fig.add_subplot(1, 3, 2)
    ax3 = fig.add_subplot(1, 3, 3)
    ax1.set_xlabel('PC 1')
    ax1.set_ylabel('PC 2')
    ax1.set_zlabel('PC 3')
    points = ax1.scatter(X_t[:,0], X_t[:,1], X_t[:,2], facecolors=["C0"]*X_t.shape[0], edgecolors=["C0"]*X_t.shape[0], picker=1)
    fc = points.get_facecolors()
elif args.n_components == 2:
    fig = plt.figure()
    ax1 = fig.add_subplot(1, 3, 1)
    ax2 = fig.add_subplot(1, 3, 2)
    ax3 = fig.add_subplot(1, 3, 3)
    ax1.set_xlabel('PC 1')
    ax1.set_ylabel('PC 2')
    points = ax1.scatter(X_t[:,0], X_t[:,1], picker=True)

if args.traverse_components:
    # Thermal curves
    fig2, axes = plt.subplots(nrows=3, ncols=args.grid_size, sharex=True, sharey=True)
    
    pc1_min = np.min(X_t[:,0])
    pc1_max = np.max(X_t[:,0])
    pc2_min = np.min(X_t[:,1])
    pc2_max = np.max(X_t[:,1])
    pc3_min = np.min(X_t[:,2])
    pc3_max = np.max(X_t[:,2])
    pc1_pts = np.linspace(pc1_min, pc1_max, args.grid_size)
    pc2_pts = np.linspace(pc2_min, pc2_max, args.grid_size)
    pc3_pts = np.linspace(pc3_min, pc3_max, args.grid_size)
    
    for j, p in enumerate(pc1_pts):
        inv = pca.inverse_transform([p, 0, 0])
        axes[0,j].step(time_bins, inv[:n_bins], where='post', linewidth=2, color='k')
        axes[0,j].set_xscale('log')
        axes[0,j].tick_params(
            axis='both',          # changes apply to the x-axis
            which='both',      # both major and minor ticks are affected
            bottom=False,      # ticks along the bottom edge are off
            top=False,         # ticks along the top edge are off
            left=False,      # ticks along the bottom edge are off
            right=False,         # ticks along the top edge are off
            labelleft=False,
            labelbottom=False) # labels along the bottom edge are off

    for j, p in enumerate(pc2_pts):
        inv = pca.inverse_transform([0, p, 0])
        axes[1,j].step(time_bins, inv[:n_bins], where='post', linewidth=2, color='k')
        axes[1,j].set_xscale('log')
        axes[1,j].tick_params(
            axis='both',          # changes apply to the x-axis
            which='both',      # both major and minor ticks are affected
            bottom=False,      # ticks along the bottom edge are off
            top=False,         # ticks along the top edge are off
            left=False,      # ticks along the bottom edge are off
            right=False,         # ticks along the top edge are off
            labelleft=False,
            labelbottom=False) # labels along the bottom edge are off

    for j, p in enumerate(pc3_pts):
        inv = pca.inverse_transform([0, 0, p])
        axes[2,j].step(time_bins, inv[:n_bins], where='post', linewidth=2, color='k')
        axes[2,j].set_xscale('log')
        axes[2,j].tick_params(
            axis='both',          # changes apply to the x-axis
            which='both',      # both major and minor ticks are affected
            bottom=False,      # ticks along the bottom edge are off
            top=False,         # ticks along the top edge are off
            left=False,      # ticks along the bottom edge are off
            right=False,         # ticks along the top edge are off
            labelleft=False,
            labelbottom=False) # labels along the bottom edge are off
    fig2.suptitle('Homogeneous Model Grid Dataset Manifold Traversal\nThermal Die-Away Curves')

    if not args.thermal_only:
        fig3, axes_ep = plt.subplots(nrows=3, ncols=args.grid_size, sharex=True, sharey=True)
        for j, p in enumerate(pc1_pts):
            inv = pca.inverse_transform([p, 0, 0])
            axes_ep[0,j].step(time_bins, inv[n_bins:], where='post', linewidth=2, color='k')
            axes_ep[0,j].set_xscale('log')
            axes_ep[0,j].tick_params(
                axis='both',          # changes apply to the x-axis
                which='both',      # both major and minor ticks are affected
                bottom=False,      # ticks along the bottom edge are off
                top=False,         # ticks along the top edge are off
                left=False,      # ticks along the bottom edge are off
                right=False,         # ticks along the top edge are off
                labelleft=False,
                labelbottom=False) # labels along the bottom edge are off

        for j, p in enumerate(pc2_pts):
            inv = pca.inverse_transform([0, p, 0])
            axes_ep[1,j].step(time_bins, inv[n_bins:], where='post', linewidth=2, color='k')
            axes_ep[1,j].set_xscale('log')
            axes_ep[1,j].tick_params(
                axis='both',          # changes apply to the x-axis
                which='both',      # both major and minor ticks are affected
                bottom=False,      # ticks along the bottom edge are off
                top=False,         # ticks along the top edge are off
                left=False,      # ticks along the bottom edge are off
                right=False,         # ticks along the top edge are off
                labelleft=False,
                labelbottom=False) # labels along the bottom edge are off

        for j, p in enumerate(pc3_pts):
            inv = pca.inverse_transform([0, 0, p])
            axes_ep[2,j].step(time_bins, inv[n_bins:], where='post', linewidth=2, color='k')
            axes_ep[2,j].set_xscale('log')
            axes_ep[2,j].tick_params(
                axis='both',          # changes apply to the x-axis
                which='both',      # both major and minor ticks are affected
                bottom=False,      # ticks along the bottom edge are off
                top=False,         # ticks along the top edge are off
                left=False,      # ticks along the bottom edge are off
                right=False,         # ticks along the top edge are off
                labelleft=False,
                labelbottom=False) # labels along the bottom edge are off
        fig3.suptitle('Homogeneous Model Grid Dataset Manifold Traversal\nEpithermal Die-Away Curves')

if args.traverse_manifold:
    # Thermal curves
    fig2, axes = plt.subplots(nrows=args.grid_size, ncols=args.grid_size, sharex=True, sharey=True)
    pc1_min = np.min(X_t[:,0])
    pc1_max = np.max(X_t[:,0])
    pc2_min = np.min(X_t[:,1])
    pc2_max = np.max(X_t[:,1])
    pc1_pts = np.linspace(pc1_min, pc1_max, args.grid_size)
    pc2_pts = np.linspace(pc2_min, pc2_max, args.grid_size)
    for i, x in enumerate(pc1_pts):
        for j, y in enumerate(pc2_pts):
            inv = pca.inverse_transform([x, y])
            axes[i,j].step(time_bins, inv[:n_bins], where='post', linewidth=2, color='k')
            axes[i,j].set_xscale('log')
            # axes[i,j].set_xlabel('Time bin (us)')
            # axes[i,j].set_ylabel('Counts')
            axes[i,j].tick_params(
                axis='both',          # changes apply to the x-axis
                which='both',      # both major and minor ticks are affected
                bottom=False,      # ticks along the bottom edge are off
                top=False,         # ticks along the top edge are off
                left=False,      # ticks along the bottom edge are off
                right=False,         # ticks along the top edge are off
                labelleft=False,
                labelbottom=False) # labels along the bottom edge are off
    fig2.suptitle('Homogeneous Model Grid Dataset Manifold Traversal\nThermal Die-Away Curves')
    # Epithermal curves
    fig3, axes_ep = plt.subplots(nrows=args.grid_size, ncols=args.grid_size, sharex=True, sharey=True)
    pc1_min = np.min(X_t[:,0])
    pc1_max = np.max(X_t[:,0])
    pc2_min = np.min(X_t[:,1])
    pc2_max = np.max(X_t[:,1])
    pc1_pts = np.linspace(pc1_min, pc1_max, args.grid_size)
    pc2_pts = np.linspace(pc2_min, pc2_max, args.grid_size)
    for i, x in enumerate(pc1_pts):
        for j, y in enumerate(pc2_pts):
            inv = pca.inverse_transform([x, y])
            axes_ep[i,j].step(time_bins, inv[n_bins:], where='post', linewidth=2, color='k')
            axes_ep[i,j].set_xscale('log')
            # axes_ep[i,j].set_xlabel('Time bin (us)')
            # axes_ep[i,j].set_ylabel('Counts')
            axes_ep[i,j].tick_params(
                axis='both',          # changes apply to the x-axis
                which='both',      # both major and minor ticks are affected
                bottom=False,      # ticks along the bottom edge are off
                top=False,         # ticks along the top edge are off
                left=False,      # ticks along the bottom edge are off
                right=False,         # ticks along the top edge are off
                labelleft=False,
                labelbottom=False) # labels along the bottom edge are off
    fig3.suptitle('Homogeneous Model Grid Dataset Manifold Traversal\nEpithermal Die-Away Curves')



# Plot original data
def plot_original(i):
    h, cl = Y[i]
    if args.dataset == 'acs':
        ax2.step(time_bins, X[i, :n_bins], where='post', linewidth=2,label=' %0.2f H  %0.2f ACS' % (h, cl))
        ax3.step(time_bins, X[i, n_bins:], where='post', linewidth=2,label=' %0.2f H  %0.2f ACS' % (h, cl))
    elif args.dataset == 'polar':
        ax2.step(time_bins, X[i, :n_bins], where='post', linewidth=2,label=' %0.2f H  %0.2f ACS' % (h, cl))
        ax3.step(time_bins, X[i, n_bins:], where='post', linewidth=2,label=' %0.2f H  %0.2f ACS' % (h, cl))
    elif args.use_restricted_bins:
        time_bins_th = np.take(time_bins, range(17,34), axis=0)
        time_bins_epi = np.take(time_bins, range(12,17), axis=0)
        ax2.step(time_bins_th, X[i, :len(time_bins_th)], where='post', linewidth=2,label=' %0.2f H  %0.2f Cl' % (h, cl))
        ax3.step(time_bins_epi, X[i, len(time_bins_th):], where='post', linewidth=2,label=' %0.2f H  %0.2f Cl' % (h, cl))
    else:
        ax2.step(time_bins, X[i, :n_bins], where='post', linewidth=2,label=' %0.2f H  %0.2f Cl' % (h, cl))
        ax3.step(time_bins, X[i, n_bins:], where='post', linewidth=2,label=' %0.2f H  %0.2f Cl' % (h, cl))
    ax2.set_xscale('log')
    ax2.legend(loc='lower left')
    ax2.set_xlabel('Time (us)')
    ax2.set_ylabel('Counts')
    ax2.set_title("Thermal Neutron Die-Away Curve")
    ax2.set_ylim(np.min(X), np.max(X))
    ax3.set_xscale('log')
    ax3.legend(loc='lower right')
    ax3.set_xlabel('Time (us)')
    ax3.set_ylabel('Counts')
    ax3.set_title("Epithermal Neutron Die-Away Curve")
    ax3.set_ylim(np.min(X), np.max(X))

def plot_curves(indexes):
    ax2.cla()
    ax3.cla()
    new_fc = fc.copy()
    for i in indexes: # might be more than one point if ambiguous click
        new_fc[i,:] = (1, 0, 0, 1)
        points._facecolor3d = new_fc
        points._edgecolor3d = new_fc
        plot_original(i)
    fig.canvas.draw_idle()
    #plt.draw()

# Allow user to click on points and print which measurement the point belongs to
def onpick(event):
    ind = event.ind
    print Y[ind]
    if args.dataset == 'dan':
        print names[ind]
    plot_curves(list(ind))

fig.canvas.mpl_connect('pick_event', onpick)

plt.show()