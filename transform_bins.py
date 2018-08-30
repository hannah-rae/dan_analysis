import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.misc import factorial
import lmfit
from scipy.stats import norm

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

# Get the centers of each time bin
# time_bins_sim_c = [(time_bins_sim[i] + time_bins_sim[i+1])/2. for i in range(len(time_bins_sim)-1)]
# time_bins_dan_c = [(time_bins_dan[i] + time_bins_dan[i+1])/2. for i in range(len(time_bins_dan)-1)]

counts_th = np.loadtxt('/Users/hannahrae/Desktop/counts_th.txt')
counts_epi = np.loadtxt('/Users/hannahrae/Desktop/counts_epi.txt')

fig, (ax1, ax2) = plt.subplots(2)
def y(x, x1, x2, y1, y2):
    b = (x - x1)/(x2 - x1)
    a = 1 - b
    return a*y1 + b*y2
ax1.step(time_bins_sim, counts_th, where='post', linewidth=2, label='Simulation data')
for i in range(len(time_bins_sim)-1):
    x_interp = np.arange(time_bins_sim[i], time_bins_sim[i+1], 0.01)
    y_interp = [y(x=x, x1=time_bins_sim[i], x2=time_bins_sim[i+1], y1=counts_th[i], y2=counts_th[i+1]) for x in x_interp]
    #y_interp = np.linspace(counts_th[i], counts_th[i+1], len(x_interp))
    ax1.plot(x_interp, y_interp, 'r-', lw=1)
ax1.set_ylabel('Thermal Neutron Counts')
ax1.set_xlabel('Time bin (us)')
ax1.legend(loc='upper right')

ax2.step(time_bins_sim, counts_epi, where='post', linewidth=2, label='Simulation data')
for i in range(len(time_bins_sim)-1):
    x_interp = np.arange(time_bins_sim[i], time_bins_sim[i+1], 0.01)
    y_interp = [y(x=x, x1=time_bins_sim[i], x2=time_bins_sim[i+1], y1=counts_epi[i], y2=counts_epi[i+1]) for x in x_interp]
    #y_interp = np.linspace(counts_epi[i], counts_epi[i+1], len(x_interp))
    ax2.plot(x_interp, y_interp, 'r-', lw=1)
ax2.set_ylabel('Epithermal Neutron Counts')
ax2.set_xlabel('Time bin (us)')
ax2.legend(loc='upper right')
plt.show()