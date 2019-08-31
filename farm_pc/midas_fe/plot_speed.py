import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as color
from matplotlib.ticker import FormatStrFormatter

blue = (57/255, 106/255, 177/255)
orange = (218/255, 124/255, 48/255)
green = (62/255, 150/255, 81/255)
red = (204/255, 37/255, 41/255)
black = (83/255, 81/255, 84/255)
black_bar = (128/255, 133/255, 133/255)

total_data_rate = 3.2 * 10 ** 7 # 32 MBit

miss_counts = np.load("miss_counts_2.npy")
miss_counts_half = np.load("miss_counts_half_2.npy")
rate = np.load("rate_total_2.npy")
rate_half = np.load("rate_half_total_2.npy")
data_loss = np.load("data_loss_2.npy")

default_idx = np.argwhere(miss_counts == -999)
default_idx_half = np.argwhere(miss_counts_half == -999)

miss_counts = np.delete(miss_counts, default_idx)
miss_counts_half = np.delete(miss_counts_half, default_idx_half)
rate = np.delete(rate, default_idx)
rate_half = np.delete(rate_half, default_idx_half)
data_loss = np.delete(data_loss, default_idx)

x = 0
for i, j in enumerate(miss_counts_half):
    if j != 0:
        x = rate_half[i-1]/1000
        break
if x == 0: x = np.max(rate_half)/1000

ber = []
for idx, value in enumerate(data_loss):
	if value == 0 or miss_counts[idx] == 0:
		ber.append(3/total_data_rate)
	else:
		tmp_total_rate = value + total_data_rate - miss_counts[idx] * 256
		ber.append(value/tmp_total_rate)

# make two subplots for splitting the y axis
f, (ax, ax2) = plt.subplots(2, 1, sharex=True)

# plot the same data on both axes
ax.scatter(rate/1000, ber, 
	marker='x', 
	#facecolors='none', 
	color=blue, 
	label='data rate scan')
ax.scatter(rate_half/1000, [3/total_data_rate for i in rate_half], 
	marker='.', 
	facecolors='none', 
	color=green, 
	label='date rate with halffull flag')

ax2.scatter(rate/1000, ber, 
	marker='x', 
	#facecolors='none', 
	color=blue, 
	label='data rate scan')
ax2.scatter(rate_half/1000, [3/total_data_rate for i in rate_half], 
	marker='o', 
	facecolors='none', 
	color=green, 
	label='date rate with halffull flag')

# zoom-in / limit the view to different portions of the data
ax.set_ylim(.78, 1.)  # rate scan only
ax2.set_ylim(0.8*1/10**7, 1.1*1/10**7)  # half full flag data

ax.set_xlim(20, 64)
ax2.set_xlim(20, 64)

# hide the spines between ax and ax2
ax.spines['bottom'].set_visible(False)
ax2.spines['top'].set_visible(False)
ax.xaxis.tick_top()
ax.tick_params(labeltop='off')  # don't put tick labels at the top
ax2.xaxis.tick_bottom()

d = .015  # how big to make the diagonal lines in axes coordinates
# arguments to pass to plot, just so we don't keep repeating them
kwargs = dict(transform=ax.transAxes, color='k', clip_on=False)
ax.plot((-d, +d), (-d, +d), **kwargs)        # top-left diagonal
ax.plot((1 - d, 1 + d), (-d, +d), **kwargs)  # top-right diagonal

kwargs.update(transform=ax2.transAxes)  # switch to the bottom axes
ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)  # bottom-left diagonal
ax2.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)  # bottom-right diagonal

ax.axvline(x=-999, color=red, label="max. data rate " + str(round(x)) + " GB/s")
ax2.axvline(x=x, color=red, label="max. data rate " + str(round(x)) + " GB/s")

#ax.legend()
ax2.legend(loc='lower right')
ax.set_title("DMA speed test with 32 MBit of data")
plt.xlabel("GB/s")
ax.set_ylabel("BER")
ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
ax2.set_ylabel("upper limit BER 95 % CL")
plt.savefig('dma_speed_test.pdf')
plt.show()
plt.close()
