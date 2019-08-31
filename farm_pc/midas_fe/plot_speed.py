import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as color

blue = (57/255, 106/255, 177/255)
orange = (218/255, 124/255, 48/255)
green = (62/255, 150/255, 81/255)
red = (204/255, 37/255, 41/255)
black = (83/255, 81/255, 84/255)
black_bar = (128/255, 133/255, 133/255)

miss_counts = np.load("miss_counts.npy")
miss_counts_half = np.load("miss_counts_half.npy")
miss_counts = miss_counts[-len(miss_counts_half):]
rate = np.load("rate_total.npy")
rate = rate[-len(miss_counts_half):]
rate_half = np.load("rate_half_total.npy")
print(rate)
x = 0
for i, j in enumerate(miss_counts_half):
    if j != 0:
        x = rate_half[i-1]/1000
        break


plt.plot(rate/1000, miss_counts, 'o', color=blue, label='frac scan');
plt.plot(rate_half/1000, miss_counts_half, 'o', color=green, label='use halfful');
plt.axvline(x=x, color=red, label=str(round(x)) + " GB/s")
plt.legend()
plt.title("DMA speed")
plt.xlabel("GB/s")
plt.ylabel("# miss counts")
plt.show()
plt.close()
