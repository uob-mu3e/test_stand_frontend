import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import collections
import sys
import os


miss_counts = []
miss_counts_half = []
rate = []
rate_half = []
for idx in range(1, 256):

    if not os.path.isfile("../../build/farm_pc/midas_fe/half_memory_content_"+ str(idx) + ".txt"): continue
    if not os.path.isfile("../../build/farm_pc/midas_fe/memory_content_"+ str(idx) + ".txt"): continue

    df_half = pd.read_csv("../../build/farm_pc/midas_fe/half_memory_content_"+ str(idx) + ".txt", sep="\t", encoding="ISO-8859-1")
    df = pd.read_csv("../../build/farm_pc/midas_fe/memory_content_"+ str(idx) + ".txt", sep="\t", encoding="ISO-8859-1")

    if df_half.shape[0] == 0: continue
    if df.shape[0] == 0: continue

    print("File: " + str(idx))

    df = df[df["counter"] != '00000000']
    df = df[df["counter"] != 0]

    df_half = df_half[df_half["counter"] != '00000000']
    df_half = df_half[df_half["counter"] != 0]

    counter = [int(str(value), 16) for value in df["counter"].to_list()]

    counter_half = [int(str(value), 16) for value in df_half["counter"].to_list()]
    halfful_half = [int(str(value).replace(".0",""), 16) for value in df_half["halfful"].to_list()]
    nothalfful_half = [int(str(value), 16) for value in df_half["nothalfful"].to_list()]

    cnt = counter[0]
    wrong_count = 0
    for jdx, value in enumerate(counter):
        if (int(counter[jdx-1]) < int(value)):
            cnt = value
        if(int(cnt) != int(value)):
            wrong_count += 1
        cnt += 1

    cnt_half = counter_half[0]
    wrong_count_half = 0
    for kdx, value in enumerate(counter_half):
        if (int(counter_half[kdx-1]) < int(value)):
            cnt_half = value
        if(int(cnt_half) != int(value)):
            wrong_count_half += 1
        cnt_half += 1

    if wrong_count == 1: wrong_count = 0
    if wrong_count_half == 1: wrong_count_half = 0

    miss_counts.append(wrong_count)
    miss_counts_half.append(wrong_count_half)

    print("rate: " + str(float(idx*250)))
    print("rate_half: " + str(float(idx*250 - (np.max(halfful_half) - np.min(halfful_half))/(np.max(halfful_half) - np.min(halfful_half) + np.max(nothalfful_half) - np.min(nothalfful_half)) * 256)))

    rate.append(float(idx*250))
    rate_half.append(float(idx*250 - (np.max(halfful_half) - np.min(halfful_half))/(np.max(halfful_half) - np.min(halfful_half) + np.max(nothalfful_half) - np.min(nothalfful_half)) * 256))

np.save("miss_counts", miss_counts)
np.save("miss_counts_half", miss_counts_half)

np.save("rate_total", rate)
np.save("rate_half_total", rate_half)

plt.plot(rate, miss_counts, 'o', color='black');
plt.show()
plt.close()

plt.plot(rate_half, miss_counts_half, 'o', color='black');
plt.show()
plt.close()
