import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from multiprocessing import Pool





def get_rates(idx):
    if not os.path.isfile("../../build/farm_pc/midas_fe/half_memory_content_" + str(idx) + ".txt"): return -999, -999, -999, -999, -999
    if not os.path.isfile("../../build/farm_pc/midas_fe/memory_content_" + str(idx) + ".txt"): return -999, -999, -999, -999, -999

    df_half = pd.read_csv("../../build/farm_pc/midas_fe/half_memory_content_"+ str(idx) + ".txt", sep="\t", encoding="ISO-8859-1")
    df = pd.read_csv("../../build/farm_pc/midas_fe/memory_content_"+ str(idx) + ".txt", sep="\t", encoding="ISO-8859-1")

    if df_half.shape[0] == 0: return -999, -999, -999, -999, -999
    if df.shape[0] == 0: return -999, -999, -999, -999, -999

    print("File: " + str(idx))

    counter = [int(str(value), 16) for value in df["counter"].to_list()]

    counter_half = [int(str(value), 16) for value in df_half["counter"].to_list()]
    halfful_half = [int(str(value).replace(".0",""), 16) for value in df_half["halfful"].to_list()]
    nothalfful_half = [int(str(value), 16) for value in df_half["nothalfful"].to_list()]

    cnt = counter[0]
    wrong_count = 0
    list_of_data_loss = []
    for jdx, value in enumerate(counter):
        if (int(counter[jdx-1]) < int(value)):
            cnt = value
        if(int(cnt) != int(value)):
            wrong_count += 1
            list_of_data_loss.append(counter[jdx-1] - counter[jdx] + 1) # counting from 0
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


    tmp_rate = float(idx*256)
    tmp_rate_half = float(idx*256 - np.max(halfful_half)/(np.max(halfful_half) + np.max(nothalfful_half)) * idx * 256)

    tmp_data_loss = np.sum(list_of_data_loss) * 256

    return wrong_count, wrong_count_half, tmp_rate, tmp_rate_half, tmp_data_loss


with Pool(10) as p:
    data = p.map(get_rates, range(1, 256))

    miss_counts = []
    miss_counts_half = []
    rate = []
    rate_half = []
    data_loss = []
    
    for value in data:
        miss_counts.append(value[0])
        miss_counts_half.append(value[1])
        rate.append(value[2])
        rate_half.append(value[3])
        data_loss.append(value[4])

    np.save("miss_counts_2", miss_counts)
    np.save("miss_counts_half_2", miss_counts_half)

    np.save("rate_total_2", rate)
    np.save("rate_half_total_2", rate_half)

    np.save("data_loss_2", data_loss)

plt.plot(rate, data_loss)
plt.show()
plt.close()

plt.plot(rate, miss_counts, 'o', color='black');
plt.show()
plt.close()

plt.plot(rate_half, miss_counts_half, 'o', color='black');
plt.show()
plt.close()
