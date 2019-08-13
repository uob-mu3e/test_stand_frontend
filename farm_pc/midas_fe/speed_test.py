import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import collections

for number in range(150, 151):
    df = pd.read_csv("../../build/farm_pc/midas_fe/memory_content_"+ str(number) + ".txt", sep="\t", encoding="ISO-8859-1")
    
    print("number: " + str(number))

    df = df[df["counter"] != '00000000']
    df = df[df["counter"] != 0]

    counter = [int(str(value), 16) for value in df["counter"].to_list()]
    halfful = [int(str(value).replace(".0",""), 16) for value in df["halfful"].to_list()]
    nothalfful = [int(str(value), 16) for value in df["nothalfful"].to_list()]

    print("len: " + str(len(counter)))

    min_idx = np.argmin(counter)
    max_idx = np.argmax(counter)
    if len(counter) == len(counter[min_idx:]):
        print("right order")
    else:
        counter = np.concatenate((counter[min_idx:], counter[:max_idx+1]), axis=None)

    cnt = 0
    min_value = counter[min_idx]
    for value in counter:
        if(cnt != value - min_value):
            print("wrong count")
            break
        cnt += 1



    print("Data rate: " + str(number / int(str('FF'), 16) * 250*256 - (np.max(halfful) - np.min(halfful))/(np.max(nothalfful) - np.min(nothalfful)) * 256))

    plt.plot(counter)
    plt.savefig("plot_" + str(number) + ".pdf")
    plt.close()