import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import collections
import sys

if (str(sys.argv[2]) == str(1)):
    print("half")
    df = pd.read_csv("../../build/farm_pc/midas_fe/half_memory_content_"+ str(sys.argv[1]) + ".txt", sep="\t", encoding="ISO-8859-1")
else:
    df = pd.read_csv("../../build/farm_pc/midas_fe/memory_content_"+ str(sys.argv[1]) + ".txt", sep="\t", encoding="ISO-8859-1")
#    print(df)
print("number: " + str(sys.argv[1]))

df = df[df["counter"] != '00000000']
df = df[df["counter"] != 0]

counter = [int(str(value), 16) for value in df["counter"].to_list()]
halfful = [int(str(value).replace(".0",""), 16) for value in df["halfful"].to_list()]
nothalfful = [int(str(value), 16) for value in df["nothalfful"].to_list()]

print("len: " + str(len(counter)))

#    plt.plot(counter)
#    plt.show()
#    plt.close()

cnt = counter[0]
wrong_count = 0
for idx, value in enumerate(counter):
    if (int(counter[idx-1]) < int(value)):
        cnt = value
    if(int(cnt) != int(value)):
#            print(value - int(counter[min_idx]))
#            print(counter[idx-1] - int(counter[min_idx]))
#            print(counter[idx])
        wrong_count += 1





    cnt += 1





print("Number wrong counts: " + str(wrong_count))
if (str(sys.argv[2]) == str(1)):
    print("rate_half: " + str(float(int(sys.argv[1])*250 - (np.max(halfful) - np.min(halfful))/(np.max(halfful) - np.min(halfful) + np.max(nothalfful) - np.min(nothalfful)) * 256)))
else:
    print("rate: " + str(float(int(sys.argv[1])*250)))

plt.plot(counter)
if (str(sys.argv[2]) == str(1)):
    plt.savefig("half_plot_" + str(sys.argv[1]) + ".pdf")
else:
    plt.savefig("plot_" + str(sys.argv[1]) + ".pdf")
plt.close()
