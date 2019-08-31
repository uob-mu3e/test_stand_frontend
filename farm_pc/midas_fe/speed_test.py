import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys

if (str(sys.argv[2]) == str(1)):
    print("halffull mode")
    path = "../../build/farm_pc/midas_fe/half_memory_content_"
    path += str(sys.argv[1]) + ".txt"
else:
    print("normal mode")
    path = "../../build/farm_pc/midas_fe/memory_content_"
    path += str(sys.argv[1]) + ".txt"
df = pd.read_csv(path, sep="\t", encoding="ISO-8859-1")

print("number: " + str(sys.argv[1]))


#df = df[df["counter"] != '00000000']
#df = df[df["counter"] != 0]

counter = [int(str(value), 16) for value in df["counter"].to_list()]
halfful = [int(str(value).replace(".0", ""), 16) for value in df["halfful"].to_list()]
nothalfful = [int(str(value), 16) for value in df["nothalfful"].to_list()]

print("len: " + str(len(counter)))

print("Max halffull: " + str(np.max(halfful)))
print("Max nothalffull: " + str(np.max(nothalfful)))

cnt = counter[0]
wrong_count = 0
list_of_data_loss = []
tmp_idx_wrong_count = 0
for idx, value in enumerate(counter):
    if (int(counter[idx-1]) < int(value)):
        cnt = value
    if(int(cnt) != int(value)):
        print(counter[idx-1])
        print(counter[idx])
        list_of_data_loss.append(counter[idx-1] - counter[idx] + 1) # counting from 0
        tmp_idx_wrong_count = idx
        wrong_count += 1
    cnt += 1
print("Total amount of data loss: " + str(np.sum(list_of_data_loss[1:]) * 256))

print("Number wrong counts: " + str(wrong_count))
if (str(sys.argv[2]) == str(1)):
    print("rate_half: " + str(float(int(sys.argv[1]) * 256 - np.max(halfful)/(np.max(halfful) + np.max(nothalfful)) * int(sys.argv[1]) * 256)))
else:
    print("rate: " + str(float(int(sys.argv[1])*256)))

plt.plot(counter)
if (str(sys.argv[2]) == str(1)):
    plt.savefig("half_plot_" + str(sys.argv[1]) + ".pdf")
else:
    plt.savefig("plot_" + str(sys.argv[1]) + ".pdf")
plt.show()
plt.close()
