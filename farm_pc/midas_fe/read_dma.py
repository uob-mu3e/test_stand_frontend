import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import collections

df = pd.read_csv("../../build/farm_pc/midas_fe/memory_content.txt", sep="\t", encoding="ISO-8859-1")

#df["newFraction"] = df["newFraction"].apply(lambda x: int(x, 16))

df["mod_8"] = df["idx"].mod(8)


state = np.array(df.loc[(df.mod_8 == 2)]["data"].tolist())
ram_add = df.loc[(df.mod_8 == 3)]["data"].tolist()
fifo = df.loc[(df.mod_8 == 4)]["data"].tolist()
gen_data = df.loc[(df.mod_8 == 5)]["data"].tolist()
data = np.array(df.loc[(df.mod_8 == 5)]["data"].tolist())
length = np.array(df.loc[(df.mod_8 == 6)]["data"].tolist())

print(length)
print(data)
read_idx = 1
events_right = 0
events_wrong = 0
counter = 0

while read_idx + 2 < len(length):
    if data[read_idx + 1] == "0000009C" and data[read_idx] == "0000009C" and data[read_idx + 2] == "E80000BC":
        events_right += 1
        if events_right == 2:
            print("WRONG")
            print(events_right)
            print(length[read_idx])
            print(length[read_idx-7:read_idx+25])
            print(data[read_idx-7:read_idx+25])
            print(read_idx)
    else:
        events_wrong += 1
        if events_wrong == 2:
            print("WRONG")
            print(events_right)
            print(length[read_idx])
            print(length[read_idx-7:read_idx+21])
            print(data[read_idx-7:read_idx+21])
            print(read_idx)
    read_idx += int(length[read_idx], 16)




print(events_right)
print(events_wrong)

list_of_length = []
list_of_counts = []
old_length = 0
counter = 0
for idx, value in enumerate(data):
    if idx == len(data) - 1:
        continue
    if value == "0000009C" and data[idx+1] == "0000009C":
        list_of_length.append(old_length)
        list_of_counts.append(counter)
        counter = 0
    counter += 1
    old_length = int(length[idx], 16)

plt.plot(range(len(list_of_counts)), list_of_counts, label="COUNTS")
plt.plot(range(len(list_of_length)), list_of_length, label="LENGTH")
plt.legend()
plt.show()

"""















#plt.hist(state)
#plt.show()
#plt.close()

#d = {"data": data[:1000], "length": length[:1000]}
#new_df = pd.DataFrame(data=d)

#new_df.to_html('/home/martin/test.html')

list_of_idx = []
list_of_length = []
idxlength = 0
saw_9c = 0
event_id_miss_count = []
idx_last_misscount = 0




for idx, value in enumerate(data):
    if value == "0000009C":
        for le in range(int(length[idx - 1], 16)):



    idxlength += 1
    if value == "0000009C" and saw_9c == 0:
        if idxlength is not int(length[idx - 1], 16):

            idxfrom = idx - 20
            idxto = idx + 20

            print(length[idx-1], length[idx], length[idx+1])
            print(data[idx-1], data[idx], data[idx+1])
            print(data[idxfrom:idxto])
            print(length[idxfrom:idxto])
            print(idxlength)
            print(int(length[idx - 1], 16))
            print(idx)
            print(len(list_of_idx))
            #if idx_last_misscount != 0:
            event_id_miss_count.append(float(len(list_of_idx) - idx_last_misscount))
            idx_last_misscount = len(list_of_idx)
        saw_9c = 1
        list_of_idx.append(idxlength)
        list_of_length.append(int(length[idx - 1], 16))


        if len(list_of_idx) == 3:
            print("NORMAL")
            print(length[idx-1], length[idx], length[idx+1])
            print(data[idx-1], data[idx], data[idx+1])
            print(data[idxfrom:idxto])
            print(length[idxfrom:idxto])
            print(idxlength)
            print(int(length[idx - 1], 16))
            print(idx)
            print(len(list_of_idx))
        idxlength = 0

    else:
        saw_9c = 0

plt.hist(event_id_miss_count)
plt.show()
plt.close()

print("Are counts and data equal? : " + str(collections.Counter(list_of_idx[-1:]) == collections.Counter(list_of_length[-1:])))
plt.plot(range(len(list_of_idx)), list_of_idx, label="IDX")
plt.plot(range(len(list_of_length)), list_of_length, label="LENGTH")
plt.legend()
plt.show()
plt.close()
plt.hist2d(list_of_idx, list_of_length, bins=100, norm=mcolors.PowerNorm(0.8))
plt.show()
"""
