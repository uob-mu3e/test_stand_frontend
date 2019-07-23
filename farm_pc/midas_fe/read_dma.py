import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import collections

df = pd.read_csv("memory_content.txt", sep="\t", encoding="ISO-8859-1")

#df["newFraction"] = df["newFraction"].apply(lambda x: int(x, 16))

df["mod_8"] = df["idx"].mod(8)

data = df.loc[(df.mod_8 == 6)]["data"].tolist()
length = df.loc[(df.mod_8 == 7)]["data"].tolist()

d = {"data": data[:1000], "length": length[:1000]}
new_df = pd.DataFrame(data=d)

new_df.to_html('/home/martin/test.html')

list_of_idx = []
list_of_length = []
idxlength = 0
saw_9c = 0
for idx, value in enumerate(data):
    idxlength += 1
    if value == "0000009C" and saw_9c == 0:
        saw_9c = 1
        list_of_idx.append(idxlength)
        list_of_length.append(int(length[idx - 1], 16))
        idxlength = 0
    else:
        saw_9c = 0

print("Are counts and data equal? : " + str(collections.Counter(list_of_idx[-1:]) == collections.Counter(list_of_length[-1:])))
plt.plot(range(len(list_of_idx)), list_of_idx, label="IDX")
plt.plot(range(len(list_of_length)), list_of_length, label="LENGTH")
plt.legend()
plt.show()
