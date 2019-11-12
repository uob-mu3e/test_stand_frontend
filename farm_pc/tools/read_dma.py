import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.animation as animation
import numpy as np
import collections
import binascii

fig = plt.figure()
ax1 = fig.add_subplot(1,1,1)

def animate(i):
	df = pd.read_csv("../../build/farm_pc/midas_fe/memory_content.txt", sep="\t", encoding="ISO-8859-1")
	data = np.array(df["data"])
	course_counter = []
	course_fine = []
	next = 0
	for d in data:
		if "0000039C" == d:
			next = 0
		if next == 1:
			binary = "{0:8b}".format(int(d, 16))
			if len(binary) == 27:
				binary_list = np.array(list(binary))
				course_fine.append(int(''.join(binary_list[21:26]),2))
				course_counter.append(int(''.join(binary_list[7:21]),2))
		if "E80020BC" == d:
			next = 1
	bins = np.linspace(0, 40, 100)

	ax1.clear()
	ax1.hist(course_counter, bins=bins, alpha=0.5, label='course')
	ax1.hist(course_fine, bins=bins, alpha=0.5, label='fine')
	ax1.legend(loc='upper right')

ani = animation.FuncAnimation(fig, animate, interval=1000)
plt.show()

"""

df = pd.read_csv("../../build/farm_pc/midas_fe/memory_content.txt", sep="\t", encoding="ISO-8859-1")

#df["event_length"].plot.hist(bins=12, alpha=0.5)
#plt.show()
#plt.close()

print(df)

#df["event_length"].plot()
#plt.show()
#plt.close()

data = np.array(df["data"])

 = 0

course_counter = []
course_fine = []
hit_type = []
energy_flag = []
ch_number = []
tmp_bad = []

for d in data:
	if "0000039C" == d:
		next = 0
	if next == 1:
		binary = "{0:8b}".format(int(d, 16))
		if len(binary) == 27:
			binary_list = np.array(list(binary))
			
			hit_type.append(list(binary_list)[0])
			ch_number.append(int(''.join(list(binary_list)[1:8]),2))
			tmp_bad.append(list(binary_list)[21])
			energy_flag.append(list(binary_list)[-1])
			course_fine.append(int(''.join(binary_list[21:26]),2))
			course_counter.append(int(''.join(binary_list[7:21]),2))
	if "E80020BC" == d:
		next = 1


bins = np.linspace(0, 40, 100)


plt.hist(course_counter, bins=bins, alpha=0.5, label='course')
plt.hist(course_fine, bins=bins, alpha=0.5, label='fine')
plt.legend(loc='upper right')
plt.show()
"""
"""
plt.hist(course_counter)
plt.show()
plt.hist(course_fine)
plt.show()
plt.hist(hit_type)
plt.show()
plt.hist(energy_flag)
plt.show()
plt.hist(ch_number)
plt.show()
plt.hist(tmp_bad)
plt.show()
"""
"""
length = np.array(df["event_length"])
print(data)

list_of_length = []
list_of_counts = []
counter = 1
for idx, value in enumerate(data):
    if value == "0000039C":
        list_of_counts.append(counter)
        list_of_length.append(length[idx])

        if counter != length[idx]:
            print(data[idx-35:idx+20])
            print(length[idx-20:idx+20])
        counter = 1
    else:
        counter += 1

plt.plot(range(len(list_of_counts)), list_of_counts, label="COUNTS")
plt.plot(range(len(list_of_length)), list_of_length, '--', label="LENGTH")
plt.legend()
plt.show()


read_idx = 0
events_right = 0
events_wrong = 0
list_of_length = []
list_of_counts = []
while read_idx < len(length):
    if data[read_idx] == "E80000BC" and data[read_idx - 1] == "0000009C":
        events_right += 1
    else:
        events_wrong += 1
        print(read_idx)
        while True:
            if data[read_idx] == "E80000BC" and data[read_idx - 1] == "0000009C":
                if length[read_idx] < 30:
                    break
            read_idx += 1
            if (read_idx == len(data)):
                break
    list_of_length.append(length[read_idx])


    read_idx += length[read_idx]



print("Number Events / Right Counts: " + str(events_right/len(list_of_length)))


print(events_right)
print(events_wrong)


#read_idx = 0
#for idx, value in enumerate(data):
#    if value == "0000009C":
#        read_idx = idx
#        break

#list_of_length = []
#list_of_counts = []
#old_length = 0
#counter = 0
#while read_idx + 1 < len(length):
#    if data[read_idx] == "0000009C" and data[read_idx + 1] == "E80000BC":
#        list_of_length.append(old_length)
#        list_of_counts.append(counter)
#        counter = 0
#    counter += 1
#    read_idx += int(length[read_idx], 16)
#    old_length = int(length[read_idx], 16)


"""

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
