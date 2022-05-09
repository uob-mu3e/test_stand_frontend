import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('small_content.txt', sep='\t')["00000001"].values

febDict = {0:[],1:[],2:[],3:[],4:[],5:[],6:[],7:[],8:[],9:[]}
febCnt = {0:0,1:0,2:0,3:0,4:0,5:0,6:0,7:0,8:0,9:0}
febCntHits = {0:0,1:0,2:0,3:0,4:0,5:0,6:0,7:0,8:0,9:0}
febTS = {0:[],1:[],2:[],3:[],4:[],5:[],6:[],7:[],8:[],9:[]}

last_d = 0
curFEB = -999
for d in data:
    if curFEB != -999:
        curEvent.append(hex(int(d, 16)))
    if d == "FC00009C":
        febDict[curFEB].append(curEvent)
        curFEB = -999

    if last_d == '00000000':
        if len(d.split('E8')) == 2:
            if len(d.split('E8')[1].split('BC')) == 2:
                curFEB = int(d.split('E8')[1].split('BC')[0].split('0')[2])
                curEvent = [hex(int(d, 16))]
    last_d = d

for feb in febDict:
    for event in febDict[feb]:
        febCnt[feb] += 1
        if event[0] != hex(int("E8100" + str(feb) + "BC", 16)):
            print(f"FebCnt: {febCnt[feb]} of FEB: {feb} had no header")
        febTS[feb].append((int(event[2], 16) >> 16) & 0xFFFF)#(int(event[1], 16) << 5) | (int(event[2], 16) >> 27))

        if str(event[3]) != hex(0xfe00000):
            print(f"FebCnt: {febCnt[feb]} of FEB: {feb} did not start with subheader")
        startSubheader = 0
        for idx, hit in enumerate(event[3:]):
            if hex((int(hit, 16) >> 21) & 0x7F) == hex(0x7F):
                curSubheader = ((int(hit, 16) >> 28) << 5) | ((int(hit, 16) >> 16) & 0x1F)
                if curSubheader != startSubheader:
                    print(f"FebCnt: {febCnt[feb]} of FEB: {feb} Skip Subheader subheader {curSubheader} {startSubheader}")
                    for i in range(-30, 20): print(event[3+i+idx], "chipID: ", (int(event[3+i+idx], 16) >> 21) & 0x7F)
                    break
                startSubheader += 1
            else:
                if hit != hex(0xFC00009C): febCntHits[feb] += 1
        if startSubheader != 128:
            print(f"FebCnt: {febCnt[feb]} of FEB: {feb} wrong subheader ending {startSubheader}")

for i in range(10): print("Events: " + str(febCnt[i]), "Hits: " + str(febCntHits[i]))

plt.plot(febTS[5])
plt.plot(febTS[6])
plt.plot(febTS[7])
plt.plot(febTS[8])
plt.plot(febTS[9])
plt.xlabel("Events")
plt.ylabel("HeadTS")
plt.show()

