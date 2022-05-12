import pandas as pd
import matplotlib.pyplot as plt
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("-f", "--file", help="get path to memory file", type=str)
parser.add_argument("-n", "--nrows", help="num rows", type=int)

args = parser.parse_args()

if args.nrows == -1:
    data = pd.read_csv(args.file, sep='\t', header=None)[1].values
else:
    data = pd.read_csv(args.file, sep='\t', header=None, nrows=args.nrows)[1].values
print(data)

febDict = {0:[],1:[],2:[],3:[],4:[],5:[],6:[],7:[],8:[],9:[]}
febCnt = {0:0,1:0,2:0,3:0,4:0,5:0,6:0,7:0,8:0,9:0}
febCntHits = {0:0,1:0,2:0,3:0,4:0,5:0,6:0,7:0,8:0,9:0}
febTS = {0:[],1:[],2:[],3:[],4:[],5:[],6:[],7:[],8:[],9:[]}
febTrailerCnt = {0:[],1:[],2:[],3:[],4:[],5:[],6:[],7:[],8:[],9:[]}
headerList=["E80000BC","E80001BC","E80002BC","E80003BC","E80004BC","E81005BC","E81006BC","E81007BC","E81008BC","E81009BC","E81000BC"]

last_d = 0
curFEB = -999
for idx, d in enumerate(data):
    if d == "FFFFFFFF" and data[idx+1] == "FFFFFFFF": break
    if curFEB != -999:
        curEvent.append(hex(int(d, 16)))
    if d == "FC00019C" or d == "FC00009C":
        febTrailerCnt[curFEB].append(hex(((int(data[idx+1], 16) >> 8) & 0xFFFF)-1))
        febDict[curFEB].append(curEvent)
        curFEB = -999

    if last_d == '00000000' and d in headerList:
        curFEB = int(d.split('E8')[1].split('BC')[0].split('0')[2])
        curEvent = [hex(int(d, 16))]
        febTS[curFEB].append((int(data[idx+1], 16) << 5) | (int(data[idx+2], 16) >> 27))
    last_d = d

for feb in febDict:
    if len(febDict[feb]) > 0:
        startEventCnt = int(febDict[feb][0][2], 16) & 0xFFFF
        trailerEventCnt = 0
    for idx_e, event in enumerate(febDict[feb]):
#print("Trailer Ecnt", febTrailerCnt[feb][idx_e], "Start Cnt", hex(startEventCnt))
        febCnt[feb] += 1
        if event[0] != hex(int("E8100" + str(feb) + "BC", 16)):
            print(f"FebCnt: {febCnt[feb]} of FEB: {feb} had no header")

        if str(event[3]) != hex(0xfe00000):
            print(f"FebCnt: {febCnt[feb]} of FEB: {feb} did not start with subheader")
        startSubheader = 0
        for idx, hit in enumerate(event[3:]):
            if hex((int(hit, 16) >> 21) & 0x7F) == hex(0x7F):
                curSubheader = ((int(hit, 16) >> 28) << 5) | ((int(hit, 16) >> 16) & 0x1F)
                if curSubheader != startSubheader:
                    print(f"FebCnt: {febCnt[feb]} of FEB: {feb} Skip Subheader subheader {curSubheader} {startSubheader}")
                    for i in range(-50, 20): 
                        print(event[3+i+idx], "chipID: ", (int(event[3+i+idx], 16) >> 21) & 0x7F, 3+i+idx)
                    break
                startSubheader += 1
            else:
                if hit != hex(0xFC00019C) or hit != hex(0xFC00009C): febCntHits[feb] += 1
        if startSubheader != 128:
            print(f"FebCnt: {febCnt[feb]} of FEB: {feb} wrong subheader ending {startSubheader}")
#if (int(event[2], 16) & 0xFFFF) != startEventCnt: print("We miss an event")

        startEventCnt += 1

for i in range(10): print("Events: " + str(len(febTS[i])) + " " + str(len(febDict[i])), "Hits: " + str(febCntHits[i]))

#4D2C F811A59F
#4D2C F020A59E

plt.plot(febTS[5], label="FEB5")
plt.plot(febTS[6], label="FEB6")
plt.plot(febTS[7], label="FEB7")
plt.plot(febTS[8], label="FEB8")
plt.plot(febTS[9], label="FEB9")
plt.xlabel("Events")
plt.ylabel("HeadTS")
plt.legend()
plt.savefig("headerTS.pdf")

