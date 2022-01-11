import pandas as pd
import argparse


def checkSubH(v):
    return [i for i in "{:08b}".format(v)][:6] == ['1', '1', '1', '1', '1', '1']

def getSubH(v):
    bitList = [int(i) for i in "{:08b}".format(v)][9:16]
    out = 0
    for bit in bitList:
        out = (out << 1) | bit
    return out

parser = argparse.ArgumentParser(description='Process file name.')
parser.add_argument('--file', type=str, action='store', default='memory_content.txt',
                    help='get file name for memory_content.txt')

args = parser.parse_args()

df = pd.read_csv(args.file, sep="\t")


# loop throw the file and check the event
# all events start with
# 0x00000001 0
# 0x00000001 1
# 0x00000002 2
# 0x00000230 3
# 0x00000228 4
# 0x00000031 5
# 0x31444350 6
# 0x00000006 7
# 0x00000218 8
# 0x00000000 9
# 0xE80000BC <- first value to check
# than two times ts should come
# FC000000 <- should start with this and then count until FC7F0000
# FC00009C <- should ent with this
# AFFEAFFE <- padding amount for n%256

doubleHits = 0
wrongCntOfSubheader = 0
wrongFirstSubheader = 0
noHeader = 0
noTrailer = 0
tsWrong = 0
nEvent = -1
curSubheader = 0
wrongPadding = 0
oldWord = 0
checkTrailer = False
checkPadding = False
buffer = []
nEvents = 0
for idx, d in enumerate(df["data"]):
    d = int(d, 16)
    if d != 0xAFFEAFFE and d == oldWord and nEvent >= 10:
        #print(d, oldWord, nEvent)
        doubleHits += 1
    nEvent += 1
    if nEvent < 10:
        nEvents += 1
        buffer.append([hex(d), nEvent])
        continue
    elif nEvent == 10:
        if d != 0xE80000BC:
            noHeader += 1
    elif nEvent == 11:
        continue
    elif nEvent == 12:
        continue
    elif nEvent == 13:
        if d != 0xFC000000:
            wrongFirstSubheader += 1
        curSubheader = 0
        buffer = []
    elif getSubH(d) == 0x7F:
        checkTrailer = True
    elif checkTrailer:
        if d != 0xFC00009C:
            noTrailer += 1
        checkPadding = True
        checkTrailer = False
    elif checkPadding:
        if not (d == 0xAFFEAFFE or d == 0x00000001):
            wrongPadding += 1
        if d == 0x00000001:
            nEvent = 0
            checkPadding = False
    elif checkSubH(d):
        if d == 0xFC00009C:
            wrongCntOfSubheader += 1
            checkPadding = True
            continue
        if d != 0xFC000000:
            curSubheader += 1
        if curSubheader != getSubH(d):
            #print(hex(oldoldWord), hex(oldWord), hex(d), getSubH(oldoldWord), hex(curSubheader))
            wrongCntOfSubheader += 1
    elif d == 0xAFFEAFFE:
        noTrailer += 1
        checkPadding = True

    oldoldWord = oldWord
    oldWord = d


print("Sorter Check Report")
print("-------------------")
print("nEvents={}".format(nEvents))
print("doubleHits={}".format(doubleHits))
print("wrongCntOfSubheader={}".format(wrongCntOfSubheader))
print("wrongFirstSubheader={}".format(wrongFirstSubheader))
print("noHeader={}".format(noHeader))
print("noTrailer={}".format(noTrailer))
print("wrongPadding={}".format(wrongPadding))
