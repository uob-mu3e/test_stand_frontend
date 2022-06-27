# -*- coding: utf-8 -*-
import json

target_rate = 500 #dummy
inputfilename = 'thre_scan_test.json'

outputfilename=inputfilename+'.sh'
outputfile = open(outputfilename, 'w')

with open(inputfilename) as inputfile:

   data = json.load(inputfile)
   inputfile.close()

for entry in data:
    if len(entry) == 1:
        channel = entry
        thresholds = data[entry]["Threshold"]
        rates = data[entry]["Rate"]
        thr_target_rate = 0
        for i in range(len(rates)):
            if rates[i]>target_rate and i>0:
                thr_target_rate=thresholds[i-1]
                break
            string = "odbedit -d \"/Equipment/SciFi/Settings/ASICs/Channels/{0}\" -c \"set tthresh {1}\""
            print(string.format(channel,thr_target_rate),file = outputfile)

outputfile.close()
