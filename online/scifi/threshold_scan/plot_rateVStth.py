import json
import matplotlib.pyplot as plt

with open('thr_scan_test6Jul_allch.json') as jsonfile:
    data = json.load(jsonfile)
    jsonfile.close()

    for entry in data:
        if len(entry) ==1:
            channel = entry
            thresholds = data[entry]['Threshold']
            rate = data[entry]['Rate']
            plt.grid(True)
            plt.plot(thresholds,rate, color='maroon', marker='o')
    plt.xlabel('TTH [dac value]')
    plt.ylabel('rate [Hz]')
    plt.show()


