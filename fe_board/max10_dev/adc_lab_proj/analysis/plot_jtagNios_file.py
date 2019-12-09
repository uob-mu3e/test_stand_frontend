import numpy as np
import matplotlib.pyplot as plt
import scipy.odr as odr
fpath="./data/"
fname="jtagNiosRamp5std_failedafter1std"
data,b=np.loadtxt(fpath+fname+".txt",skiprows=17,unpack=True,delimiter=",")

data=[int(round(d/1000)) >> 4 for d in data]
time=np.linspace(0,1,len(data))
fig,ax=plt.subplots(figsize=(24,16))
ax.plot(time, data)
plt.legend(loc="best    ")
plt.savefig("time_"+fname+".png")
