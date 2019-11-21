import numpy as np
import matplotlib.pyplot as plt
import scipy.odr as odr

data=np.genfromtxt("test.txt",skiprows=5,comments='#', usecols=(0))


def linearslope(B,x):
    #B[0]= start
    #B[1]= end
    #B[2]= offset
    #B[3] = slope
    return np.concatenate(  (np.zeros((len(x)-len(np.where(x>B[0])[0])))    ,B[3]*(x[np.where(x>B[0])]-B[0]),   np.zeros((len(x)-1-len(B[2]+0*np.where(x>B[1])[0]))))  , axis=0)
print(data.shape)
print(linearslope([1,1,1,1],np.array([1])).shape)
print(linearslope([3,6,2,100],np.array([1,2,3,4,5,6,7,8])))

time = np.linspace(0,1,len(data))
time_err = 0.002
data_err = 0.01


slope = odr.Model(linearslope)
mydata = odr.RealData(time, data, sx=time_err, sy=data_err)
myodr = odr.ODR(mydata, slope, beta0=[0,1,0.5,0.0001])
myoutput = myodr.run()


fig,ax=plt.subplots(figsize=(24,16))
ax.hist(data,bins=500)
plt.savefig("hist.svg")

fig,ax=plt.subplots(figsize=(24,16))
ax.plot(time ,data,".")
ax.plot(time, linearslope(myoutput, time), label="fit")
plt.legend(loc="best    ")
plt.savefig("time.svg",dpi=300)
