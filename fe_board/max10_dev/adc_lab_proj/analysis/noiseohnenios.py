import numpy as np

def noise(x,x2):
    n=0x1ffffff
    s2=(x2/n-(x/n)**2)*n/(n-1)
    m=x/n
    return m,np.sqrt(s2)
#no nios
# w jtag plugged 1v 4v wo jtag plugged 1v 4v
x=np.array([0x65E177DD0,0x197804aa2d,0x65c265793,0x1977f4b149])
x2=np.array([0x1445dfaadf28,0x14454a06b88b4,0x143983f9c840,0x14453098431ae])

print(noise(x,x2))
