import sys
import json
import matplotlib.pyplot as plt


f=open(sys.argv[1])
    
data = json.load(f)
 
X = data[sys.argv[2]]['Output'][sys.argv[3]]
Y = data[sys.argv[2]]['Output'][sys.argv[4]]

plt.scatter(X,Y)
plt.xlabel(sys.argv[3])  
plt.ylabel(sys.argv[4])  
plt.show()
plt.savefig('FileName.pdf') 

# Closing file
f.close()
