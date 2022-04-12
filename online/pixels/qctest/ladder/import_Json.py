import sys
import json
import matplotlib.pyplot as plt

def plot_json_fields(field,x,y,file):
    # Opening JSON file
    # returns JSON object as a dictionary
    data = json.load(file)
 
    X = data[field]['Output'][x]
    Y = data[field]['Output'][y]

    plt.scatter(X,Y)
    plt.xlabel(x)  
    plt.ylabel(y)  
    plt.show()

    
f=open(sys.argv[1])
plot_json_fields('IV','Voltage','Current',f)
plt.savefig('FileName.pdf') 

# Closing file
f.close()
