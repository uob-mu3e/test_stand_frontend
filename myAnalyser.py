
import sys
sys.path.append('/midas_v1/mu3eMSci/packages/midas/python/')
import midas.file_reader
import matplotlib.pyplot as plt

# DATAFRAME INITIALIZATION
data = { 'Temp' : [] , 
         'Flow' : [] ,
         'Press' : [] ,
         'A' : [],
         'Setpoint' : [],
         'Relative Humidity' : [],
         'Ambient Temperature' : []    
         } 

# FILE I/O
mfile = midas.file_reader.MidasFile("run00134.mid")
  
# READ file
while mfile.read_next_event_header():
    
    header = mfile.event.header
    if header.is_midas_internal_event():
        # Skip over events that contain midas messages or ODB dumps
        continue
    
    mfile.read_this_event_body()     

    data_tmp = []
    for name, bank in mfile.event.banks.items():
        this_data = [x for x in bank.data] 
        data_tmp.append(this_data)
    data["Temp"].append(data_tmp[0])
    data["Flow"].append(data_tmp[1])
    data["Press"].append(data_tmp[2])
    data["A"].append(data_tmp[3])
    data["Setpoint"].append(data_tmp[4])
    data["Relative Humidity"].append(data_tmp[5])
    data["Ambient Temperature"].append(data_tmp[6])

plt.scatter(range(1,len(data["Temp"])+1), data["Temp"], s=10, c='steelblue', marker = 'x', label='Temperature')
plt.scatter(range(1,len(data["Flow"])+1), data["Flow"], s=10, c='forestgreen', marker = 'x', label='Flow')
plt.scatter(range(1,len(data["Press"])+1), data["Press"], s=10, c='orangered', marker = 'x', label='Pressure')
plt.scatter(range(1,len(data["A"])+1), data["A"], s=10, c='violet', marker = 'x', label='A')
plt.scatter(range(1,len(data["Setpoint"])+1), data["Setpoint"], s=10, c='pink', marker = 'x', label='Setpoint')
plt.scatter(range(1,len(data["Relative Humidity"])+1), data["Relative Humidity"], s=10, c='black', marker = 'x', label='Relative Humidity')
plt.scatter(range(1,len(data["Ambient Temperature"])+1), data["Ambient Temperature"], s=10, c='cyan', marker = 'x', label='Ambient Temperature')
plt.legend(loc=(0.75,0.75))
plt.xlabel("time (s)")
#plt.xlim()
plt.ylim(-5, 50)
#plt.show()
plt.savefig('run134.png')  
  


    # print("Overall size of event,type ID and bytes")
    # print((header.serial_number, header.event_id, header.event_data_size_bytes))
        # if isinstance(bank.data, tuple) and len(bank.data):
        #     # A tuple of ints/floats/etc (a tuple is like a fixed-length list)
        #     type_str = "tuple of %s containing %d elements" % (type(bank.data[0]).__name__, len(bank.data))
        # elif isinstance(bank.data, tuple):
        #     # A tuple of length zero
        #     type_str = "empty tuple"
        # elif isinstance(bank.data, str):
        #     # Of the original data was a list of chars, we convert to a string.
        #     type_str = "string of length %d" % len(bank.data)
        # else:
        #     # Some data types we just leave as a set of bytes.
        #     type_str = type(bank.data[0]).__name__
        
        # print("  - bank %s contains %d bytes of data. Python data type: %s" % (name, bank.size_bytes, type_str))



