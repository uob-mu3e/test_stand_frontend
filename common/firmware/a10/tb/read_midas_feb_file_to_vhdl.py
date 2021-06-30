import midas.file_reader

mfile = midas.file_reader.MidasFile('run00198.mid')

list_feb0 = []
list_feb2 = []

def to_vhdl(d, i):
    
    value_str = str(hex(d)).split('x')[1]
    
    while len(list(value_str)) != 8:
        value_str = "0" + value_str

    outfile = 'data_feb' + i + ' <= x\"' + value_str + '\";\n'
    if d == 0xe8feb0bc or d == 0xe8feb2bc or d == 0xFC00009C:
        outfile += 'datak_feb' + i + ' <= "0001";\n'
    else:
        outfile += 'datak_feb' + i + ' <= "0000";\n'
    outfile += 'wait until rising_edge(clk);\n'
    return outfile

counter = 0

for event in mfile:
    bank_names = ", ".join(b.name for b in event.banks.values())
    if counter == 100: break 
    for bank_name, bank in event.banks.items(): 
        if 'PCD1' == bank_name:
            counter += 1
            for d in bank.data:
                if d == 0xAFFEAFFE: continue
                if bank.data[0] == 0xe8feb0bc:
                    list_feb0.append(to_vhdl(d, "0"))
                else:
                    list_feb2.append(to_vhdl(d, "2"))
            if counter == 100: break

file=open('f0_sim.vhd','w')
for items in list_feb0:
    file.writelines([items])
file.close()

file=open('f1_sim.vhd','w')
for items in list_feb2:
    file.writelines([items])
file.close()
