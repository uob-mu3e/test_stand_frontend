import sys
import datetime

print(len(sys.argv))
if len(sys.argv) < 2:
    raise Exception('Give me a mudaq_registers.vhd file')

outfile = '/************************************************\n'
outfile += '* Register map header file\n'
outfile += '* Automatically generated from ' + str(sys.argv[1]) + '\n'
outfile += '* On ' + str(datetime.datetime.now()) + '\n'
outfile += '/************************************************\n\n'

outfile += '#ifndef MUDAQ_REGISTERS__H\n'
outfile += '#define MUDAQ_REGISTERS__H\n\n\n'

with open(sys.argv[1], 'r') as file:
    for line in file:
        current_line = line.replace("\n", "").replace("", "").replace("\t", "")
        if "s.all;" in current_line:
            continue
        if 'constant' in line.replace("\n", "").replace("", ""):
            variable = current_line.split('constant')[1].split('integer')[0].replace(" ", "").replace(":", "")
            value = current_line.split('constant')[1].split('integer')[1].replace(" ", "").replace(":", "").replace(";", "").replace("=", "")
            if "16#" in value:
                value = value.replace("16#", "0x").replace("#", "")
            print(current_line.split('constant')[1].split('integer'))
            outfile += '#define ' + variable + ' ' + value + '\n'

        if 'subtype' in line.replace("\n", "").replace("", ""):
            print(current_line.split('subtype'))

outfile += '\n'
outfile += '#endif  //#ifndef MUDAQ_REGISTERS__H'

print(outfile)
