# M. Mueller, March 2022

print("generating lookup table for reverse col row transformation")
print("https://www.physi.uni-heidelberg.de/Forschung/he/mu3e/restricted/notes/Mu3e-Note-0052-MuPix10_Documentation.pdf")

vhdl_input_order = []
vhdl_output_order = []

def convert(col,row):
  invrow = 0;
  newcol = 0;
  newrow = 0;
  newcol = col*2;
  invrow = (0x1FF & ~row)
  if invrow > 380:
      newcol +=1;
      newrow = (499-invrow)/2;
      if (499-invrow)%2 == 1:
          newrow +=60;
  elif invrow > 255:
      newrow = (380-invrow)/2;
      if (380-invrow)%2 == 0:
          newrow += 62;
  elif invrow > 124:
      newrow = (255-invrow)/2;
      newcol += 1;
      newrow += 119;
      if (255-invrow)%2 == 1:
          newrow += 66;
  else :
      newrow = (124-invrow)/2;
      newrow += 125;
      if (124-invrow)%2 == 0:
          newrow += 62;



  # firmware gets the physical col row as input
  # and wants to know the location in memory (= digital addr.) where it should write it to.
  vhdl_input_order.append(newcol*250 + newrow)
  vhdl_output_order.append(row)
  #print(str(newcol*256+newrow) + ',')
  # print(col, row, newcol ,newrow);
  # print('when x"' + str(hex(newcol*250 + newrow)).replace('0x', '') + '" => mem_addr <= x"' + str(hex(row)).replace('0x', '') + '";')

for col in range(0, 1):
    for row in range(11, 512):
        convert(col,row)
#print(vhdl_input_order)
#print(vhdl_output_order)

vhdl_inout = list(zip(vhdl_input_order,vhdl_output_order))
vhdl_inout_sorted = sorted(vhdl_inout, reverse = True)
print(vhdl_inout_sorted)
print(zip(*vhdl_inout_sorted)[1])