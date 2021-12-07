import tkinter as tk
import numpy as np


grid = np.zeros([128, 500])

def show_entry_fields():
    row = int(e1.get())
    col = int(e2.get())
    if col%2 != 0:
        row += 250
    #tk.Label(app, text="row").grid(row=int(e1.get()), col=int(e2.get()))
    grid[col//2, row] = 1
    print(col%2)
    print("Row: {} Col: {}".format(row, col//2))
    print(grid)
    
def clear():
    global grid
    grid = np.zeros([128, 500])
    print(grid)

def transform_col_row_MPX(col, row):
    newcol = col*2
    row = 0x1FF & (~row)

    if row > 380:
        newcol += 1
        newrow = (499-row)/2
        if ((499-row)%2) == 1:
            newrow += 60
    elif row > 255:
        newrow = (380-row)/2
        if ((380-row)%2) == 0:
            newrow += 62
    elif row > 124:
        newrow = (255-row)/2
        newcol += 1
        newrow += 119
        if ((255-row)%2) == 1:
            newrow += 66
    else:
        newrow = (124-row)/2
        newrow += 125
        if ((124-row)%2) == 0:
            newrow+=62
    return int(newcol), int(newrow)
    
def create():
    outStr  = "#include <vector>\n"
    outStr += "#include <cstdint>\n\n"
    outStr += "uint32_t default_config_mupix[127] = {\n"
    for col in range(128):
        if np.unique(grid[col]).size > 1:
            outDict[col] = grid[col].tolist()
    print(outDict)

row = []
col = []
ilist = []
jlist = []
for i in range(128):
    for j in range(500):
        c, r = transform_col_row_MPX(i, j)
        if r <= 249 and r >= 0 and c <= 255 and c >= 0:
            if j not in jlist:
                if r not in row:
                    row.append(r)
                    jlist.append(j)
            if i not in ilist:
                col.append(c)
                ilist.append(i)
            else:
                if c not in col:
                    col.append(c)

print(np.array(ilist))
print(np.array(jlist))
exit()
app = tk.Tk()
tk.Label(app, text="row").grid(row=0)
tk.Label(app, text="col").grid(row=1)

e1 = tk.Entry(app)
e2 = tk.Entry(app)

e1.grid(row=0, column=1)
e2.grid(row=1, column=1)

tk.Button(app, text='Clear', command=clear).grid(row=3, column=0, sticky=tk.W, pady=4)
tk.Button(app, text='Set', command=show_entry_fields).grid(row=3, column=1, sticky=tk.W, pady=4)
tk.Button(app, text='Create', command=create).grid(row=3, column=2, sticky=tk.W, pady=4)

tk.mainloop()
