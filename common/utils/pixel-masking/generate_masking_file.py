import tkinter as tk
import numpy as np
import pandas as pd


grid = np.zeros([256, 250])

def set_phy_addr():
    row = int(e1.get())
    col = int(e2.get())
    grid[col, row] = 1
    print("Row: {} Col: {}".format(row, col))
    print(grid)
    
def clear():
    global grid
    grid = np.zeros([256, 250])
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
    outList = []
    curNBits = 0
    curWord = 0
    for col in range(128):
        curList = [col]
        for row in range(500):
            for b in range(7):
                curNBits += 1
                pcol, prow = transform_col_row_MPX(col, row)
                if b == 6 and grid[pcol, prow] == 1:
                    curWord = curWord | (1 << curNBits)
                if curNBits == 31:
                    curList.append(hex(curWord))
                    curNBits = 0
                    curWord = 0
        outList.append(curList)
    df = pd.DataFrame(outList)
    print(df)
    df.to_csv("tdacs_mupix_{}.csv".format(e3.get()), index=False)

app = tk.Tk()
tk.Label(app, text="row").grid(row=0)
tk.Label(app, text="col").grid(row=1)
tk.Label(app, text="chip").grid(row=2)

e1 = tk.Entry(app)
e2 = tk.Entry(app)
e3 = tk.Entry(app)

e1.grid(row=0, column=1)
e2.grid(row=1, column=1)
e3.grid(row=2, column=1)

tk.Button(app, text='Clear', command=clear).grid(row=3, column=0, sticky=tk.W, pady=4)
tk.Button(app, text='Set', command=set_phy_addr).grid(row=3, column=1, sticky=tk.W, pady=4)
tk.Button(app, text='Create', command=create).grid(row=3, column=2, sticky=tk.W, pady=4)

tk.mainloop()
