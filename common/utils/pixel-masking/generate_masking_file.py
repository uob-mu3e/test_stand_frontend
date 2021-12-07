import tkinter as tk
import numpy as np
import json


grid = np.zeros([128, 500])

def show_entry_fields():
    row = int(e1.get())
    col = int(e2.get())
    if col%2 != 0: row+=250
    #tk.Label(app, text="row").grid(row=int(e1.get()), col=int(e2.get()))
    grid[col//2, row] = 1
    print(col%2)
    print("Row: {} Col: {}".format(row, col//2))
    print(grid)
    
def clear():
    global grid
    grid = np.zeros([128, 500])
    print(grid)
    
def create():
    outDict = {}
    for col in range(128):
        if np.unique(grid[col]).size > 1:
            outDict[col] = grid[col].tolist()
    print(outDict)

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
