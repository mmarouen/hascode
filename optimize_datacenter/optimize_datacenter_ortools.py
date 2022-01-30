from numpy.core.fromnumeric import shape
from ortools.sat.python import cp_model
from time import time
from tabulate import tabulate
import numpy as np
import matplotlib.pyplot as plt

filenames = ['example.in', 'dc.in']
index = 0
file_url = 'optimize_datacenter/data/' + filenames[index]
view = True
# parse input
file = open(file_url, 'r')
lines = file.readlines()
unavailable = []
servers = []
for l, line in enumerate(lines):
    line = line.rstrip('\r\n')
    values = [int(val) for val in line.split(' ')]
    if(l == 0):
        n_rows = values[0]
        n_slots = values[1]
        n_unavailable = values[2]
        n_pools = values[3]
        n_servers = values[4]
    elif(l <= n_unavailable):
        unavailable.append((values[0], values[1]))
    else:
        servers.append((values[0], values[1]))
file.close()

print(f'Problem setup for file {filenames[index]}:')
print(f'Datacenter:\n-Rows {n_rows} x Slots {n_slots}\n-Unavailable slots {n_unavailable}')
print(f'Pools {n_pools}, Servers {n_servers}')
if view:
    print(f'Unavailable slots {unavailable}')
    print(f'Servers: {servers}')
    datacenter = np.array([[' ' for _ in range(n_slots)] for _ in range(n_rows)])
    for u in unavailable:
        datacenter[u[0]][u[1]] = 'X'
    table = tabulate(datacenter, tablefmt="fancy_grid")
    print('Datacenter rows x slots')
    print(table)
print('----------------------------')

model = cp_model.CpModel()

"""
create decision variables:
- x_r,c,s: whether cell (r,c) corresponds to the upper left corner of shape s
"""
x = []
