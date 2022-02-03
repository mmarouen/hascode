from numpy.core.fromnumeric import shape, size
from ortools.sat.python import cp_model
from time import time
from tabulate import tabulate
import numpy as np
#import matplotlib.pyplot as plt

filenames = ['example.in', 'dc.in']
index = 1
file_url = 'optimize_datacenter/data/' + filenames[index]
view = False
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

total_capacity = sum([servers[s][1] for s in range(n_servers)])
min_capacity = min([servers[s][1] for s in range(n_servers)])
max_capacity = max([servers[s][1] for s in range(n_servers)])
uu = [(0, 0), (0, 1), (2, 2), (2, 4), (2, 5)]
unavailable_per_row = [sum([unavailable[u][0] == r for u in range(len(unavailable))]) for r in range(n_rows)]
print(f'Problem setup for file {filenames[index]}:')
print(f'Datacenter:\n-Rows {n_rows} x Slots {n_slots}\n-Unavailable slots {n_unavailable}')
print(f'Pools {n_pools}, Servers {n_servers}')
print(f'Servers: total capacity {total_capacity}, min capacity {min_capacity}, max capacity {max_capacity}')

if view:
    print(f'Unavailable slots {unavailable}')
    print(f'unavailable slots per row {unavailable_per_row}')
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
- x_r,s,m,p: whether cell (r,s) of the datacenter corresponds to the leftmost cell of server m in pool p
"""
x = []
for r in range(n_rows):
    x_r = []
    for s in range(n_slots):
        x_s = []
        for m in range(n_servers):
            x_m = []
            for p in range(n_pools):
                x_m.append(model.NewBoolVar(f'x[{r}][{s}][{m}][{p}]'))
            x_s.append(x_m)
        x_r.append(x_s)
    x.append(x_r)

"""
Derived functions
capacity_p: total pool capacity
pool_row_capacity_p,r: remaining capacity in pool 'p' after row 'r' drops
gc_p:guaranteed capacity for each pool
"""
capacity = []
pool_row_capacity = []
for p in range(n_pools):
    capacity.append(model.NewIntVar(min_capacity, total_capacity, f'capacity[{p}]'))
    model.Add(capacity[p] == sum([x[r][s][m][p] * servers[m][1] for r in range(n_rows) for s in range(n_slots) for m in range(n_servers)]))
    tmp = []
    for r in range(n_rows):
        tmp.append(model.NewIntVar(min_capacity, total_capacity, f'pool_row_max_capacity[{p}][{r}]'))
        model.Add(tmp[r] == capacity[p] - sum(x[r][s][m][p] * servers[m][1] for s in range(n_slots) for m in range(n_servers)))
    pool_row_capacity.append(tmp)

gc = []
for p in range(n_pools):
    gc.append(model.NewIntVar(min_capacity, total_capacity, f'gc[{p}]'))
    tmp = model.NewIntVar(min_capacity, total_capacity, f'')
    model.AddMinEquality(gc[p], [pool_row_capacity[p][r] for r in range(n_rows)])

#Constraints
#C1
for m in range(n_servers):
    size_m = servers[m][0]
    for r in range(n_rows):
        for s in range(n_slots):
            for p in range(n_pools):
                for m_ in range(n_servers):
                    size_m_ = servers[m_][0]
                    for p_ in range(n_pools):
                        model.Add(sum([x[r][s + i][m_][p_] for i in range(-size_m_ + 1, size_m)\
                            if ((s + i < n_slots) and (s + i >= 0) and (i != 0 or m_ != m))]) == 0)\
                                .OnlyEnforceIf(x[r][s][m][p])
                        #for i in range(-size_m_ + 1, size_m):
                        #    if i == 0 and m_ == m:
                        #        continue
                        #    if s + i < n_slots and s + i >= 0:
                        #        model.Add(x[r][s + i][m_][p_] == 0).OnlyEnforceIf(x[r][s][m][p])
                
#C2
for (r, s) in unavailable:
    for m in range(n_servers):
        size_m = servers[m][0]
        for p in range(n_pools):
            model.Add(sum([x[r][s - i][m][p] for i in range(size_m) if s - i >= 0]) == 0)
            #for i in range(size_m):
            #    if(s - i >= 0):
            #        model.Add(x[r][s - i][m][p] == 0)

#C3
for m in range(n_servers):
    size_m = servers[m][0]
    for p in range(n_pools):
        for r in range(n_rows):
            model.Add(sum([x[r][n_slots - 1 - i][m][p] for i in range(size_m - 1)]) == 0)
            #for i in range(size_m - 1):
            #    model.Add(x[r][n_slots - 1 - i][m][p] == 0)

#C4
for m in range(n_servers):
    model.Add(sum([x[r][s][m][p] for r in range(n_rows) for s in range(n_slots) for p in range(n_pools)]) <= 1)

#Objective
objective = model.NewIntVar(min_capacity, total_capacity, 'minimum_gc')
model.AddMinEquality(objective, gc)
model.Maximize(objective)

#Break symmetry


"""
Model solve and display
"""
solver = cp_model.CpSolver()
solver.parameters.add_lp_constraints_lazily = True
status = solver.Solve(model)
if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
    print("Solutions found!")
    print(f'Optimal total value: {solver.ObjectiveValue()}.')
    cells = [["" for s in range(n_slots)] for r in range(n_rows)]
    #fig, ax = plt.subplots()
    #ax.xaxis.set_visible(False) 
    #ax.yaxis.set_visible(False)
    colors = ['r', 'g', 'b', 'c', 'm', 'y']
    table_colors = [['w' for s in range(n_slots)] for r in range(n_rows)]
    for (row, slot) in unavailable:
        cells[row][slot] = "X"
    for p in range(n_pools):
        col_index = p % len(colors)
        for m in range(n_servers):
            size_m = servers[m][0]
            for s in range(n_slots):
                for r in range(n_rows):
                    if(solver.Value(x[r][s][m][p])):
                        print(f'({r},{s},{m},{p})')
                        for i in range(size_m):
                            table_colors[r][s + i] = colors[col_index]
                            cells[r][s + i] = f's{m}'
    #ax.table(cellText=cells, loc='center', cellColours=table_colors)
    #ax.set_title(filenames[index])
    #plt.savefig('optimize_datacenter/results/' + filenames[index] + '.jpg')
    #if view:
    #    plt.show()
else:
    print('No solution found.')