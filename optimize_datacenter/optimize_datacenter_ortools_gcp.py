from numpy.core.fromnumeric import shape, size
from numpy.core.function_base import add_newdoc
from numpy.core.shape_base import block
from ortools.sat.python import cp_model
from time import time
from tabulate import tabulate
import numpy as np
# import matplotlib.pyplot as plt

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
- x_r,m = s cell slot index 's' the leftmost cell of server m in row 'r'
- y_m,p: whether server m belongs to pool p
"""
x = []
for r in range(n_rows):
    x_r = []
    for m in range(n_servers):
        x_r.append(model.NewIntVar(-1, n_slots - 1, f'x[{r}][{m}]'))
    x.append(x_r)

y = []
for m in range(n_servers):
    y_m = []
    for p in range(n_pools):
        y_m.append(model.NewBoolVar(f'y[{m}][{p}]'))
    y.append(y_m)

"""
Derived functions
z_m,r: whether server 'm' is in row 'r'
capacity_p: total pool capacity
pool_row_capacity_p,r: remaining capacity in pool 'p' after row 'r' drops
gc_p:guaranteed capacity for each pool
"""

z = []
for m in range(n_servers):
    z_m = []
    for r in range(n_rows):
        z_m.append(model.NewBoolVar(f'z[{m}][{r}]'))
        model.Add(x[r][m] > -1).OnlyEnforceIf(z_m[r])
        model.Add(x[r][m] == -1).OnlyEnforceIf(z_m[r].Not())
    z.append(z_m)

capacity = []
pool_row_capacity = []
for p in range(n_pools):
    capacity.append(model.NewIntVar(min_capacity * n_rows, total_capacity // n_pools, f'capacity[{p}]'))
    model.Add(capacity[p] == sum([y[m][p] * servers[m][1] for m in range(n_servers)]))
    pool_row_capacity_r = []
    for r in range(n_rows):
        pool_row_capacity_r.append(model.NewIntVar(min_capacity * (n_rows - 1), total_capacity // n_pools, f'pool_row_max_capacity[{p}][{r}]'))
        tmp = []
        for m in range(n_servers):
            tmp.append(model.NewBoolVar(f'tmp[{m}][{r}][{p}]'))
            model.AddMultiplicationEquality(tmp[m], [z[m][r], y[m][p]])
        model.Add(pool_row_capacity_r[r] == capacity[p] - sum([tmp[m] * servers[m][1] for m in range(n_servers)]))
    pool_row_capacity.append(pool_row_capacity_r)

gc = []
for p in range(n_pools):
    gc.append(model.NewIntVar(min_capacity * (n_rows - 1), total_capacity // n_pools, f'gc[{p}]'))
    model.AddMinEquality(gc[p], [pool_row_capacity[p][r] for r in range(n_rows)])

#Constraints
#C1
for m in range(n_servers):
    size_m = servers[m][0]
    for r in range(n_rows):
        for s in range(n_slots):
            stop_variable = model.NewBoolVar('stop_variable')
            model.Add(x[r][m] == s).OnlyEnforceIf(stop_variable)
            model.Add(x[r][m] != s).OnlyEnforceIf(stop_variable.Not())
            for m_ in range(n_servers):
                size_m_ = servers[m_][0]
                blocked_indices = [s + i for i in range(-size_m_ + 1, size_m)\
                        if ((s + i < n_slots) and (s + i >= 0) and (i != 0 or m_ != m))]
                if len(blocked_indices) == 0 or m_ == m:
                    continue
                tmp_min = model.NewBoolVar('min_cutoff')
                model.Add(x[r][m_] < min(blocked_indices)).OnlyEnforceIf(tmp_min)
                model.Add(x[r][m_] >= min(blocked_indices)).OnlyEnforceIf(tmp_min.Not())
                tmp_max = model.NewBoolVar('max_cutoff')
                model.Add(x[r][m_] > max(blocked_indices)).OnlyEnforceIf(tmp_max)
                model.Add(x[r][m_] <= max(blocked_indices)).OnlyEnforceIf(tmp_max.Not())
                model.AddBoolOr([tmp_min, tmp_max]).OnlyEnforceIf(stop_variable)

#C2
for (r, s) in unavailable:
    for m in range(n_servers):
        size_m = servers[m][0]
        for i in range(size_m):
            if s - i < 0:
                continue
            model.Add(x[r][m] != s - i)
#C3
for m in range(n_servers):
    size_m = servers[m][0]
    for r in range(n_rows):
        for i in range(size_m - 1):
            model.Add(x[r][m] != n_slots - 1 - i)

#C4
for m in range(n_servers):
    model.Add(sum([z[m][r] for r in range(n_rows)]) == sum([y[m][p] for p in range(n_pools)]))

#C5
for m in range(n_servers):
    model.Add(sum([y[m][p] for p in range(n_pools)]) <= 1)


#Objective
objective = model.NewIntVar(min_capacity * (n_rows - 1), total_capacity // n_pools, 'minimum_gc')
model.AddMinEquality(objective, gc)
model.Maximize(objective)

#Break symmetry
#S1
capacity_list = [servers[m][1] for m in range(n_servers)]
for p in range(n_pools):
    imax = np.argmax(capacity_list)
    model.Add(y[imax][p] == 1)
    capacity_list[imax] = 0

#S2
for r in range(n_rows):
    for m in range(n_servers):
        size_m = servers[m][0]
        for m_ in range(n_servers):
            size_m_ = servers[m_][0]
            consecutive = model.NewBoolVar('consecutive')
            model.Add(x[r][m_] == x[r][m] + size_m).OnlyEnforceIf(consecutive)
            model.Add(x[r][m_] != x[r][m] + size_m).OnlyEnforceIf(consecutive.Not())
            sorted_size = model.NewBoolVar('sorted_size')
            model.Add(size_m_ <= size_m).OnlyEnforceIf(sorted_size)
            model.Add(size_m_ > size_m).OnlyEnforceIf(sorted_size.Not())
            consecutive_sorted = model.NewBoolVar('consecutive_sorted')
            model.AddImplication(consecutive_sorted, z[m][r])
            model.AddImplication(consecutive_sorted, z[m_][r])
            model.AddImplication(consecutive_sorted, sorted_size)
            model.AddImplication(consecutive_sorted, consecutive)

"""
Model solve and display
"""
solver = cp_model.CpSolver()
solver.parameters.num_search_workers = 16
solver.parameters.linearization_level = 2
status = solver.Solve(model)
if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
    print("Solutions found!")
    print(f'Optimal total value: {solver.ObjectiveValue()}.')
    """
    cells = [["" for s in range(n_slots)] for r in range(n_rows)]
    fig, ax = plt.subplots()
    ax.xaxis.set_visible(False) 
    ax.yaxis.set_visible(False)
    colors = ['r', 'g', 'b', 'c', 'm', 'y']
    table_colors = [['w' for s in range(n_slots)] for r in range(n_rows)]
    for (row, slot) in unavailable:
        cells[row][slot] = "X"
    for p in range(n_pools):
        col_index = p % len(colors)
        for m in range(n_servers):
            size_m = servers[m][0]
            for r in range(n_rows):
                if(solver.Value(z[m][r]) * solver.Value(y[m][p])):
                    print(f'({r},{m},{p})')
                    s = solver.Value(x[r][m])
                    for i in range(size_m):
                        table_colors[r][s + i] = colors[col_index]
                        cells[r][s + i] = f's{m}'
    ax.table(cellText=cells, loc='center', cellColours=table_colors)
    ax.set_title(filenames[index])
    plt.savefig('optimize_datacenter/results/' + filenames[index] + '.jpg')
    if view:
        plt.show()
    """
else:
    print('No solution found.')