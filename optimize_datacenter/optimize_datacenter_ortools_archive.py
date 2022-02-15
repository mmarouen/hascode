from numpy.core.fromnumeric import shape, size
from numpy.core.function_base import add_newdoc
from ortools.sat.python import cp_model
from time import time
from tabulate import tabulate
import numpy as np
gcp_mode = True
if not gcp_mode:
    import matplotlib.pyplot as plt

filenames = ['example.in', 'dc.in']
index = 1
file_url = 'optimize_datacenter/data/' + filenames[index]
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
median_capacity = np.median([servers[s][1] for s in range(n_servers)])
mean_capacity = total_capacity / n_servers
min_size = min([servers[s][0] for s in range(n_servers)])
max_size = max([servers[s][0] for s in range(n_servers)])
unavailable_per_row = [sum([unavailable[u][0] == r for u in range(len(unavailable))]) for r in range(n_rows)]
print(f'Problem setup for file {filenames[index]}:')
print(f'Datacenter:\n-Rows {n_rows} x Slots {n_slots}\n-Unavailable slots {n_unavailable}')
print(f'Pools {n_pools}, Servers {n_servers}')
print('Servers:')
print(f'--Capacity: total capacity {total_capacity}, min capacity {min_capacity},\
median capacity {median_capacity}, mean capacity {mean_capacity} max capacity {max_capacity}')
print(f'--Size: min size {min_size}, max size {max_size}')

model = cp_model.CpModel()

"""
create decision variables:
- x_r,s,m: whether cell (r,s) of the datacenter corresponds to the leftmost cell of server m
- y_m,p: whether server m belongs to pool p
"""
x = []
for r in range(n_rows):
    x_r = []
    for s in range(n_slots):
        x_s = []
        for m in range(n_servers):
            x_s.append(model.NewBoolVar(f'x[{r}][{s}][{m}]'))
        x_r.append(x_s)
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
        model.Add(sum([x[r][s][m] for s in range(n_slots)]) > 0).OnlyEnforceIf(z_m[r])
        model.Add(sum([x[r][s][m] for s in range(n_slots)]) == 0).OnlyEnforceIf(z_m[r].Not())
    z.append(z_m)

capacity = []
pool_row_capacity = []
min_capa_per_row_per_pool = median_capacity
min_capacity_per_pool = int(median_capacity * n_rows)
min_gc_per_pool = int(min_capa_per_row_per_pool * (n_rows - 1))
# min_capa_per_row_per_pool = 0.5 * (median_capacity + min_capacity)
max_capa_per_pool = total_capacity // n_pools
max_capa_per_pool_per_row = int(total_capacity / n_pools / n_rows)
max_gc_capa_per_pool = int(total_capacity * (n_pools - 1) / n_pools)
print(f'mini capa per pool per row {min_capa_per_row_per_pool}, max capa per pool {max_capa_per_pool}')
for p in range(n_pools):
    capacity.append(model.NewIntVar(min_capacity_per_pool, max_capa_per_pool, f'capacity[{p}]'))
    model.Add(capacity[p] == sum([y[m][p] * servers[m][1] for m in range(n_servers)]))
    pool_row_capacity_r = []
    for r in range(n_rows):
        pool_row_capacity_r.append(model.NewIntVar(min_gc_per_pool, max_gc_capa_per_pool, f'pool_row_max_capacity[{p}][{r}]'))
        tmp = []
        for m in range(n_servers):
            tmp.append(model.NewBoolVar(f'tmp[{m}][{r}][{p}]'))
            model.AddMultiplicationEquality(tmp[m], [z[m][r], y[m][p]])
        model.Add(pool_row_capacity_r[r] == capacity[p] - sum([tmp[m] * servers[m][1] for m in range(n_servers)]))
    pool_row_capacity.append(pool_row_capacity_r)

gc = []
for p in range(n_pools):
    gc.append(model.NewIntVar(min_gc_per_pool, max_gc_capa_per_pool, f'gc[{p}]'))
    model.AddMinEquality(gc[p], [pool_row_capacity[p][r] for r in range(n_rows)])
print('finished problem formulation')

#Constraints
#C1
filter_by_occupied = False
for r in range(n_rows):
    u_r = [unavailable[u][1] for u in range(n_unavailable) if unavailable[u][0] == r]
    u_r.sort()
    filter_by_occupied == len(u_r) > 0
    for s in range(n_slots):
        left_bound = 0
        right_bound = n_slots - 1
        if filter_by_occupied:
            if s in u_r:
                continue
            if u_r[0] < s:
                left_bound = max([u for u in u_r if u - s < 0])
            if u_r[-1] > s:
                right_bound = min([u for u in u_r if u - s > 0])
        for m in range(n_servers):
            size_m = servers[m][0]
            for m_ in range(n_servers):
                size_m_ = servers[m_][0]
                # Symmetry breakers
                lower_bound = -size_m_ + 1
                upper_bound = size_m
                if size_m_ < size_m:
                    lower_bound = min(lower_bound, left_bound)
                else:
                    upper_bound = max(upper_bound,right_bound)
                blocked_indices = [s + i for i in range(lower_bound, upper_bound)\
                                  if ((s + i < n_slots) and (s + i >= 0) and (i != 0 or m_ != m))]
                model.Add(sum([x[r][i][m_] for i in blocked_indices]) == 0).OnlyEnforceIf(x[r][s][m])
print('finished C1')

#C2
for (r, s) in unavailable:
    for m in range(n_servers):
        size_m = servers[m][0]
        model.Add(sum([x[r][s - i][m] for i in range(size_m) if s - i >= 0]) == 0)
print('finished C2')

#C3
for m in range(n_servers):
    size_m = servers[m][0]
    for r in range(n_rows):
        model.Add(sum([x[r][n_slots - 1 - i][m] for i in range(size_m - 1)]) == 0)
print('finished C3')

#C4 / C5
for m in range(n_servers):
    model.Add(sum([x[r][s][m] for r in range(n_rows) for s in range(n_slots)]) == sum([y[m][p] for p in range(n_pools)]))
    model.Add(sum([y[m][p] for p in range(n_pools)]) <= 1)
print('finished C4')

#Objective
objective = model.NewIntVar(min_gc_per_pool, max_gc_capa_per_pool, 'minimum_gc')
model.AddMinEquality(objective, gc)
model.Maximize(objective)

#Break symmetry
#S1
capacity_list = [servers[m][1] for m in range(n_servers)]
for p in range(n_pools):
    imax = np.argmax(capacity_list)
    model.Add(y[imax][p] == 1)
    capacity_list[imax] = 0
print('finished model setup\nSolving...')

"""
Model solve and display
"""
solver = cp_model.CpSolver()
solver.parameters.num_search_workers = 100
solver.parameters.linearization_level = 2
status = solver.Solve(model)
if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
    print("Solutions found!")
    print(f'Optimal total value: {solver.ObjectiveValue()}.')
    if not gcp_mode:
        for p in range(n_pools):
            print(f'capacity pool {p}: {solver.Value(capacity[p])}')
            for r in range(n_rows):
                print(f'pool[{p}][{r}]:{solver.Value(pool_row_capacity[p][r])}')
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
                for s in range(n_slots):
                    for r in range(n_rows):
                        if(solver.Value(x[r][s][m]) * solver.Value(y[m][p])):
                            print(f'({r},{s},{m},{p})')
                            for i in range(size_m):
                                table_colors[r][s + i] = colors[col_index]
                                cells[r][s + i] = f's{m}'
        ax.table(cellText=cells, loc='center', cellColours=table_colors)
        ax.set_title(filenames[index])
        plt.savefig('optimize_datacenter/results/' + filenames[index] + '.jpg')
        plt.show()
else:
    print('No solution found.')