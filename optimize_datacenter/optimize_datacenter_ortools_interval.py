from numpy.core.fromnumeric import shape, size
from numpy.core.function_base import add_newdoc
from numpy.core.shape_base import block
from numpy.lib.type_check import _nan_to_num_dispatcher
from ortools.sat.python import cp_model
from time import time
from tabulate import tabulate
import numpy as np
from joblib import Parallel, delayed
import collections

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

# n_pools = 1

total_capacity = sum([servers[s][1] for s in range(n_servers)])
min_capacity = min([servers[s][1] for s in range(n_servers)])
max_capacity = max([servers[s][1] for s in range(n_servers)])
median_capacity = np.median([servers[s][1] for s in range(n_servers)])
mean_capacity = total_capacity / n_servers
min_size = min([servers[s][0] for s in range(n_servers)])
max_size = max([servers[s][0] for s in range(n_servers)])
if len(unavailable) > 0:
    unavailable_per_row = [sum([unavailable[u][0] == r for u in range(len(unavailable))]) for r in range(n_rows)]
print(f'Problem setup for file {filenames[index]}:')
print(f'Datacenter:\n-Rows {n_rows} x Slots {n_slots}\n-Unavailable slots {n_unavailable}')
print(f'Pools {n_pools}, Servers {n_servers}')
print('Servers:')
print(f'--Capacity: total capacity {total_capacity}, min capacity {min_capacity},\
median capacity {median_capacity}, mean capacity {mean_capacity} max capacity {max_capacity}')
print(f'--Size: min size {min_size}, max size {max_size}')

if not gcp_mode:
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
- x_r,m = dictionary containing server 'r' presence and slot index on row 'r'
- y_m,p: whether server m belongs to pool p
"""

server_allocation = collections.namedtuple('server_allocation', 'start end presence interval')
x = {}
for r in range(n_rows):
    for m in range(n_servers):
        slot_index = model.NewIntVar(0, n_slots, f'start[{r}][{m}]')
        end_index = model.NewIntVar(0, n_slots, f'end[{r}][{m}]')
        presence_var = model.NewBoolVar(f'presence[{r}][{m}]')
        interval_var = model.NewOptionalIntervalVar(slot_index, servers[m][0], end_index, presence_var, f'interval[{r}][{m}]')
        x[r, m] = server_allocation(start=slot_index, end=end_index, presence=presence_var, interval=interval_var)

y = []
for m in range(n_servers):
    y_m = []
    for p in range(n_pools):
        y_m.append(model.NewBoolVar(f'y[{m}][{p}]'))
    y.append(y_m)

"""
Derived functions
capacity_p: total pool capacity
pool_row_capacity_p,r: remaining capacity in pool 'p' after row 'r' drops
gc_p:guaranteed capacity for each pool
"""

capacity = []
pool_row_capacity = []
# min_capa_per_row_per_pool = min_capacity
min_capa_per_row_per_pool = int(0.5 * (median_capacity + min_capacity))
min_capacity_per_pool = int(min_capa_per_row_per_pool * n_rows)
min_gc_per_pool = int(min_capa_per_row_per_pool * (n_rows - 1))
max_capa_per_pool = int(1.5 * total_capacity / n_pools)
max_capa_per_pool_per_row = int(1.5 * total_capacity / n_pools / n_rows)
max_gc_capa_per_pool = int(max_capa_per_pool_per_row * (n_rows - 1))
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
            model.AddMultiplicationEquality(tmp[m], [x[r, m].presence, y[m][p]])
        model.Add(pool_row_capacity_r[r] == capacity[p] - sum([tmp[m] * servers[m][1] for m in range(n_servers)]))
    pool_row_capacity.append(pool_row_capacity_r)

gc = []
for p in range(n_pools):
    gc.append(model.NewIntVar(min_gc_per_pool, max_gc_capa_per_pool, f'gc[{p}]'))
    model.AddMinEquality(gc[p], [pool_row_capacity[p][r] for r in range(n_rows)])

#Constraints
print('formulate C1')
#C1
#for r in range(n_rows):
#    model.AddNoOverlap([x[r, m].interval for m in range(n_servers)])
model.AddNoOverlap([x[0, m].interval for m in range(n_servers)])
model.AddNoOverlap([x[1, m].interval for m in range(n_servers)])

print('formulate C2')
#C2
for (r, s) in unavailable:
    for m in range(n_servers):
        size_m = servers[m][0]
        for i in range(size_m):
            if s - i < 0:
                continue
            model.Add(x[r, m].start != s - i)

print('formulate C3')
#C3
#for m in range(n_servers):
#    size_m = servers[m][0]
#    for r in range(n_rows):
#        for i in range(size_m - 1):
#            model.Add(x[r, m].start != n_slots - 1 - i)

print('formulate C4')
#C4
for m in range(n_servers):
    model.Add(sum([x[r, m].presence for r in range(n_rows)]) == sum([y[m][p] for p in range(n_pools)]))

print('formulate C5')
#C5
for m in range(n_servers):
    model.Add(sum([y[m][p] for p in range(n_pools)]) <= 1)

print('formulate objective')
#Objective
objective = model.NewIntVar(min_gc_per_pool, max_gc_capa_per_pool, 'minimum_gc')
model.AddMinEquality(objective, gc)
model.Maximize(objective)

print('formulate decision stragies')
model.AddDecisionStrategy([x[r, m].start for r in range(n_rows) for m in range(n_servers)], cp_model.CHOOSE_FIRST,
                        cp_model.SELECT_MIN_VALUE)
model.AddDecisionStrategy([y[m][p] for m in range(n_servers) for p in range(n_pools)], cp_model.CHOOSE_FIRST,cp_model.SELECT_MIN_VALUE)
model.AddDecisionStrategy([pool_row_capacity[p][r] for p in range(n_pools) for r in range(n_rows)], cp_model.CHOOSE_FIRST,cp_model.SELECT_MAX_VALUE)


# symmetry break
capacity_list = [servers[m][1] for m in range(n_servers)]
for p in range(n_pools):
    imax = np.argmax(capacity_list)
    model.Add(y[imax][p] == 1)
    capacity_list[imax] = 0

"""
Model solve and display
"""
print('finished problem formulation\nSolving...')
solver = cp_model.CpSolver()
solver.parameters.num_search_workers = 64
solver.parameters.linearization_level = 2
solver.parameters.search_branching = cp_model.FIXED_SEARCH

now = time()
status = solver.Solve(model)
if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
    print(f'total runtime {int(time() - now)}s')
    print("Solutions found!")
    print(f'Optimal total value: {solver.ObjectiveValue()}.')
    if not gcp_mode:
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
                    if(solver.Value(x[r, m].presence) * solver.Value(y[m][p])):
                        print(f'(row: {r}, machine: {m}, pool: {p})')
                        s = solver.Value(x[r, m].start)
                        for i in range(size_m):
                            table_colors[r][s + i] = colors[col_index]
                            cells[r][s + i] = f's{m}'
        ax.table(cellText=cells, loc='center', cellColours=table_colors)
        ax.set_title(filenames[index])
        plt.savefig('optimize_datacenter/results/' + filenames[index] + '.jpg')
        plt.show()
else:
    print('No solution found.')