from numpy.lib.arraysetops import unique
from ortools.sat.python import cp_model
from time import time
from tabulate import tabulate
import numpy as np
import collections

gcp_mode = True
if not gcp_mode:
    import matplotlib.pyplot as plt

class VarArraySolutionPrinterWithLimit(cp_model.CpSolverSolutionCallback):
  """Print intermediate solutions + early breaking"""

  def __init__(self, variables, limit):
    cp_model.CpSolverSolutionCallback.__init__(self)
    self.__variables = variables
    self.__solution_count = 0
    self.__solution_limit = limit

  def on_solution_callback(self):
    self.__solution_count += 1
    print('Solution %i' % self.__solution_count)
    print('objective value = %i' % self.ObjectiveValue())
    for v in self.__variables:
      print('%s=%i' % (v, self.Value(v)), end=' ')
    print()
    if self.__solution_count >= self.__solution_limit:
      print('Stop search after %i solutions' % self.__solution_limit)
      self.StopSearch()

  def solution_count(self):
    return self.__solution_count

def main():
    filenames = ['example.in', 'dc.in']
    index = 0
    if gcp_mode:
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
    print(f'Problem setup for file {filenames[index]}:')
    print(f'Datacenter:\n-Rows {n_rows} x Slots {n_slots}\n-Unavailable slots {n_unavailable}')
    print(f'Pools {n_pools}, Servers {n_servers}')
    print('Servers:')
    print(f'--Capacity: total capacity {total_capacity}, min capacity {min_capacity},\
    median capacity {median_capacity}, mean capacity {mean_capacity} max capacity {max_capacity}')
    print(f'--Size: min size {min_size}, max size {max_size}')
    all_sizes = list(set([servers[m][0] for m in range(n_servers)]))
    all_sizes.sort()
    unique_servers = collections.defaultdict(list)
    for s in all_sizes:
        servers_list = [m for m in range(n_servers) if servers[m][0] == s]
        relevant_capacities = [servers[m][1] for m in servers_list]
        servers_idxs = np.argsort(-np.asarray(relevant_capacities))
        servers_list_sorted = [servers_list[idx] for idx in servers_idxs]
        unique_servers[s] = servers_list_sorted
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
            size_m = servers[m][0]
            slot_index = model.NewIntVar(0, n_slots - size_m, f'start[{r}][{m}]')
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
    precedence_r,m: which server preceeds
    """

    capacity = []
    pool_row_capacity = []
    # min_capa_per_row_per_pool = min_capacity
    # min_capa_per_row_per_pool = int(0.5 * (median_capacity + min_capacity))
    min_capa_per_row_per_pool = int(0.9 * median_capacity)
    min_capacity_per_pool = int(min_capa_per_row_per_pool * n_rows)
    min_gc_per_pool = int(min_capa_per_row_per_pool * (n_rows - 1))
    max_capa_per_pool_per_row = int(1.5 * total_capacity / n_pools / n_rows)
    max_capa_per_pool = int(max_capa_per_pool_per_row * n_rows)
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
    for r in range(n_rows):
        model.AddNoOverlap([x[r, m].interval for m in range(n_servers)])

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

    print('formulate decision strategies')
    #for r in range(n_rows):
    #    model.AddDecisionStrategy([x[r, m].start for m in range(n_servers)],
    #                            cp_model.CHOOSE_FIRST, cp_model.SELECT_MIN_VALUE)
    #model.AddDecisionStrategy([y[m][p] for m in range(n_servers) for p in range(n_pools)],
    #                         cp_model.CHOOSE_FIRST, cp_model.SELECT_MIN_VALUE)
    #for p in range(n_pools):
    #    for r in range(n_rows):
    #        model.AddDecisionStrategy([pool_row_capacity[p][r]], cp_model.CHOOSE_FIRST, cp_model.SELECT_MAX_VALUE)


    # symmetry break
    print('formulate symmetry break S1')
    # S1: distribute biggest servers in size on pools
    capacity_list = [servers[m][1] for m in range(n_servers)]
    for p in range(n_pools):
        imax = np.argmax(capacity_list)
        model.Add(y[imax][p] == 1)
        capacity_list[imax] = 0

    print('formulate symmetry break S2')
    # S2: use identical servers in order, force placing servers by increasing capacity size first
    for s in unique_servers.keys():
        servers_list = unique_servers[s]
        if len(servers_list) > 1:
            for i in range(1, len(servers_list)):
                used_server_current = model.NewBoolVar(f' ')
                m = servers_list[i]
                model.Add(sum([x[r, m].presence for r in range(n_rows)]) == 1).OnlyEnforceIf(used_server_current)
                model.Add(sum([x[r, m].presence for r in range(n_rows)]) == 0).OnlyEnforceIf(used_server_current.Not())
                used_server_prev = model.NewBoolVar(f' ')
                m_ = servers_list[i - 1]
                model.Add(sum([x[r, m_].presence for r in range(n_rows)]) == 1).OnlyEnforceIf(used_server_prev)
                model.Add(sum([x[r, m_].presence for r in range(n_rows)]) == 0).OnlyEnforceIf(used_server_prev.Not())
                model.AddImplication(used_server_current, used_server_prev)

    print('formule symmetry break S3')
    # S3: assign values to pool once the previous one was started
    for p in range(1, n_pools):
        started_previous_pool = model.NewBoolVar(f'started_prev_pool[{p}]')
        model.Add(sum([y[m][p - 1] for m in range(n_servers)]) > 0).OnlyEnforceIf(started_previous_pool)
        model.Add(sum([y[m][p - 1] for m in range(n_servers)]) == 0).OnlyEnforceIf(started_previous_pool.Not())
        started_current_pool = model.NewBoolVar(f'started_curr_pool[{p}]')
        model.Add(sum([y[m][p] for m in range(n_servers)]) > 0).OnlyEnforceIf(started_current_pool)
        model.Add(sum([y[m][p] for m in range(n_servers)]) == 0).OnlyEnforceIf(started_current_pool.Not())
        model.AddImplication(started_current_pool, started_previous_pool)

    print('formulate symmetry break S4')
    # S4: assign values to rows once the previous one was started
    for r in range(1, n_rows):
        started_previous_row = model.NewBoolVar(f'started_row[{r}]')
        model.Add(sum([x[r - 1, m].presence for m in range(n_servers)]) > 0).OnlyEnforceIf(started_previous_row)
        model.Add(sum([x[r - 1, m].presence for m in range(n_servers)]) == 0).OnlyEnforceIf(started_previous_row.Not())
        started_current_row = model.NewBoolVar(f'started_current_row[{r}]')
        model.Add(sum([x[r, m].presence for m in range(n_servers)]) > 0).OnlyEnforceIf(started_current_row)
        model.Add(sum([x[r, m].presence for m in range(n_servers)]) == 0).OnlyEnforceIf(started_current_row.Not())
        model.AddImplication(started_current_row, started_previous_row)
    
    print('formulate hints')
    # hints
    for p in range(n_pools):
        model.AddHint(gc[p], max_gc_capa_per_pool)
        model.AddHint(capacity[p], max_capa_per_pool)
        servers_per_pool = model.NewIntVar(1, n_servers, f'server_per_pool[{p}]')
        model.Add(servers_per_pool == sum([y[m][p] for m in range(n_servers)]))
        model.AddHint(servers_per_pool, n_servers // n_pools)
        for r in range(n_rows):
            model.AddHint(pool_row_capacity[p][r], max_capa_per_pool_per_row)

    for r in range(n_rows):
        servers_per_row = model.NewIntVar(1, n_servers, f'server_per_row[{r}]')
        model.Add(servers_per_row == sum([x[r, m].presence for m in range(n_servers)]))
        model.AddHint(servers_per_row, n_servers // n_rows)


    """
    Model solve and display
    """
    print('finished problem formulation\nSolving...')
    variables_list = [gc[p] for p in range(n_pools)]
    solution_printer = VarArraySolutionPrinterWithLimit(variables_list, 1)
    solver = cp_model.CpSolver()
    # solver.parameters.enumerate_all_solutions = True
    solver.parameters.num_search_workers = 64
    solver.parameters.linearization_level = 1
    solver.parameters.search_branching = cp_model.PORTFOLIO_SEARCH

    now = time()
    status = solver.Solve(model, solution_printer)
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

if __name__ == '__main__':
    main()