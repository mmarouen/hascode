import numpy as np
from ortools.sat.python import cp_model
from tabulate import tabulate

# inputs
n_machines = 3
jobs = [[(0, 3), (1, 2), (2, 2)],
        [(0, 2), (2, 1), (1, 4)],
        [(1, 4), (2, 3)]]
n_jobs = len(jobs)
n_tasks = max([len(tasks) for tasks in jobs])
machines = np.zeros((n_jobs, n_tasks, n_machines), dtype=int)
durations = np.zeros((n_jobs, n_tasks),dtype=int)
for t in range(n_tasks):
    for j in range(n_jobs):
        if(j + 1 > len(jobs[j]) and t == 2):
            continue
        machines[j, t, jobs[j][t][0]] = 1
        durations[j, t] = jobs[j][t][1]
n_days = int(max(np.sum(durations, axis=1)) + 5)
model = cp_model.CpModel()

"""
create decision variables:
x_j,t,d: whether task "t" from job "j" runs at day "d"
"""
x = []
for j in range(n_jobs):
    x_t = []
    for t in range(n_tasks):
        x_d = []
        for d in range(n_days):
            x_m = []
            for m in range(n_machines):
                x_m.append(model.NewBoolVar(f'x[{j}][{t}][{d}][{m}]'))
            x_d.append(x_m)
        x_t.append(x_d)
    x.append(x_t)

# constraint C3
for j in range(n_jobs):
    for t in range(n_tasks):
        for m in range(n_machines):
            if(machines[j, t, m] == 0):
                for d in range(n_days):
                    model.Add(x[j][t][d][m] == 0)

# constraint C4
for d in range(n_days):
    for m in range(n_machines):
        model.Add(sum(x[j][t][d][m] for t in range(n_tasks) for j in range(n_jobs)) <= 1)
"""
# constraint C2
for t in range(1, n_tasks):
    for j in range(n_jobs):
        duration_tj = durations[j, t]
        m_j = np.argmax(machines[j, t, :])
        m_j_1 = np.argmax(machines[j, t - 1, :])
        for d in range(n_days):
            model.Add(x[j][t][d][m_j] == 0).OnlyEnforceIf(model.AddBoolAnd([x[j][t - 1][i][m_j_1].Not() for i in range(d - 1)]))
"""

"""
# constraint C1
for j in range(n_jobs):
    for t in range(n_tasks):
            model.Add(sum(x[j][t][d][m] for d in range(n_days) for m in range(n_machines)) == durations[j, t])
"""


# constraint C1
for t in range(n_tasks):
    for j in range(n_jobs):
        duration_tj = durations[j, t]
        products = []
        m_jt = np.argmax(machines[j, t, :])
        for d in range(n_days - duration_tj):
            tomultiply = [x[j][t][i][m_jt] for i in range(d, d + duration_tj)]
            multiply_var = model.NewIntVar('multiply_var')
            #model.AddMultiplicationEquality(multiply_var, tomultiply)
            products.append(tomultiply)
        #model.AddBoolOr(products)
        model.Add(sum(x[j][t][d][m_jt] for d in range(n_days)) == duration_tj)

# objective
max_makespan = model.NewIntVar(1, n_days, 'max_makespan')
model.AddMaxEquality(max_makespan, [d * x[j][t][d][m] for d in range(n_days) for t in range(n_tasks) for j in range(n_jobs) for m in range(n_machines)])
model.Minimize(max_makespan + 1)

# Creates the solver and solve.
solver = cp_model.CpSolver()
status = solver.Solve(model)

if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
    print('day x machine table:')
    # Create per machine output lines.
    tab = np.zeros((n_machines, n_days), dtype=object)
    for d in range(n_days):
        for m in range(n_machines):
            var_msg = ''
            for j in range(n_jobs):
                for t in range(n_tasks):
                    value = solver.Value(x[j][t][d][m])
                    if(value):
                        tab[m, d] = f'{j}-{t}'
    table = tabulate(tab, range(n_days), tablefmt="fancy_grid", showindex=True)
    print(table)
    # Finally print the solution found.
    print(f'Optimal Schedule Length: {solver.ObjectiveValue()}')
else:
    print('No solution found.')

# Statistics.
print('\nStatistics')
print('  - conflicts: %i' % solver.NumConflicts())
print('  - branches : %i' % solver.NumBranches())
print('  - wall time: %f s' % solver.WallTime())
