import numpy as np
from numpy.core.fromnumeric import prod
from numpy.lib.function_base import append
from ortools.sat.python import cp_model
from tabulate import tabulate

# inputs
n_days = 7
books = [1, 2, 3, 4, 5, 6]
libraries = [ # foreach library: [t_l, n_l, list_of_books]
    [2, 2, [0, 1, 2, 3, 4]],
    [3, 1, [0, 2, 3, 5]]
]
n_books = len(books)
n_libs = len(libraries)
model = cp_model.CpModel()

"""
create decision variables:
- x_d,b,l: whether book "b" from library "l" gets scanned on day "d"
- y_d,l: whether library "l" gets selected on day "d"
- y_l: whether libeary "l" gets selected or not
"""
x = []
for d in range(n_days):
    x_d = []
    for b in range(n_books):
        x_b = []
        for l in range(n_libs):
            x_b.append(model.NewBoolVar(f'x[{d}][{b}][{l}]'))
        x_d.append(x_b)
    x.append(x_d)

y = []
for d in range(n_days):
    y_d = []
    for l in range(n_libs):
        y_d.append(model.NewBoolVar(f'y[{d}][{l}]'))
    y.append(y_d)

y_l = []
for l in range(n_libs):
    y_l.append(model.NewBoolVar(f'y_l[{l}]'))

"""
Derived useful functions:
x_b: whether book "b" gets scanned 
"""
x_b = []
for b in range(n_books):
    x_b.append(model.NewBoolVar(f'x_b[{b}]'))
    model.Add(sum(x[d][b][l] for l in range(n_libs) for d in range(n_days)) > 0).OnlyEnforceIf(x_b[b])
    model.Add(sum(x[d][b][l] for l in range(n_libs) for d in range(n_days)) == 0).OnlyEnforceIf(x_b[b].Not())

# Constraint C1
for b in range(n_books):
    for l in range(n_libs):
        l_books = libraries[l][2]
        if (b not in l_books):
            for d in range(n_days):
                model.Add(x[d][b][l] == 0)

# Constraint C2
for b in range(n_books):
    # model.Add(x_b[b] <= 1)
    model.Add(sum(x[d][b][l] for l in range(n_libs) for d in range(n_days)) <= 1)

# Constraint C3
for d in range(n_days):
    model.Add(sum(y[d][l] for l in range(n_libs)) <= 1)

# Constraint C4
for l in range(n_libs):
    t_l = libraries[l][0]
    model.Add(sum(y[d][l] for d in range(n_days)) <= y_l[l] * t_l)

# Constraint C5
for d in range(n_days):
    for l in range(n_libs):
        n_l = libraries[l][1]
        model.Add(sum(x[d][b][l] for b in range(n_books)) <= n_l * y_l[l])

# Constrainz C6
for l in range(n_libs):
    n_l = libraries[l][1]
    t_l = libraries[l][0]
    argmax_d = model.NewIntVar(t_l - 1, n_days, 'argmax_d')
    days_libs = []
    for d in range(n_days):
        days_libs.append(model.NewIntVar(0, n_days, f'days_libs[{d}]'))
        model.Add(days_libs[d] == d * y[d][l])
    model.AddMaxEquality(argmax_d, days_libs)
    for d in range(t_l):
        model.Add(sum(x[d][b][l] for b in range(n_books)) == 0)
    for d in range(t_l, n_days):
        bool_var = model.NewBoolVar('bool_var')
        model.Add(argmax_d < d).OnlyEnforceIf(bool_var)
        model.Add(argmax_d >= d).OnlyEnforceIf(bool_var.Not())
        model.Add(sum(x[d][b][l] for b in range(n_books)) > 0).OnlyEnforceIf(bool_var)
        model.Add(sum(x[d][b][l] for b in range(n_books)) == 0).OnlyEnforceIf(bool_var.Not())

# Constraint C7: library integrated in consecutive days
for l in range(n_libs):
    integrated = []
    t_l = libraries[l][0]
    pos = 0
    for d in range(t_l, n_days):
        integrated.append(model.NewBoolVar(f'integrated_{pos}'))
        for i in range(d - t_l, d):
            model.AddImplication(integrated[pos], y[i][l])
        pos += 1
    model.Add(sum(integrated[pos] for pos in range(len(integrated))) == 1).OnlyEnforceIf(y_l[l])
    model.Add(sum(integrated[pos] for pos in range(len(integrated))) == 0).OnlyEnforceIf(y_l[l].Not())

# objective
total_score = model.NewIntVar(0, 200, 'total_score')
model.Add(total_score == sum(books[b] * x_b[b] for b in range(n_books)))
model.Maximize(total_score)

# Creates the solver and solve.
solver = cp_model.CpSolver()
status = solver.Solve(model)

if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
    print(f'Solution found: total score {solver.Value(total_score)}')
    print("Day x libraries tables")
    tab = np.zeros((n_libs, n_days), dtype=object)
    for l in range(n_libs):
        if(not solver.Value(y_l[l])):
            continue
        for d in range(n_days):
            txt = ''
            if(solver.Value(y[d][l])):
                txt = f'int l_{l}'
            for b in range(n_books):
                if(solver.Value(x[d][b][l])):
                    txt += f'b_{b}'
            tab[l, d] = txt
    table = tabulate(tab,
                     [f'd_{d}' for d in range(n_days)],
                     tablefmt="fancy_grid",
                     showindex=[f'l_{l}' for l in range(n_libs)])
    print(table)
else:
    print('No solution found.')

# Statistics.
print('\nStatistics')
print('  - conflicts: %i' % solver.NumConflicts())
print('  - branches : %i' % solver.NumBranches())
print('  - wall time: %f s' % solver.WallTime())
