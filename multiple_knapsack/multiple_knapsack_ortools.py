import numpy as np
from ortools.sat.python import cp_model
from tabulate import tabulate

values = [
    10, 30, 25, 50, 35, 30, 15, 40, 30, 35, 45, 10, 20, 30, 25
]
weights = [
    48, 30, 42, 36, 36, 48, 42, 42, 36, 24, 30, 30, 42, 36, 36
]
capacities = 5 * [100]
n_sacks = len(values)
n_bins = len(capacities)
print(f'values {len(values)}, weights {len(weights)}, capacities {capacities}')
model = cp_model.CpModel()

"""
Decision variables:
x_s,b = {0,1} whether sack 's' is included in bin 'b'
"""
x = []
for s in range(n_sacks):
    x_s = []
    for b in range(n_bins):
        x_s.append(model.NewBoolVar(f'x[{s}][{b}]'))
    x.append(x_s)

"""
Derived variables:
x_b = {0 .... sum(values)} total value per bin
"""
x_bin = []
for b in range(n_bins):
    x_bin.append(model.NewIntVar(0, sum(values),f'x_bin[{b}]'))
    model.Add(x_bin[b] == sum(x[s][b] * values[s] for s in range(n_sacks)))

"""
Objective function
"""
model.Maximize(sum(x_bin[b] for b in range(n_bins)))

"""
Constraints
"""
# C1
for b in range(n_bins):
    model.Add(sum(x[s][b] * weights[s] for s in range(n_sacks)) <= capacities[b])

# C2
for s in range(n_sacks):
    model.Add(sum(x[s][b] for b in range(n_bins)) <= 1)

"""
Model solve and display
"""
solver = cp_model.CpSolver()
status = solver.Solve(model)
if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
    print("Solutions found!")
    print(f'Optimal total value: {solver.ObjectiveValue()}\nValue/Weight per bin')
    for b in range(n_bins):
        value = sum(solver.Value(x[s][b]) * values[s] for s in range(n_sacks))
        weight = sum(solver.Value(x[s][b]) * weights[s] for s in range(n_sacks))
        print(f'bin {b}:\nTotal Weight:{weight}, Total value:{value}')
else:
    print('No solution found.')