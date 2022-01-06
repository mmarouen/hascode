import numpy as np
from ortools.sat.python import cp_model

# a_example
n_slices = 17
pizza = [2, 5, 6, 8]
# b_example
n_slices = 100
pizza = [4, 14, 15, 18, 29, 32, 36, 82, 95, 95]
# c_medium
n_slices = 4500
pizza = [7, 12, 12, 13, 14, 28, 29, 29, 30, 32, 32, 34, 41, 45, 46,
         56, 61, 61, 62, 63, 65, 68, 76, 77, 77, 92, 93, 94, 97, 103, 113, 114, 114,
         120, 135, 145, 145, 149, 156, 157, 160, 169, 172, 179, 184, 185, 189, 194, 195, 195]
# a parser should be added to sread the files

n_pizzas = len(pizza)
model = cp_model.CpModel()

"""
create decision variables:
- x_m: whether pizza "m" gets selected
"""
x = []
for m in range(n_pizzas):
    x.append(model.NewBoolVar(f'x[{m}]'))

# Constraint C1: satisfied by selecting booleans

# Constraint C2: satisfied by defining an upperbound for the score function

# objective
total_score = model.NewIntVar(0, n_slices, 'total_score')
model.Add(total_score == sum(x[m] * pizza[m] for m in range(n_pizzas)))
model.Maximize(total_score)

# Creates the solver and solve.
solver = cp_model.CpSolver()
status = solver.Solve(model)

if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
    print(f'Solution found: total score {solver.Value(total_score)}')
    output = 'selected pizzas models: '
    for m in range(n_pizzas):
        if(solver.Value(x[m])):
            output += f'{m} '
    print(output)
else:
    print('No solution found.')
