import numpy as np
from ortools.sat.python import cp_model

filenames = ['a_example.in', 'b_small.in', 'c_medium.in', 'd_quite_big.in', 'e_also_big.in']
index = 4
file_url = 'more_pizza/data/' + filenames[index]
# parse input
file = open(file_url, 'r')
lines = file.readlines()
for l, line in enumerate(lines):
    if(l == 0):
        n_slices = int(line.split(' ')[0])
    if(l == 1):
        line = line.rstrip('\r\n')
        pizza = [int(p) for p in line.split(' ')]
file.close()
n_pizzas = len(pizza)

print(f'Problem setup for file {filenames[index]}:')
print(f'Maxmimum slices to order {n_slices}, out of {n_pizzas} variants')
print(f'Pizza slices count per type: {pizza}')
print('----------------------------')

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
    n_ordered_pizzas = 0
    output = 'selected pizzas models: '
    for m in range(n_pizzas):
        if(solver.Value(x[m])):
            output += f'{m} '
            n_ordered_pizzas += 1
    print(f'Solution found:\nTotal pizza slices {solver.Value(total_score)} from a total of {n_ordered_pizzas} pizzas.\nTarget is {n_slices} slices')
    print(output)
else:
    print('No solution found.')
