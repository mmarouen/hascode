from numpy.core.fromnumeric import shape
from ortools.sat.python import cp_model
from time import time
from tabulate import tabulate
import numpy as np
import matplotlib.pyplot as plt

filenames = ['example.in', 'small.in', 'medium.in', 'big.in']
index = 1
file_url = 'pizza/data/' + filenames[index]
view = True
# parse input
file = open(file_url, 'r')
lines = file.readlines()
ingredients = np.zeros((10,))
for l, line in enumerate(lines):
    line = line.rstrip('\r\n')
    if(l == 0):
        values = [int(val) for val in line.split(' ')]
        n_rows = values[0]
        n_cols = values[1]
        min_val = values[2]
        max_val = values[3]
        ingredients = np.zeros((n_rows, n_cols))
    if(l > 0):
        ingredients_list = list(line)
        ingredients[l - 1, :] = 1 * ["M" == ingredient for ingredient in ingredients_list]
file.close()

# create the list of shapes
shapes = []
for r in range(min(max_val, n_rows)): # a shape cannot exceed grid dimensions
    for c in range(min(max_val, n_cols)):
        s = (r + 1) * (c + 1)
        if(s <= max_val and s > 2 * min_val):
            shapes.append((r + 1, c + 1, s))
n_shapes = len(shapes)

print(f'Problem setup for file {filenames[index]}:')
print(f'Rows {n_rows}, cols {n_cols}, L {min_val}, H {max_val}')
print(f'{n_shapes} shapes were identified')
if view:
    print(f'shapes: {shapes}')
    print('pizza ingredients: (0: tomatoes, 1:mushrooms)')
    table = tabulate(ingredients, tablefmt="fancy_grid")
    print(table)
print('----------------------------')

model = cp_model.CpModel()

"""
create decision variables:
- x_r,c,s: whether cell (r,c) belongs to shape s
-occupancy_r,c: whether cell (r,c) is occupied
"""
x = []
occupancy = {}
for r in range(n_rows):
    x_r = []
    for c in range(n_cols):
        x_c = []
        for s in range(n_shapes):
            x_c.append(model.NewBoolVar(f'x[{r}][{c}][{s}]'))
            occupancy[(r, c, s)] = model.NewBoolVar(f'occurancy[({r},{c},{s})]')
        x_r.append(x_c)
    x.append(x_r)

y = []
for s in range(n_shapes):
    y.append(model.NewIntVar(0, 10, f'y[{s}]'))
    model.Add(y[s] == sum(x[r][c][s] for r in range(n_rows) for c in range(n_cols)))

# constraints
#C1: already satsified by variable definitions

#C2
for r in range(n_rows):
    for c in range(n_cols):
        model.Add(sum(x[r][c][s] for s in range(n_shapes)) <= 1)
        model.Add(sum(occupancy[(r, c, s)] for s in range(n_shapes)) <= 1)

#C3
for s in range(n_shapes):
    n_rs = shapes[s][0]
    n_cs = shapes[s][1]
    for r in range(n_rows):
        for c in range(n_cols):
            for i in range(-n_rs + 1, n_rs):
                for j in range(-n_cs + 1, n_cs):
                    if (i == 0 and j == 0):
                        model.Add(occupancy[(r, c, s)] == 1).OnlyEnforceIf(x[r][c][s])
                        continue
                    if (r + i < n_rows and c + j < n_cols and r + i >= 0 and c + j >= 0):
                        model.Add(x[r + i][c + j][s] == 0).OnlyEnforceIf(x[r][c][s])
                    if(i >= 0 and j >=0 and r + i < n_rows and c + j < n_cols):
                            model.Add(occupancy[(r + i, c + j, s)] == 1).OnlyEnforceIf(x[r][c][s])

#C4
for s in range(n_shapes):
    n_rs = shapes[s][0]
    n_cs = shapes[s][1]
    for r in range(n_rows - n_rs + 1, n_rows):
        for c in range(n_cols):
            model.Add(x[r][c][s] == 0)
    for c in range(n_cols - n_cs + 1, n_cols):
        for r in range(n_rows):
            model.Add(x[r][c][s] == 0)

#C5
b_ingredients = {}
for s in range(n_shapes):
    n_rs = shapes[s][0]
    n_cs = shapes[s][1]
    surface_s = shapes[s][2]
    for r in range(n_rows - n_rs + 1):
        for c in range(n_cols - n_cs + 1):
            n_mushroom = sum(ingredients[r + i, c + j] for i in range(n_rs) for j in range(n_cs))
            n_tomatoes = surface_s - n_mushroom
            b_ingredients[(r, c, s)] = model.NewBoolVar(f'b_ingredients[({r},{c},{s})]')
            bool_test = 1* (n_mushroom >= min_val and n_tomatoes >= min_val)
            model.Add(b_ingredients[(r, c, s)] == bool_test)
            model.AddImplication(x[r][c][s], b_ingredients[(r, c, s)])

# objective function
objective = model.NewIntVar(1, n_rows * n_cols, 'objective')
model.Add(objective == sum(y[s] * shapes[s][2] for s in range(n_shapes)))
model.Maximize(objective)

"""
Model solve and display
"""
solver = cp_model.CpSolver()
status = solver.Solve(model)
if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
    print("Solutions found!")
    print(f'Optimal total value: {solver.ObjectiveValue()}.')
    cells = [["" for r in range(n_cols)] for c in range(n_rows)]
    fig, ax = plt.subplots()
    ax.xaxis.set_visible(False) 
    ax.yaxis.set_visible(False)
    colors = ['r', 'g', 'b', 'c', 'm', 'y']
    table_colors = [['w' for c in range(n_cols)] for r in range(n_rows)]
    for s in range(n_shapes):
        n_rs = shapes[s][0]
        n_cs = shapes[s][1]
        col_index = s % len(colors)
        for r in range(n_rows - n_rs + 1):
            for c in range(n_cols - n_cs + 1):
                if(solver.Value(x[r][c][s])):
                    print(f'({r},{c},{s})')
                    for i in range(n_rs):
                        for j in range(n_cs):
                            table_colors[r + i][c + j] = colors[col_index]
                            cells[r + i][c + j] = f's{s}'
    ax.table(cellText=cells, loc='center', cellColours=table_colors)
    ax.set_title(filenames[index])
    plt.savefig('pizza/results/' + filenames[index] + '.jpg')
    plt.show()
else:
    print('No solution found.')