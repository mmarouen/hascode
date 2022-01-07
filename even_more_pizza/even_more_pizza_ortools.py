import numpy as np
from ortools.sat.python import cp_model
from tabulate import tabulate

filenames = ['a_example.in', 'b_little_bit_of_everything.in', 'c_many_ingredients.in']
index = 0
view = True
file_url = 'even_more_pizza/data/' + filenames[index]

# parse input
file = open(file_url, 'r')
lines = file.readlines()
teams = []
ingredients = []
for l, line in enumerate(lines):
    line_vectorized = line.rstrip('\r\n').split(' ')
    if(l == 0):
        n_pizzas = int(line_vectorized[0])
        t_2 = int(line_vectorized[1])
        t_3 = int(line_vectorized[2])
        t_4 = int(line_vectorized[3])
        for t in range(t_2):
            teams.append(2)
        for t in range(t_3):
            teams.append(3)
        for t in range(t_4):
            teams.append(4)
    if(l > 0):
        ingredients.append(line_vectorized[1:])
file.close()
n_teams = len(teams)
ingredients_unique = [ingredient for sublist in ingredients  for ingredient in sublist]
ingredients_unique = list(set(ingredients_unique))
n_ingredients = len(ingredients_unique)
recipe = np.zeros((n_pizzas, n_ingredients), dtype=int)
for p in range(n_pizzas):
    p_ingredients = ingredients[p]
    for ingredient in p_ingredients:
        i = ingredients_unique.index(ingredient)
        recipe[p, i] = 1

# problem setup
if view:
    print('Recipe table (R):\nR(p,i)=1 if pizza p contains ingredient i, 0 else')
    recipe_tab = tabulate(recipe,
                        [f'{ingredients_unique[i]}' for i in range(n_ingredients)],
                        tablefmt="fancy_grid",
                        showindex=[f'p_{m}' for m in range(n_pizzas)])
    print(recipe_tab)
    print('teams vector:\nteams[t] = count of persons in team t')
    print(teams)
print(f'Summary:\npizzas count {n_pizzas}\nunique ingredients count {n_ingredients}\nteams count {n_teams}')

model = cp_model.CpModel()

"""
Decision variables:
x_t,p = {0...teams[t]} count of pizza 'p' delivered to team 't'
y_t = {0,1} whether team 't' gets served or not
"""
x = []
y = []
for t in range(n_teams):
    y.append(model.NewBoolVar(f'y[{t}]'))
    x_p = []
    for p in range(n_pizzas):
        x_p.append(model.NewIntVar(0, teams[t],f'x[{t}][{p}]'))
    x.append(x_p)

"""
Derived functions:
ind_t,p = 0 if xt,p == 0, 1 else: whether team 't' gets pizze 'p' or not
diversity_t: count of unique ingredients served for team t
"""
ind = []
for t in range(n_teams):
    ind_p = []
    for p in range(n_pizzas):
        ind_p.append(model.NewBoolVar(f'ind[{t}][{p}]'))
    ind.append(ind_p)
for t in range(n_teams):
    for p in range(n_pizzas):
        model.Add(x[t][p] > 0).OnlyEnforceIf(ind[t][p])
        model.Add(x[t][p] == 0).OnlyEnforceIf(ind[t][p].Not())

diversity = []
for t in range(n_teams):
    diversity.append(model.NewIntVar(0, n_ingredients, f'diversity[{t}]'))
    tmp = []
    for i in range(n_ingredients):
        tmp.append(model.NewBoolVar(f'tmp[{i}]'))
        model.Add(sum(ind[t][p] * recipe[p, i] for p in range(n_pizzas)) > 0).OnlyEnforceIf(tmp[i])
        model.Add(sum(ind[t][p] * recipe[p, i] for p in range(n_pizzas)) <= 0).OnlyEnforceIf(tmp[i].Not())
    model.Add(diversity[t] == sum(tmp[i] for i in range(n_ingredients)))

"""
Constraints
"""
# C1
for t in range(n_teams):
    model.Add(sum(x[t][p] for p in range(n_pizzas)) == y[t] * teams[t])

# C2
for p in range(n_pizzas):
    model.Add(sum(ind[t][p] for t in range(n_teams)) <= 1)

"""
Cost function
"""
cost = []
for t in range(n_teams):
    cost.append(model.NewIntVar(0, n_ingredients * n_ingredients, f'cost[{t}]'))
    model.AddProdEquality(cost[t], [diversity[t], diversity[t]])
model.Maximize(sum(cost[t] for t in range(n_teams)))

"""
Model solve and display
"""
solver = cp_model.CpSolver()
status = solver.Solve(model)
if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
    print("Solutions found!")
    print(f'Optimal score: {solver.ObjectiveValue()}')
    if view:
        print('teams pizzas')
        for t in range(n_teams):
            if (not solver.Value(y[t])):
                print(f'team {t}: no pizza')
            else:
                output = f'team {t} total unique ingredients: {solver.Value(diversity[t])}. Pizzas list x quantity: '
                for p in range(n_pizzas):
                    if(solver.Value(ind[t][p])):
                        output += f'(pizza index: {p} x qty:{solver.Value(x[t][p])}) '
                print(output)

else:
    print('No solution found.')

# Statistics.
print('\nStatistics')
print('  - conflicts: %i' % solver.NumConflicts())
print('  - branches : %i' % solver.NumBranches())
print('  - wall time: %f s' % solver.WallTime())
