from ortools.sat.python import cp_model
import numpy as np
from tabulate import tabulate

filenames = ['a_an_example.in', 'b_little_bit_of_everything.in', 'c_many_ingredients.in']
index = 0
view = True
file_url = 'one_pizza/data/' + filenames[index]

# parse input
file = open(file_url, 'r')
lines = file.readlines()

preference = []
# read number of customers
line_vectorized = lines[0].rstrip('\r\n').split(' ')
n_clients = int(line_vectorized[0])
# parse customer preferences
for l in range(n_clients):
    client_l = []
    line_1 = lines[2 * l + 1].rstrip('\r\n').split(' ')
    line_2 = lines[2 * (l + 1)].rstrip('\r\n').split(' ')
    client_l.append(line_1[1:])
    client_l.append(line_2[1:])
    preference.append(client_l)
file.close()
ingredients = []
for likes, dislikes in preference:
    for like in likes:
        ingredients.append(like)
    for dislike in dislikes:
        ingredients.append(dislike)
ingredients = list(set(ingredients))
n_ingredients = len(ingredients)
pref = np.zeros((n_clients, n_ingredients))
for c, (likes, dislikes) in enumerate(preference):
    for like in likes:
        pref[c, ingredients.index(like)] = 1
    for dislike in dislikes:
        pref[c, ingredients.index(dislike)] = -1

# problem setup
if view:
    print('Preference table (P):\nP(c,i)=-1 if customer doesnt like ingredient i, 1 if he likes it, 0 if neither is true')
    pref_tab = tabulate(pref,
                        [f'{ingredients[i]}' for i in range(n_ingredients)],
                        tablefmt="fancy_grid",
                        showindex=[f'c_{c}' for c in range(n_clients)])
    print(pref_tab)
print(f'Summary:\nclients count {n_clients}\nunique ingredients count {n_ingredients}')

model = cp_model.CpModel()

"""
Decision variables
x_i=1...I: whether ingredient 'i' is included in pizza or not
"""
x = []
for i in range(n_ingredients):
    x.append(model.NewBoolVar(f'x[{i}]'))

"""
Derived useful functions
satisfaction function defined in https://docs.google.com/document/d/10lSriEA-ZhRUpy6hpYDGBIKT4ojdnWTN7MqjMwwDMUg/edit#
"""
satisfaction = []
for c in range(n_clients):
    likes = np.where(pref[c,:] > 0)[0]
    dislikes = np.where(pref[c,:] < 0)[0]
    satisfaction.append(model.NewBoolVar(f'satisfaction[{c}]'))
    for like in likes:
        model.AddImplication(satisfaction[c], x[like])
    for dislike in dislikes:
        model.AddImplication(satisfaction[c], x[dislike].Not())

"""
cost function
"""
model.Maximize(sum(satisfaction[c] for c in range(n_clients)))

"""
Model solve and display
"""
solver = cp_model.CpSolver()
status = solver.Solve(model)
if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
    print("Solutions found!")
    print(f'Optimal total value: {solver.ObjectiveValue()}.\nIncluded ingredients:')
    output = ''
    for i in range(n_ingredients):
        if(solver.Value(x[i])):
            output += f'{ingredients[i]}-'
    print(output)
    print('Customers served:')
    output = ''
    for c in range(n_clients):
        served = 1
        likes = np.where(pref[c,:] > 0)[0]
        dislikes = np.where(pref[c,:] < 0)[0]
        for like in likes:
            served *= solver.Value(x[like])
        for dislike in dislikes:
            served *= (1- solver.Value(x[dislike]))
        if served:
            output += f'{c}-'
    print(output)
else:
    print('No solution found.')