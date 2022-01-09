from ortools.sat.python import cp_model

weights = [48, 30, 19, 36, 36, 27, 42, 42, 36, 24, 30]
capacities = len(weights) * [100]
n_sacks = len(weights)
n_bins = len(weights)
print(f'Minimize the distribution of {n_sacks} on bins with capacity {capacities[0]}')
print(f'total weights {sum(weights)}')
model = cp_model.CpModel()

"""
Decision variables:
x_s,b = {0,1} whether sack 's' is included in bin 'b'
y_b = {0,1} whether bin 'b' gets selected or not
"""
x = []
y = []
for s in range(n_sacks):
    y.append(model.NewBoolVar(f'y[{s}]'))
    x_s = []
    for b in range(n_bins):
        x_s.append(model.NewBoolVar(f'x[{s}][{b}]'))
    x.append(x_s)

"""
Objective function
"""
model.Minimize(sum(y[b] for b in range(n_bins)))

"""
Constraints
"""
# C1
for b in range(n_bins):
    model.Add(sum(x[s][b] * weights[s] for s in range(n_sacks)) <= y[b] * capacities[b])

# C2
for s in range(n_sacks):
    model.Add(sum(x[s][b] for b in range(n_bins)) == 1)

"""
Model solve and display
"""
solver = cp_model.CpSolver()
status = solver.Solve(model)
if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
    print("Solutions found!")
    print(f'Optimal total value: {solver.ObjectiveValue()}')
    print(f'total bins used {sum(solver.Value(y[b]) for b in range(n_bins))}')
    for b in range(n_bins):
        if(solver.Value(y[b])):
            print(f'bin {b}: {sum(solver.Value(x[s][b]) * weights[s] for s in range(n_sacks))}')
        else:
            print(f'bin {b}: Not assigned')
else:
    print('No solution found.')