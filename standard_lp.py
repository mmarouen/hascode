from ortools.linear_solver import pywraplp
from ortools.init import pywrapinit
import matplotlib.pyplot as plt
import numpy as np

def main():
    # Create the linear solver with the GLOP backend.
    solver = pywraplp.Solver.CreateSolver('GLOP')


    """
    Formalization of the optimization problem:
     max cost[0] * x + cost[1] * y
     under constraints: 
        x_bounds[0] <= x <= x_bounds[1]
        y_bounds[0] <= y <= y_bounds[1]
        constraint[0] * x + constraint[1] * y <= constraint[2]
    """
    x_bounds = [40, 200] # x_bounds[0] <= x <= x_bounds[1]
    y_bounds = [25, 200] # y_bounds[0] <= y <= y_bounds[1]
    cost = [225, 200] # maximize cost[0] * x + cost[1] * y
    constraint = [1, 1, 150] # under constraint: constraint[0] * x + constraint[1] * y <= constraint[2]

    x = solver.NumVar(x_bounds[0], x_bounds[1], 'x')
    y = solver.NumVar(y_bounds[0], y_bounds[1], 'y')

    print('Number of variables =', solver.NumVariables())

    # Create a linear constraint, 0 <= x + y <= 2.
    ct = solver.Constraint(0, constraint[2], 'ct')
    ct.SetCoefficient(x, constraint[0])
    ct.SetCoefficient(y, constraint[1])

    print('Number of constraints =', solver.NumConstraints())

    # Create the objective function, 3 * x + y.
    objective = solver.Objective()
    objective.SetCoefficient(x, cost[0])
    objective.SetCoefficient(y, cost[1])
    objective.SetMaximization()
    solver.Solve()

    print('Solution:')
    print('Objective value =', objective.Value())
    print('x =', x.solution_value())
    print('y =', y.solution_value())

    x_coords = np.linspace(x_bounds[0], x_bounds[1], 20)
    #max_vals = [20000, 30000, 40000]
    max_vals = []
    x_vals = [x_bounds[0], constraint[2] - x_bounds[0], ]
    for x_val in x_vals:
        max_vals.append(cost[0] * x_val + cost[1] * (constraint[2] - constraint[0] * x_val) / constraint[1])
    #max_vals.append(cost[0] * x_bounds[0] + cost[1] * 5 * y_bounds[0])
    #max_vals.append(cost[0] * 5 * x_bounds[0] + cost[1] * y_bounds[0])
    cols = ["r", "g", "yellow"]
    plt.hlines(y=y_bounds[1], xmin=x_bounds[0],xmax=x_bounds[1], colors="b", linestyles="--", label="y={}".format(y_bounds[1]))
    plt.hlines(y=y_bounds[0], xmin=x_bounds[0],xmax=x_bounds[1], colors="b", linestyles="--", label="y={}".format(y_bounds[0]))
    plt.vlines(x=x_bounds[1], ymin=y_bounds[0], ymax=y_bounds[1], colors="b", linestyles="--", label="x={}".format(x_bounds[1]))
    plt.vlines(x=x_bounds[0], ymin=y_bounds[0], ymax=y_bounds[1], colors="b", linestyles="--", label="x={}".format(x_bounds[0]))
    plt.plot(x_coords, (constraint[2] - constraint[0] * x_coords) / constraint[1], '--b', label="{}x+{}y={}".format(constraint[0], constraint[1], constraint[2]))
    for i, max_val in enumerate(max_vals):
        plt.plot(x_coords, (max_val - cost[0] * x_coords) / cost[1], linestyle="-", color=cols[i], label="{}x+{}y={}".format(cost[0], cost[1], max_val))
    plt.legend(loc='upper left')
    plt.grid()
    plt.show()

if __name__ == '__main__':
    pywrapinit.CppBridge.InitLogging('basic_example.py')
    cpp_flags = pywrapinit.CppFlags()
    cpp_flags.logtostderr = True
    cpp_flags.log_prefix = False
    pywrapinit.CppBridge.SetFlags(cpp_flags)

    main()