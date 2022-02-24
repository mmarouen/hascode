from numpy.core.fromnumeric import size
from numpy.lib.arraysetops import unique
from ortools.sat.python import cp_model
from time import time
from tabulate import tabulate
import numpy as np
import collections

gcp_mode = True
if not gcp_mode:
    import matplotlib.pyplot as plt

class VarArraySolutionPrinterWithLimit(cp_model.CpSolverSolutionCallback):
  """Print intermediate solutions + early breaking"""

  def __init__(self, variables, limit):
    cp_model.CpSolverSolutionCallback.__init__(self)
    self.__variables = variables
    self.__solution_count = 0
    self.__solution_limit = limit

  def on_solution_callback(self):
    self.__solution_count += 1
    print('Solution %i' % self.__solution_count)
    print('objective value = %i' % self.ObjectiveValue())
    for v in self.__variables:
      print('%s=%i' % (v, self.Value(v)), end=' ')
    print()
    if self.__solution_count >= self.__solution_limit:
      print('Stop search after %i solutions' % self.__solution_limit)
      self.StopSearch()

  def solution_count(self):
    return self.__solution_count

def main():
    filenames = ['a_an_example.in.txt', 'b_better_start_small.in.txt']
    index = 0
    file_url = '2022_challenge/data/' + filenames[index]
    # parse input
    file = open(file_url, 'r')
    lines = file.readlines()
    n_contributors = 0
    skills_matrix = []
    n_projects = 0
    contributors = collections.defaultdict(list)
    for l, line in enumerate(lines):
        line = line.rstrip('\r\n')
        if(l == 0):
            values = [int(val) for val in line.split(' ')]
            n_contributors = values[0]
            n_projects = values[1]
        else:
            values = [val for val in line.split(' ')]
            c = 0
            while(c < n_contributors):
                n_skills = int(values[1])
                contributor_name = values[0]
                contributors[values[0]] = [contributor_name]
                s_index = 0
                while(s_index < n_skills):
                    contributors[contributor_name].append((values[0], values[1]))
                    s_index += 1
                c += 1

    print(contributors)

    """
    Model solve and display
    """
    print('finished problem formulation\nSolving...')
    """    #variables_list = [gc[p] for p in range(n_pools)]
    solution_printer = VarArraySolutionPrinterWithLimit(variables_list, 1)
    solver = cp_model.CpSolver()
    # solver.parameters.enumerate_all_solutions = True
    solver.parameters.num_search_workers = 64
    solver.parameters.linearization_level = 1
    solver.parameters.search_branching = cp_model.FIXED_SEARCH

    now = time()
    status = solver.Solve(model, solution_printer)
    if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
        print(f'total runtime {int(time() - now)}s')
        print("Solutions found!")
        print(f'Optimal total value: {solver.ObjectiveValue()}.')
    else:
        print('No solution found.')
    """
if __name__ == '__main__':
    main()