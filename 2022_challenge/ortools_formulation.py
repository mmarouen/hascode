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
    projects = collections.defaultdict(list)
    line_index = 0
    n_days = 100
    for l, line in enumerate(lines):
        line = line.rstrip('\r\n')
        print(l)
        if(l == 0):
            values = [int(val) for val in line.split(' ')]
            n_contributors = values[0]
            n_projects = values[1]
            line_index += 1
        else:
            for c in range(n_contributors):
                c_line = lines[line_index]
                values = [val for val in c_line.split(' ')]
                n_skills = int(values[1])
                contributor_name = values[0]
                contributors[contributor_name] = [n_skills]
                line_index += 1
                for s in range(n_skills):
                    s_line = lines[line_index]
                    values = [val for val in s_line.split(' ')]
                    contributors[contributor_name].append((values[0], int(values[1])))
                    line_index += 1
            for p in range(n_projects):
                p_line = lines[line_index]
                values = [val for val in p_line.split(' ')]
                n_project_roles = int(values[4])
                p_name = values[0]
                projects[p_name] = [int(values[1]), int(values[2]), int(values[3]), n_project_roles]
                line_index += 1
                project_roles_list = []
                for r in range(n_project_roles):
                    p_roles_line = lines[line_index]
                    values = [val for val in p_roles_line.split(' ')]
                    project_roles_list.append((values[0], int(values[1])))
                    line_index += 1
                projects[p_name].append(project_roles_list)
            break
    print(projects)
    print(contributors)
    unique_skills = []
    for c in contributors.keys():
        if len(contributors[c]) > 1:
            skills = [s[0] for s in contributors[c][1:]]
        unique_skills.append(skills)
    unique_skills = [item for sublist in unique_skills for item in sublist]
    unique_skills = list(set(unique_skills))

    skills_matrix = np.zeros((n_contributors, len(unique_skills)))
    for i, c in enumerate(contributors.keys()):
        for j, s in enumerate(unique_skills):
            skill_list_c = [s[0] for s in contributors[c][1:]]
            loc = np.where(np.asarray(skill_list_c) == s)
            if len(loc[0]) > 0:
                skills_matrix[i, j] = contributors[c][loc[0][0] + 1][1]
    print(skills_matrix)

    project_matrix = np.zeros((n_projects, len(unique_skills)))
    for i, p in enumerate(projects.keys()):
        print(f'project {p}: {projects[p]}')
        for j, s in enumerate(unique_skills):
            project_list_p = [s[0] for s in projects[p][4:][0]]
            print(f'skill {s}, project skills: {project_list_p}')
            loc = np.where(np.asarray(project_list_p) == s)
            if len(loc[0]) > 0:
                print(loc[0][0])
                print(projects[p][4])
                project_matrix[i, j] = projects[p][4][loc[0][0]][1]
    print(project_matrix)
    
    model = cp_model.CpModel()
    project_allocation = collections.namedtuple('project_allocation', 'start end presence interval score')
    x = {}
    for i, p in enumerate(projects.keys):
        p_duration = projects[p][0]
        p_score = projects[p][1]
        slot_index = model.NewIntVar(0, n_days - p_duration, f'start[{p}]')
        end_index = model.NewIntVar(0, n_days, f'end[{p}]')
        presence_var = model.NewBoolVar(f'presence[{p}]')
        interval_var = model.NewOptionalIntervalVar(slot_index, p_duration, end_index, presence_var, f'interval[{p}]')
        x[p] = project_allocation(start=slot_index, end=end_index, presence=presence_var, interval=interval_var, score=p_score)

    """
    cost
    """

    """
    Model solve and display
    """
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