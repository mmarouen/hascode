import numpy as np
from djkstra import Dijkstra
from tabulate import tabulate

# define input
data = np.genfromtxt('search_algorithms/adjacency_matrix/data_1.csv',
                     delimiter=", ", dtype=int, autostrip=True)
start_node = (2, 2)
end_node = (6, 6)

# plot input data
table = tabulate(data, tablefmt="fancy_grid")
print(table)

dijkstra = Dijkstra(start_node, end_node, data)
path = dijkstra.getPathFromSrcToDst()
dijkstra.printSolution(path)