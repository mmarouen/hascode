# code for the dijkstra based on the wikipedia page pseudo code
# https://en.wikipedia.org/wiki/Dijkstra%27s_algorithm

import numpy as np
from tabulate import tabulate

inf = 1e5

class Dijkstra():
	"""! Dijkstra implementation with tools
	Finds the optimal path (assuming it exists) between a starting point and an end point
	The map between source and target is a grid containing:
	 - obstacles: zero value
	 - transitions from cell i -> j: positive value
	This implicitly implies that its only possible to move from one cell to another adjacent to it.
	Additionally, the current implementation eliminates diagonal moves.
	Examples:
	 - Direct transition from (0,0) -> (2, 3) is impossible
	 - Direct transition from (2, 3) is limited to the following cells [(1, 3), (3, 3), (2, 4), (2, 1)]
	Source and destination points have to be within the grid and not an obstacle point
	"""
	def __init__(self, start_node, end_node, input_data):
		"""! Disjkstra class initializer.
		@param start_node  the source of the path to compute
		@param end_node  the destination of the path
		@param input_data graph describing blocked cells (value=0) and transition costs
		@return  Instance of Dijkstra class
		"""
		self.start_node = start_node
		self.end_node = end_node
		self.graph = input_data
		self.distances_matrix = inf * np.ones_like(input_data)
		self.distances_matrix[start_node[0], start_node[1]] = 0

	def findNextMin(self, dist_matrix, visited_set):
		"""! Finds the cell with min distance that is not yet examined
		@param dist_matrix  weights matrix
		@param visited_set the record of visited cells
		@return pair containing the coordinates of the cell with minimum distance
		"""
		tmp_dist = dist_matrix.copy()
		tmp_dist[visited_set == 0] = 1e4
		return np.unravel_index(tmp_dist.argmin(), tmp_dist.shape)

	def isValidNeighbor(self, u, adjancency, visited_set):
		"""! Identifies whether the input cell index is valid or not.
		Validity is based on:
		 - Cell within grid
		 - Cell is not an obstacle
		@param u cell index to examine
		@param adjancency adjancency matrix
		@param visited_set record of visited cells
		@return boolean
		"""
		x_lim = adjancency.shape[0]
		y_lim = adjancency.shape[1]
		is_valid = u[0] < x_lim and u[0] >= 0
		is_valid *= u[1] < y_lim and u[1] >= 0
		if(is_valid):
			is_valid *= adjancency[u[0], u[1]] > 0
			is_valid *= visited_set[u[0], u[1]] > 0
		return is_valid

	def getNeighbors(self, u, adjancency, visited_set):
		"""! Finds the valid neighbours for a given cell
		@param u cell to find the neighbours to
		@param adjancency adjancency matrix
		@param visited_set record of visited cells
		@return list of neighboours
		"""
		x_1 = u[0] - 1
		x_2 = u[0] + 1
		y_1 = u[1] - 1
		y_2 = u[1] + 1
		valid_neighbors = []
		candidate_list = [(x_2, u[1]), (x_1, u[1]), (u[0], y_1), (u[0], y_2)]
		for candidate in candidate_list:
			if(self.isValidNeighbor(candidate, adjancency, visited_set)):
				valid_neighbors.append(candidate)
		return valid_neighbors

	def getShortestPaths(self):
		"""! Runs the Dijkstra algorithm on the class input
		The implementation is based on the pseudo code provided on the
		wiki page https://en.wikipedia.org/wiki/Dijkstra%27s_algorithm
		@return dict of previous cells for each visited cell
		"""
		prevs = {}
		current_node = self.start_node
		working_set = np.ones_like(self.graph)
		while (current_node[0] != self.end_node[0] or current_node[1] != self.end_node[1]):
			current_node = self.findNextMin(self.distances_matrix, working_set)
			working_set[current_node[0], current_node[1]] = 0
			neighbors = self.getNeighbors(current_node, self.graph, working_set)
			for neighbor in neighbors:
				dist_potential = self.distances_matrix[current_node[0], current_node[1]] + self.graph[neighbor[0], neighbor[1]]
				if (dist_potential  < self.distances_matrix[neighbor[0], neighbor[1]]):
					self.distances_matrix[neighbor[0], neighbor[1]] = dist_potential
					prevs[neighbor] = current_node
		return prevs
	
	def getPathFromSrcToDst(self):
		"""! Given a computed previous dict and the class setup,
		the function computes the path from source -> destination
		@return list of cells traversal from source -> destination
		"""
		prevs = self.getShortestPaths()
		current_node = self.end_node
		path = []
		path.append(self.end_node)
		while (current_node[0] != self.start_node[0] or current_node[1] != self.start_node[1]):
			path.append(prevs[current_node])
			current_node = prevs[current_node]
		path.reverse()
		return path

	def printSolution(self, path):
		"""! Plots the path within the grid from source to destination
		"""
		tab = np.array([['' for d in range(self.graph.shape[1])] for d in range(self.graph.shape[0])], dtype=object)
		for p in path:
			tab[p[0], p[1]] = "x"
		tab[self.start_node[0], self.start_node[1]] = "Start"
		tab[self.end_node[0], self.end_node[1]] = "End"
		tab[self.graph == 0] = "Obs"
		table = tabulate(tab, tablefmt="fancy_grid")
		print(table)
