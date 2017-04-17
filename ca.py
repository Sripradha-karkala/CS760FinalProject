import numpy as np
import random
import csv
import sys
import math
import argparse

a = np.array([1.,2.,3.])
b = np.array([2.,4.,6.])

data = np.array([a,b])

class Data:
	def __init__(self, file_name, testset_split):
		with open(file_name) as input_file:
			csv.reader(input_file, delimiter = ',')
			data = list(input_file)
		data = data[12:] # first 12 lines have comments
		data = [numbers.split(',') for numbers in data] 
		d = np.array(data)[:,1:].astype(int) # first column has date
		trainset_size = int(math.ceil((1.0 - testset_split) * len(d)))
		self.train_data = d[:trainset_size,:]
		self.test_data = d[trainset_size:,:] 

class UpdateRule:
	def __init__(self, neighborhood_size):
		self.weights = np.random.rand(neighborhood_size+2) # +1 for self, +1 for bias
	
	def __call__(self, cell_value, neighbor_values):
		return self.weights[0] * cell_value \
			+ self.weights[1:-1].dot(neighbor_values) + self.weights[-1]

	
class CellularAutomaton:
	def __init__(self, initial_values, update_rule):
		self.values = initial_values
		self.update_rule = update_rule

	def get_neighbors(self, i):
		return np.append(self.values[:i], self.values[i+1:])

	def update(self):
		for i in range(len(self.values)):
			neighbors = self.get_neighbors(i)
			self.values[i] = self.update_rule(self.values[i], neighbors)

	def get_values(self):
		return self.values

class Evaluator:
	def __init__(self, data, batch_size, neighborhood_size):
		update_rule = UpdateRule(neighborhood_size)
		ca = CellularAutomaton(data[0], update_rule)
		error = 0.0
		for t in range(batch_size):
			ca.update()
			error += sum(abs(ca.get_values() - data[t]))
			print ca.get_values()

def main(args):
	data = Data(args.input_file, args.split)
	Evaluator(data.train_data, args.batch, args.neighbor)

def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('input_file', type = str,  
			help='Specify the path/filename of the input data.')
	parser.add_argument('-s', '--split', type = float, default = 0.3,  
			help = 'Specify the portion of data to use as testset, e.g. 0.3.')
	parser.add_argument('-n', '--neighbor', type = int, default = 2,
			help = 'Specify the number of neighbors to use.')
	parser.add_argument('-b', '--batch', type = int, default = 2,
			help = 'Specify the number of batches to run.')
	return parser.parse_args()

if __name__ == '__main__':
	main(parse_args())

