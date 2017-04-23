import numpy as np
from trainer import Trainer, basic_train
import ca
import sklearn

class RegressionTrainter(Trainer):
	def __init__(self, args):
		pass

	
	def train(self, intervals, graph):
		num_cities = len(graph)
		num_pairs = sum((len(x)-2) * num_cities for x in intervals)
		y = np.zeros(num_pairs)		
		x = np.zeros(shape = (num_pairs, 2))
		i = 0
		for fold in intervals:
			for t, time in enumerate(fold):
				if t < 2:
					continue
				else:
					for c, city in enumerate(time):
						y[i] = city
						x[i][0] = fold[t-1][c]
						x[i][1] = np.dot(fold[t-1], graph[c])
						#x[i][2] = fold[t-2][c]
						i += 1
		
		self.model = sklearn.linear_model.LinearRegression()
		self.model.fit(x,y)

		update_rule = ca.UpdateRule(2, np.append(self.model.coef_, self.model.intercept_))

		return update_rule
