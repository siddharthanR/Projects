import numpy as np
from data import Data as dt
from plot import Plotting as pt
class Model:

	#initialise model parameters
	def __init__(self):
		f, f1 = dt.load_data(self)
		self.x = f
		self.y = f1
		self.weight = np.zeros(len(self.x[0]))
		self.epoch = 1000
		self.learning_rate = 0.0001
		self.lamda = 1 / self.epoch
		self.error = [ ]
		pt.visualize_data(self)

	#gradient descent 
	def gradient_descent(self):
		pred = 0
		loss = 0
		w = np.ones(len(self.x[0]))
		loss = self.hinge_loss(w)
		pred = self.predict(w)
		print("before gradient descent: error:{0} p:{1}".format(loss, pred))
		e = 0
		for j in range(0, self.epoch, 1):
			for i in range(0, len(self.x), 1):
				if (self.y[i] * np.dot(self.x[i], self.weight)) < 1:
					self.weight += self.learning_rate * ((self.y[i] * self.x[i])+ (-2 * self.lamda * self.weight))
					e = 1
				else:
					self.weight += self.learning_rate *(-2 * self.lamda * self.weight)
			self.error.append(e)
		loss = self.hinge_loss(self.weight)
		pred = self.predict(self.weight)
		print("after gradient descent: error:{0} p:{1}".format(loss, pred))

	#cost function
	def hinge_loss(self, w):
		error = 0
		for i in range(0, len(self.x), 1):
			error += 1 - (self.y[i] * np.dot(self.x[i], w))
		return round(error, 4)

	#predicting
	def predict(self, w):
		p = 0
		for i in range(0, len(self.x), 1):
			p += self.x[i] * w
		return p
 
#main()
if __name__ == "__main__":
	m = Model()
	m.gradient_descent()