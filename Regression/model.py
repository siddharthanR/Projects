from data import Data as dt
from plot import Plotting as pt
class Model:

	#initialise variables and dataset
	def __init__(self):
		features, labels = dt.load_data(self)
		self.x = features
		self.y = labels

	#cost function
	def sum_squared_error(self, m, b):
		error = 0
		error_list = [ ]
		for i in range(0, len(self.x), 1):
			error += (self.y[i] - ((m * self.x[i]) + b)) ** 2
			error_list.append(error)
		sse = round((error / len(self.x)), 3)
		return sse, error_list

	#step gradient
	def step_gradient(self, m, b, learning_rate):
		new_m = 0
		new_b = 0
		m_updated = 0
		b_updated = 0
		n = len(self.x)
		for i in range(0, n, 1):
			#gradient of m
			m_grad = -((2/n) * (self.y[i] - (m * self.x[i]) + b) * self.x[i])
			#gradient of b
			b_grad = -((2/n) * (self.y[i] - (m * self.x[i]) + b)) 
			new_m = new_m + m_grad
			new_b = new_b + b_grad
		#updated hyperparameters
		m_updated = (m - (learning_rate * new_m))
		b_updated = (b - (learning_rate * new_b))
		return m_updated, b_updated

	#making prediction
	def model_predict(self, m, b):
		prediction = [ ]
		for i in range(0, len(self.x), 1):
			y = m * self.x[i] + b
			prediction.append(y)
		return prediction

	#gradient descent 
	def gradient_descent(self, m, b, learning_rate, iterations):
		#error before performing gradient descent
		err = self.sum_squared_error(m, b)
		print("m:{0} b:{1}".format(m, b))
		print("error before gradient descent:{0}".format(err[0]))
		
		#iterations of gradient descent
		for i in range(0, iterations, 1):
			m, b = self.step_gradient(m, b, learning_rate)

		#error after performing gradient descent
		new_err = self.sum_squared_error(m, b)
		print("m:{0} b:{0}".format(m, b))
		print("error after gradient descent:{0}".format(new_err[0]))

if __name__ == "__main__":
	m = Model()
	m.gradient_descent(0, 0, 0.0001, 1000)