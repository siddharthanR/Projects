import matplotlib.pyplot as plt
from data import Data as dt
class Plotting:

	def __init__(self):
		f, l = dt.load_data(self)
		self.x = f
		self.y = l

	#lets visualize the data
	def visualize_data(self):
		plt.xlabel('feature 1')
		plt.ylabel('feature 2')
		plt.title('Data')
		for i in range(0, len(self.x), 1):
			if self.y[i] < 1:
				plt.scatter(self.x[i][0], self.x[i][1], marker = 'p', color = 'red')
			else:
				plt.scatter(self.x[i][0], self.x[i][1], marker = 'P', color = 'blue')
		plt.plot([-2, 6],[6, 1])
		plt.show()

	#lets visualize the error
	def visualize_error(self, error):
		plt.title('Error')
		plt.xlabel('iterations')
		plt.ylabel('error')
		plt.plot(range(len(error)), error)
		plt.show()

if __name__ == "__main__":
	p = Plotting()
	p.visualize_data()