import matplotlib.pyplot as plt
import numpy as np
class Plotting:

	#initialise variables
	def __init__(self):
		features, labels = dt.load_data(self)
		self.x = features
		self.y =  labels
		
	#lets visualize the data
	def visualize_data(self):
		plt.scatter(self.x, self.y)
		plt.title('ADR vs Rating')
		plt.xlabel('ADR')
		plt.ylabel('Rating')
		plt.show()

	#lets visualize the model
	def visualize_model(self, pred):
		plt.title("ADR vs Rating")
		plt.xlabel("ADR")
		plt.ylabel("Rating")
		plt.scatter(self.x, self.y, label = 'Data', color = 'BLUE')
		plt.plot(self.x, pred, label = 'Best Fit', color = 'RED')
		plt.show()

	def visualize_error(self, error):
		iterations = range(len(error[1]))
		plt.title('Error Steps')
		plt.xlabel('Iterations')
		plt.ylabel('Error')
		plt.plot(iterations, error[1])
		plt.text(10, 130, 'Minimum Error:'+str(min(error[1])))
		plt.show()

if __name__ == "__main__":
	Plotting()