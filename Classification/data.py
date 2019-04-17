import numpy as np
class Data:

	def load_data(self):
		x = np.array([[-2,4],[-4,1],[1, 6],[2, 4],[6, 2]])
		y = np.array([-1,-1,1,1,1])
		return x, y

if __name__ == "__main__":
	d = Data()
	d.load_data()