import pandas as pd
class Data:
	def __init__(self):
		self.features = None
		self.labels = None

	def load_data(self):
		data = pd.read_csv('ADRvsRating.csv')
		features = list(data['ADR'])
		labels = list(data['Rating'])
		return features, labels