import numpy

class Expert:

	def __init__(self):
		self.predictions = []
		self.losses = []

	def updateLoss(self, label):
		self.losses.append(abs(self.prediction - label))

	def getLoss(self):
		return self.losses[-1]

	def getLosses(self):
		return self.losses

	def getLossSum(self):
		return float(sum(self.losses))

	# Let default be optimistic
	def updatePrediction(self):
		self.predictions.append(1)

	def getPrediction(self):
		if len(self.predictions) == 0:
			self.updatePrediction()

		return self.predictions[-1]

class OptimisticExpert(Expert):

	# Redundant, since default is optimistic, but oh well...
	def updatePrediction(self):
		self.predictions.append(1)

class PessimisticExpert(Expert):

	def updatePrediction(self):
		self.predictions.append(0)

class AmbivalentExpert(Expert):
	
	def updatePrediction(self):
		if len(self.predictions) == 0:
			self.predictions.append(0)
		else:
			self.predictions.append((self.predictions[-1] + 1) % 2)
