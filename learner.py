import "utils.py"

class WeightedMajorityAlgorithmLearner:

	experts = []
	weights = []

	def __init__(self, world, experts, eta=0.5):
		self.world = world
		self.experts = experts
		self.eta = eta

		self.weights = len(self.experts) * [1]
		self.losses = []
		self.predictions = []

	def updateExpertPredictions(self):

		for expert in self.experts:
			expert.updatePrediction()

	def getExpertPredictions(self):

		expertPredictions = []
		for expert in self.experts:
			expertPredictions.append(expert.getPrediction())

	def updateLoss(self):
		self.losses.append(abs(self.prediction - self.worldLabel))

	def updatePrediction(self):
		self.predictions.append(round(float(numpy.dot(self.weights, self.expertPredictions)) / sum(self.weights)))

	def getPrediction(self):
		return self.predictions[-1]

	def getLoss(self):
		return self.losses[-1]

	def getLossSum(self):
		return float(sum(self.losses))

	def getWeights(self):
		return self.weights

	def getAverageCumulativeRegret(self):
		return (self.getLossSum()/len(self.losses)) - self.getBestExpertSumLoss(self.experts)

	def updateWeights(self):
		
		expertLosers = []
		for expert in self.experts:
			if self.world.getLabel() == self.expert.getPrediction(): # the expert is a winner
				expertLosers.append(0) # 0 because we don't want to apply eta
			else: # the expert is a loser
				expertLosers.append(1) # 1 because we want to apply eta

		self.weightUpdater = numpy.multiply(([self.eta] * len(self.weights)), expertLosers) # Apply eta only to the losers
		self.weights = numpy.multiply(self.weights, self.weightUpdater)

class RandomizedWeightedMajorityAlgorithmLearner(WeightedMajorityAlgorithmLearner):
