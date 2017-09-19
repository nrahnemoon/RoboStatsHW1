from utils import *
import random
import numpy

####################
### WMA Learner ####
####################

class WeightedMajorityAlgorithmLearner:

	def __init__(self, world, experts, eta=0.5):

		self.experts = []
		self.weights = []

		self.world = world
		self.experts = experts
		self.eta = eta

		self.weights = len(self.experts) * [1]
		self.losses = []
		self.predictions = []

	def reset(self):
		self.weights = len(self.experts) * [1]
		self.losses = []
		self.predictions = []

	def setWorld(self, world):
		self.world = world


	def setExperts(self, experts):
		self.experts = experts

	def setEta(self, eta):
		self.eta = eta

	def getEta(self):
		return self.eta

	def updateLoss(self):
		self.losses.append(abs(self.getPrediction() - self.world.getLabel()))

	def updatePrediction(self):
		self.predictions.append(round(float(numpy.dot(self.weights, getExpertPredictions(self.experts))) / sum(self.weights)))

	def getPrediction(self):
		return self.predictions[-1]

	def getLoss(self):
		return self.losses[-1]

	def getLossSum(self):
		return float(sum(self.losses))

	def getWeights(self):
		return self.weights

	def getAverageCumulativeRegret(self):
		# print "Best expert loss: " + str(getBestExpertSumLoss(self.experts))
		# print "Num losses: " + str(len(self.losses))
		# print "Loss Sum: " + str(self.getLossSum())
		return (self.getLossSum() - getBestExpertSumLoss(self.experts))/len(self.losses)

	def getCumulativeLoss(self):
		return numpy.cumsum(self.losses)

	def updateWeights(self):
		
		expertLosers = numpy.array([])
		for expert in self.experts:
			if self.world.getLabel() == expert.getPrediction(): # the expert is a winner
				expertLosers= numpy.append(expertLosers, 0) # 0 because we don't want to apply eta
			else: # the expert is a loser
				expertLosers= numpy.append(expertLosers, 1) # 1 because we want to apply eta

		expertWinners = 1 - expertLosers
		self.weightUpdater = expertWinners + numpy.multiply(([self.eta] * len(self.weights)), expertLosers) # Apply eta only to the losers
		self.weights = numpy.multiply(self.weights, self.weightUpdater)

	def getName(self):
		return "WMA Learner"

###############################
### Randomized WMA Learner ####
###############################

class RandomizedWeightedMajorityAlgorithmLearner(WeightedMajorityAlgorithmLearner):

	def updatePrediction(self):
		
		randomSelector = random.random()
		normalizedWeights = numpy.divide(self.weights, float(sum(self.weights)))
		weightSum = 0
		
		for idx, weight in enumerate(normalizedWeights):
			weightSum += weight
			if randomSelector < weightSum:
				self.predictions.append(self.experts[idx].getPrediction())

	def getName(self):
		return "Randomized WMA Learner"
