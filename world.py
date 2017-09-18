import "utils.py"
import random

class World:

	self.labels = []

	# Let the default world be the stochastic world
	def updateLabel(self):
		self.labels.append(round(random.random()))

	def getLabel():
		if len(self.labels) == 0:
			self.updateLabel()
		return self.labels[-1]

class StochasticWorld:

	# Redundant, since default is stochastic, but oh well...
	def updateLabel(self):
		self.labels.append(round(random.random()))

class DeterministicWorld:

	def updateLabel(self):
		if len(self.labels) == 0:
			self.labels.append(0)
		else: 
			self.labels.append((self.labels[-1] + 1) % 2)

class AdversarialWorld:

	def __init__(self, leaner, experts):
		self.learner = learner
		self.experts = experts

	def updateLabel(self):
		expertPredictions = getExpertPredictions(self.experts)
		learnerWeights = self.learner.getWeights()
		self.labels.append(expertPredictions[learnerWeights.index(max(learnerWeights))])
