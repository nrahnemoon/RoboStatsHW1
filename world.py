from utils import *
import random
import numpy

######################
### Default World ####
######################

class World:

	def __init__(self):
		self.labels = []

	def reset(self):
		self.labels = []

	# Let the default world be the stochastic world
	def updateLabel(self):
		self.labels.append(round(random.random()))

	def getLabel(self):
		if len(self.labels) == 0:
			self.updateLabel()
		return self.labels[-1]

	def getName(self):
		return "Default World"

#########################
### Stochastic World ####
#########################

class StochasticWorld(World):

	# Redundant, since default is stochastic, but oh well...
	def updateLabel(self):
		self.labels.append(round(random.random()))

	def getName(self):
		return "Stochastic World"

###########################
### Determinstic World ####
###########################

class DeterministicWorld(World):

	def updateLabel(self):
		if len(self.labels) == 0:
			self.labels.append(0)
		else: 
			self.labels.append((self.labels[-1] + 1) % 2)


	def getName(self):
		return "Deterministic World"


##########################
### Adversarial World ####
##########################

class AdversarialWorld(World, object):

	def __init__(self, learner, experts):
		super(self.__class__, self).__init__()
		self.learner = learner
		self.experts = experts

	def setLearner(self, learner):
		self.learner = learner

	def setExperts(self, experts):
		self.experts = experts

	def updateLabel(self):
		expertPredictions = getExpertPredictions(self.experts)
		learnerWeights = list(self.learner.getWeights())
		self.labels.append(expertPredictions[learnerWeights.index(max(learnerWeights))])

	def getName(self):
		return "Adversarial World"


###########################
### Feature-rich World ####
###########################

class FeatureRichWorld(World, object):

	def __init__(self, sunnyProbability, homeProbability, minWinsForStreak, winWhenSunnyProbability, winWhenHomeProbability, winWhenWinStreakProbability):

		super(self.__class__, self).__init__()

		self.sunnyProbability = sunnyProbability
		self.homeProbability = homeProbability
		self.minWinsForStreak = minWinsForStreak
		
		self.winWhenSunnyProbability = winWhenSunnyProbability
		self.winWhenHomeProbability = winWhenHomeProbability
		self.winWhenWinStreakProbability = winWhenWinStreakProbability

		self.updateFeatures()

	def updateFeatures(self):

		self.isSunny = (random.random() < self.sunnyProbability)
		self.isHome = (random.random() < self.homeProbability)
		self.isWinStreak = (sum(self.labels[(-1 * self.minWinsForStreak):]) == self.minWinsForStreak)

	def getWinProbability(self):
		winProbabilities = [
			(self.isSunny * self.winWhenSunnyProbability) + ((not self.isSunny) * (1 - self.winWhenSunnyProbability)),
			(self.isHome * self.winWhenHomeProbability) + ((not self.isHome) * (1 - self.winWhenHomeProbability)),
			(self.isWinStreak * self.winWhenWinStreakProbability) + ((not self.isWinStreak) * (1 - self.winWhenWinStreakProbability))
		]
		return numpy.mean(winProbabilities)

	def isItSunny(self):
		return self.isSunny

	def isItHome(self):
		return self.isHome
	
	def isItWinStreak(self):
		return self.isWinStreak

	def updateLabel(self):
		self.updateFeatures()
		self.labels.append(int(random.random() < self.getWinProbability()))

	def getName(self):
		return "Feature Rich World"
