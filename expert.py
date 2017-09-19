import numpy
from world import *

#######################
### Default Expert ####
#######################
class Expert:

	def __init__(self):
		self.predictions = []
		self.losses = []

	def reset(self):
		self.losses = []
		self.predictions = []

	def updateLoss(self, label):
		self.losses.append(abs(self.getPrediction() - label))

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

	def getCumulativeLoss(self):
		return numpy.cumsum(self.losses)

	def getName(self):
		return "Default Expert"

##########################
### Optimistic Expert ####
##########################

class OptimisticExpert(Expert):

	# Redundant, since default is optimistic, but oh well...
	def updatePrediction(self):
		self.predictions.append(1)


	def getName(self):
		return "Optimistic Expert"


###########################
### Pessimistic Expert ####
###########################

class PessimisticExpert(Expert):

	def updatePrediction(self):
		self.predictions.append(0)

	def getName(self):
		return "Pessimistic Expert"

##########################
### Ambivalent Expert ####
##########################

class AmbivalentExpert(Expert):
	
	def updatePrediction(self):
		if len(self.predictions) == 0:
			self.predictions.append(0)
		else:
			self.predictions.append((self.predictions[-1] + 1) % 2)

	def getName(self):
		return "Ambivalent Expert"

#######################
### Feature Expert ####
#######################

class FeatureExpert(Expert, object):
	
	def __init__(self, featureRichWorld, featureProbability):
		self.predictions = []
		self.losses = []
		self.world = featureRichWorld
		self.featureProbability = featureProbability

	def setWorld(self, world):
		self.world = world

	# Default is sunny...
	def featureActive(self):
		return self.world.isItSunny()

	def getWinBeliefProbability(self):
		return (self.featureActive() * self.featureProbability) + ((not self.featureActive()) * (1 - self.featureProbability))

	def updatePrediction(self):
		self.predictions.append(int(random.random() < self.getWinBeliefProbability()))

	def getName(self):
		return "Default Feature Expert"

#####################
### Sunny Expert ####
#####################

class SunnyExpert(FeatureExpert):

	def featureActive(self):
		return self.world.isItSunny()

	def getName(self):
		return "Sunny Expert"

####################
### Home Expert ####
####################

class HomeExpert(FeatureExpert):

	def featureActive(self):
		return self.world.isItHome()

	def getName(self):
		return "Home Expert"

##########################
### Win Streak Expert ####
##########################

class WinStreakExpert(FeatureExpert):

	def featureActive(self):
		return self.world.isItWinStreak()

	def getName(self):
		return "Win Streak Expert"
