import "world.py"
import "expert.py"
import "learner.py"
import "utils.py"
import numpy
import matplotlib.pyplot as plt

class Experiment:

	def __init__(self, learner, experts, world, numIterations):
		self.learner = learner
		self.experts = experts
		self.world = world
		self.numIterations = numIterations
		self.averageCumulativeRegrets = []


	def updateExpertLosses(self):
		for expert in self.experts:
			expert.updateLoss(self.world.getLabel())

	def run(self):

		for iteration in range(0, numIterations):

			for expert in self.experts:
				expert.updatePrediction()

			self.learner.updatePrediction()

			self.world.updateLabel()

			for expert in self.experts:
				expert.updateLoss(self.world.getLabel())

			self.learner.updateLoss(self.world.getLabel())

			self.averageCumulativeRegrets.append(self.learner.getAverageCumulativeRegret())

			self.learner.updateWeights()

	def plotAverageCumulativeRegret(self):
		plt.plot(averageCumulativeRegrets)
		plt.ylabel('Average Cumulative Regret')
		plt.xlabel('Iteration')
		plt.show()

# Create the experts
experts = []
experts.append(OptimisticExpert())
experts.append(PessimisticExpert())
experts.append(AmbivalentExpert())

# Create the learners
wmaLearner = WeightedMajorityAlgorithmLearner(stochasticWorld, experts)

# Create the worlds
stochasticWorld = StochasticWorld()
deterministicWorld = DeterministicWorld()
wmaAdversarialWorld = AdversarialWorld(weightedMajorityAlgorithmLearner, experts)

wmaLearnerInStochasticWorldExperiment = Experiment(wmaLearner, experts, stochasticWorld, 100)