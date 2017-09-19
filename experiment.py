from world import *
from expert import *
from learner import *
import matplotlib.pyplot as plt

class Experiment:

	def __init__(self, learner, experts, world, numIterations):

		self.learner = learner
		self.experts = experts
		self.world = world
		self.numIterations = numIterations
		self.averageCumulativeRegrets = []

	def reset(self):
		self.learner.reset()
		resetExperts(self.experts)
		self.world.reset()
		self.averageCumulativeRegrets = []

	def updateExpertLosses(self):
		for expert in self.experts:
			expert.updateLoss(self.world.getLabel())

	def run(self):

		for iteration in range(0, self.numIterations):

			for expert in self.experts:
				expert.updatePrediction()

			self.learner.updatePrediction()

			self.world.updateLabel()

			for expert in self.experts:
				expert.updateLoss(self.world.getLabel())

			self.learner.updateLoss()

			self.averageCumulativeRegrets.append(self.learner.getAverageCumulativeRegret())

			self.learner.updateWeights()

	def plotAverageCumulativeRegret(self):

		plt.plot(range(1, self.numIterations + 1), self.averageCumulativeRegrets)
		plt.title('Average Cummulative Regret for a ' + self.learner.getName() + ' (eta = ' + str(self.learner.getEta()) + ') in a ' + self.world.getName())
		plt.ylabel('Average Cumulative Regret')
		plt.xlabel('Iteration')
		plt.show()

	def plotCummulativeLosses(self):

		plt.title('Cummulative Loss for a ' + self.learner.getName() + ' (eta = ' + str(self.learner.getEta()) + ') and experts in a ' + self.world.getName())

		plt.plot(range(1, self.numIterations + 1), self.learner.getCumulativeLoss(), label=self.learner.getName())

		for expert in self.experts:
			plt.plot(expert.getCumulativeLoss(), label=expert.getName())

		plt.legend(loc='lower right', shadow=True)

		plt.ylabel('Cumulative Loss')
		plt.xlabel('Iteration')
		plt.show()
