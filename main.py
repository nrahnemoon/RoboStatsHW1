from world import *
from expert import *
from learner import *
from experiment import *
from utils import *

# Create the experts for the simple version of the project (i.e., without adversaries) 
simpleExperts = []
simpleExperts.append(OptimisticExpert())
simpleExperts.append(PessimisticExpert())
simpleExperts.append(AmbivalentExpert())

# Create the stochastic world
stochasticWorld = StochasticWorld()

# Create the WMA learner
wmaLearner = WeightedMajorityAlgorithmLearner(stochasticWorld, simpleExperts, 0.5)


#######################################################################
### Experiment 1: WMA Learner with Eta = 0.5 in a Stochastic World ####
#######################################################################

wmaStochasticExperiment = Experiment(wmaLearner, simpleExperts, stochasticWorld, 100)
wmaStochasticExperiment.run()
wmaStochasticExperiment.plotAverageCumulativeRegret()
wmaStochasticExperiment.plotCummulativeLosses()

# Reset from last experiment
wmaStochasticExperiment.reset()


##########################################################################
### Experiment 2: WMA Learner with Eta = 0.5 in a Deterministic World ####
##########################################################################

# Create the deterministic world
deterministicWorld = DeterministicWorld()
wmaLearner.setWorld(deterministicWorld)

wmaDeterministicExperiment = Experiment(wmaLearner, simpleExperts, deterministicWorld, 100)
wmaDeterministicExperiment.run()
wmaDeterministicExperiment.plotAverageCumulativeRegret()
wmaDeterministicExperiment.plotCummulativeLosses()

# Reset from last experiment
wmaDeterministicExperiment.reset()


#########################################################################
### Experiment 3: WMA Learner with Eta = 0.5 in an Adversarial World ####
#########################################################################

# Create the deterministic world
adversarialWorld = AdversarialWorld(wmaLearner, simpleExperts)
wmaLearner.setWorld(adversarialWorld)

wmaAdversarialExperiment = Experiment(wmaLearner, simpleExperts, adversarialWorld, 100)
wmaAdversarialExperiment.run()
wmaAdversarialExperiment.plotAverageCumulativeRegret()
wmaAdversarialExperiment.plotCummulativeLosses()

# Reset from last experiment
wmaAdversarialExperiment.reset()


##################################################################################
### Experiment 4: Randomized WMA Learner with Eta = 0.5 in a Stochastic World ####
##################################################################################

# Create the WMA learner
randomizedWmaLearner = RandomizedWeightedMajorityAlgorithmLearner(stochasticWorld, simpleExperts, 0.5)

randomizedWmaStochasticExperiment = Experiment(randomizedWmaLearner, simpleExperts, stochasticWorld, 100)
randomizedWmaStochasticExperiment.run()
randomizedWmaStochasticExperiment.plotAverageCumulativeRegret()
randomizedWmaStochasticExperiment.plotCummulativeLosses()

# Reset from last experiment
randomizedWmaStochasticExperiment.reset()


#####################################################################################
### Experiment 5: Randomized WMA Learner with Eta = 0.5 in a Deterministic World ####
#####################################################################################

randomizedWmaLearner.setWorld(stochasticWorld)

randomizedWmaDeterministicExperiment = Experiment(randomizedWmaLearner, simpleExperts, deterministicWorld, 100)
randomizedWmaDeterministicExperiment.run()
randomizedWmaDeterministicExperiment.plotAverageCumulativeRegret()
randomizedWmaDeterministicExperiment.plotCummulativeLosses()

# Reset from last experiment
randomizedWmaDeterministicExperiment.reset()


####################################################################################
### Experiment 6: Randomized WMA Learner with Eta = 0.5 in an Adversarial World ####
####################################################################################

randomizedWmaLearner.setWorld(adversarialWorld)

randomizedWmaAdversarialExperiment = Experiment(randomizedWmaLearner, simpleExperts, adversarialWorld, 100)
randomizedWmaAdversarialExperiment.run()
randomizedWmaAdversarialExperiment.plotAverageCumulativeRegret()
randomizedWmaAdversarialExperiment.plotCummulativeLosses()

# Reset from last experiment
randomizedWmaAdversarialExperiment.reset()


#########################################################################
### Experiment 7: WMA Learner with Eta = 0.5 in a Feature-rich World ####
#########################################################################

# Create the feature-rich world were the parameters are:
# sunnyProbability: 50%
# homeProbability: 50%
# minWinsForStreak: 3
# winWhenSunnyProbability: 60%
# winWhenHomeProbability: 80%
# winWhenWinStreakProbability: 90%
featureRichWorld = FeatureRichWorld(0.5, 0.5, 3, 0.6, 0.8, 0.9)

# Create the experts for the simple version of the project (i.e., without adversaries) 
featureExperts = []
featureExperts.append(SunnyExpert(featureRichWorld, 0.7))
featureExperts.append(HomeExpert(featureRichWorld, 0.75))
featureExperts.append(WinStreakExpert(featureRichWorld, 0.95))

wmaLearner.setWorld(featureRichWorld)

wmaFeatureRichExperiment = Experiment(wmaLearner, featureExperts, featureRichWorld, 100)
wmaFeatureRichExperiment.run()
wmaFeatureRichExperiment.plotAverageCumulativeRegret()
wmaFeatureRichExperiment.plotCummulativeLosses()

# Reset from last experiment
wmaFeatureRichExperiment.reset()


####################################################################################
### Experiment 8: Randomized WMA Learner with Eta = 0.5 in a Feature-rich World ####
####################################################################################

randomizedWmaLearner.setWorld(featureRichWorld)

randomizedWmaFeatureRichExperiment = Experiment(randomizedWmaLearner, featureExperts, featureRichWorld, 100)
randomizedWmaFeatureRichExperiment.run()
randomizedWmaFeatureRichExperiment.plotAverageCumulativeRegret()
randomizedWmaFeatureRichExperiment.plotCummulativeLosses()

# Reset from last experiment
randomizedWmaFeatureRichExperiment.reset()
