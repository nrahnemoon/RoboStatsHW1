def getExpertPredictions(experts):

	expertPredictions = []
	for expert in experts:
		expertPredictions.append(expert.getPrediction())

	return expertPredictions

def getBestExpertSumLoss(experts):
	sumLosses = []
	for expert in experts:
		sumLosses.append(expert.getLossSum())
	return min(sumLosses)

def resetExperts(experts):
	for expert in experts:
		expert.reset()
