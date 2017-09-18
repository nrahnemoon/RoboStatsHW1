def getExpertPredictions(experts):

	expertPredictions = []
	for expert in experts:
		expertPredictions.append(expert.getPrediction())

	return expertPredictions

def getBestExpertSumLoss(experts):
	sumLosses = []
	for expert in experts:
		sumLosses << expert.getLossSum()
	return min(sumLosses)