from distances import euclideanDistance
import operator

def calculateNeighborsWeightened(trainingSet, testInstance, k):
    distances = []
    length = len(testInstance) - 1
    for x in range(len(trainingSet)):
        dist = euclideanDistance(trainingSet[x],testInstance,length)
        distances.append((trainingSet[x],dist))
    distances.sort(key=operator.itemgetter(1))
    neighbors = []
    for x in range(k):
        neighbors.append(distances[x])
    
    return responseWeightened(neighbors)

def responseWeightened(neighbors):
    classVotes = {}
    for x in range(len(neighbors)):
        responseWeightened = neighbors[x][0][-1]

        if responseWeightened in classVotes:
            try:
                classVotes[responseWeightened] += 1/neighbors[x][1]
            except ZeroDivisionError:
                classVotes[responseWeightened] += 999999999.0            
        else:
            try:
                classVotes[responseWeightened] = 1/neighbors[x][1]
            except ZeroDivisionError:
                classVotes[responseWeightened] = 999999999.0
    sortedVotes = sorted(classVotes.iteritems(), key = operator.itemgetter(1), reverse=True)
    return sortedVotes[0][0]