from distances import euclideanDistance
import operator

def calculateNeighborsUnWeightened(trainingSet, testInstance, k):
    distances = []
    length = len(testInstance) - 1
    for x in range(len(trainingSet)):
        dist = euclideanDistance(trainingSet[x],testInstance,length)
        distances.append((trainingSet[x],dist))
    distances.sort(key=operator.itemgetter(1))
    neighbors = []
    for x in range(k):
        neighbors.append(distances[x])
    
    return responseUnWeightened(neighbors)
def responseUnWeightened(neighbors):
    classVotes = {}
    for x in range(len(neighbors)):
        responseWeightened = neighbors[x][0][-1]        
        if responseWeightened in classVotes:
            classVotes[responseWeightened] += 1
        else:
            classVotes[responseWeightened] = 1
            
    sortedVotes = sorted(classVotes.iteritems(), key = operator.itemgetter(1), reverse=True)
    return sortedVotes[0][0]