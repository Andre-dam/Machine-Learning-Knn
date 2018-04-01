import math
import operator

def calculateNeighborsUnWeightenedVDM(trainingSet, testInstance, k, lookup_table,classes):
    distances = []
    length = len(testInstance) - 1
    
    for x in range(len(trainingSet)):
        dist = VDM(trainingSet[x],testInstance,lookup_table,classes)
        distances.append((trainingSet[x],dist))
    distances.sort(key=operator.itemgetter(1))
    neighbors = []
    for x in range(k):
        neighbors.append(distances[x])
    return responseUnWeightened(neighbors)

def VDM(instance1,instance2,lookup_table,classes):
    summation = 0
    for j in range(len(instance1[0])-1):
        summation += vdm_i(instance1[j],instance2[j],lookup_table[j])
    return math.sqrt(summation)

def vdm_i(instance1_i,instance2_i,lookup_table,classes):
    summation = 0
    for key in classes:
        summation += abs(lookup_table[instance1_i][classes] - lookup_table[instance2_i][classes])
    return summation

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
