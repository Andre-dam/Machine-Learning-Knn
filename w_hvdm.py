import math
import operator

def calculateNeighborsWeightenedHVDM(trainingSet, testInstance, k, lookup_table,classes,maxmin):
    
    distances = []
    length = len(testInstance) - 1
    
    for x in range(len(trainingSet)):        
        dist = VDM(trainingSet[x],testInstance,lookup_table,classes,maxmin)
        distances.append((trainingSet[x],dist))
    distances.sort(key=operator.itemgetter(1))
    neighbors = []
    for x in range(k):
        neighbors.append(distances[x])
    return responseWeightened(neighbors)

def VDM(instance1,instance2,lookup_table,classes,maxmin):
    summation = 0
    for j in range(len(instance1)-1):
        if instance1[j] == '?' and instance2[j] == '?':
            summation += 0 
        elif instance1[j] == '?' or instance2[j] == '?':
            summation += 1        
        elif isinstance(instance1[j],float) == True:
            summation += abs(instance1[j]-instance2[j])/maxmin[j]
        else:
            summation += vdm_i(instance1[j],instance2[j],lookup_table[j],classes)
    return math.sqrt(summation)

def vdm_i(instance1_i,instance2_i,lookup_table,classes):
    summation = 0
    for key in classes:
        p1 = lookup_table.get(instance1_i,0)
        p2 = lookup_table.get(instance2_i,0)
        if p1 != 0:
            p1 = p1.get(key,0)
        if p2 != 0:
            p2 = p2.get(key,0)
            
        summation += abs(p1 - p2)        
    return summation

def responseWeightened(neighbors):
    classVotes = {}
    for x in range(len(neighbors)):
        responseWeightened = neighbors[x][0][-1]        
        if responseWeightened in classVotes:
            classVotes[responseWeightened] += 1
        else:
            classVotes[responseWeightened] = 1
            
    sortedVotes = sorted(classVotes.iteritems(), key = operator.itemgetter(1), reverse=True)
    return sortedVotes[0][0]
