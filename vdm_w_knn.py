import math
import operator
import pandas as pd
import numpy as np

def calculateNeighborsWeightenedVDM(trainingSet, testInstance, k, lookup_table,classes):
    distances = []
    length = len(testInstance) - 1
    
    for x in range(len(trainingSet)):
        dist = VDM(trainingSet[x],testInstance,lookup_table,classes)
        #dist = euclideanDistance(trainingSet[x],testInstance,length)
        distances.append((trainingSet[x],dist))
    distances.sort(key=operator.itemgetter(1))
    neighbors = []
    for x in range(k):
        neighbors.append(distances[x])
    return responseWeightened(neighbors)

def VDM(instance1,instance2,lookup_table,classes):
    summation = 0
    #varre o numero de atributos menos a classe
    for j in range(len(instance1[0])-1):
        summation += vdm_i(instance1[j],instance2[j],lookup_table[j])
        #summation += vdm_i(instance1[j],instance2[j],[row[j] for row in trainingSet],[row[-1] for row in trainingSet])
    return math.sqrt(summation)

def vdm_i(instance1_i,instance2_i,lookup_table,classes):
    summation = 0
    for key in classes:
        summation += abs(lookup_table[instance1_i][classes] - lookup_table[instance2_i][classes])
    return summation

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
