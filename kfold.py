import numpy as np
#from random import shuffle

def partition(vector, k):
    np.random.shuffle(vector)
    trainingSets = []
    testSets = []

    for fold in range(0, k):
        size = vector.shape[0]
        start = (size/k)*fold
        end = (size/k)*(fold+1)
        
        validation = vector[start:end]
        training = np.concatenate((vector[:start], vector[end:]))

        testSets.append(validation)
        trainingSets.append(training)
            
    testSets = np.asarray(testSets)
    trainingSets = np.asarray(trainingSets)

    return trainingSets, testSets