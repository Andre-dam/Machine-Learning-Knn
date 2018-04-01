from uw_knn import calculateNeighborsUnWeightened
from w_knn import calculateNeighborsWeightened
from vdm_uw_knn import calculateNeighborsUnWeightenedVDM
from vdm_w_knn import calculateNeighborsWeightenedVDM
import misc
import numpy as np
def main():
    k = 5
    trainingSet=[]
    testSet=[]
    predictions = []

#Test unweightened knn
    dataset = misc.loadData('cm1.arff')
    trainingSet = np.asarray(trainingSet)
    testSet = np.asarray(testSet)
    predictions = []

    #gambiarra antes do kfold
    testSet = dataset[range(0,50),:]
    trainingSet = dataset[range(51,len(dataset)-1)]
    
    for x in range(0,50):
        result = calculateNeighborsUnWeightened(trainingSet, testSet[x], k)
        predictions.append(result)
        print('> predicted=' + repr(result) +', actual=' + repr(dataset[x][-1]))
    accuracy = misc.calculateAccuracy(testSet,predictions)
    print('Unweightened knn Accuracy: ' + repr(accuracy) + '%')
    print
    #end-gambiarra antes do kfold

#Test weightened knn
    dataset = misc.loadData('cm1.arff')
    trainingSet = np.asarray(trainingSet)
    testSet = np.asarray(testSet)
    predictions = []

    #gambiarra antes do kfold
    testSet = dataset[range(0,50),:]
    trainingSet = dataset[range(51,len(dataset)-1)]
    
    for x in range(0,50):
        result = calculateNeighborsWeightened(trainingSet, testSet[x], k)
        predictions.append(result)
        print('> predicted=' + repr(result) +', actual=' + repr(dataset[x][-1]))
    accuracy = misc.calculateAccuracy(testSet,predictions)
    print('Weightened knn Accuracy: ' + repr(accuracy) + '%')
    print
#Test vdm unweightened knn
    dataset = misc.loadData('chess.csv')
    trainingSet = np.asarray(trainingSet)
    testSet = np.asarray(testSet)
    predictions = []

    #gambiarra antes do kfold
    testSet = dataset[range(0,50),:]
    trainingSet = dataset[range(51,len(dataset)-1)]
    
    lookup_ = misc.buildLookUp(trainingSet)
    classes = misc.getClasses(trainingSet)

    for x in range(0,50):
        result = calculateNeighborsUnWeightenedVDM(trainingSet,testSet[x],k,lookup_,classes)
        predictions.append(result)
        print('> predicted=' + repr(result) +', actual=' + repr(testSet[x][-1]))
    accuracy = misc.calculateAccuracy(testSet,predictions)
    print('Unweightened knn with vdm Accuracy: ' + repr(accuracy) + '%')
    print

#Test vdm weightened knn
    dataset = misc.loadData('chess.csv')
    trainingSet = np.asarray(trainingSet)
    testSet = np.asarray(testSet)
    predictions = []

    #gambiarra antes do kfold
    testSet = dataset[range(0,50),:]
    trainingSet = dataset[range(51,len(dataset)-1)]
    
    lookup_ = misc.buildLookUp(trainingSet)
    classes = misc.getClasses(trainingSet)

    for x in range(0,50):
        result = calculateNeighborsWeightenedVDM(trainingSet,testSet[x],k,lookup_,classes)
        predictions.append(result)
        print('> predicted=' + repr(result) +', actual=' + repr(testSet[x][-1]))
    accuracy = misc.calculateAccuracy(testSet,predictions)
    print('Weightened knn with vdm Accuracy: ' + repr(accuracy) + '%')
    print
    
"""     lookup_ = buildLookUp(dataset[range(50,len(dataset)-1)])
    classes = getClasses(dataset[range(50,len(dataset)-1)])

    predictions = []
    for x in range(0,50):
        #trainingSet, testInstance, k, lookup_table,classes        
        neighbors = calculateNeighborsVDM(dataset[range(50,len(dataset)-1)],dataset[x],k,lookup_,classes)
        #neighbors = calculateNeighbors(trainingSet,testSet[x],k)
        result = responseUnWeightened(neighbors)
        predictions.append(result)
        print('> predicted=' + repr(result) +', actual=' + repr(dataset[x][-1]))
    accuracy = calculateAccuracy(dataset[range(0,50),:],predictions)
    print('Accuracy: ' + repr(accuracy) + '%')
 """

main()