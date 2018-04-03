from uw_knn import calculateNeighborsUnWeightened
from w_knn import calculateNeighborsWeightened
from vdm_uw_knn import calculateNeighborsUnWeightenedVDM
from vdm_w_knn import calculateNeighborsWeightenedVDM
from uw_hvdm import calculateNeighborsUnWeightenedHVDM
from w_hvdm import calculateNeighborsWeightenedHVDM

from kfold import partition
import misc
import numpy as np

def main():
    k = 5
    k_fold = 10
    trainingSet=[]
    testSet = []
    predictions = []

 
 #Test unweightened knn
    dataset = misc.loadData('pc1.arff')
    
    trainingSet = np.asarray(trainingSet)
    testSet = np.asarray(testSet)
    predictions = []

    trainingSet,testSet = partition(dataset,k_fold)
    accuracy = []

    for k in [1,2,3,5,7,9,11,13,15]:
        print "Unweightened KNN with K = "  + repr(k)
        print "K-fold = " + repr(k_fold)
        for x in range(k_fold):
            predictions = []
            for i in range(len(testSet[x])):      
                result = calculateNeighborsUnWeightened(trainingSet[x], testSet[x][i], k)
                predictions.append(result)
            accuracy.append(misc.calculateAccuracy(testSet[x],predictions))
            print 'K = '+ repr(x) + " accuracy:" + repr(accuracy[x]) + '%'
        accuracy_avg = sum(accuracy)/k_fold
        print ("Average accuracy : " + repr(accuracy_avg) + '%')
        print 
        accuracy = []
        accuracy_avg = 0
    print 

#Test weightened knn
    dataset = misc.loadData('pc1.arff')

    trainingSet = np.asarray(trainingSet)
    testSet = np.asarray(testSet)
    predictions = []

    trainingSet,testSet = partition(dataset,k_fold)
    accuracy = []

    for k in [1,2,3,5,7,9,11,13,15]:
        print "Weightened KNN with K = "  + repr(k)
        print "K-fold = " + repr(k_fold)
        for x in range(k_fold):
            predictions = []
            for i in range(len(testSet[x])):      
                result = calculateNeighborsWeightened(trainingSet[x], testSet[x][i], k)
                predictions.append(result)
            accuracy.append(misc.calculateAccuracy(testSet[x],predictions))
            #print 'K = '+ repr(x) + " accuracy:" + repr(accuracy[x]) + '%'
        accuracy_avg = sum(accuracy)/k_fold
        print ("Average accuracy : " + repr(accuracy_avg) + '%')
        print 
        accuracy = []
        accuracy_avg = 0
    print 

#Test vdm unweightened knn
    dataset = misc.loadData('chess.csv')
    
    trainingSet = np.asarray(trainingSet)
    testSet = np.asarray(testSet)
    predictions = []
    trainingSet,testSet = partition(dataset,k_fold)
    accuracy = []
    test = []
    test = testSet

    np.copyto(test,testSet)

    for k in [1,2,3,5,7,9,11,13,15]:
        print "Unweightened KNN-VDM with K = "  + repr(k)
        print "K-fold = " + repr(k_fold)
        for x in range(k_fold):
            predictions = []
            lookup_ = misc.buildLookUp(trainingSet[x])
            classes = misc.getClasses(trainingSet[x])
            for i in range(len(testSet[x])):      
                result = calculateNeighborsUnWeightenedVDM(trainingSet[x],testSet[x][i],k,lookup_,classes)
                predictions.append(result)
            accuracy.append(misc.calculateAccuracy(testSet[x],predictions))
            print 'K = '+ repr(x) + " accuracy:" + repr(accuracy[x]) + '%'
        accuracy_avg = sum(accuracy)/k_fold
        print ("Average accuracy : " + repr(accuracy_avg) + '%')
        print 
        accuracy = []
        accuracy_avg = 0
    print 

#Test vdm weightened knn
    dataset = misc.loadData('chess.csv')
    
    trainingSet = np.asarray(trainingSet)
    testSet = np.asarray(testSet)
    predictions = []

    trainingSet,testSet = partition(dataset,k_fold)
    accuracy = []

    for k in [1,2,3,5,7,9,11,13,15]:
        print "Weightened KNN-VDM with K = "  + repr(k)
        print "K-fold = " + repr(k_fold)
        for x in range(k_fold):
            predictions = []
            lookup_ = misc.buildLookUp(trainingSet[x])
            classes = misc.getClasses(trainingSet[x])
            for i in range(len(testSet[x])):      
                result = calculateNeighborsWeightenedVDM(trainingSet[x],testSet[x][i],k,lookup_,classes)
                predictions.append(result)
            accuracy.append(misc.calculateAccuracy(testSet[x],predictions))
            print 'K = '+ repr(x) + " accuracy:" + repr(accuracy[x]) + '%'
        accuracy_avg = sum(accuracy)/k_fold
        print ("Average accuracy : " + repr(accuracy_avg) + '%')
        print 
        accuracy = []
        accuracy_avg = 0
    print 

#Test hvdm unweightened knn
    dataset = misc.loadData('analysis.data')    
    trainingSet = np.asarray(trainingSet)
    testSet = np.asarray(testSet)
    predictions = []

    trainingSet,testSet = partition(dataset,k_fold)
    accuracy = []

    for k in [1,2,3,5,7,9,11,13,15]:
        print "Unweightened KNN-HVDM with K = "  + repr(k)
        print "K-fold = " + repr(k_fold)

        for x in range(k_fold):
            predictions = []
            lookup_ = misc.buildLookUp(trainingSet[x])
            classes = misc.getClasses(trainingSet[x])
            for i in range(len(testSet[x])):      
                result = calculateNeighborsUnWeightenedHVDM(trainingSet[x],testSet[x][i],k,lookup_,classes)
                predictions.append(result)
            accuracy.append(misc.calculateAccuracy(testSet[x],predictions))
            print 'K = '+ repr(x) + " accuracy:" + repr(accuracy[x]) + '%'
        accuracy_avg = sum(accuracy)/k_fold
        print ("Average accuracy : " + repr(accuracy_avg) + '%')
        print 
        accuracy = []
        accuracy_avg = 0
    print 

#Test hvdm weightened knn
    dataset = misc.loadData('analysis.data')
    
    trainingSet = np.asarray(trainingSet)
    testSet = np.asarray(testSet)
    predictions = []

    trainingSet,testSet = partition(dataset,k_fold)

    for k in [1,2,3,5,7,9,11,13,15]:
        print "Weightened KNN-HVDM with K = "  + repr(k)
        print "K-fold = " + repr(k_fold)

        for x in range(k_fold):
            predictions = []
            lookup_ = misc.buildLookUp(trainingSet[x])
            classes = misc.getClasses(trainingSet[x])
            for i in range(len(testSet[x])):      
                result = calculateNeighborsWeightenedHVDM(trainingSet[x],testSet[x][i],k,lookup_,classes)
                predictions.append(result)
            accuracy.append(misc.calculateAccuracy(testSet[x],predictions))
            print 'K = '+ repr(x) + " accuracy:" + repr(accuracy[x]) + '%'
        accuracy_avg = sum(accuracy)/k_fold
        print ("Average accuracy : " + repr(accuracy_avg) + '%')
        print 
        accuracy = []
        accuracy_avg = 0
    print
main()