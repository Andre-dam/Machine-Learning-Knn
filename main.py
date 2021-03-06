from uw_knn import calculateNeighborsUnWeightened
from w_knn import calculateNeighborsWeightened
from vdm_uw_knn import calculateNeighborsUnWeightenedVDM
from vdm_w_knn import calculateNeighborsWeightenedVDM
from uw_hvdm import calculateNeighborsUnWeightenedHVDM
from w_hvdm import calculateNeighborsWeightenedHVDM
from os import sys
from kfold import partition
import misc
import numpy as np



def main():
    
    print "--------test---------"
    print "1 - Unweightened kNN"
    print "2 - Weightened kNN"
    print "3 - Unweightened kNN-VDM"
    print "4 - Weightened kNN-VDM"
    print "5 - Unweightened kNN-HVDM"
    print "6 - Weightened kNN-HVDM"
    print "7 - all"
    option = raw_input()
    print 
    if option == '1':
        Test_unweightened_knn()
    elif option == '2':
        Test_weightened_knn()
    elif option == '3':
        Test_vdm_unweightened_knn()
    elif option == '4':
        Test_vdm_weightened_knn()
    elif option == '5':
        Test_hvdm_unweightened_knn()
    elif option == '6':
        Test_hvdm_weightened_knn()
    elif option == '7':
        Test_unweightened_knn()
        Test_weightened_knn()
        Test_vdm_unweightened_knn()
        Test_vdm_weightened_knn()
        Test_hvdm_unweightened_knn()
        Test_hvdm_weightened_knn()
    else:
        print "invalid option"
        main() 
 
def Test_unweightened_knn():
    k_fold = 10
    trainingSet=[]
    testSet = []
    predictions = []
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
            print 'iteration: '+ repr(x) + " accuracy:" + repr(accuracy[x]) + '%'
        accuracy_avg = sum(accuracy)/k_fold
        print ("Average accuracy : " + repr(accuracy_avg) + '%')
        print 
        accuracy = []
        accuracy_avg = 0
    print 

def Test_weightened_knn():
    k_fold = 10
    trainingSet=[]
    testSet = []
    predictions = []
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
            print 'iteration: '+ repr(x) + " accuracy:" + repr(accuracy[x]) + '%'
        accuracy_avg = sum(accuracy)/k_fold
        print ("Average accuracy : " + repr(accuracy_avg) + '%')
        print 
        accuracy = []
        accuracy_avg = 0
    print 

def Test_vdm_unweightened_knn():
    k_fold = 10
    trainingSet=[]
    testSet = []
    predictions = []
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
            print 'iteration: '+ repr(x) + " accuracy:" + repr(accuracy[x]) + '%'
        accuracy_avg = sum(accuracy)/k_fold
        print ("Average accuracy : " + repr(accuracy_avg) + '%')
        print 
        accuracy = []
        accuracy_avg = 0
    print 

def Test_vdm_weightened_knn():
    k_fold = 10
    trainingSet=[]
    testSet = []
    predictions = []
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
            print 'iteration: '+ repr(x) + " accuracy:" + repr(accuracy[x]) + '%'
        accuracy_avg = sum(accuracy)/k_fold
        print ("Average accuracy : " + repr(accuracy_avg) + '%')
        print 
        accuracy = []
        accuracy_avg = 0
    print 

def Test_hvdm_unweightened_knn():
    k_fold = 10
    trainingSet=[]
    testSet = []
    predictions = []
    dataset = misc.loadData('crx.data')
    
    for i in range(len(dataset)):
        if dataset[i][13] != '?':
            dataset[i][13] = float(dataset[i][13])

    trainingSet = np.asarray(trainingSet)
    testSet = np.asarray(testSet)
    predictions = []

    trainingSet,testSet = partition(dataset,k_fold)
    accuracy = []

    for k in [1,2,3,5,7,9,11,13,15]:
        print "UnWeightened KNN-HVDM with K = "  + repr(k)
        print "K-fold = " + repr(k_fold)

        for x in range(k_fold):
            predictions = []
            maxmin = misc.maxMin(trainingSet[x])
            lookup_ = misc.buildLookUp(trainingSet[x])
            classes = misc.getClasses(trainingSet[x])
            for i in range(len(testSet[x])):      
                result = calculateNeighborsUnWeightenedHVDM(trainingSet[x],testSet[x][i],k,lookup_,classes,maxmin)
                predictions.append(result)
            accuracy.append(misc.calculateAccuracy(testSet[x],predictions))
            print 'iteration: '+ repr(x) + " accuracy:" + repr(accuracy[x]) + '%'
        accuracy_avg = sum(accuracy)/k_fold
        print ("Average accuracy : " + repr(accuracy_avg) + '%')
        print 
        accuracy = []
        accuracy_avg = 0
    print  

def Test_hvdm_weightened_knn():
    k_fold = 10
    trainingSet=[]
    testSet = []
    predictions = []
    dataset = misc.loadData('crx.data')
    
    for i in range(len(dataset)):
        if dataset[i][13] != '?':
            dataset[i][13] = float(dataset[i][13])

    trainingSet = np.asarray(trainingSet)
    testSet = np.asarray(testSet)
    predictions = []

    trainingSet,testSet = partition(dataset,k_fold)
    accuracy = []

    for k in [1,2,3,5,7,9,11,13,15]:
        print "Weightened KNN-HVDM with K = "  + repr(k)
        print "K-fold = " + repr(k_fold)

        for x in range(k_fold):
            predictions = []
            maxmin = misc.maxMin(trainingSet[x])
            lookup_ = misc.buildLookUp(trainingSet[x])
            classes = misc.getClasses(trainingSet[x])
            for i in range(len(testSet[x])):      
                result = calculateNeighborsWeightenedHVDM(trainingSet[x],testSet[x][i],k,lookup_,classes,maxmin)
                predictions.append(result)
            accuracy.append(misc.calculateAccuracy(testSet[x],predictions))
            print 'iteration: '+ repr(x) + " accuracy:" + repr(accuracy[x]) + '%'
        accuracy_avg = sum(accuracy)/k_fold
        print ("Average accuracy : " + repr(accuracy_avg) + '%')
        print 
        accuracy = []
        accuracy_avg = 0
    print 

main()