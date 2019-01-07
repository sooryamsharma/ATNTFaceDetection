# -*- coding: utf-8 -*-
"""
Created on Sat Oct 6 14:14:35 2018

@author: Suryam Sharma
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt; plt.rcdefaults()
import math
import operator
from sklearn import svm
from sklearn import metrics
from sklearn.cross_validation import KFold

#Reading pixel data
faceData = pd.read_csv('ATNTFaceImages400.txt', delimiter = ',', header = -1).values

dataX = np.transpose(faceData[1:, :])
dataY = np.transpose(faceData[0, :])


def pickData(fileName, Nclass, samples, trainingInstance, testingInstance):
  data = pd.read_csv(fileName, header = -1).values
  x_data = np.transpose(data[1:, :])
  y_data = np.transpose(data[0, :])
  
  Nx = data.shape[0] - 1
  
  y_train, y_test = [], []
  x_train = np.zeros((1, Nx))
  x_test = np.zeros((1, Nx))
  
  for k in Nclass:
    i = k - 1
    x_train = np.vstack((x_train, x_data[(samples * i):((samples * i) + trainingInstance), :]))
    y_train = np.hstack((y_train, y_data[(samples * i):((samples * i) + trainingInstance)]))
    x_test = np.vstack((x_test, x_data[((samples * i) + trainingInstance + 1):((samples * i) + trainingInstance + testingInstance), :]))
    y_test = np.hstack((y_test, y_data[((samples * i) + trainingInstance + 1):((samples * i) + trainingInstance + testingInstance)]))
  
  x_train = x_train[1:, :]
  x_test = x_test[1:, :]
  
  return x_train, y_train, x_test, y_test  

def storeData (xTrain, yTrain, xTest, yTest):
  file = open("trainingData.txt", "w")
  rows = yTrain.shape[0]
  for i in range(0, rows):
      file.write(str(int(yTrain[i])))
      if (i < rows - 1):
          file.write(',')
  file.write("\n")
  xTrain = np.transpose(xTrain)
  row_x, col_x = xTrain.shape
  for i in range(0, row_x):
      for j in range(0, col_x):
          file.write(str(int(xTrain[i][j])))
          if (j < col_x - 1):
              file.write(',')
      file.write("\n")
  file.close()
  # test data
  file = open("testingData.txt", "w")
  n = yTest.shape[0]
  for i in range(0, n):
      file.write(str(int(yTest[i])))
      if (i < n - 1):
          file.write(',')
  file.write("\n")
  xTest = np.transpose(xTest)
  row_x, col_x = xTest.shape
  for i in range(0, row_x):
      for j in range(0, col_x):
          file.write(str(int(xTest[i][j])))
          if (j < col_x - 1):
              file.write(',')
      file.write("\n")
  file.close()
  
#Creating training and testing data for centroid classifier
def storeCentroidData(x_train, y_train, x_test, y_test, trainingLen):
    #training data
    file = open("trainingCentroidData.txt", "w")
    for i in range(1, 41):
        file.write(str(int(i)))
        if (i < 40):
            file.write(',')
    file.write("\n")
    x_train = np.transpose(x_train)
    row_x, col_x = x_train.shape
    for i in range(0, row_x):
        for j in range (0, 40):
            classXCen = round((float(np.sum(x_train[i][j*trainingLen:(j+1)*trainingLen])/trainingLen)), 2)
            file.write(str(classXCen))
            if (j < 39):
                file.write(',')
        file.write('\n')
    file.close()

    #testing data
    file = open("testingCentroidData.txt", "w")
    rows = y_test.shape[0]
    for i in range(0, rows):
        file.write(str(int(y_test[i])))
        if (i < rows - 1):
            file.write(',')
    file.write("\n")
    x_test = np.transpose(x_test)
    row_x, col_x = x_test.shape
    for i in range(0, row_x):
        for j in range(0, col_x):
            file.write(str(int(x_test[i][j])))
            if (j < col_x - 1):
                file.write(',')
        file.write("\n")
    file.close()

# Calculating the Euclidean distance for KNN and Centroid classifier
def calcDistance(instance1, instance2, length):
	distance = 0
	for x in range(length):
		distance += pow((instance1[x] - instance2[x]), 2) # (x1 - x2)^2
	return math.sqrt(distance)

# Finding the k nearest neighbors
def findNeighbors(trainingSet, testingInstance, k):
	distances = []
	length = len(testingInstance)-1
	for x in range(len(trainingSet)):
		dist = calcDistance(testingInstance, trainingSet[x], length)
		distances.append((trainingSet[x], dist))
	distances.sort(key=operator.itemgetter(1))   # to sort the distances from shortest to longest
	neighbors = []
	for x in range(k):
		neighbors.append(distances[x][0])
	return neighbors

# Finding the Nearest Neighbors
def nearestNeighbor(neighbors):

    classVotes = {}
    for x in range(len(neighbors)):
        response = neighbors[x][0]

        if response in classVotes:
            classVotes[response] += 1
        else:
            classVotes[response] = 1
    sortedVotes = sorted(classVotes.items(), key=operator.itemgetter(1), reverse=True)

    return sortedVotes[0][0]

# Evaluating the accuracy of the classifier
def calcAccuracy(testSet, predictions):
	correct = 0
	for x in range(len(testSet)):
		if testSet[x][0] == predictions[x]:
			correct += 1
	return (correct/float(len(testSet))) * 100.0


  
def crossVal(classifier):
  fileName = 'ATNTFaceImages400.txt';
  Nclass = [x for x in range(1, 41, 1)]   # no. of classes
  fileName
  samples = 10            # there are 10 images for each person
  trainingInstance = 8    # out of 10 images, 8 are used for training 
  testingInstance = 2     # 2 for testing
  # for splitting image data into 2 sets: training and testing 
  x_train, y_train, x_test, y_test = pickData(fileName, Nclass, samples, trainingInstance, testingInstance)
  
  xTrain = x_train
  yTrain = y_train
  xTest = x_test
  yTest = y_test  
  
  if (classifier == "KNN"):
      folds = [5]   # 5-fold cross validation
      arr = []
      c = []
      for i in folds:
          kf = KFold(len(dataX), n_folds=i, shuffle=True)
          sum_acc = 0
          
          for training_index, testing_index in kf:
              xTrain, xTest = dataX[training_index], dataX[testing_index]
              yTrain, yTest = dataY[training_index], dataY[testing_index]
              storeData(xTrain, yTrain, xTest, yTest)
              
              file1 = pd.read_csv("trainingData.txt", header=-1)
              trainingData = file1.values
              trainingData = trainingData.transpose()
              
              file2 = pd.read_csv("testingData.txt", header=-1)
              testingData = file2.values
              testingData = testingData.transpose()
              
              predictions=[]
              k = 5
              for x in range(len(testingData)):
                  neighbors = findNeighbors(trainingData, testingData[x], k)
                  nearestClass = nearestNeighbor(neighbors)
                  predictions.append(nearestClass)
              accuracy = calcAccuracy(testingData, predictions)
              print("Accuracy using KNN and {:d}-fold cross validation: {:05.2f}%".format(i, accuracy))
              sum_acc = sum_acc + accuracy
          avg_acc = sum_acc / i
          print("Average Accuracy: {:05.2f}%".format(avg_acc));
          arr.append(avg_acc)
      c = c + [j for j in arr]

  elif (classifier == "Centroid"):
      folds = [5]
      arr = []
      c = []
      for i in folds:
          kf = KFold(len(dataX), n_folds=i, shuffle=True)
          sumacc = 0
          for training_index, testing_index in kf:
              xTrain, xTest = dataX[training_index], dataX[testing_index]
              yTrain, yTest = dataY[training_index], dataY[testing_index]

              storeCentroidData(xTrain, yTrain, xTest, yTest, trainingInstance)

              file1 = pd.read_csv("trainingCentroidData.txt", header=-1)
              trainingData = file1.values
              trainingData = trainingData.transpose()
              
              file2 = pd.read_csv("testingCentroidData.txt", header=-1)
              testingData = file2.values
              testingData = testingData.transpose()
              
              predictions=[]
              k = 5
              for x in range(len(testingData)):
                  neighbors = findNeighbors(trainingData, testingData[x], k)
                  nearestClass = nearestNeighbor(neighbors)
                  predictions.append(nearestClass)
              accuracy = calcAccuracy(testingData, predictions)
              print("Accuracy using Centroid and {:d}-fold cross validation: {:05.2f}%".format(i, accuracy))
              sumacc = sumacc + accuracy
          avg_acc = sumacc / i
          print("Average Accuracy: {:05.2f}%".format(avg_acc))
          arr.append(avg_acc)
      c = c + [j for j in arr]

  elif (classifier == "SVM"):
      folds = [5]
      arr = []
      c = []
      for i in folds:
          kf = KFold(len(dataX), n_folds=i, shuffle=True)
          sumacc = 0
          for training_index, testing_index in kf:
              x_train, x_test = dataX[training_index], dataX[testing_index]
              y_train, y_test = dataY[training_index], dataY[testing_index]
              svmclassifier = svm.LinearSVC()
              svmclassifier.fit(x_train, y_train)
              # calculating prediction
              predicted = svmclassifier.predict(x_test)
              
              actual = y_test
              accuracy = metrics.accuracy_score(actual, predicted) * 100
              # printing accuracy
              print("Accuracy using SVM and {:d}-fold cross validation: {:05.2f}%".format(i, accuracy))
              sumacc = sumacc + accuracy
          avg_acc = sumacc / i
          print("Average Accuracy: {:05.2f}%".format(avg_acc))
          arr.append(avg_acc)
      c = c + [j for j in arr]
      
  elif (classifier == "Linear Regression"):
      folds = [5]
      arr = []
      c = []
      for i in folds:
          kf = KFold(len(dataY), n_folds=i, shuffle=True)
          sumacc = 0
          for training_index, testing_index in kf:
              x_train, x_test = dataX[training_index], dataX[testing_index]
              y_train, y_test = dataY[training_index], dataY[testing_index]
              rx_train, cx_train = x_train.transpose().shape
              rx_test, cx_test = x_test.transpose().shape

              Atrain = np.ones((1,cx_train ))
              Atest = np.ones((1,cx_test))
              Xtrain_padding = np.row_stack((x_train.transpose(),Atrain))
              Xtest_padding = np.row_stack((x_test.transpose(), Atest)) # computing the regression coefficients

              B_padding = np.dot(np.linalg.pinv(Xtrain_padding.T), y_train.T)

              ytest_padding = np.dot(B_padding.T,Xtest_padding)
              ytest_padding_argmax = np.argmax(ytest_padding,axis=0)+1
              err_test_padding = y_test - ytest_padding_argmax
              
              accuracy = (float((np.nonzero(err_test_padding)[0].size-1)/len(err_test_padding)))*100
              print("Accuracy using LR and {:d}-fold cross validation: {:05.2f}%".format(i, accuracy))
              sumacc = sumacc + accuracy
          avg_acc = sumacc / i
          print("Average Accuracy: {:05.2f}%".format(avg_acc))
          arr.append(avg_acc)
      c = c + [j for j in arr]
  return c



y_knn = crossVal("KNN")
y_centroid = crossVal("Centroid")
y_svm = crossVal("SVM")
y_lr = crossVal("Linear Regression")


print("KNN: {:05.2f}%".format(y_knn[0]))
print("Centroid: {:05.2f}%".format(y_centroid[0]))
print("SVM: {:05.2f}%".format(y_svm[0]))
print("Linear Regression: {:05.2f}%".format(y_lr[0]))


# Plotting performance

classifiers = np.array(['KNN', 'Centroid', 'SVM', 'Lin Reg.'])
#y_pos = np.arange(len(classifiers))
y_pos = np.arange(len(classifiers))
performance = np.array([y_knn[0], y_centroid[0], y_svm[0], y_lr[0]])
 
plt.barh(y_pos, performance, align='center', color='blue')
plt.yticks(y_pos, classifiers)
plt.xlabel('Prediction Accuracy')
plt.title('Classifier Performance Measure on ATNT FaceData')
plt.savefig('ClassifierPerformance.png', bbox_inches='tight') 
plt.show()
