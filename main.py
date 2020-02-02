import ctypes as c
import csv
from igb import ml

dataset_path = "D:/_Matireals/mnist/mnist/"

trainInputs = ml.csv2dataset(csv.reader(open(dataset_path + "trainInputs.csv", "r")))
trainOutputs = ml.csv2dataset(csv.reader(open(dataset_path + "trainOutputs.csv", "r")))
testInputs = ml.csv2dataset(csv.reader(open(dataset_path + "testInputs.csv", "r")))
testOutputs = ml.csv2dataset(csv.reader(open(dataset_path + "testOutputs.csv", "r")))

igb = ml.igboost(5,10,1,0.1,0.8)
igb.fit(trainInputs,trainOutputs)

ml.accuracy(igb, testInputs, testOutputs)


