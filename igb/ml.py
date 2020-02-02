import ctypes as c
import csv


def csv2dataset(tab):
    return [[ float(i) for i in row ] for row in tab]

def csvsize(lll):
    return (len(lll),len(lll[0]))

def d2d(dataset):
    size = csvsize(dataset)
    s = size[0]*size[1]
    x = [0.0]*s
    X = (c.c_double*s)(*x)
    for i in range(size[0]):
        for j in range(size[1]):
            X[i*size[1] + j] = dataset[i][j]
    return X;

def argmax(arr):
    max = -777
    maxind = 0
    for i in range(len(arr)):
        if(max<arr[i]):
            max= arr[i]
            maxind = i
    return maxind


def accuracy(ml, inputs, outputs):
    sum = 0.0;
    for i in range(len(inputs)):
        res = ml.compute(inputs[i])
        if(argmax(res)==argmax(outputs[i])) :sum = sum+1

    print(sum/len(inputs))

class igboost:

    def __init__(self, itter,trees,minpoints,subset,f):
        #self.dll = c.CDLL("D:\\_R&D\\_C++\\C_test2\\x64\Release\\C_test2.dll")
        self.dll = c.CDLL("C_test2.dll")
        self.dll.IgBoost10_create(itter,trees,minpoints,c.c_double(subset),c.c_double(f))

    def fit(self, inputs, outputs):
        self.inpdim = len(inputs[0])
        self.outdim = len(outputs[0])
        self.dll.IgBoost10_fit(d2d(inputs), d2d(outputs), len(inputs), self.inpdim, self.outdim)

    def compute(self, vector):
        X = (c.c_double * self.inpdim)(*vector)
        res = (c.c_double * self.outdim)()
        self.dll.IgBoost10_comp(X, res)       
        return list(res)

    

