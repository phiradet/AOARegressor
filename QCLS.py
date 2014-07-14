import numpy as np
import math
import pylab as pl
import random
from FeatureExtractor import Extractor

def drange(start, stop, step):
    output = []
    r = start
    while r <= stop:
        output.append(r)
        r += step
    return output

def GenData(fn, start=0, end=500, step=1 ,n=-1):
    X = drange(start,end, step)
    numRemove = len(X)-n
    if n>0:
        for i in range(numRemove):
            removePos = random.randint(0,len(X)-1)
            X.pop(removePos)
    Y= []
    for x in X:
        Y.append(fn(x))
    return np.matrix(X).transpose(), np.matrix(Y).transpose()


def Distance(A,B):
        return abs(A-B)
    
def Gaussian(A,B, bandwidth=1.0):
    lower = 2.0*bandwidth*bandwidth
    dist = Distance(A,B)
    upper = -1.0*dist*dist
    output= math.exp(upper/lower)
    return output

def GetK(train, predict,bandwidth):
    K_array = []
    rTrain,cTrain = train.shape
    rPredict, cPredict = predict.shape
    
    for i in range(rPredict):
        K_array.append([])
        for j in range(rTrain):
            gaussianVal = Gaussian(train.item(j,0), predict.item(i,0), bandwidth=bandwidth)
            K_array[-1].append(gaussianVal)
    K = np.matrix(K_array)
    
    return K

def AddNoiseByPercentage(Y, amplitude, percentage):
    R,C = Y.shape
    choice = range(0,R)
    numOfNoise = float(percentage)/100.0*R
    for i in range(int(numOfNoise)):
        noisePos = random.choice(choice)
        choice.remove(noisePos)
        Y[noisePos][0] = Y[noisePos][0]+(random.random()*amplitude*2)-(amplitude)
    return Y

def GenerateRandomPSD(size, MAX=-10.0, MIN=10.0):
    Z_array = []
    for i in range(size):
        Z_array.append([])
        for j in range(size):
            randomNum = random.random()*(MAX-MIN)+MIN
            Z_array[-1].append(randomNum)
    Z = np.matrix(Z_array)
    #print np.linalg.eig(Z*(Z.T))
    return (Z.T)*Z

def GenerateI(size, diagVal=1.0):
    Z_array = []
    for i in range(size):
        Z_array.append([])
        for j in range(size):
            if i==j:
                Z_array[-1].append(diagVal)
            else:
                Z_array[-1].append(0.0)
    Z = np.matrix(Z_array)
    return (Z.T)*Z

def GenerateIPattern(size, diagVal=2.0):
    Z_array = []
    for i in range(size):
        Z_array.append([])
        for j in range(size):
            if i==j:
                #Z_array[-1].append(i/diagVal)
                Z_array[-1].append(0)
            else:
                Z_array[-1].append(diagVal*i)
    Z = np.matrix(Z_array)
    return (Z.T)*Z

def GetL_QCLS(K, R, Lambda):
    K_t = K.T
    return ((K_t*K+Lambda*R).I)*K_t

def GetL_LS(K):
    return K.I

def is_pos_def(x):
    return np.all(np.linalg.eigvals(x) >= 0)

def isR_OK(param, R):
    return np.all(param.T*R*param >= 0)

def main():
    ###################
    ## CONFIGURATION ##
    start = 0
    end = 50
    step = 0.5
    stepTest = 0.1
    n = -1

    bandwidth = 1.0

    isNoise = True
    noiseAmplitude = 0.3
    percentage = 50.0

    Lambda = 0.0

    #GenRFn = GenerateRandomPSD
    GenRFn = GenerateI
    #GenRFn = GenerateIPattern

    fnLabel = "sin(x)+cos(2x)"
    genDataFn = lambda x: np.cos(x*2.0)+np.sin(x)
    
    #fnLabel = "log10(x+1)"
    #genDataFn = lambda x: math.log10((x+1))
    
    #fnLabel = "sin(x)"
    #genDataFn = lambda x: np.sin(x)

    #fnLabel = "4x^3+x^2-2x+4"
    #genDataFn = lambda x: 4*(x**3)+(x**2)-(2*x)+4

    #fnLabel = "Random function"
    #genDataFn = lambda x: 3*random.random()

    #genDataFn = lambda x: np.cos(x*2.0)+2.0*np.sin(x)
    #genDataFn = lambda x: np.cos(x/2.0)
    ###################


    ## GENERATE DATA ##
    X, Y = GenData(genDataFn,start=start, end=end, step=step, n=n)
    Y=AddNoiseByPercentage(Y, noiseAmplitude, percentage)

    R = GenRFn(Y.shape[0])
    print "size of R"
    print R.sum()


    ### Parameter Estimation ###
    K = GetK(X,X,bandwidth)
    L = GetL_QCLS(K, R, Lambda)
    L_LS = GetL_LS(K)
    param = L*Y
    if not isR_OK(param, R):
        print 'try new R'
        return
    else:
        print "OK, R"
    param_LS = L_LS*Y
    ###########################


    ############ TEST ##############
    outputX = drange(start, end, stepTest)
    outputX = np.matrix(outputX).T
    K_test = GetK(X,outputX,bandwidth)
    outputY = (K_test*param).T
    outputY_LS = (K_test*param_LS).T
    ################################

    X_array = np.squeeze(np.asarray(X))
    Y_array = np.squeeze(np.asarray(Y))
    outputX_array = np.squeeze(np.asarray(outputX))
    outputY_array = np.squeeze(np.asarray(outputY))
    outputY_LS_array = np.squeeze(np.asarray(outputY_LS))

    pl.scatter(X_array, Y_array, c='k', label='input')
    #pl.plot(X_array, Y_array, c='k', linestyle='--', label='input')
    pl.hold('on')
    pl.plot(outputX_array, outputY_array, c='r', label='QCLS')
    #pl.plot(outputX_array, outputY_LS_array, c='g', linestyle="--", label='LS')
    pl.xlabel('x')
    pl.ylabel('f(x)')
    pl.title("%s"%(fnLabel))
    pl.figtext(0.4, .02, "bandwidth={0:.2f}, noise=[-{2:.2f},{2:.2f}], noise percentage={4:.2f}%, #sample={3}, Rfn={5}".format(bandwidth,isNoise,noiseAmplitude,len(Y_array),percentage, GenRFn.__name__))
    pl.legend()
    pl.show()
        

if __name__ == "__main__":
    #R = GenerateRandomPSD(3, MAX=10,MIN=-10)
    #print is_pos_def(R)
    #print R
    main()

