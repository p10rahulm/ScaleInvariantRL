import numpy as np
import numpy.random as random
from numpy import linalg as LA

# This is a helper file which contains various utility functions
# It is used in various parts of our code. Main functions used are twoNorm and twoNormSq

def infNorm(someVector):
    return np.max(np.abs(someVector))


def twoNormSq(someVector):
    return np.sum(np.square(someVector))


def twoNorm(someVector):
    return np.sqrt(np.sum(np.square(someVector)))


def determinant(someMatrix):
    return np.linalg.det(someMatrix)


def GeneratePSD(size, lo=-5, hi=5):
    # https://math.stackexchange.com/questions/796476/generate-random-symmetric-positive-definite-matrix
    A = random.uniform(-np.sqrt(abs(lo)), np.sqrt(abs(hi)), (size, size))
    AA = A.dot(A.T)
    return AA


def RandomB(size, lo=-5, hi=5):
    return random.uniform(lo, hi, (size, 1))


def getEig(A):
    eigVals, eigVecs = LA.eig(A)
    eigVals.sort()
    return eigVals


def AbdivByANorm(A_input, b_input):
    row2Norm = np.apply_along_axis(twoNorm, 1, A_input)
    obrow2Norm = np.diag((1 / row2Norm))

    # Processing variables
    A_proc = obrow2Norm @ A_input
    b_proc = (obrow2Norm @ b_input).reshape(b_input.shape[0])

    return A_proc, b_proc

def prettyPrintNpArray(arr):
    print("[ ",end='')
    for i in range(len(arr)):
        if(i==len(arr)-1):
            print("%0.3e" % arr[i], end=' ]')
            return
        print("%0.3e"%arr[i], end=', ')