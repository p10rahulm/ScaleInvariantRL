import numpy as np, numpy.random as random, numpy.linalg as la
import utils

# This is a helper file that is used to generate all random inputs that we will use.
# Some examples are generating initial iterate and generating random matrices and vectors that form the linear system

def generateRandomNormalMatrix(rows, cols, mean=0, standardDevn=1):
    randMatrix = random.normal(mean, standardDevn, size=(rows, cols))
    return randMatrix

def generateRandomUniformMatrix(rows, cols, min=-1, max=1):
    if (min >= max):
        return
    randMatrix = min+random.random((rows, cols))*(max-min)
    return randMatrix


def GenerateRandomMatrix(rows, cols, max=10):
    randMatrix = random.random((rows, cols))
    randMatrix -= 0.5
    randMatrix *= 2 * max
    return randMatrix


def GenerateRandomVector(length, max=10, positive=False):
    if positive:
        return max * (random.rand(length))
    return max * 2 * (random.rand(length) - 0.5)


def GeneratePSD(size, lo=-5, hi=5):
    A = random.uniform(-np.sqrt(abs(lo)), np.sqrt(abs(hi)), (size, size))
    AA = A.dot(A.T)
    return AA


def getTransitionMatrix(numStates):
    transitionMatrix = random.rand(numStates, numStates)
    transitionMatrix = (transitionMatrix.T / np.sum(transitionMatrix, axis=1)).T
    return transitionMatrix


def getStationaryDistribution(transitionMatrix):
    numStates = len(transitionMatrix)
    eigVals, eigVectors = np.linalg.eig(transitionMatrix.T)
    # print("eigVectors=", eigVectors)
    stationaryDist = eigVectors[:, abs(eigVals - 1) < 1e-5]
    stationaryDist = np.real(stationaryDist)
    stationaryDist = stationaryDist / np.sum(stationaryDist)
    # stationaryDist = stationaryDist / sum(stationaryDist)

    assert utils.infNorm(stationaryDist - (stationaryDist.T @ transitionMatrix).T) < 1e-6
    return stationaryDist.reshape(numStates)
