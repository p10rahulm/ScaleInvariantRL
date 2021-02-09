import numpy as np, numpy.random as random, numpy.linalg as la
import utils

# This is a helper file that helps us compute the errors.
# Given an iterate and a (ovrrdetermined) linear system,
# we may have many errors to compute and this file helps us compute these

# This splits the error amongst various methods
def getErrorMethod(A, b, x, probDist, errorMethod, norm=2):
    if errorMethod == 1:
        return getErrorfromAb_SumDistanceFromHPs(A, b, x, probDist)
    if errorMethod == 2:
        return getErrorfromAb_SumNormedDistanceFromHPs(A, b, x, probDist, norm)
    if errorMethod == 8:
        return getDistanceFromIdealx_SumDistanceFromHPs(A, b, x, probDist, norm)
    else:
        return getErrorfromAb(A, b, x, probDist)


# This error is the weighted sum of squares error.
# This will be the default that we use
def getErrorfromAb(A, b, x, probDist):
    x_flat = x.reshape(x.shape[0])
    b_flat = b.reshape(b.shape[0])
    error_per_coordinate = np.abs(A.dot(x_flat) - b_flat)
    diagprobDist = np.diag(probDist)
    weighted_error_per_coordinate = diagprobDist @ error_per_coordinate
    return utils.twoNorm(weighted_error_per_coordinate)


# Finds first instance of number in error vector less than certain quantity
def getErrorThresholdsfromErrorVector(errorVec):
    error_range = np.array([10 ** -i for i in range(1, 7)])
    threshold_values = len(errorVec) - np.searchsorted(errorVec, error_range, side='left',
                                                       sorter=np.arange(len(errorVec) - 1, -1, -1))
    thresholds = dict(zip(error_range, threshold_values))
    return thresholds


def printErrorThresholds(thresholds, failedThreshold):
    print("Error Value, Iterations at which reached")
    for key, value in thresholds.items():
        if (value < failedThreshold):
            print("%.2e,%d" % (key, value))


def printErrorThresholdsforSim(thresholds, failedThreshold):
    ender = ", "
    for key, value in thresholds.items():
        if key == 1e-6:
            ender = " || "
        if (value < failedThreshold):
            print("%d" % (value), end=ender)
        else:
            print("N/A", end=ender)



def getIdealX_SumDistanceFromHPs(A, b, probDist):
    m, n = A.shape
    AiAiT = np.zeros((n, n))
    for i in range(m):
        Ai = A[i].reshape((n,1))
        Mi = (Ai @ Ai.T)/(np.linalg.norm(Ai,2)**2)
        AiAiT += Mi * probDist[i]


    biAi = np.zeros(n)

    for i in range(len(A)):
        Ai = A[i]
        bi = b[i]
        vi = (Ai * bi)/ np.linalg.norm(Ai, 2)**2

        biAi += vi * probDist[i]

    InvAiAiT = np.linalg.inv(AiAiT)
    finalX = InvAiAiT @ biAi
    return finalX

def getDistanceFromIdealx_SumDistanceFromHPs(A, b, x, probDist,norm=2):
    idealX = getIdealX_SumDistanceFromHPs(A,b,probDist)
    return np.linalg.norm(x-idealX,norm)




def getErrorfromAb_SumDistanceFromHPs(A, b, x, probDist):
    x_flat = x.reshape(x.shape[0])
    b_flat = b.reshape(b.shape[0])
    numRows = b.shape[0]
    for i in range(numRows):
        Ai_twonorm = utils.twoNorm(A[i, :])
        A[i, :] = A[i, :] / Ai_twonorm
        b_flat[i] = b_flat[i] / Ai_twonorm

    error_per_coordinate = A.dot(x_flat) - b_flat
    diagStatDist = np.diag(probDist)
    weighted_error_per_coordinate = diagStatDist @ error_per_coordinate
    return utils.twoNorm(weighted_error_per_coordinate)


def getErrorfromAb_SumNormedDistanceFromHPs(A, b, x, probDist, norm=2):
    if not isinstance(norm, int):
        print("Please input integer norm")
        return 0
    x_flat = x.reshape(x.shape[0])
    b_flat = b.reshape(b.shape[0])

    row2Norm = np.apply_along_axis(utils.twoNorm, 1, A)
    obrow2Norm = np.diag((1 / row2Norm))

    distanceToHPNotNormed = np.abs(A.dot(x_flat) - b_flat)
    distanceToHP = obrow2Norm @ distanceToHPNotNormed
    weighted_error_per_coordinate = distanceToHP ** norm
    total_probWtdError = probDist.dot(weighted_error_per_coordinate)
    return total_probWtdError ** (1 / norm)


def getErrorfromPhi(weights, phi, cost, gamma, tm):
    numStates = len(phi)
    A = (np.eye(numStates) - gamma * tm) @ phi
    Adotx = A.dot(weights)
    b = cost
    return utils.infNorm(Adotx - b)


def getWeightedErrorfromPhi(weights, phi, cost, gamma, tm, statDist, errorNorm=2):
    A = phi - gamma * tm @ phi

    row2Norm = np.apply_along_axis(utils.twoNorm, 1, (phi - gamma * tm @ phi))
    obrow2Norm = np.diag((1 / row2Norm))
    b = np.array(cost, copy=True)
    weights = weights.reshape(weights.shape[0])
    b = b.reshape(b.shape[0])

    numRows = A.shape[0]
    # BIG NOTE: PLEASE CHECK WHETHER ROW NORMALIZATION IN CHECKING ERRORS IS OK>
    # ROW NORMALIZATION LEADS TO PURE DISTANCES BETWEEN HYPERPLANES. SO IS A GOOD MEASURE.
    for i in range(numRows):
        Ai_twonorm = utils.twoNorm(A[i, :])
        A[i, :] = A[i, :] / Ai_twonorm
        b[i] = b[i] / Ai_twonorm


    Adotx = A.dot(weights)
    errors = np.abs(Adotx - b).reshape(b.shape[0])
    weightedError = np.Infinity
    try:
        weightedError = np.diag(statDist) @ errors
    except:
        print("An exception occurred")
        print("A=", A.shape, "b=", b.shape, "adotx", Adotx.shape, "errors = ", errors.shape)

    if errorNorm == np.inf:
        return np.linalg.norm(weightedError, ord=np.inf)
    elif errorNorm == 2:
        return utils.twoNorm(weightedError)
    else:
        return np.mean(weightedError)
