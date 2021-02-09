import numpy as np, numpy.random as random

# This file is used to generate the examples in Appendix A of the paper.
# We generate two sets of errors, one for our error, and the other for least squares
def ourErrors(phi, V=np.array([2, 2, 2])):
    V = np.array([2] * len(phi))
    N = np.diag(1 / phi)
    phiNew = N.dot(phi)
    vNew = N.dot(V)
    w = 1 / (phiNew.dot(phiNew)) * phiNew.dot(vNew)
    print("our w=", w)
    vEstimate = w * phi
    errors = V - vEstimate
    print("our vEstimate:", vEstimate)
    print("our errors:", errors)
    return (w, vEstimate, errors)


def lsErrors(phi, V=np.array([2, 2, 2])):
    V = np.array([2] * len(phi))
    N = np.eye(len(phi))
    phiNew = N.dot(phi)
    vNew = N.dot(V)
    w = 1 / (phiNew.dot(phiNew)) * phiNew.dot(vNew)
    print("LS w=", w)
    vEstimate = w * phi
    errors = V - vEstimate
    print("LS VEstimate:", vEstimate)
    print("LS errors:", errors)
    return (w, vEstimate, errors)


def getAllErrors(phi, V=np.array([2, 2, 2])):
    print("Our Estimates")
    ourW, _, ourErrorArray = ourErrors(phi, V)
    print("LS Estimates")
    lsW, _, lsErrorArray = lsErrors(phi, V)
    return ourW,lsW,ourErrorArray, lsErrorArray


def runAvgSequence(size,numIters):
    totalourW, totallsW, totalOurErrorArray, totalLsErrorArray = 0, 0, np.zeros(size), np.zeros(size)
    for i in range(numIters):
        phi = np.append(random.normal(1, 0.05, size - 1), 5)
        totalourW, totallsW, ourErrorArray, lsErrorArray = getAllErrors(phi)
        totalOurErrorArray += ourErrorArray
        totalLsErrorArray += lsErrorArray
    meanOurW = totalourW / numIters
    meanLSW = totallsW / numIters
    meanOurError = totalOurErrorArray / numIters
    meanLSError = totalLsErrorArray / numIters

    print("mean Our W = ", meanOurW)
    print("mean LS W = ", meanLSW)

    print("mean Our Error = ", meanOurError)
    print("mean LS Error = ", meanLSError)


if __name__ == "__main__":
    float_formatter = "{:.2e}".format
    # np.set_printoptions(formatter={'float_kind': float_formatter})
    np.set_printoptions(precision=2)
    phi = np.array([1.01, .99, 2])
    getAllErrors(phi)
    random.seed(8)
    size = 50
    numIters=1000
    runAvgSequence(size, numIters)
