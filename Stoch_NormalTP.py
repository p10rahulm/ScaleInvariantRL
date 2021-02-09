import numpy as np, numpy.random as random, numpy.linalg as la
import utils, generator, errorCalcs, TPCore, DrawingCharting

# This file is used to run Plain Total Projections in the stochastic case
# To run this type "python Stoch_NormalTP.py"
# You can pipe the output as follows
# "python Stoch_NormalTP.py > outputs/imsaves/LongS_PlainTP/LongS_PlainTPLog.txt"

def runTP_Sims(A_input, b_input, x_input, probDist, maxIters, errorType=0, utType="Reg", tpType="Reg"):
    numStates,numFeatures = A_input.shape

    A_proc, b_proc = np.array(A_input, copy=True), np.array(b_input, copy=True)
    x_first = np.array(x_input, copy=True).reshape(numFeatures)
    x_last = x_first
    x_proc = np.array(x_input, copy=True).reshape(numFeatures)
    x_prev = x_proc
    # Error vectors
    errors = np.zeros(maxIters)
    errors[0] = errorCalcs.getErrorMethod(A_proc, b_proc, x_proc, probDist, errorType, norm=2)
    iters = 1
    while iters < maxIters:
        numSamples = 3
        sampledRows = random.choice(numStates,numSamples,p=probDist)
        tp1 = TPCore.TPAlgosampledRows(A_proc, b_proc, x_proc,sampledRows=sampledRows)

        alpha = 1/(iters+1)
        x_proc = x_proc + alpha * (tp1-x_proc)

        errors[iters] = errorCalcs.getErrorMethod(A_proc, b_proc, x_proc, probDist, errorType, norm=2)
        iters += 1

    return x_proc, errors


if __name__ == "__main__":
    random.seed(8)
    numStates = 300
    numFeatures = 30
    filesLocation = "Outputs/imSaves/LongS_PlainTP/"
    print("Error LeastSq,Error TP, Error LstSq - Error TP,Iterations for lower error than Least Squares")
    for iter in range(10):
        A = generator.GenerateRandomMatrix(numStates, numFeatures, max=10)
        b = generator.GenerateRandomVector(numStates)
        start_x = generator.GenerateRandomVector(numFeatures)
        probDist = 1 / numStates * np.ones(numStates)
        errorType = 8
        maxIters = 5000

        np.set_printoptions(precision=3)
        lstSq = la.inv(A.T @ A) @ (A.T @ b)

        lstSqError = errorCalcs.getErrorMethod(A, b, lstSq, probDist, errorType, norm=2)
        x_out, errors = runTP_Sims(A, b, start_x, probDist, maxIters, errorType=errorType, utType="Reg",
                                         tpType="Reg")
        print("Errors for iteration: %d" % (iter + 1))
        for i in range(len(errors)):
            print("%d,%.2e" % (i + 1, errors[i]))
        simName = str(iter + 1) + "Long Stochastic Plain TP"
        chartTitle = "Errors for Long Stochastic Plain TP"
        DrawingCharting.drawSavePlot(maxIters, errors, chartTitle, label='Total Projections Error',
                                                       color="mediumorchid", logErrors=True, plotShow=False,
                                                       location=filesLocation, fileName=simName + ".png")

        LStSqBetterIters = maxIters
        for i in range(maxIters):
            if (errors[i] - lstSqError < 0):
                LStSqBetterIters = i
                break
        ourError = errorCalcs.getErrorMethod(A, b, x_out, probDist, errorType, norm=2)
        errorDiff = ourError - lstSqError
        print("%0.3e,%0.3e,%0.3e,%d" % (lstSqError, ourError, errorDiff, LStSqBetterIters))
