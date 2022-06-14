import numpy as np, numpy.random as random, numpy.linalg as la
import utils, generator, errorCalcs, TPCore, DrawingCharting

# This file is used to run curvature-step Total Projections in the non-stochastic case
# To run this type "python NStoch_CurvatureStepTP.py"
# You can pipe the output as follows
# "python NStoch_CurvatureStepTP.py > outputs/imsaves/LongNS_CurvatureStepTP/LongNS_CurvatureStepTPLog.txt"


def runTP_Sims(A_input, b_input, x_input, probDist, maxIters, errorType=0, utType="Reg", tpType="Reg"):
    numStates, numFeatures = A_input.shape

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
        tp1 = TPCore.TPAlgo(A_proc, b_proc, x_proc)
        tp2 = TPCore.TPAlgo(A_proc, b_proc, tp1)
        dTP1 = tp1 - x_proc                                 # r'(t)
        dTP2 = tp2 - tp1                                    # r'(t+1)
        ddTP = dTP2 - dTP1                                  # r''(t)
        kappa = utils.twoNorm(ddTP) / utils.twoNorm(dTP1)   # ||r''(t)||/||r'(t)||
        alpha = 1 / kappa                                   # ||r'(t)||/||r''(t)||
        x_proc = x_proc + alpha * (tp1 - x_proc)            # ||r'(t)||/||r''(t)|| * r'(t)

        errors[iters] = errorCalcs.getErrorMethod(A_proc, b_proc, x_proc, probDist, errorType, norm=2)
        iters += 1

    return x_proc, errors


if __name__ == "__main__":
    random.seed(8)
    numStates = 100
    numFeatures = 30
    # numStates = 30
    # numFeatures = 10
    filesLocation = "Outputs/imSaves/LongNS_CurvatureStepTP/"
    print("Error LeastSq,Error TP, Error LstSq - Error TP,Iterations for lower error than Least Squares")
    for iter in range(10):
        A = generator.GenerateRandomMatrix(numStates, numFeatures, max=10)
        _, s, _ = np.linalg.svd(A)
        b = generator.GenerateRandomVector(numStates)
        start_x = generator.GenerateRandomVector(numFeatures)
        probDist = 1 / numStates * np.ones(numStates)
        errorType = 8
        maxIters = 500

        np.set_printoptions(precision=3)
        lstSq = la.inv(A.T @ A) @ (A.T @ b)

        lstSqError = errorCalcs.getErrorMethod(A, b, lstSq, probDist, errorType, norm=2)
        x_out, errors = runTP_Sims(A, b, start_x, probDist, maxIters, errorType=errorType, utType="Reg",
                                   tpType="Reg")
        print("Errors for iteration: %d, Kappa = %.1f" % (iter + 1, s[0]/s[-1]))
        for i in range(len(errors)):
            print("%d,%.2e" % (i + 1, errors[i]))
        simName = str(iter + 1) + "Long Non Stochastic CurvatureStep TP"
        chartTitle = "Errors for Long Non Stochastic CurvatureStep TP"
        DrawingCharting.drawSavePlotwithHorizontalLine(maxIters, errors, chartTitle, label='Total Projections Error',
                                                       color="mediumorchid", logErrors=True, plotShow=False,
                                                       location=filesLocation, fileName=simName + ".png",
                                                       lineVal=lstSqError, lineLabel="Least Squares")

        LStSqBetterIters = maxIters
        for i in range(maxIters):
            if (errors[i] - lstSqError < 0):
                LStSqBetterIters = i
                break
        ourError = errorCalcs.getErrorMethod(A, b, x_out, probDist, errorType, norm=2)
        errorDiff = ourError - lstSqError
        print("%0.3e,%0.3e,%0.3e,%d" % (lstSqError, ourError, errorDiff, LStSqBetterIters))
