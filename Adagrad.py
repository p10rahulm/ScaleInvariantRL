import numpy as np, numpy.random as random, numpy.linalg as la
import utils, generator, errorCalcs, TPCore, DrawingCharting

# This file is used to run Adagrad
# To run this type "python Adagrad.py"
# You can pipe the output as follows
# "python Adagrad.py > outputs/imsaves/Adagrad/AdagradLog.txt"

# This momentum method does not need any additional parameters



def Adagrad(A_input, b_input, x_input, probDist, maxIters, errorType=0, utType="Reg", tpType="Reg",momentumMult=0.1):
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
    gamma = 0.9
    Mean_Squared_gradient = np.zeros(x_proc.shape)
    while iters < maxIters:
        numSamples = 3
        sampledRows = random.choice(numStates,numSamples,p=probDist)
        tp1 = TPCore.TPAlgosampledRows(A_proc, b_proc, x_proc,sampledRows=sampledRows)
        tp2 = TPCore.TPAlgosampledRows(A_proc, b_proc, tp1,sampledRows=sampledRows)
        dTP1 = tp1 - x_proc
        dTP2 = tp2 - tp1
        ddTP = dTP2 - dTP1

        kappa = utils.twoNorm(ddTP) / utils.twoNorm(dTP1)
        # Radius of osculating circle
        radius = 1 / kappa

        momentumTerm = x_proc - x_prev
        x_prev = x_proc

        alpha = 1/(iters*numSamples/numStates+1)
        step = (alpha * radius* (dTP1))
        Mean_Squared_gradient = Mean_Squared_gradient + dTP1**2
        epsilon = 1e-6*np.ones(Mean_Squared_gradient.shape)
        Delta = (step)/(Mean_Squared_gradient+epsilon)**0.5
        # Delta = ((dTP1))/Mean_Squared_gradient**0.5
        x_proc = x_proc + Delta


        errors[iters] = errorCalcs.getErrorMethod(A_proc, b_proc, x_proc, probDist, errorType, norm=2)
        iters += 1

    return x_proc, errors


if __name__ == "__main__":
    random.seed(8)
    numStates = 25
    numFeatures = 10
    filesLocation = "Outputs/imSaves/Adagrad/"
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
        x_out, errors = Adagrad(A, b, start_x, probDist, maxIters, errorType=errorType, utType="Reg",
                                         tpType="Reg")
        print("Errors for iteration: %d" % (iter + 1))
        for i in range(len(errors)):
            print("%d,%.2e" % (i + 1, errors[i]))
        simName = str(iter + 1) + "Long Stochastic CurvatureStep TP with Adagrad Momentum"
        chartTitle = "Errors for Long Stochastic CurvatureStep TP with Adagrad Momentum"
        DrawingCharting.drawSavePlotwithHorizontalLine(maxIters, errors, chartTitle, label='Total Projections Error',
                                                       color="mediumorchid", logErrors=True, plotShow=False,
                                                       location=filesLocation, fileName=simName + ".png",
                                                       lineVal=lstSqError,lineLabel="Least Squares Error")

        LStSqBetterIters = maxIters
        for i in range(maxIters):
            if (errors[i] - lstSqError < 0):
                LStSqBetterIters = i
                break
        ourError = errorCalcs.getErrorMethod(A, b, x_out, probDist, errorType, norm=2)
        errorDiff = ourError - lstSqError
        print("%0.3e,%0.3e,%0.3e,%d" % (lstSqError, ourError, errorDiff, LStSqBetterIters))
