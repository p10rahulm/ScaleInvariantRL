# a moving average of squared gradients to normalize the gradient.
import numpy as np, numpy.random as random, numpy.linalg as la
import utils, generator, errorCalcs, TPCore, DrawingCharting

# This file is used to run Nadam
# To run this type "python NesterovAdam.py"
# You can pipe the output as follows
# "python NesterovAdam.py > outputs/imsaves/NesterovAdam/NesterovAdamLog.txt"

# The default parameters here are beta1 = 0.9,beta2=0.999

def NADAM(A_input, b_input, x_input, probDist, maxIters, errorType=0, utType="Reg", tpType="Reg", momentumMult=0.1):
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
    beta2 = 0.999
    beta1 = 0.9
    meanSquareGradientAccumulator = np.zeros(x_proc.shape)
    momentumAccumulator = np.zeros(x_proc.shape)
    while iters < maxIters:
        numSamples = 3

        sampledRows = random.choice(numStates, numSamples, p=probDist)
        tp1 = TPCore.TPAlgosampledRows(A_proc, b_proc, x_proc, sampledRows=sampledRows)
        tp2 = TPCore.TPAlgosampledRows(A_proc, b_proc, tp1, sampledRows=sampledRows)
        dTP1 = tp1 - x_proc
        dTP2 = tp2 - tp1
        ddTP = dTP2 - dTP1


        kappa = utils.twoNorm(ddTP) / (utils.twoNorm(dTP1)) ** 2
        # Radius of osculating circle
        radius = 1 / kappa
        radiusByNorm_dTP1 = radius / utils.twoNorm(dTP1)

        alpha = 1 / (iters * numSamples / numStates + 1)
        alpha = alpha * radiusByNorm_dTP1
        # Notice that we have multiplied and divided by utils.twoNorm(dTP1) one,
        # which was done for clarity and may be skipped.
        momentumAccumulator = beta1 * momentumAccumulator + (1 - beta1) * dTP1
        meanSquareGradientAccumulator = beta2 * meanSquareGradientAccumulator + (1 - beta2) * dTP1 ** 2
        mHat = momentumAccumulator / (1 - beta1 ** (iters + 1))
        vHat = meanSquareGradientAccumulator / (1 - beta2 ** (iters + 1))
        epsilon = 1e-6
        mUpdater = beta1 * mHat + (1 - beta1) / (1 - beta1 ** (iters + 1)) * dTP1
        momentumTerm = alpha * mUpdater / ((vHat) ** 0.5 + epsilon) - alpha * dTP1

        x_prev = x_proc
        x_proc = x_proc + momentumTerm


        errors[iters] = errorCalcs.getErrorMethod(A_proc, b_proc, x_proc, probDist, errorType, norm=2)
        iters += 1

    return x_proc, errors



if __name__ == "__main__":
    random.seed(8)
    numStates = 25
    numFeatures = 10
    filesLocation = "Outputs/imSaves/nesterovAdam/"
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
        momentumChoices = [0.5]
        for momentum in momentumChoices:
            x_out, errors = NADAM(A, b, start_x, probDist, maxIters, errorType=errorType, utType="Reg", tpType="Reg")
            print("Errors for iteration: %d,momentum = %.2f" % (iter + 1, momentum))
            for i in range(len(errors)):
                print("%d,%.2e" % (i + 1, errors[i]))
            #
            simName = str(iter + 1) + "Long Stochastic CurvatureStep TP with Nesterov Adam Momentum"
            chartTitle = "Errors for Long Stochastic CurvatureStep TP with Nesterov Adam Momentum"
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
