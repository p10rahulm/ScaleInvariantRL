# a moving average of squared gradients to normalize the gradient.
import numpy as np, numpy.random as random, numpy.linalg as la
import utils, generator, errorCalcs, TPCore, DrawingCharting


# This file is used to run curvature-step Total Projections in the non-stochastic case
# with heavy ball momentum, with beta values taken between 0 to 1
# To run this type "python ConstMomentumBeta0to1_NS.py"
# You can pipe the output as follows
# "python ConstMomentumBeta0to1_NS.py > outputs/imsaves/ConstantMomentumBeta0to1_NS/ConstantMomentumBeta0to1_NSLog.txt"


def ConstantMomentum(A_input, b_input, x_input, probDist, maxIters, errorType=0, utType="Reg", tpType="Reg",
                     momentumMult=0.1):
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
        dTP1 = tp1 - x_proc
        dTP2 = tp2 - tp1
        ddTP = dTP2 - dTP1

        kappa = utils.twoNorm(ddTP) / utils.twoNorm(dTP1)
        # Radius of osculating circle
        radius = 1 / kappa
        momentumTerm = x_proc - x_prev
        x_prev = x_proc
        alpha = 1 / (iters + 1)
        x_proc = x_proc + alpha * radius * (dTP1) + momentumMult * (momentumTerm)

        errors[iters] = errorCalcs.getErrorMethod(A_proc, b_proc, x_proc, probDist, errorType, norm=2)
        iters += 1

    return x_proc, errors


if __name__ == "__main__":
    numStates = 25
    numFeatures = 10
    filesLocation = "Outputs/imSaves/ConstantMomentumBeta0to1_NS/"
    for momentum in np.linspace(0, 1, 21):
        random.seed(8)
        print("Error LeastSq,Error TP, Error LstSq - Error TP,Iterations for lower error than Least Squares")
        errorType = 8
        maxIters = 5000
        totalErrors, meanErrors = np.zeros(maxIters), np.zeros(maxIters)
        totalIterations=10
        for iter in range(totalIterations):
            A = generator.GenerateRandomMatrix(numStates, numFeatures, max=10)
            b = generator.GenerateRandomVector(numStates)
            start_x = generator.GenerateRandomVector(numFeatures)
            probDist = 1 / numStates * np.ones(numStates)


            np.set_printoptions(precision=3)
            lstSq = la.inv(A.T @ A) @ (A.T @ b)

            lstSqError = errorCalcs.getErrorMethod(A, b, lstSq, probDist, errorType, norm=2)

            x_out, errors = ConstantMomentum(A, b, start_x, probDist, maxIters, errorType=errorType, utType="Reg",
                                             tpType="Reg",momentumMult=momentum)
            totalErrors = totalErrors+errors

            simName = str(iter + 1) + "Long Non-Stochastic CurvatureStep TP with Constant Momentum, Beta " + str(momentum)
            chartTitle = "Errors for Long Non-Stochastic CurvatureStep TP with \nConstant Momentum Beta " + str(momentum)
            DrawingCharting.drawSavePlotwithHorizontalLine(maxIters, errors, chartTitle,
                                                           label='Total Projections Error',
                                                           color="mediumorchid", logErrors=True, plotShow=False,
                                                           location=filesLocation, fileName=simName + ".png",
                                                           lineVal=lstSqError, lineLabel="Least Squares Error")

            LStSqBetterIters = maxIters
            for i in range(maxIters):
                if (errors[i] - lstSqError < 0):
                    LStSqBetterIters = i
                    break

            ourError = errorCalcs.getErrorMethod(A, b, x_out, probDist, errorType, norm=2)
            errorDiff = ourError - lstSqError
            print("%0.3e,%0.3e,%0.3e,%d" % (lstSqError, ourError, errorDiff, LStSqBetterIters))

        meanErrors = totalErrors / totalIterations
        print("Errors for momentum = %.2f" % (momentum))
        for i in range(len(meanErrors)):
            print("%d,%.2e" % (i + 1, meanErrors[i]))
