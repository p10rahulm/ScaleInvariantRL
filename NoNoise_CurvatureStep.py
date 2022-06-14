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


        rprime_t = tp1  -x_proc
        rprime_tplus1 = tp2 - tp1
        rprime_Avg = (rprime_t + rprime_tplus1) / 2

        tangent_t = rprime_t/utils.twoNorm(rprime_Avg)

        # tangent_tplus1 = rprime_tplus1 / utils.twoNorm(rprime_tplus1)
        tangent_tplus1 = rprime_tplus1 / utils.twoNorm(rprime_Avg)

        # print("utils.twoNorm(tangent_t)",utils.twoNorm(tangent_t))
        # print("utils.twoNorm(tangent_tplus1)", utils.twoNorm(tangent_tplus1))

        tangent_prime_t =tangent_tplus1-tangent_t
        normal =tangent_prime_t/utils.twoNorm(tangent_prime_t)

        rdoubleprime_t = rprime_tplus1 - rprime_t

        rprime_t_norm =utils.twoNorm(rprime_t)
        rdoubleprime_t_norm = utils.twoNorm(rdoubleprime_t)

        # Curvature = rprime_t_norm**3/(rprime_t_norm**2*rdoubleprime_t_norm**2 - (np.dot(rprime_t,rdoubleprime_t))**2)**0.5


        Curvature = utils.twoNorm(tangent_prime_t)/utils.twoNorm(rprime_Avg)
        alpha = 1 / Curvature
        x_proc = x_proc + alpha * (rprime_Avg) / utils.twoNorm(rprime_Avg)

        # Curvature = utils.twoNorm(tangent_prime)/utils.twoNorm(rprime_t)
        # alpha = Curvature
        alpha = 1 / Curvature
        # dTP1 = tp1 - x_proc
        # dTP2 = tp2 - tp1
        # ddTP = dTP2 - dTP1
        # kappa = utils.twoNorm(ddTP) / utils.twoNorm(dTP1)
        # alpha = 1 / kappa
        x_proc = x_proc + alpha * (rprime_t)/utils.twoNorm(rprime_t)*1/(iters+1)
        # x_proc = x_proc + alpha * (rprime_t)

        errors[iters] = errorCalcs.getErrorMethod(A_proc, b_proc, x_proc, probDist, errorType, norm=2)
        print("iters=",iters,"errors[iters]",errors[iters])
        iters += 1

    return x_proc, errors


if __name__ == "__main__":
    random.seed(8)
    # numStatesArray = [100,1000,10000]
    # numFeaturesArray = [30,300,3000]
    numStatesArray = [100, 500, 1000]
    numFeaturesArray = [25, 125, 250]
    for paramCounter in range(3):
        numStates = numStatesArray[paramCounter]
        numFeatures = numFeaturesArray[paramCounter]
        filesLocation = "Outputs/imSaves/NoNoise_CurvatureStep/"
        print("Error LeastSq,Error TP, Error LstSq - Error TP,Iterations for lower error than Least Squares")
        numtrials = 10
        maxIters = 500
        ErrorHolder = np.zeros((maxIters,numtrials))
        for iter in range(numtrials):
            A = generator.GenerateRandomMatrix(numStates, numFeatures, max=10)
            _, s, _ = np.linalg.svd(A)
            b = generator.GenerateRandomVector(numStates)
            start_x = generator.GenerateRandomVector(numFeatures)
            probDist = 1 / numStates * np.ones(numStates)
            errorType = 8


            np.set_printoptions(precision=3)
            lstSq = la.inv(A.T @ A) @ (A.T @ b)

            lstSqError = errorCalcs.getErrorMethod(A, b, lstSq, probDist, errorType, norm=2)
            x_out, errors = runTP_Sims(A, b, start_x, probDist, maxIters, errorType=errorType, utType="Reg",
                                       tpType="Reg")
            print("Errors for iteration: %d, Kappa = %.1f" % (iter + 1, s[0]/s[-1]))
            for i in range(len(errors)):
                ErrorHolder[i,iter] = errors[i]
                print("%d,%.2e" % (i + 1, errors[i]))
            simName = str(iter + 1) + "TP for curvatureStep with no noise, m" + str(numStates) + "n" + str(numFeatures)
            chartTitle = "Errors for TP for curvatureStep with no noise, m = " + str(numStates) + " , n = " + str(numFeatures)
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
        AverageError = np.mean(ErrorHolder,axis=1)
        print("\nAverage Errors over iterations for m = %d, n= %d"%(numStates,numFeatures))
        for i in range(len(AverageError)):
            print("%d,%.2e" % (i + 1, AverageError[i]))