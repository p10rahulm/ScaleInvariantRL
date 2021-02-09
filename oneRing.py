import numpy as np, numpy.random as random, numpy.linalg as la
import interface, generator, errorCalcs, DrawingCharting, TPCore, utils

# This is a core file for the interface.py pipeline. interface.py calls runChoices.py which calls oneRing.py

# The below file is a single file that can run all the required options
def CodeRunner(A_input, b_input, x_input, probDist, maxIters, errorType=0, stochasticType="Stochastic",
               stepSizeType="Curvature Step", momentumType="Heavy Ball Momentum",
               momentumParam=0.5, momentumParam2=0.99):
    # We first collect the size of the linear system
    numStates, numFeatures = A_input.shape

    # We use the below two placeholders so as to not touch the original matrices A and b
    A_proc, b_proc = np.array(A_input, copy=True), np.array(b_input, copy=True)

    # Our working iterate is called x_proc
    x_proc = np.array(x_input, copy=True).reshape(numFeatures)
    # Our previous iterate is stored in x_prev
    x_prev = x_proc

    # Error vectors. We will store our errors in these
    errors = np.zeros(maxIters)
    errors[0] = errorCalcs.getErrorMethod(A_proc, b_proc, x_proc, probDist, errorType, norm=2)
    iters = 1

    # We will initialize two holders that will be used in some of the momentum methods
    meanSquareGradientAccumulator = np.zeros(x_proc.shape)
    momentumAccumulator = np.zeros(x_proc.shape)
    # We run iterations updating x_proc each time
    while iters < maxIters:
        # First we check for stochastic type. Non stochastic does not sample the rows
        if stochasticType == "Non-Stochastic":
            tp1 = TPCore.TPAlgo(A_proc, b_proc, x_proc)
            tp2 = TPCore.TPAlgo(A_proc, b_proc, tp1)
            # alpha = 1 / (iters + 1)
            # No need for dividing by iters as we have full information here.
            alpha = 1
        # We then check for stochastic case
        else:
            numSamples = 3
            sampledRows = random.choice(numStates, numSamples, p=probDist)
            tp1 = TPCore.TPAlgosampledRows(A_proc, b_proc, x_proc, sampledRows=sampledRows)
            tp2 = TPCore.TPAlgosampledRows(A_proc, b_proc, tp1, sampledRows=sampledRows)
            alpha = 1 / (iters * numSamples / numStates + 1)

        # dTP1 is our main update
        dTP1 = tp1 - x_proc
        # In case of curvature step, there are two further variables that we will need to modify the step size
        # We will not be modifying the Non-Curvature step as we see good results without dividing further by m
        if stepSizeType == "Curvature Step":
            dTP2 = tp2 - tp1
            ddTP = dTP2 - dTP1

            kappa = utils.twoNorm(ddTP) / (utils.twoNorm(dTP1)) ** 2
            # Radius of osculating circle
            radius = 1 / kappa
            radiusByNorm_dTP1 = radius / utils.twoNorm(dTP1)
            # Notice that we have multiplied and divided by utils.twoNorm(dTP1) one,
            # which was done for clarity and may be skipped.
            alpha = alpha * radiusByNorm_dTP1

        # We will now compute our momentum term. First the placeholder for momentum
        momentumTerm = np.zeros(x_proc.shape)
        if momentumType == "No Momentum":
            # In case no momentum, we simply return 0s
            momentumTerm = np.zeros(x_proc.shape)
        elif momentumType == "Heavy Ball Momentum":
            # In case of heavy-ball momentum, the momentum is (current iterate - previous iterate)*constant
            momentumTerm = x_proc - x_prev
            momentumTerm = momentumParam * momentumTerm
            x_prev = x_proc
        elif momentumType == "RMSProp":
            # In case of RMSProp, a modification of Adagrad, we use a multiplier in each direction
            # given by the meanSquareGradientAccumulator. This is multiplied with our original update rule
            # Notice that we subtract the update given by the first term
            # alpha * dTP1 so that we only have an RMSProp update
            # notice that meanSquareGradientAccumulator is a vector that we are dividing by.
            # Thus it is different along each axis
            meanSquareGradientAccumulator = momentumParam * meanSquareGradientAccumulator + \
                                            (1 - momentumParam) * dTP1 ** 2
            epsilon = 1e-6 * np.ones(meanSquareGradientAccumulator.shape)
            momentumTerm = alpha * dTP1 / ((meanSquareGradientAccumulator) ** 0.5 + epsilon) - alpha * dTP1
        elif momentumType == "Adagrad":
            # Adagrad is similar to RMSProp, except that the accumulator just keeps increasing.
            # There is no parameter required here
            meanSquareGradientAccumulator = meanSquareGradientAccumulator + dTP1 ** 2
            epsilon = 1e-6 * np.ones(meanSquareGradientAccumulator.shape)
            momentumTerm = alpha * dTP1 / ((meanSquareGradientAccumulator) ** 0.5 + epsilon) - alpha * dTP1

        elif momentumType == "ADAM":
            # Adam is closer the heavy-ball type of updates than adagrad and RMSProp.
            # Here the iterate moves in a direction given by mHat.
            momentumAccumulator = momentumParam * momentumAccumulator + (1 - momentumParam) * dTP1
            meanSquareGradientAccumulator = momentumParam2 * meanSquareGradientAccumulator + \
                                            (1 - momentumParam2) * dTP1 ** 2
            mHat = momentumAccumulator/(1-momentumParam**(iters+1))
            vHat = meanSquareGradientAccumulator / (1 - momentumParam2**(iters+1))
            epsilon = 1e-6
            momentumTerm = alpha * mHat / ((vHat) ** 0.5 + epsilon) - alpha * dTP1

        elif momentumType == "Nadam":
            # This makes a small change to Adam to move in the direction
            # given by a "more current estimate" of mHat.
            # Many more details are found in: https://ruder.io/optimizing-gradient-descent/
            momentumAccumulator = momentumParam * momentumAccumulator + (1 - momentumParam) * dTP1
            meanSquareGradientAccumulator = momentumParam2 * meanSquareGradientAccumulator + \
                                            (1 - momentumParam2) * dTP1 ** 2
            mHat = momentumAccumulator / (1 - momentumParam**(iters+1))
            vHat = meanSquareGradientAccumulator / (1 - momentumParam2**(iters+1))
            mUpdater =  momentumParam*mHat + (1-momentumParam)/(1-momentumParam**(iters+1))*dTP1
            epsilon = 1e-6
            momentumTerm = alpha * mUpdater / ((vHat) ** 0.5 + epsilon) - alpha * dTP1
        else:
            # In case no good option, we just add 0's
            momentumTerm = np.zeros(x_proc.shape)

        # We do the actual update as a sum of the step size times current gradient  and the momentum term
        x_proc = x_proc + alpha * (dTP1) + momentumTerm

        errors[iters] = errorCalcs.getErrorMethod(A_proc, b_proc, x_proc, probDist, errorType, norm=2)
        iters += 1

    return x_proc, errors
