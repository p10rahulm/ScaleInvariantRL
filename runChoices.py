import numpy as np, numpy.random as random, numpy.linalg as la
import interface, generator, errorCalcs, DrawingCharting, oneRing

# This is a core file that is called from interface.py, our main user facing file

# We can get the size of the system as a tuple from a string of the form "25x10" with the below function
def getSizeofLinearSystemFromSizeString(sizeofLinearSystem):
    return tuple(map(lambda x: int(x), sizeofLinearSystem.split("x")))

# We use the below function to get tuple of rows and columns of linear sustem given size choice
def getSizeofLinearSystem(sizeofLinearSystemChoice, sizeofLinearSystemList):
    if sizeofLinearSystemChoice == 1:
        return 15, 5
    if sizeofLinearSystemChoice == 2:
        return 25, 10
    if sizeofLinearSystemChoice == 3:
        return 100, 30

# We use the below function to generate the matrices as per user inputs for randomness
def generateAbasPerChoiceInputs(randomInputsTypeChoice, numStates, numFeatures):
    # randomInputTypeList = ["Uniform [-1,1]", "Uniform [-5,5]", "Normal (0,1)", "Normal (0,5)"]
    if randomInputsTypeChoice == 1:
        A = generator.generateRandomUniformMatrix(numStates, numFeatures, min=-1, max=1)
        b = generator.generateRandomUniformMatrix(numStates, 1, min=-1, max=1).reshape(numStates)
        return A, b
    if randomInputsTypeChoice == 2:
        A = generator.generateRandomUniformMatrix(numStates, numFeatures, min=-5, max=5)
        b = generator.generateRandomUniformMatrix(numStates, 1, min=-5, max=5).reshape(numStates)
        return A, b
    if randomInputsTypeChoice == 3:
        A = generator.generateRandomNormalMatrix(numStates, numFeatures, mean=0, standardDevn=1)
        b = generator.generateRandomNormalMatrix(numStates, 1, mean=0, standardDevn=1).reshape(numStates)
        return A, b
    if randomInputsTypeChoice == 4:
        A = generator.generateRandomNormalMatrix(numStates, numFeatures, mean=0, standardDevn=5)
        b = generator.generateRandomNormalMatrix(numStates, 1, mean=0, standardDevn=5).reshape(numStates)
        return A, b
    else:
        return np.zeros((numStates, numFeatures)), np.zeros(numStates)

# We use the below function to generate the name of the charts based on the choices
def getChartName(sizeofLinearSystem, stochasticType, randomInputType, stepSizeType, momentumType, momentumTypeChoice,
                 momentumParam, momentumParam2):
    outName = "Errors for "
    outName += sizeofLinearSystem + " Random Linear System (sampled from "
    outName += randomInputType + ")\nfor "
    outName += stochasticType
    outName += " TP: "
    outName += stepSizeType + " with "
    outName += momentumType
    if (momentumTypeChoice == 2 or momentumTypeChoice == 6):
        outName += " with parameter = "
        outName += str(momentumParam)
    elif (momentumTypeChoice == 3 or momentumTypeChoice == 4):
        outName += " with parameters = ("
        outName += str(momentumParam) + "," + str(momentumParam2) + ")"
    return outName


# We use the below function to generate the name of the log file and also the names of the charts we create
def getSimName(sizeofLinearSystem, stochasticType, randomInputType, stepSizeType, momentumType,
               momentumTypeChoice, momentumParam, momentumParam2):
    outName = ""
    outName += sizeofLinearSystem + "|Random("
    outName += randomInputType + ")System|"
    outName += stochasticType
    outName += " TP|"
    outName += stepSizeType + "|"
    outName += momentumType
    if (momentumTypeChoice == 2 or momentumTypeChoice == 6):
        outName += ",Parameter-"
        outName += str(momentumParam)
    elif (momentumTypeChoice == 3 or momentumTypeChoice == 4):
        outName += ",Parameters-"
        outName += str(momentumParam)
        outName += ","
        outName += str(momentumParam2)
    outName = outName.replace(" ", "")
    outName = outName.replace("|", "~")
    return outName

# The below function is called from "interface.py"
# and will be the main function used to run the choices
# This function will call the function CodeRunner from
# file oneRing.py which runs Total Projections with all the choices
def runChoices(numberofRunsChoice, maxIterationsChoice, stochasticTypeChoice,
               sizeofLinearSystemChoice, randomInputsTypeChoice, stepSizeTypeChoice,
               momentumTypeChoice, momentumParamChoice, momentumParam2Choice):
    # We create lists for possible choices below:
    numberofRunsQues, numberofRunsList, maxIterationsQues, maxIterationsList, stochasticTypeQues, stochasticTypeList, \
    sizeofLinearSystemQues, sizeofLinearSystemList, randomInputTypeQues, randomInputTypeList, \
    stepSizeTypeQues, stepSizeTypeList, momentumTypeQues, momentumTypeList, \
    momentumParamQues, momentumParamList, momentumParam2Ques, momentumParam2List = interface.choicesData()

    # We retrieve all user inputs
    numberofRuns = numberofRunsList[numberofRunsChoice - 1]
    maxIterations = maxIterationsList[maxIterationsChoice - 1]
    stochasticType = stochasticTypeList[stochasticTypeChoice - 1]
    sizeofLinearSystem = sizeofLinearSystemList[sizeofLinearSystemChoice - 1]
    randomInputType = randomInputTypeList[randomInputsTypeChoice - 1]
    stepSizeType = stepSizeTypeList[stepSizeTypeChoice - 1]
    momentumType = momentumTypeList[momentumTypeChoice - 1]
    momentumParam = momentumParamList[momentumParamChoice - 1]
    momentumParam2 = momentumParam2List[momentumParam2Choice - 1]

    # We get the size of the system
    numStates, numFeatures = getSizeofLinearSystemFromSizeString(sizeofLinearSystem)

    # We use seed so that all files generated use same seed for repeatability
    random.seed(8)
    # Below is the location where the log files and charts generated will be stored
    filesLocation = "Outputs/runAllOutputs/"
    simName = getSimName(sizeofLinearSystem, stochasticType, randomInputType, stepSizeType, momentumType,
                         momentumTypeChoice, momentumParam, momentumParam2)
    logFileName = simName + "_Log.txt"
    # We now create the file
    f = open(filesLocation + logFileName, "w")
    f.close()
    errorType = 8
    # We set some basic variables below
    maxIters = maxIterations
    totalErrors, meanErrors = np.zeros(maxIters), np.zeros(maxIters)
    totalIterations = numberofRuns
    # We loop for each iteration below
    for iter in range(totalIterations):
        # We generate the required matrices
        A, b = generateAbasPerChoiceInputs(randomInputsTypeChoice, numStates, numFeatures)
        start_x = generator.GenerateRandomVector(numFeatures)
        # This is the transition probability matrix, now set to uniform
        probDist = 1 / numStates * np.ones(numStates)
        # We set the print rules
        np.set_printoptions(precision=3)
        # This is the main caller. It calls CodeRunner from oneRing.py which retrieves all errors for choices.
        x_out, errors = oneRing.CodeRunner(A, b, start_x, probDist, maxIters, errorType=errorType,
                                           stochasticType=stochasticType, stepSizeType=stepSizeType,
                                           momentumType=momentumType, momentumParam=momentumParam,
                                           momentumParam2=momentumParam2)
        # We now write these errors to file.
        f = open(filesLocation + logFileName, "a")
        f.write("Errors for iteration: %d\n" % (iter + 1))
        for i in range(len(errors)):
            f.write("%d,%.2e\n" % (i + 1, errors[i]))
        f.close()
        # We compute the total error so far
        totalErrors = totalErrors + errors

        # We plot the errors into chart
        chartTitle = getChartName(sizeofLinearSystem, stochasticType, randomInputType, stepSizeType, momentumType,
                                  momentumTypeChoice, momentumParam, momentumParam2)
        chartFileName = str(iter + 1) + "-" + simName

        DrawingCharting.drawSavePlot(maxIters, errors, chartTitle,
                                     label='Total Projections Error',
                                     color="mediumorchid", logErrors=True, plotShow=False,
                                     location=filesLocation, fileName=chartFileName + ".png")

    # We now calculate the mean error over all the iterations
    meanErrors = totalErrors / totalIterations
    # We paste these mean errors into the log file
    f = open(filesLocation + logFileName, "a")
    f.write("Average Errors\n")
    for i in range(len(meanErrors)):
        f.write("%d,%.2e\n" % (i + 1, meanErrors[i]))
    f.close()
