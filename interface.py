import numpy as np, numpy.random as random
import runChoices

# This is the main client facing file. This will call runchoices which contains the work-horse functions

#
# # This is our interface file which will be used to generate other inputs
# # We will be using this file to generate all other outputs.
# # For each input, we will give options, one of which the user has to choose and then run the preferred file of
# # their choice.
#
# Number of runs:
# 1. 1
# 2. 5
# 3. 10
# 4. 25
# 5. 50
#
#
# Max Iterations:
# 1. 500
# 2. 1000
# 3. 5000
# 4. 10000
# 5. 25000
#
#
# Stochastic vs Non-stochastic:
# 1. Non-Stochastic
# 2. Stochastic
#
# Size of A (in Ax=b)
# 1. 15x5
# 2. 25x10 (default)
# 3. 100x30
#
#
# Random Inputs:
# 1. Uniform [-1,1] (default)
# 2. Uniform [-5,5]
# 3. Normal (0,1)
# 4. Normal (0,5)
#
#
# Step Size:
# 1. Regular TP Step
# 2. Curvature Step
#
# Momentum Type:
# 1. No Momentum
# 2. Heavy Ball Momentum (default)
# 3. Adam
# 4. Nadam
# 5. Adagrad
# 6. RMSProp
#
# Momentum Parameter:
# 1. 0.01
# 2. 0.05
# 3. 0.1
# 4. 0.25
# 5. 0.5
# 6. 0.75
# 7. 0.9
# 8. 0.95
# 9. 0.99
#


# The below function will be used to get user input
def getUserInput(question, listOfChoices, recursionNum=0):
    if (recursionNum > 10):
        print("Sorry. You have exceeded the maximum number of tries.")
        return -1
    print(question)
    print("The choices are:")
    for index in range(len(listOfChoices)):
        print("%d. %s" % (index + 1, listOfChoices[index]))
    choice = input("Please enter your choice of input:")
    try:
        intChoice = int(choice)
        if intChoice in list(range(1, len(listOfChoices) + 1)):
            print("Your choice for \"", question, "\" is: ", listOfChoices[intChoice - 1], "\n", sep="")
        else:
            print("You have entered an invalid input")
            print("Please enter a number between %d and %d!\n\n" % (1, len(listOfChoices)))
            choice = getUserInput(question, listOfChoices, recursionNum + 1)
    except:
        print("You have not entered a number choice.")
        print("Please enter a number between %d and %d!\n\n" % (1, len(listOfChoices)))
        choice = getUserInput(question, listOfChoices, recursionNum + 1)
    return int(choice)

# The below function is used as a user check
def errorCheck(error, input):
    if error == 1 or input < 0:
        return 1
    else:
        return 0

# The below function is used to store the questions and choice data
def choicesData():
    numberofRunsQues = "How many runs of the simulation do you wish to run?"
    numberofRunsList = [1, 5, 10, 25, 50]
    maxIterationsQues = "What is the maximum number of iterations you wish to run per run"
    maxIterationsList = [500, 1000, 5000, 10000, 25000]
    stochasticTypeQues = "Do you wish to run in stochastic mode or non-stochastic?"
    stochasticTypeList = ["Non-Stochastic", "Stochastic"]
    sizeofLinearSystemQues = "What is the size of the linear system you wish to work with?"
    sizeofLinearSystemList = ["15x5", "25x10", "100x30"]
    randomInputTypeQues = "What distribution do you wish the random linear system to be sampled from?"
    randomInputTypeList = ["Uniform [-1,1]", "Uniform [-5,5]", "Normal (0,1)", "Normal (0,5)"]
    stepSizeTypeQues = "What is the step size type you wish to choose?"
    stepSizeTypeList = ["Regular TP Step", "Curvature Step"]
    momentumTypeQues = "What is the momentum type you wish to choose?"
    momentumTypeList = ["No Momentum", "Heavy Ball Momentum", "ADAM", "Nadam", "Adagrad", "RMSProp"]
    momentumParamQues = "Please choose a momentum parameter choice"
    momentumParamList = [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]
    momentumParam2Ques = "Please choose the second momentum parameter choice"
    momentumParam2List = [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 0.999]

    return numberofRunsQues, numberofRunsList, maxIterationsQues, maxIterationsList, stochasticTypeQues, \
           stochasticTypeList, sizeofLinearSystemQues, sizeofLinearSystemList, randomInputTypeQues, \
           randomInputTypeList, stepSizeTypeQues, stepSizeTypeList, momentumTypeQues, momentumTypeList, \
           momentumParamQues, momentumParamList, momentumParam2Ques, momentumParam2List


# The below function is the main program that is run
def getRunInputs():
    error = 0
    # We create lists for possible choices below:
    numberofRunsQues, numberofRunsList, maxIterationsQues, maxIterationsList, stochasticTypeQues, stochasticTypeList, \
    sizeofLinearSystemQues, sizeofLinearSystemList, randomInputTypeQues, randomInputTypeList, stepSizeTypeQues, \
    stepSizeTypeList, momentumTypeQues, momentumTypeList, momentumParamQues, momentumParamList, \
    momentumParam2Ques, momentumParam2List = choicesData()
    numberofRunsChoice = getUserInput(numberofRunsQues, numberofRunsList, 0)
    print("numberofRunsChoice:", numberofRunsChoice)
    if errorCheck(error, numberofRunsChoice):
        return -1

    maxIterationsChoice = getUserInput(maxIterationsQues, maxIterationsList, 0)
    print("maxIterationsChoice:", maxIterationsChoice)
    if errorCheck(error, maxIterationsChoice):
        return -1

    stochasticTypeChoice = getUserInput(stochasticTypeQues, stochasticTypeList, 0)
    print("stochasticTypeChoice:", stochasticTypeChoice)
    if errorCheck(error, stochasticTypeChoice):
        return -1

    sizeofLinearSystemChoice = getUserInput(sizeofLinearSystemQues, sizeofLinearSystemList, 0)
    print("sizeofLinearSystemChoice:", sizeofLinearSystemChoice)
    if errorCheck(error, sizeofLinearSystemChoice):
        return -1

    randomInputsTypeChoice = getUserInput(randomInputTypeQues, randomInputTypeList, 0)
    print("randomInputsTypeChoice:", randomInputsTypeChoice)
    if errorCheck(error, randomInputsTypeChoice):
        return -1

    stepSizeTypeChoice = getUserInput(stepSizeTypeQues, stepSizeTypeList, 0)
    print("stepSizeTypeChoice:", stepSizeTypeChoice)
    if errorCheck(error, stepSizeTypeChoice):
        return -1

    momentumTypeChoice = getUserInput(momentumTypeQues, momentumTypeList, 0)
    print("momentumTypeChoice:", momentumTypeChoice)
    if errorCheck(error, momentumTypeChoice):
        return -1

    momentumParamChoice = -1
    if (momentumTypeChoice == 2 or momentumTypeChoice == 3 or momentumTypeChoice == 4 or momentumTypeChoice == 6):
        momentumParamChoice = getUserInput(momentumParamQues, momentumParamList, 0)
        print("momentumParamChoice:", momentumParamChoice)
        if errorCheck(error, momentumParamChoice):
            return -1

    # Used for Adam and Nadam
    momentumParam2Choice = -1
    if (momentumTypeChoice == 3 or momentumTypeChoice == 4):
        momentumParam2Choice = getUserInput(momentumParam2Ques, momentumParam2List, 0)
        print("momentumParam2Choice:", momentumParam2Choice)
        if errorCheck(error, momentumParam2Choice):
            return -1

    runChoices.runChoices(numberofRunsChoice, maxIterationsChoice, stochasticTypeChoice,
                          sizeofLinearSystemChoice,randomInputsTypeChoice, stepSizeTypeChoice,
                          momentumTypeChoice, momentumParamChoice,momentumParam2Choice)


if __name__ == "__main__":
    # When this file is run, the getRunInputs() function is called
    getRunInputs()
