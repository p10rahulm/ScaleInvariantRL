import matplotlib.pyplot as plt

# This is a helper file that helps us to draw our charts using matplotlib.
# There are some default colors that we use that can be changed

def SaveFigure(figure, location="Outputs/imSaves/", fileName="image.png"):
    if location[-1] != '/':
        location += '/'
    if fileName[-4:] != ".png":
        fileName += ".png"
    locationString = str(location) + str(fileName)
    figure.savefig(locationString, edgecolor=figure.get_edgecolor())


def drawSavePlot(maxIters, errors, title, label="Plot of Errors", color="orangered",
                 logErrors=True, plotShow=False, location="Outputs/imSaves/", fileName="image.png"):
    maxIters = int(maxIters)
    fig = plt.figure(linewidth=10, edgecolor="#04253a")
    plt.plot(range(maxIters), errors, color= color,label= label)
    plt.legend()
    fig.suptitle(title, fontsize=12, wrap=True)
    axes = plt.gca()
    axes.set_xlim([-1, maxIters])
    if logErrors:
        axes.set_yscale('log')
    if plotShow:
        fig.show()
    SaveFigure(fig, location, fileName)
    plt.close(fig)


def drawSavePlotwithHorizontalLine(maxIters, errors, title, label="Plot of Errors", color="orangered",
                 logErrors=True, plotShow=False, location="Outputs/imSaves/", fileName="image.png",
                                   lineVal=0,lineLabel="Comparison Line"):
    maxIters = int(maxIters)
    fig = plt.figure(linewidth=10, edgecolor="#04253a")
    plt.plot(range(maxIters), errors, color= color,label= label)
    plt.axhline(y=lineVal,color="orangered",label=lineLabel, linewidth=0.5,linestyle='--')
    plt.legend()
    fig.suptitle(title, fontsize=12, wrap=True)
    axes = plt.gca()
    axes.set_xlim([-1, maxIters])
    if logErrors:
        axes.set_yscale('log')
    if plotShow:
        fig.show()
    SaveFigure(fig, location, fileName)
    plt.close(fig)

def DrawSingleError(maxIters, errors, title, label="Plot of Errors", color="orangered", logErrors=True, plotShow=True):
    DrawErrors(maxIters, [errors], title, [label], [color], logErrors, plotShow)


def DrawErrors(maxIters, errors, title, labels, colors, logErrors=True, plotShow=True):
    maxIters = int(maxIters)
    for i in range(len(errors)):
        plt.plot(range(maxIters), errors[i], colors[i], label=labels[i])
    plt.legend()
    plt.title(title)
    axes = plt.gca()
    axes.set_xlim([-1, maxIters])
    if logErrors:
        axes.set_yscale('log')
    if plotShow:
        plt.show()
    return plt


def AddToPlot(plot, maxIters, error, color, label):
    plot.plot(range(maxIters), error, color, label)
    plot.show()
    return plot


def createOutputs(td0_err, totalProj_err, numStates, numFeatures, gamma, maxIters, consistent, chartFlag):
    errors = [td0_err, totalProj_err]
    final_errors = list(map(lambda x: x[-1], errors))
    least = final_errors.index(min(final_errors))
    leastErrorMethod = {0: "TD0", 1: "Total Projections"}

    if consistent:
        titleC = "Overdetermined consistent system."
    else:
        titleC = "Overdetermined inconsistent system."

    titleDetails = "numstates=%d,numFeatures=%d,gamma=%0.2f:" % (numStates, numFeatures, gamma)
    print(titleC, titleDetails,
          "td0Err=%.2e" % td0_err[-1],
          "Total Projections error=%.2e" % totalProj_err[-1],
          "Least Method: ", leastErrorMethod[least])

    if chartFlag:
        colors = ['orangered', 'mediumorchid']
        labels = ['TDO error', 'Total Projections Error']
        plotTitle = "Plot for error with iterations for " + titleC + "\n" + titleDetails
        DrawErrors(maxIters, errors, plotTitle, labels, colors, True)
