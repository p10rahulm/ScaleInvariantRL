import os.path
from numpy import loadtxt

# This is a helper file used for reading files
def readFile(inputFileName):
    if not os.path.isfile(inputFileName):
        print('File does not exist.')
    else:
        with open(inputFileName) as f:
            content = f.read().splitlines()
    return content

def loadIntoNpArray(filename,delimiter =",",commentChar="#"):
    npArray = loadtxt(filename, comments=commentChar, delimiter=delimiter)
    return npArray

if __name__=="__main__":
    inputFileName = "Outputs/imSaves/Selected/selectedNamesParsed.txt"
    npArr = loadIntoNpArray(inputFileName)
    print(npArr)
    print(npArr.shape)
