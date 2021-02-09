
# Scale Invariant RL

This project can be used to run the Total Projections algorithm to 
find solutions to overdetermined systems. We have provided an interface that enables
easy access to the various modules in the project.

To get started, you can start this interface by typing the following:

    python interface.py

## Requirements
The requirements needed to run the algorithms are
- `numpy` (we use version 1.20.1)
- `matplotlib` (we use version 3.3.4) 

Other versions of these should work just fine, as long as you use Python 3
## Core Pipeline

When you call `interface.py`, the pipeline uses the following files
- `Interface.py`: This runs the main interface and passes the requirements to runChoices.py
- `runChoices.py`: This parses the choices and calls oneRing.py to run the iterations
- `oneRing.py`: This runs the iterations and gets the error based on the user inputs

## Total Projections File
The core file that runs the Total Projections algorithm is the following:
- `TPCore.py`
It is called within each file where required

## Helper Files
The following are the main helper files we use

- `DrawingCharting.py`: This is used for plotting the results
- `errorCalcs.py`: This is used for computing the errors given an iterate 
and a linear system
- `generator.py`: This is used to generate various types of random inputs 
like random vectors and matrices
- `readFile.py`: This is used to read and write output to disk
- `utils.py`: This covers major utilities like norms and linear algebra 
functions that are used in other files

Short descriptions of the functions in these are given in the respective files

## Sample Files
The actual files used to generate the graphs in the paper were 
generated by the following files. The details for what is run within each file
are given within the files themselves

- `Stoch_NormalTP.py`: This runs the Total Projections in the stochastic case with 
normal step size and without momentum
- `Stoch_CurvatureStepTP.py`: This file runs Total Projections in the stochastic case 
with our curvature step
- `Stoch_CurvatureStepTP_withMomentum.py`: This file runs Total Projections in 
the stochastic case with our curvature step, 
using constant beta heavy ball momentum. We test for the beta values of 0.1 and 0.5 here

- `NStoch_NormalTP.py`: This file runs Total Projections in the non-stochastic case 
with normal step size and without momentum
- `NStoch_CurvatureStepTP.py`:  This file runs total projections in the non-stochastic
case with curvature-step size and without momentum

Further we run a few simulations using different types of constant beta values:
- `ConstMomentumBeta0to1_NS.py`:  We do a deeper dive with tests for 
Total Projections with curvature-step and heavy ball momentum,  
on a whole range of momentum beta values between 0 and 1. 
This file checks for the non-stochastic case. 
We find beta value of 0.5 to be most suitable
 
- `ConstMomentumBeta0to1.py`:  Here we test for various beta values for Total Projections
using curvature-step and heavy ball momentum in the stochastic case. 
We test for many values between 0 and 1 and find beta value of 0.5 to be most suitable.

- `ConstMomentumBeta0.1.py`: We check for Total Projections using curvature step 
in the stochastic case using heavy-ball momentum with a constant beta value of 0.1   
- `ConstMomentumBeta0.5.py`: We check for Total Projections using curvature step 
in the stochastic case using heavy-ball momentum with a constant beta value of 0.5 
       

### Different Types of Momentum:
The details for the below files are provided within the files themselves.
We use the reference: 
[Sebastian Ruder: An overview of gradient descent optimization algorithms](https://ruder.io/optimizing-gradient-descent/)

- `Adagrad.py`
- `Adam.py`
- `NesterovAdam.py`
- `RMSPropElementWise.py`
- `RMSPropWithNorm.py`

## Miscellaneous files
There are two files used to generate some parts of the appendix. These are:

- `infiniteSum.py`: Used to verify some parts of Appendix G
- `ourErrorvsLSError.py`: Used to check our error vs the least squares error 
for random systems with outliers in Appendix A