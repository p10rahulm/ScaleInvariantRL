import numpy as np

# Here we test whether summation in the momentum convergence proof sums to 0 in Appendix G
def infiSumBetaPowerKminusiByKminusi(k=10,beta=0.5):
    total = 0
    for i in range(k):
        total  +=beta**(i)/(k-i)
    return total

if __name__ == "__main__":
    np.random.seed(8)
    for k in range(0,100000001,100000000):
        total = infiSumBetaPowerKminusiByKminusi(k,0.99999999)
        print("(k,total)=(%d,%.7f)"%(k,total))