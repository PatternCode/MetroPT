import numpy as np
from functions import MulVarGauss
ter = MulVarGauss([1,1],[[1,0],[0,1]],20)
t = np.array([2,3,4])
print(ter)