import numpy as np
from numba import jit, f8,i8,b1,void
import argparse

#Metropolis
@jit(nopython=True)
def metro_heis(S,exps,L):
    N=L^2
    rand_l = np.array(np.random.rand(N))
    rand_config = rand_l.reshape(L,L)
    for i in range(L):
        for j in range(L):
            h = (S[i,j]*(S[(i + 1)%L,j] + S[i,(j + 1)%L] + S[(i - 1 + L)%L,j] + S[i,(j - 1 + L )%L]) + 4)//2
            if rand_l[i,j] < exps[h, (S[i,j]+1)//2]:
                S[i,j] *= -1
    return S
        
    

