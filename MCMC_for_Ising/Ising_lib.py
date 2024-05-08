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
## Heat bath
@jit(nopython=True)
def heatbath(S,exps,L):
    N = L**2
    ran = np.random.rand(N).reshape(L,L)
    for i in range(L):
        for j in range(L):
            hm = ((S[(i + 1)%L,j] + S[i,(j + 1)%L] + S[(i - 1 + L)%L,j] + S[i,(j - 1 + L )%L]) + 4)//2
            if ran[i,j] < exps[hm, (S[i,j]+1)//2]:
                S[i,j] = 1
            else:
                S[i,j] = -1
    return S


## For Swendsen-Wang algorithm
@jit(nopython=True)
def get_cluster_num(num,cluster_num):
    if num == cluster_num[num]:
        return num
    else:
        return get_cluster_num(cluster_num[num],cluster_num)


@jit(nopython=True)
def update_cluster_num(ni,nj,cluster_num):
    ci = get_cluster_num(ni,cluster_num)
    cj = get_cluster_num(nj,cluster_num)
    if ci < cj:
        cluster_num[cj] = ci
    else:
        cluster_num[ci] = cj

@jit(nopython=True)
def make_bond(S,prob,L):
    N = L**2
    bond = np.zeros((L,L,2),dtype=np.int64)
    ran = np.random.rand(2*N).reshape(L,L,2)
    for i in range(L):
        for j in range(L):
            if S[i,j] *S[(i+1)%L, j] > 0: ## spins are parallel to horizontal(neighbour to rigth)
                if ran[i,j,0] < prob:
                    bond[i,j,0] = 1
            if S[i,j] *S[i, (j+1)%L] > 0: ## spins are parallel to vertical(rigth above)
                if ran[i,j,1] < prob:
                    bond[i,j,1] = 1
    return bond

@jit(nopython=True)
def make_cluster(bond,L):
    N = L**2
    cluster_num = np.arange(N)
    for i in range(L):
        for j in range(L):
            if bond[i,j,0] >0: ## connected to horizontal
                ni = i + j *L
                nj = (i + 1)%L + j * L
                update_cluster_num(ni,nj,cluster_num)
            if bond[i,j,1] >0: ## connected to vertical
                ni = i + j *L
                nj =  i+ (j + 1)%L * L
                update_cluster_num(ni,nj,cluster_num)
        ## count total cluster number
        
    cluster_num_count = np.zeros(N,dtype=np.int64)
    for i in range(N):
        nc = get_cluster_num(i,cluster_num)
        cluster_num[i] = nc
        cluster_num_count[nc] += 1

    total_cluster_num = 0

    true_cluster_num = np.zeros(N,dtype=np.int64)
    true_cluster_num_count = np.zeros(N,dtype=np.int64)
    for nc in range(N):
        if cluster_num_count[nc] > 0:
            true_cluster_num[nc] = total_cluster_num
            true_cluster_num_count[total_cluster_num] = cluster_num_count[nc]
            total_cluster_num += 1

    for i in range(N):
        cluster_num[i] = true_cluster_num[cluster_num[i]]
    return cluster_num.reshape(L,L), true_cluster_num_count[:total_cluster_num]

@jit(nopython=True)
def flip_spin(S, cluster_num,cluster_num_count,flip,L):    
    total_cluster_num = cluster_num_count.shape[0]
    ran = np.random.rand(total_cluster_num)
    spin_direction = np.zeros(total_cluster_num,dtype=np.int64)


    for i in range(total_cluster_num):
        if ran[i] < 1.0 / (1.0 + np.exp(-2.0 * flip * cluster_num_count[i])):
            spin_direction[i] = 1
        else:
            spin_direction[i] = -1

    for i in range(L):
        for j in range(L):
            S[i,j] = spin_direction[cluster_num[i,j]]

def Swendsen_Wang(S,prob,flip,L):
    N = L**2
    
    ## make bond configulations    
    bond = make_bond(S,prob,L)

    ## make clusters
    cluster_num, cluster_num_count = make_cluster(bond,L)
    ## update spin

    flip_spin(S,cluster_num,cluster_num_count,flip,L)
    ## for imporoved estimator
    Nc2 = np.sum(cluster_num_count.astype(float)**2)
    Nc4 = np.sum(cluster_num_count.astype(float)**4)

    return S,Nc2/N**2, (3 * Nc2**2 - 2*Nc4)/float(N)**4

    

