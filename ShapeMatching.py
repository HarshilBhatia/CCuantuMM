"""
@author: Harshil Bhatia
"""

import numpy as np
import dimod
import neal
from utils import *
import numpy as np
import time 
import config
from numba import jit
import TwoShapes
import ThreeShapes as ThreeShapes
from dwave.system import DWaveSampler, EmbeddingComposite,VirtualGraphComposite, FixedEmbeddingComposite

def getFuncTime(func):
    def wrapper(*args, **kwargs):
        before = time.perf_counter()
        value = func(*args, **kwargs)
        after = time.perf_counter()
        fname = func.__name__
        print(f'{fname}: {after - before} s')
        return value

    return wrapper

@jit(nopython = True,cache = True)
def getNextPairs(firstPairs):
    # exit()
    nextPairs= np.zeros(firstPairs.shape,dtype = np.int64)
    nextPairs[0,0],nextPairs[0,1] = firstPairs[0][0],firstPairs[1][1]
    itr = 1 
    if len(firstPairs)>2:
        nextPairs[itr,0],nextPairs[itr,1] = firstPairs[0][1], firstPairs[2][1]  
        itr += 1
        for k in range(2,len(firstPairs)-1):
            nextPairs[itr,0], nextPairs[itr,1] = firstPairs[k-1][0], firstPairs[k+1][1]
            itr += 1

        nextPairs[itr,0],nextPairs[itr,1] = firstPairs[firstPairs.shape[0]-2][0], firstPairs[firstPairs.shape[0]-1][0]
    else:
        nextPairs[itr,0],nextPairs[itr,1] = firstPairs[0][1],firstPairs[1][0] 
    return nextPairs

    
def cos(A, B): 
    return (A*B).sum(axis=1) / (A*A).sum(axis=1) ** .5 / (B*B).sum(axis=1) ** .5

def csm(A,B):
    num=np.dot(A,B.T)
    p1=np.sqrt(np.sum(A**2,axis=1))[:,np.newaxis]
    p2=np.sqrt(np.sum(B**2,axis=1))[np.newaxis,:]
    return num/(p1*p2)

@jit(nopython = True,cache = True)
def getCycleLists(worst_matches):

    N = len(worst_matches)
    elem_to_add = np.zeros((N//2,2),dtype = np.int64)

    for i in range(N//2):
        elem_to_add[i,0] = worst_matches[2*i]
        elem_to_add[i,1] = worst_matches[2*i+1]

    elem_to_add =  np.random.permutation( elem_to_add)
    
    cycle_lists = np.zeros((N-1,elem_to_add.shape[0],elem_to_add.shape[1]),dtype = np.int64)
    itr = 0

    cycle_lists[itr,:,:] = elem_to_add
    currentPairs=getNextPairs(elem_to_add)

    for _ in range(N-2):
        itr += 1
        cycle_lists[itr] = currentPairs
        currentPairs=getNextPairs(currentPairs)

    return cycle_lists


@jit(nopython = True,cache = True)
def updatePermutationTable(perm,new_perm, upd_decision ):
    result = np.copy(perm)

    for count, cycle in enumerate(new_perm):
        if upd_decision[count]==1:
            for iteration,elem in enumerate(cycle):
                result[elem]= perm[cycle[(iteration+1)% len(cycle)]]
        
    return result
    
class ShapeMatching():
    # corr = []
    geodesicsX = []
    geodesicsZ = []
    curr_perm_XY = []
    curr_perm_YZ = []
        
    solve = 'QUBO'

    pre_decided_linear_ids_alpha = []
    pre_decided_linear_ids_beta = []
    annealer_ids_alpha = []
    annealer_ids_beta = []
    decision = [] 
    dim_alpha = 0
    dim_beta = 0
    
    def __init__(self,geodesicsX=None,geodesicsY=None,geodesicsZ=None) -> None:

        self.geodesicsX = geodesicsX
        self.geodesicsY = geodesicsY
        self.geodesicsZ = geodesicsZ
        pass

    
    def preprocessBiasAndCouplings(self,R1,R2,R3):
        [We_1,ce_1] = R1
        [We_2,ce_2] = R2
        b_alpha, b_beta, C_alpha, C_beta, alpha_beta = R3
        self.dim_alpha = int(b_alpha.shape[0])
        self.dim_beta = int(b_beta.shape[0])

        self.bias_alpha,self.bias_beta = b_alpha,b_beta
        self.coupling_alpha,self.coupling_beta, self.alpha_beta = np.zeros((self.dim_alpha,self.dim_alpha)),np.zeros((self.dim_beta,self.dim_beta)),np.zeros((self.dim_alpha,self.dim_beta))

        for i in range(self.dim_alpha):
            self.bias_alpha[i] = b_alpha[i] + C_alpha[i, i] + We_1[i,i] + ce_1[i]

        for j in range(self.dim_beta):
            self.bias_beta[j] = b_beta[j] + C_beta[j, j] + We_2[j,j] + ce_2[j]

        for i in range(self.dim_alpha):
            for j in range(i + 1, self.dim_alpha):
                self.coupling_alpha[i,j] = C_alpha[i, j] +  We_1[i,j] 
                self.coupling_alpha[j,i] = C_alpha[j, i] + We_1[j,i]

        for i in range(self.dim_beta):
            for j in range(i + 1, self.dim_beta):
                self.coupling_beta[i,j] = C_beta[i, j] +  We_2[i,j]  
                self.coupling_beta[j,i] = C_beta[j, i] + We_2[j,i]
        
        for i in range(self.dim_alpha):
            for j in range(self.dim_beta):
                self.alpha_beta[i,j] = alpha_beta[i,j] 

        self.g_bias_alpha, self.g_bias_beta, self.g_coupling_alpha, self.g_coupling_beta, self.g_alpha_beta = self.bias_alpha, self.bias_beta, self.coupling_alpha, self.coupling_beta, self.alpha_beta

    
    def subsampleBiasAndCouplings(self):

        self.bias_alpha = self.bias_alpha[self.annealer_ids_alpha]
        self.bias_beta = self.bias_beta[self.annealer_ids_beta]

        self.coupling_alpha = self.coupling_alpha[self.annealer_ids_alpha,:]
        self.coupling_alpha = self.coupling_alpha[:,self.annealer_ids_alpha]
        
        self.coupling_beta = self.coupling_beta[self.annealer_ids_beta,:]
        self.coupling_beta = self.coupling_beta[:,self.annealer_ids_beta]

        self.alpha_beta = self.alpha_beta[self.annealer_ids_alpha,:]
        self.alpha_beta = self.alpha_beta[:,self.annealer_ids_beta]

    
    def dropVars(self):

        for i in range(self.dim_alpha):
            val = 0 
            for j in range(self.dim_alpha):
                val += np.abs(self.coupling_alpha[i,j]) + np.abs(self.alpha_beta[i,j]) 

            if( 2*np.abs(val) < self.bias_alpha[i]): 
                self.decision[i] = -min(self.bias_alpha[i]/np.abs(self.bias_alpha[i]),0)
                self.pre_decided_linear_ids_alpha.append(i)
        
        for i in range(self.dim_beta):
            val = 0 
            for j in range(self.dim_beta):
                val +=  np.abs(self.coupling_beta[i,j])+ np.abs(self.alpha_beta[i,j])

            if( 2 * np.abs(val) < np.abs(self.bias_beta[i])): 
                self.decision[i+self.dim_alpha] = -min(self.bias_beta[i]/np.abs(self.bias_beta[i]),0)
                self.pre_decided_linear_ids_beta.append(i)
    
    def linearAndQUBOindexes(self,linear_ids,dim):
        
        temp_flag = [0]*dim
        for i in linear_ids:
            temp_flag[i] = 1
        annealer_ids = []
        for i,elem in enumerate(temp_flag):
            if(elem == 0):
                annealer_ids.append(i)
        return annealer_ids
    

    def processbiasandcoupling(self,R1,R2,R3):

        self.preprocessBiasAndCouplings(R1,R2,R3)
        if config.args['dropvars']: self.dropVars()
        self.annealer_ids_alpha = self.linearAndQUBOindexes(self.pre_decided_linear_ids_alpha,self.dim_alpha)
        self.annealer_ids_beta = self.linearAndQUBOindexes(self.pre_decided_linear_ids_beta,self.dim_beta)
        self.subsampleBiasAndCouplings()


    def createPolynomial(self):
                
        poly_l = dict()
        poly_q = dict()

        dim_alpha, dim_beta = len(self.bias_alpha),len(self.bias_beta)

        for i in range(dim_alpha):
            poly_l[("alpha[{}]".format(i))] = self.bias_alpha[i]

        for j in range(dim_beta):
            poly_l[("beta[{}]".format(j))] = self.bias_beta[j] 
            
        for i in range(dim_alpha):
            for j in range(dim_beta):
                poly_q[("alpha[{}]".format(i), "beta[{}]".format(j))] = self.alpha_beta[i, j] 

        for i in range(dim_alpha):
            for j in range(i + 1, dim_alpha):
                poly_q[("alpha[{}]".format(i), "alpha[{}]".format(j))] = self.coupling_alpha[i, j] + self.coupling_alpha[j, i]

        for i in range(dim_beta):
            for j in range(i + 1, dim_beta):
                poly_q[("beta[{}]".format(i), "beta[{}]".format(j))] = self.coupling_beta[i, j] + self.coupling_beta[j, i] 

        return poly_l,poly_q
    
    def constructSolution(self,solution):
        
        poly = dict()
        for i in range(self.dim_alpha):
            poly["alpha[{}]".format(i)] = solution[i]

        for i in range(self.dim_beta):
            poly["beta[{}]".format(i)] = solution[self.dim_alpha + i]

        return poly

    
    def RunAnneal(self,poly_l,poly_q):

        quad = dimod.BinaryQuadraticModel(poly_l,poly_q, dimod.BINARY)
        
        
        if config.args['qpu']:

            chain = 1.0001*max(np.max(np.abs(list(poly_l.values()))),np.max(np.abs(list(poly_q.values()))))

            t = time.perf_counter()
            sampleset = self.qpusolver.sample(quad,auto_scale=True ,chain_strength=chain,num_reads=config.args['num_reads'],return_embedding = True)
            print(time.perf_counter() - t, end = ' ')
            Result = [sampleset.first.sample,sampleset.first.energy]

            sampleset_serialize = sampleset.to_serializable()

        else:
            solver = neal.SimulatedAnnealingSampler()
            response = solver.sample(quad, num_reads=config.args['num_reads'], num_sweeps=config.args['num_sweeps'])
            Result = [response.first.sample, response.first.energy]
            sampleset_serialize = response.to_serializable()

        return [Result,sampleset_serialize] 
        

    def linearSolve(self,b_alpha,b_beta):
        change_linear_alpha = [1 if elem < 0 else 0 for elem in b_alpha]
        change_linear_beta =[1 if elem < 0 else 0 for elem in b_beta]
        return change_linear_alpha,change_linear_beta

    def getDecision(self,R1,R2,R3):
        
        self.processbiasandcoupling(R1,R2,R3)

        if(self.solve == 'linear'):
            Result = self.linearSolve(self.g_bias_alpha,self.g_bias_beta)
            return Result
        else:
            poly_l,poly_q = self.createPolynomial()
            Result = self.RunAnneal(poly_l,poly_q)
            return Result

    def optimize(self,C_XY,C_YZ,worst_matches_XY,worst_matches_YZ,descX,descY,descZ,iteration_number):

        self.curr_perm_XY = np.array(C_XY[:,1],dtype = np.int64)
        self.curr_perm_YZ = np.array(C_YZ[:,1],dtype = np.int64)
        self.curr_perm_XZ = np.array([self.curr_perm_YZ[i] for i in self.curr_perm_XY])

        if config.args['qpu']: self.qpusolver = EmbeddingComposite(DWaveSampler())

        self.iteration_number = iteration_number
        
        cycle_lists_XY = getCycleLists(worst_matches_XY)
        cycle_lists_YZ = getCycleLists(worst_matches_YZ)

        for itr_cycle1 in range(len(cycle_lists_XY)):
            itr_cycle2 = itr_cycle1

            self.decision = -10 * np.ones(len(cycle_lists_XY[itr_cycle1])*2)

            self.pre_decided_linear_ids_alpha = []
            self.pre_decided_linear_ids_beta = []
            self.annealer_ids_alpha = []
            self.annealer_ids_beta = []
            
            R1 = TwoShapes.getQUBO(self.curr_perm_XY,cycle_lists_XY[itr_cycle1],self.geodesicsX,self.geodesicsY)
            R2 = TwoShapes.getQUBO(self.curr_perm_YZ,cycle_lists_YZ[itr_cycle2],self.geodesicsY,self.geodesicsZ)
            R3 = ThreeShapes.getQUBO(self.curr_perm_XY,self.curr_perm_YZ,cycle_lists_XY[itr_cycle1],cycle_lists_YZ[itr_cycle2],self.geodesicsX,self.geodesicsZ)

            dim = len(cycle_lists_XY[itr_cycle1])
            [Result,info] = self.getDecision(R1,R2,R3)
            
                
            for i in range(len(self.annealer_ids_alpha)):
                self.decision[self.annealer_ids_alpha[i]]= Result[0]['alpha[{}]'.format(i)]

            for i in range(len(self.annealer_ids_beta)):
                self.decision[self.annealer_ids_beta[i]+dim] = Result[0]['beta[{}]'.format(i)]

            
            self.curr_perm_XY =  updatePermutationTable(self.curr_perm_XY,cycle_lists_XY[itr_cycle1],self.decision[:dim])
            self.curr_perm_YZ =  updatePermutationTable(self.curr_perm_YZ,cycle_lists_YZ[itr_cycle2],self.decision[dim:2*dim])


        return self.curr_perm_XY,self.curr_perm_YZ,info
 


 
