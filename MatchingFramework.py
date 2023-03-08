"""
@author: Harshil Bhatia
"""
import numpy as np
from scipy.optimize import linear_sum_assignment
import scipy.io
import ShapeMatching
import pickle
import time
import config 
import utils
import random
from numba import njit,jit
import random

def getFuncTime(func):
    def wrapper(*args, **kwargs):
        before = time.perf_counter()
        value = func(*args, **kwargs)
        after = time.perf_counter()
        fname = func.__name__
        print(f'{fname}: {after - before} s')
        return value

    return wrapper

# @getFuncTime
def calc_worst_match(cnt,C_XY,C_YZ,C_XZ,Xgeodesics,Ygeodesics,Zgeodesics):
    
    finalWorstVertices = np.zeros((2,config.args['nrWorst']),dtype = np.int64)

    if cnt == 0 :
        
        worst_matches_XY_1 = utils.evaluateCorrespondencesVertices(C_XY, Xgeodesics, Ygeodesics)
        worst_matches_YZ_1 = C_XY[worst_matches_XY_1,1]

        finalWorstVertices[0] = worst_matches_XY_1
        finalWorstVertices[1] = worst_matches_YZ_1
    
    elif cnt == 1:
        worst_matches_YZ_2 = utils.evaluateCorrespondencesVertices(C_YZ, Ygeodesics, Zgeodesics)
        worst_matches_XY_2 = np.zeros(len(worst_matches_YZ_2))

        dict = {}

        for elem in C_XY:
            for vert in worst_matches_YZ_2:
                if(elem[1] == vert):
                    dict[vert] = elem[0]

        for i in range(len(worst_matches_YZ_2)):
            worst_matches_XY_2[i] = dict[worst_matches_YZ_2[i]]
        
        finalWorstVertices[0] = worst_matches_XY_2
        finalWorstVertices[1] = worst_matches_YZ_2
        
    elif cnt == 2:
        
        worst_matches_XZ_3 = utils.evaluateCorrespondencesVertices(C_XZ, Xgeodesics, Zgeodesics)
        worst_matches_YZ_3 = np.copy(C_XY[worst_matches_XZ_3,1])
        
        finalWorstVertices[0] = worst_matches_XZ_3
        finalWorstVertices[1] = worst_matches_YZ_3

    return finalWorstVertices

@jit(nopython = True,cache = True)
def cij(ci,cj):
    num_vertices = ci.shape[0]
    _cj = np.copy(ci[:,1])                            
    _cj[np.array([k for k in cj[:,1]],dtype = np.int64)] = np.arange(0,num_vertices,1,dtype = np.int64)
    
    C_ij = np.copy(ci)
    C_ij[:,1] = np.array([_cj[int(k)] for k in ci[:,1]],dtype = np.int64)

    return C_ij

# @getFuncTime
@jit(nopython = True,cache = True)
def getEnergy(C,master,geodesics,num_shapes):
    itr = 0
    energy = np.zeros(int(num_shapes*(num_shapes-1)/2))
    
    num_vertices = C[0].shape[0]
    for i in range(num_shapes):
        for j in range(i+1,num_shapes):
            if i != master:
                if j!= master:
                    C_ij = cij(C[i],C[j])                    
                    p = utils.evaluateCorrespondencesScore(C_ij,geodesics[i],geodesics[j])

                else:
                    p = utils.evaluateCorrespondencesScore(C[i],geodesics[i],geodesics[j])
                    
            else:
                C_ij = np.copy( C[(master+1)%num_shapes])
                C_ij[np.array([k for k in C[j][:,1]],dtype = np.int64) , 1] = np.arange(0,num_vertices,1,dtype = np.int64)
                p = utils.evaluateCorrespondencesScore(C_ij,geodesics[i],geodesics[j])

            energy[itr] = p
            itr += 1

    return energy


class MatchingFramework():
    descriptors = []
    geodesics = [] 

    def __init__(self) -> None:
        pass
    
    def loadDescriptors(self,Xdescriptors,Ydescriptors):
        softC = Xdescriptors.dot(Ydescriptors.transpose())
        rows, cols = linear_sum_assignment(-softC)
        return np.array([rows,cols]).transpose()

    def getMasterNode(self):
        
        W = np.zeros((config.args['num_shapes'],config.args['num_shapes']))
        for i in range(config.args['num_shapes']):
            for j in range(i+1,config.args['num_shapes']):
                C = self.loadDescriptors(self.descriptors[i],self.descriptors[j])
                W[i,j] = utils.evaluateCorrespondencesScore(C,self.geodesics[i],self.geodesics[j])
                
        val = np.zeros(len(self.descriptors))

        for i in range(config.args['num_shapes']):
            for j in range(config.args['num_shapes']):
                val[i] += W[i,j] + W[j,i]

        ids = np.argsort(val)
        return ids[0]

    def perform_matching(self,C,geodesics,master,offset):
        
        nodes = np.arange(config.args['num_shapes'],dtype = np.int64)
        nodes = np.concatenate((nodes[:master],nodes[master+1:]))

        samples = np.copy(nodes)
        np.random.shuffle(samples)
        
        shapes =[samples[-1],samples[0]]
        samplesItr = 1

        shapes = np.array(shapes,dtype = np.int64)
        C = np.array(C,dtype = np.int64)
        num_vertices = C[0].shape[0]

        Identity_Mapping = np.arange(num_vertices,dtype = np.int64)
        if config.args['computeEnergy']: energy = getEnergy(C,master,self.geodesics,config.args['num_shapes'])

        if offset == 0:
            steps = int((config.args['num_shapes']-1)*config.args['steps'] / 9)
        else:
            steps = int(config.args['num_shapes']-1)

        for _steps in range(offset, offset + steps):
            
            if samplesItr == len(samples):
                random.shuffle(samples)
                shapes =[samples[-1],samples[0]]
                samplesItr = 1

            s = time.perf_counter()

            multishape = ShapeMatching.ShapeMatching(geodesics[shapes[0]],geodesics[master],geodesics[shapes[1]])
            
            for cnt in range(3):
                C_XY = np.copy(C[shapes[0]])
                C_YZ = np.copy(C[shapes[1]])

                C_YZ[C[shapes[1]][:,1] , 1] = Identity_Mapping

                C_XZ = np.copy(C_YZ)
                C_XZ[:, 1] = np.array([C_YZ[i,1] for i in C_XY[:,1]])

                d1 = calc_worst_match(cnt,C[shapes[0]],C_YZ,C_XZ,geodesics[shapes[0]],geodesics[master],geodesics[shapes[1]])
                worst_matches_XY,worst_matches_YZ = d1[0],d1[1]
                new_perm_1,new_perm_2,info = multishape.optimize(C[shapes[0]],C_YZ,worst_matches_XY,worst_matches_YZ,self.descriptors[shapes[0]],self.descriptors[master],self.descriptors[shapes[1]],_steps)

                C[shapes[0]][:, 1] = new_perm_1
                C[shapes[1]][new_perm_2 , 1] = Identity_Mapping

            shapes = np.array(shapes,dtype = np.int64)
            t2 = time.perf_counter()

            if config.args['computeEnergy']:
                energy = getEnergy(C,master,self.geodesics,config.args['num_shapes'])
                print(np.sum(energy),t2-s)
                pickle.dump([C,master,shapes,energy,t2-s],open(config.args['saveDir'] + "ITR/" + "Ite" + str(_steps) + ".p",'wb'))
            
            else:
                pickle.dump([C,master,shapes,t2-s],open(config.args['saveDir'] + "ITR/" + "Ite" + str(_steps) + ".p",'wb'))

            shapes[0] = shapes[1]
            shapes[1] = samples[samplesItr]

            samplesItr += 1
            offset = _steps

        return C,offset + 1

    def initC(self):
        if config.args['usedescriptors']:
            master = self.getMasterNode()
            C = np.zeros((config.args['num_shapes'],config.args['vertices'],2),dtype =np.int32)
            
            for i in range(config.args['num_shapes']):
                if i != master: 
                    C[i] = self.loadDescriptors(self.descriptors[i],self.descriptors[master])


        if config.args['loadcorr']:
            x = pickle.load(open(config.args['loadcorrpath'],'rb'))
            master = x[1]
            C = x[0]

        
        return C,master

    def computeVariance(self,max_geo):
        start = 0.25*max_geo
        end = 0.05*max_geo
    
        n = int(config.args['steps']/9)

        d = np.arange(n)

        c = np.log(end/start)/((1/n-1))
        k = start/(np.exp(c))
        var_x = np.vectorize(lambda t: k*np.exp((1/(t+1))*c)) 
        variance = var_x(d)
        return variance
    
    
    def match(self,shape_list = None):
        
        geodesics = []
        descriptors = []

        if config.args['dataset'] == 'FAUST':
            print("Shapes Used:",shape_list)
            for i in range(len(shape_list)):
                if config.args['noise']:
                    geo,desc = utils.load_mesh_noise(shape_list[i],config.args['noise_variance'])

                else:
                    geo,desc = utils.load_mesh(shape_list[i])

                geodesics.append(geo)
                descriptors.append(desc)

        if config.args['dataset'] == 'TOSCA':

            if config.args['interclass']:
                geodesics,descriptors,vert = self.interclassTOSCA()
                
            else:
                geodesics , descriptors = utils.load_mesh_TOSCA(config.args['classvalue'])
                
       
        if config.args['dataset'] == 'SMAL':

            self.num_shapes = config.args['num_shapes']
            for i in range(config.args['num_shapes']):
                geo,desc = utils.load_mesh_SMAL(i)
                geodesics.append(geo)
                descriptors.append(desc)
        
        self.geodesics = np.array(geodesics,dtype = 'float64')
        self.descriptors = np.array(descriptors,dtype = 'float64')

        C,master = self.initC()        

        offset = 0
        C,offset = self.perform_matching(C,geodesics,master,offset)
        
        max_geo = np.max(geodesics)
        variance = self.computeVariance(max_geo)        
        print(variance)
        for var in variance:
            print(var)
            gaussians = []
            for geo in geodesics:
                gaussians.append(utils.gaussian_geodesics(geo,var))

            gaussians = np.array(gaussians,dtype = 'float64')
            self.geodesics = gaussians

            C,offset = self.perform_matching(C,gaussians,master,offset)
