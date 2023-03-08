"""
@author: Harshil Bhatia
"""
import config
import os
import scipy.io
import numpy as np
from numba import jit

@jit(nopython = True)
def gaussian(x, variance):
    return 1 / (np.sqrt(2*np.pi)*variance) * np.exp(- x**2 / (variance**2))

def load_mesh_noise(problem_id,var):
               
    if config.args['noise'] == 1:
        S = scipy.io.loadmat(f'/workspace/storage/Dataset/tr_reg_0{problem_id}_var_{var}.mat')

    geodesics = S['geodesics']
    descriptors = S['descriptors']
    return geodesics,descriptors


def load_mesh(problem_id):
    
    if config.args['LRLabels'] == 1:
        S = scipy.io.loadmat(f'FAUSTLR/tr_reg_lr_0{problem_id}.mat')
    else:
        S = scipy.io.loadmat(f'/workspace/data/FaustDS/tr_reg_0{problem_id}.mat')
        
    geodesics = S['geodesics']
    descriptors = S['hks']
    return geodesics, descriptors

def load_mesh_TOSCA(classvalue):

    if config.args['LRLabels'] == 1:
        S = scipy.io.loadmat(f'/workspace/data/LRTOSCA/LR_{classvalue}.mat')
    else:
        S = scipy.io.loadmat(f'/workspace/data/TOSCA/{classvalue}.mat')

    config.args['num_shapes'] = S['N'][0][0]
    config.args['vertices'] = S['id0']['geodesics'][0][0].shape[0]

    geodesics = []
    descriptors = []
    for i in range(config.args['num_shapes']):
        geodesics.append(S[f'id{i}']['geodesics'][0][0])
        descriptors.append(S[f'id{i}']['descriptors'][0][0])

    return geodesics, descriptors

def load_mesh_SMAL(itr):

    _cls = config.args['classvalue']
    S = scipy.io.loadmat(f'/workspace/storage/SMALDataset/{_cls}_{itr}.mat')['x']
    geodesics = S['geodesics'][0][0]

    descriptors = S['descriptors'][0][0]
    config.args['vertices'] = geodesics.shape[0]
    
    return geodesics, descriptors


@jit(nopython = True)
def gaussian_geodesics(geodesicsX,variance):
    mx = np.max(geodesicsX)
    gg = np.copy(geodesicsX)
    for i in range(len(geodesicsX)):
        for j in range(len(geodesicsX[i])):
            if gg[i,j] == mx:
                gg[i,j] = 0
            gg[i, j] = gaussian(gg[i, j],variance)
    return gg
    
@jit(nopython = True)
def evaluateCorrespondencesScore(C, Xgeodesics, Ygeodesics ):
    score = np.zeros(C.shape[0])
    for i in range(C.shape[0]):
        for j in range(i+1, C.shape[0]):
            score[i] += np.abs(Xgeodesics[C[i, 0], C[j, 0]] -
                            Ygeodesics[C[i, 1], C[j, 1]])
            score[j] += np.abs(Xgeodesics[C[i, 0], C[j, 0]] -
                            Ygeodesics[C[i, 1], C[j, 1]])
    return np.sum(score)

@jit(nopython = True)
def evaluateCorrespondencesVertices(C, Xgeodesics, Ygeodesics, n = config.args['nrWorst']):
    score = np.zeros(C.shape[0])
    for i in range(C.shape[0]):
        for j in range(i+1, C.shape[0]):
            score[i] += np.abs(Xgeodesics[C[i, 0], C[j, 0]] -
                            Ygeodesics[C[i, 1], C[j, 1]])
            score[j] += np.abs(Xgeodesics[C[i, 0], C[j, 0]] -
                            Ygeodesics[C[i, 1], C[j, 1]])
    ids = np.argsort(score)
    ids = ids[-n:]
    return ids

@jit(nopython = True)
def geodesic_error(C,geodesics):
    err = 0
    for i,elem in enumerate(C[:,1]):
        err += geodesics[elem,i]
    mx = np.max(geodesics)
    err /= mx
    return err

def CreatePaths(path):
    if not os.path.exists(path):
        os.makedirs(path)
    
def getBins(mx):
    num_bins = 100
    bins = np.linspace(0,mx,num_bins)
    return bins

def getPCK(geoErr,bins):
    num_bins = len(bins)
    boxes = [0]*num_bins
    for elem in geoErr:
        for j in range(num_bins -1):
            if elem < bins[j+1] and elem >= bins[j]:
                boxes[j] += 1
    
    for j in range(1,num_bins):
        boxes[j] += boxes[j-1]
    boxes /= np.max(boxes)
    return 100*np.array(boxes)
