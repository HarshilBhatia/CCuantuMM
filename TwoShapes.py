"""
@author: Harshil Bhatia
"""

from numba import jit
import numpy as np

fl = np.array([-1,1,1,-1])

@jit(nopython = True, cache = True)
def _getindexes_(cycle,perm):
    c1,c2 = cycle[0], cycle[1]
    c_id1,c_id2 = perm[cycle[0]],perm[cycle[1]]
    return np.array([[c1,c_id1],[c1,c_id2],[c2,c_id1],[c2,c_id2]])

@jit(nopython = True, cache = True)
def _c1c2mult_(cycle1,cycle2,perm,geodesicsX,geodesicsZ):
    l1,l2 = _getindexes_(cycle1,perm),_getindexes_(cycle2,perm)
    value = 0
    for i,[c_elem1,c_elem2] in enumerate(l1):
        for j,[b_elem1,b_elem2] in enumerate(l2):
            value += fl[i]*fl[j]*np.abs(geodesicsX[c_elem1,b_elem1] - geodesicsZ[c_elem2,b_elem2])
    return value

@jit(nopython = True, cache = True)
def _p1c1mult_(cycle1,perm,geodesicsX,geodesicsZ):
    value = 0
    l1 = _getindexes_(cycle1,perm)
    for i,id in enumerate(perm):
        for j,[c_elem1,c_elem2] in enumerate(l1):
            value += fl[j]*np.abs(geodesicsX[c_elem1,i] - geodesicsZ[c_elem2,id])
            value += fl[j]*np.abs(geodesicsX[i,c_elem1] - geodesicsZ[id,c_elem2])
    return value


@jit(nopython = True, cache = True)
def getQUBO(perm,cyclelist,geodesicsX,geodesicsZ):

    dim = cyclelist.shape[0]
    Wn = np.zeros((dim,dim))
    for i, cycle1 in enumerate(cyclelist):
        for j,cycle2 in enumerate(cyclelist):
            Wn[i,j] = _c1c2mult_(cycle1,cycle2,perm,geodesicsX,geodesicsZ)

    Cn = np.zeros(dim)
    for i,cycle1 in enumerate(cyclelist):
        Cn[i] = _p1c1mult_(cycle1,perm,geodesicsX,geodesicsZ)

    return Wn,Cn


