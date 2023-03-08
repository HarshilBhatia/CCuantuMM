"""
@author: Harshil Bhatia
"""

import numpy as np
from numba import jit, prange


mult = np.array([-1,1,1,-1])
N = 0

@jit(nopython = True,cache = True)
def _p2p1_(perm1,perm2):
    N = perm1.shape[0]
    temp_perm1_perm2 = np.zeros(N,dtype = np.int64)
    for i in range(N):
        temp_perm1_perm2[i] = perm2[perm1[i]]

    return temp_perm1_perm2

@jit(nopython = True,cache = True)
def p1cj(cycle,perm1,perm2):
    c1,c2 = cycle[0],cycle[1]
    # p2c1,p2c2 = perm1.index(c1),perm1.index(c2)
    p2c1,p2c2 = np.where(perm1 == c1)[0][0],np.where(perm1 == c2)[0][0]
    idc1,idc2 = perm2[c1],perm2[c2]
    coeff = np.array([[p2c1,idc1],[p2c1,idc2],[p2c2,idc1],[p2c2,idc2]])
    return coeff

@jit(nopython = True,cache = True)
def cip2(cycle,perm1_perm2):
    c1,c2 = cycle[0],cycle[1]
    idc1,idc2 = perm1_perm2[c1],perm1_perm2[c2]

    coeff = np.array([[c1,idc1],[c1,idc2],[c2,idc1],[c2,idc2]])
    return coeff

@jit(nopython = True,cache = True)
def _c1c2_(cycle1,cycle2,perm1,perm2):
    c11,c12 = cycle2[0],cycle2[1]

    id11,id12 = perm2[c11],perm2[c12]

    c21,c22 = cycle1[0],cycle1[1]
    id21,id22 = perm1[c21],perm1[c22]

    coeff = np.array([[c21,id11],[c21,id12],[c22,id11],[c22,id12]])

    if( (id21 == c11 or id21 == c12 ) and (id22 == c11 or id22 == c12)):
        l = np.array([-2,2,2,-2])
    elif(id21 == c11):
        l = np.array([1,-1,-1,1])
        # l = [-1,1,1,-1]
    elif(id22 == c11):
        # l = [1,-1,-1,1]
        l = np.array([-1,1,1,-1])
    elif(id21 == c12):
        l = np.array([-1,1,1,-1])
    elif(id22 == c12):
        l = np.array([1,-1,-1,1])
    else:
        l = np.array([0,0,0,0])

    return coeff,l

@jit(nopython = True,cache = True)
def _p2p1xci_(cycle,perm1,perm2,perm1_perm2,geodesicsX,geodesicsZ):
    c = cip2(cycle,perm1_perm2)
    value = 0 
    for i,id in enumerate(perm1_perm2):
        for j,[c_elem1,c_elem2] in enumerate(c):
            value += mult[j]*np.abs(geodesicsX[c_elem1,i] - geodesicsZ[c_elem2,id])
            value += mult[j]*np.abs(geodesicsX[i,c_elem1] - geodesicsZ[id,c_elem2])
    return value


@jit(nopython = True,cache = True)
def _p2p1xcj_(cycle,perm1,perm2,perm1_perm2,geodesicsX,geodesicsZ):
    coeffs = p1cj(cycle,perm1,perm2)
    value = 0 
    for i,id in enumerate(perm1_perm2):
        for j,[c_elem1,c_elem2] in enumerate(coeffs):
            value += mult[j]*np.abs(geodesicsX[c_elem1,i] - geodesicsZ[c_elem2,id])
            value += mult[j]*np.abs(geodesicsX[i,c_elem1] - geodesicsZ[id,c_elem2])
    return value

@jit(nopython = True,cache = True)   
def p1p2xcicj(cycle1,cycle2,perm1,perm2,perm1_perm2,geodesicsX,geodesicsZ):
    coeffs,_mult = _c1c2_(cycle1,cycle2,perm1,perm2)
    value = 0 
    if(_mult[0] == 0): return value

    for i,id in enumerate(perm1_perm2):
        for j,[c_elem1,c_elem2] in enumerate(coeffs):
            value += _mult[j]*np.abs(geodesicsX[c_elem1,i] - geodesicsZ[c_elem2,id])
            value += _mult[j]*np.abs(geodesicsX[i,c_elem1] - geodesicsZ[id,c_elem2])
    return value

@jit(nopython = True,cache = True)
def _cixcj_(cycle1,cycle2,perm1,perm2,perm1_perm2,geodesicsX,geodesicsZ):
    coeffs1 = cip2(cycle1,perm1_perm2)
    coeffs2 = p1cj(cycle2,perm1,perm2)
    value = 0
    for i,[c_elem1,c_elem2] in enumerate(coeffs1):
        for j,[b_elem1,b_elem2] in enumerate(coeffs2):
            value += mult[i]*mult[j]*np.abs(geodesicsX[c_elem1,b_elem1] - geodesicsZ[c_elem2,b_elem2])
    return value

@jit(nopython = True,cache = True)
def _cjxci_(cycle1,cycle2,perm1,perm2,perm1_perm2,geodesicsX,geodesicsZ):
    coeffs2 = cip2(cycle1,perm1_perm2)
    coeffs1 = p1cj(cycle2,perm1,perm2)
    value = 0
    for i,[c_elem1,c_elem2] in enumerate(coeffs1):
        for j,[b_elem1,b_elem2] in enumerate(coeffs2):
            value += mult[i]*mult[j]*np.abs(geodesicsX[c_elem1,b_elem1] - geodesicsZ[c_elem2,b_elem2])
    return value

@jit(nopython = True,cache = True)
def _cixci_(cycle1,cycle2,perm1,perm2,perm1_perm2,geodesicsX,geodesicsZ):
    coeffs1 = cip2(cycle1,perm1_perm2)
    coeffs2 = cip2(cycle2,perm1_perm2)
    value = 0
    for i,[c_elem1,c_elem2] in enumerate(coeffs1):
        for j,[b_elem1,b_elem2] in enumerate(coeffs2):
            value += mult[i]*mult[j]*np.abs(geodesicsX[c_elem1,b_elem1] - geodesicsZ[c_elem2,b_elem2])
    return value

@jit(nopython = True,cache = True)
def _cjxcj_(cycle1,cycle2,perm1,perm2,perm1_perm2,geodesicsX,geodesicsZ):
    coeffs1 = p1cj(cycle1,perm1,perm2)
    coeffs2 = p1cj(cycle2,perm1,perm2)
    value = 0
    for i,[c_elem1,c_elem2] in enumerate(coeffs1):
        for j,[b_elem1,b_elem2] in enumerate(coeffs2):
            value += mult[i]*mult[j]*np.abs(geodesicsX[c_elem1,b_elem1] - geodesicsZ[c_elem2,b_elem2])
    return value

@jit(nopython = True,cache = True)
def cicjxci(cycle1,cycle2,cycle3,perm1,perm2,perm1_perm2,geodesicsX,geodesicsZ):
    coeffs1,_mult = _c1c2_(cycle1,cycle2,perm1,perm2)
    coeffs2 = cip2(cycle3,perm1_perm2)
    value = 0
    if(_mult[0] == 0): return value
    for i,[c_elem1,c_elem2] in enumerate(coeffs1):
        for j,[b_elem1,b_elem2] in enumerate(coeffs2):
            value += _mult[i]*mult[j]*np.abs(geodesicsX[c_elem1,b_elem1] - geodesicsZ[c_elem2,b_elem2])
            value += _mult[i]*mult[j]*np.abs(geodesicsX[b_elem1,c_elem1] - geodesicsZ[b_elem2,c_elem2])
    return value


@jit(nopython = True,cache = True)
def cicjxcj(cycle1,cycle2,cycle3,perm1,perm2,perm1_perm2,geodesicsX,geodesicsZ):
    coeffs1,_mult = _c1c2_(cycle2,cycle1,perm1,perm2)
    coeffs2 = p1cj(cycle3,perm1,perm2)
    value = 0
    if(_mult[0] == 0): return value
    for i,[c_elem1,c_elem2] in enumerate(coeffs1):
        for j,[b_elem1,b_elem2] in enumerate(coeffs2):
            value += _mult[i]*mult[j]*np.abs(geodesicsX[c_elem1,b_elem1] - geodesicsZ[c_elem2,b_elem2])
            value += _mult[i]*mult[j]*np.abs(geodesicsX[b_elem1,c_elem1] - geodesicsZ[b_elem2,c_elem2])
    return value

@jit(nopython = True,cache = True)
def cicjxcicj(cycle1,cycle2,cycle3,cycle4,perm1,perm2,perm1_perm2,geodesicsX,geodesicsZ):
    coeffs1,_mult1 = _c1c2_(cycle1,cycle2,perm1,perm2)
    coeffs2,_mult2 = _c1c2_(cycle3,cycle4,perm1,perm2)

    value = 0   
    if(_mult1[0] == 0 or _mult2[0] == 0): return value

    for i,[c_elem1,c_elem2] in enumerate(coeffs1):
        for j,[b_elem1,b_elem2] in enumerate(coeffs2):
            value += _mult1[i]*_mult2[j]*np.abs(geodesicsX[c_elem1,b_elem1] - geodesicsZ[c_elem2,b_elem2])
    return value

@jit(nopython = True)
#, parallel = True)
def getQUBO(perm1,perm2,cycle_list1,cycle_list2,geodesicsX,geodesicsZ):
    dim1 = cycle_list1.shape[0]
    dim2 = cycle_list2.shape[0]

    N = perm1.shape[0]

    perm1_perm2 = _p2p1_(perm1,perm2)

    bias_alpha,bias_beta,alpha_beta,C_alpha,C_beta = np.zeros(dim1),np.zeros(dim2),np.zeros((dim1,dim2)),np.zeros((dim1,dim1)),np.zeros((dim2,dim2))


    for i1 in range(dim1):
        cycle1 = cycle_list1[i1]
        bias_alpha[i1]+= _p2p1xci_(cycle1,perm1,perm2,perm1_perm2,geodesicsX,geodesicsZ) 
        
        for j1 in range(dim1):
            cycle2 = cycle_list1[j1]
            C_alpha[i1,j1] += _cixci_(cycle1,cycle2,perm1,perm2,perm1_perm2,geodesicsX,geodesicsZ)

        for j2 in range(dim2):
            cycle2 = cycle_list2[j2]
            alpha_beta[i1,j2] += _cjxci_(cycle1,cycle2,perm1,perm2,perm1_perm2,geodesicsX,geodesicsZ) + _cixcj_(cycle1,cycle2,perm1,perm2,perm1_perm2,geodesicsX,geodesicsZ) + p1p2xcicj(cycle1,cycle2,perm1,perm2,perm1_perm2,geodesicsX,geodesicsZ)
            alpha_beta[i1,j2] += cicjxci(cycle1,cycle2,cycle1,perm1,perm2,perm1_perm2,geodesicsX,geodesicsZ) + cicjxcj(cycle2,cycle1,cycle2,perm1,perm2,perm1_perm2,geodesicsX,geodesicsZ)
            alpha_beta[i1,j2] += cicjxcicj(cycle1,cycle2,cycle1,cycle2,perm1,perm2,perm1_perm2,geodesicsX,geodesicsZ)

    for i1 in range(dim2):
        cycle1 = cycle_list2[i1]
        bias_beta[i1] += _p2p1xcj_(cycle1,perm1,perm2,perm1_perm2,geodesicsX,geodesicsZ)
        for j1 in range(dim2):
            cycle2 = cycle_list2[j1]
            C_beta[i1,j1] += _cjxcj_(cycle1,cycle2,perm1,perm2,perm1_perm2,geodesicsX,geodesicsZ)

    return bias_alpha,bias_beta,C_alpha,C_beta,alpha_beta


