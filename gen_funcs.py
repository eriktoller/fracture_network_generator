# -*- coding: utf-8 -*-
"""
Created on Tue Feb 16 11:16:00 2021

Author: Erik Toller
"""

import numpy as np

def rand_gen(n,xy_ax):
    x = np.random.uniform(xy_ax[0],xy_ax[1]*1.1,[n,1])
    x = np.sort(x, axis=0)
    y = np.random.uniform(xy_ax[2],xy_ax[3],[n,1])
    z = x + y*1j
    return z
    
def length(z1, z2):
    """
    Calculates the length between two complex points.

    """
    L = np.sqrt( z1+z2*np.conj(z1+z2) )
    return np.real(L)

def distance(z):
    dist = np.linalg.norm(z - z[:,None], axis=-1)
    return dist

