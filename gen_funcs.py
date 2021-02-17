# -*- coding: utf-8 -*-
"""
Created on Tue Feb 16 11:16:00 2021

Author: Erik Toller
"""

import numpy as np

def rand_gen(n,xy_ax):
    # Generate random x values between the give axis and sort them
    x = np.random.uniform(xy_ax[0],xy_ax[1]*1.1,[n,1])
    x = np.sort(x, axis=0)
    # generate random y values between the givven axis
    y = np.random.uniform(xy_ax[2],xy_ax[3],[n,1])
    z = x + y*1j
    return z
    
def length(z1, z2):
    # Get the length of each fracture
    L = np.sqrt((z1 - z2)*np.conj(z1 - z2))
    return np.real(L)

def distance(z):
    # get the distance between all points in a nxn martix where n=length(z)
    dist = np.linalg.norm(z - z[:,None], axis=-1)
    return dist

def collect_frac(z, num, xy_ax):
    # Calculate the distances between all points
    dist = distance(z)
    z1 = np.zeros(num) + np.zeros(num)*1j
    z2 = np.zeros(num) + np.zeros(num)*1j
    for pos in range(num):
        z1[pos] += z[pos]
        pos_min = np.where(dist[pos] == np.min(np.delete(dist[pos], np.arange(pos+1))))
        z2[pos] += z[pos_min]
    
    # Check if any x-positions are outside the boundary and if so move them to the boundary
    if any(z1.real > xy_ax[1]) == True:
        z1[z1.real > xy_ax[1]] = z1[z1.real > xy_ax[1]].imag*1j + xy_ax[1]
    
    return z1,z2