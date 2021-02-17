# -*- coding: utf-8 -*-
"""
Created on Tue Feb 16 11:16:00 2021

Author: Erik Toller
"""

import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import display
import time
import csv

def input_func():
    # Create the x-axis slider
    sliderx = widgets.FloatRangeSlider(
        value=[-1, 1],
        min=-10,
        max=10,
        step=0.1,
        description='x-axis:',
        disabled=False,
        continuous_update=False,
        orientation='horizontal',
        readout=True,
        readout_format='.1f',
    )
    
    # Create the y-axis slider
    slidery = widgets.FloatRangeSlider(
        value=[-1, 1],
        min=-10,
        max=10,
        step=0.1,
        description='y-axis:',
        disabled=False,
        continuous_update=False,
        orientation='horizontal',
        readout=True,
        readout_format='.1f',
    )
    
    # Create the number of fractures slider
    slidernum = widgets.IntSlider(
        value=10,
        min=10,
        max=10000,
        step=1,
        description='Fractures:',
        disabled=False,
        continuous_update=False,
        orientation='horizontal',
        readout=True,
        readout_format='d'
    )
    
    # Creat the textbox for filename
    textbox = widgets.Text(
        value='frac_coord.csv',
        placeholder='Type filename',
        description='Filename:',
        disabled=False
    )
    return sliderx, slidery, slidernum, textbox

def rand_gen(n,xy_ax):
    # Assign 10 % more fractures to get ridd of boundary issues
    n = int(n*1.1+1)
    
    # Generate random x values between the give axis and sort them
    # the x-axis on the right is increased by 10 % to make the netwokr fill the x-axis
    x = np.random.uniform(xy_ax[0],xy_ax[1]*1.1,[n,1])
    x = np.sort(x, axis=0)
    
    # Generate random y values between the givven axis
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

def collect_frac(num, xy_ax):
    # Get random complex points z
    z = rand_gen(num, xy_ax)
    
    # Calculate the distances between all points
    dist = distance(z)
    
    # Set up the arras as 0+0j initaly
    z1 = np.zeros(num) + np.zeros(num)*1j
    z2 = np.zeros(num) + np.zeros(num)*1j
    
    # Setup progress bar
    f = widgets.IntProgress(min=0, max=num, description="Generating") # instantiate the bar
    display(f) # display the bar
    
    # Iterator to find the nearest un-used point
    for pos in range(num):
        # Assing the starting point of fracutr n as z[n]
        z1[pos] += z[pos]
        # Find the nearest neighboor, but exclude all z that has been assigned as z1
        pos_min = np.where(dist[pos] == np.min(np.delete(dist[pos], np.arange(pos+1))))
        z2[pos] += z[pos_min]
        # update the bar
        f.value += 1 # signal to increment the progress bar
    f.bar_style = "success"
    
    # Check if any x-positions are outside the boundary and if so move them to the boundary
    if any(z1.real > xy_ax[1]) == True:
        z1[z1.real > xy_ax[1]] = z1[z1.real > xy_ax[1]].imag*1j + xy_ax[1]
    
    return z1,z2

def plot_frac(z1, z2):
    # Convert the complex vectors to vectors of x and y
    X1 = [z1.real for x in z1]
    Y1 = [z1.imag for x in z1]
    X2 = [z2.real for x in z2]
    Y2 = [z2.imag for x in z2]
    
    # Change the figure size and set the axis as equal
    plt.rcParams['figure.figsize'] = [15, 15]
    plt.gca().set_aspect('equal')
    
    # Setup progress bar
    f = widgets.IntProgress(min=0, max=len(X1), description="Plotting") # instantiate the bar
    display(f) # display the bar
    
    # Plot each fracture as a line
    for pos in range(len(X1)):
        plt.plot([X1[0][pos],X2[0][pos]],[Y1[0][pos],Y2[0][pos]], color='black')
        f.value += 1 # signal to increment the progress bar
    f.bar_style = "success"
    
    # Turn off the axis and show the plot
    plt.axis('off')
    plt.show()
    
    
def plot_length(z1, z2):
    # Get the lengt hof all fractures
    L = length(z1, z2)
    
    # Plot the lengths in a histogram
    plt.rcParams['figure.figsize'] = [15, 7]
    plt.hist(L, bins=30)
    plt.xlabel('Length');
    strL = 'Number of fractures:' + str(len(L))
    plt.title(strL)
    
def save_frac(z1,z2,name):
    # Make the z1 z2 into on matrix
    a = np.array(np.array([z1,z2]).T)
    
    # Save the matrix as a csv
    np.savetxt(name, a, delimiter=",")