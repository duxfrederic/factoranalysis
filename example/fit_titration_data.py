# -*- coding: utf-8 -*-

# use numpy to load the content of the textfiles to arrays
import numpy as np 

# append the parent folder to the path
import sys
sys.path.insert(0,'..')

# import the needed classes
from simulateSpectrum import Component, System, Titration

# %% 
"""
LOADING AND SAMPLING THE DATA
"""
folder = 'titration_data/'
# ligand/metal ratios at the different points of the titration:
eqs = np.loadtxt(folder+'eqs')

# span of the UV-VIS spectrum: 
span = np.loadtxt(folder+'span')

# values of the UV-VIS spectrum: (contained by columns in the file)
uvs  = np.loadtxt(folder+'uvs')

#%% 
"""
CREATING THE SPECIES, ARRANGING THEM IN A SYSTEM, INITIALIAZING THE TITRATION
"""
# constructor of a component :
# Component( guess for equilibrium constant, initial concentration, {named arguments})
# (see docstring)
# if the Component is a building-block (M or L), the value of the equilibrium constant
# does not matter. (below, 0 is given as the initial guess.)
M    = Component(0,    1.,  uv_known=True,  uvvis=uvs[:,0], eqconst_known=True, \
                 name='M')

L    = Component(0,    0.,  uv_known=True,  uvvis=np.zeros_like(span), eqconst_known=True,\
                 name='L',  titrant=True) # L is the titrant.

ML   = Component(10., 1.,  buildblocks=[M,L],  coeffs=[1,1], uv_known=False,  uvvis=uvs[:,-1], 
                 eqconst_known=False)

#      System(list of building blocks, list of components, span of the uv-vis spectra)
S    = System([M,L], [ML], span=span)

#      Titration(system, M/L ratios, experimental uv-vis spectra)
T    = Titration(S, eqs, uvs)

# OPTIMIZE 
T.optimize()
# plot and print the result
T.plotCurrentModel()
T.printCurrentModel()

