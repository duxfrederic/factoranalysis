# -*- coding: utf-8 -*-
import numpy as np 
import sys
sys.path.insert(0,'..')
from simulateSpectrum import Component, System, Titration

# %% 
"""
LOADING AND SAMPLING THE DATA
"""
folder = 'titration_data/'
# concentrations of the titrant (L):
eqs = np.loadtxt(folder+'eqs')

# span of the UV-VIS spectrum: 
span = np.loadtxt(folder+'span')

# values of the UV-VIS spectrum:
uvs  = np.loadtxt(folder+'uvs')

#%% 
"""
CREATING THE SPECIES, ARRANGING THEM IN A SYSTEM, INITIALIAZING THE TITRATION
"""#             keq conc_ini
M    = Component(0,    1.,  uv_known=True,  uvvis=uvs[:,0], eqconst_known=True, \
                 name='M')

L    = Component(0,    0.,  uv_known=True,  uvvis=np.zeros_like(span), eqconst_known=True,\
                 name='L',  titrant=True) # L is the titrant.

ML   = Component(5, 1.,  [M,L],  [1,1], uv_known=False,  uvvis=uvs[:,-1], eqconst_known=False)


S    = System([M,L],[ML], span=span)

T    = Titration(S,eqs,uvs)

# OPTIMIZE AND RETRIEVE THE RESULT IN A 2D ARRAY
T.optimize()
# plot and print the result
T.plotCurrentModel()
T.printCurrentModel()

