This repository implements a simulation of a titration, for a system of two main components,
$N​$ species and $N-1$ equilibrium constants. Some species (composed of a combination of a metal (M) and of a ligand (L), with a given or unknown UV-VIS spectrum) are contained in a flask. More of a solution of a ligand or of the metal (the titrating agent, or titrant) is added to the flask, modifying the concentrations of the species depending on the metal/ligand concentrations ratio.  The UV-VIS spectra are recorded experimentally at each step of the titration, and the routines contained in this script try to obtain a best fit of the data by finding the equilibrium constants that will describe best the concentrations of the species. If the UV-VIS spectra of a given species is not known, its values will be added to the variables to be optimized.

### Dependencies 

- Python 2.7
- Scipy
- Numpy
- Matplotlib

###Usage example

An example is given in the file example/fit_titration_data.py.
In this example, a flask contains a solution of M.  A solution of L is added to the flask, generating the species ML and ML$_2$.

The spectra of L and M are known, but not those of ML and ML$_2$.
The initial guess for ML$_2$ is given by the end result of the titration, when
L is in large excess compared to L.

The required data files are listed below: 

- a list of the ligand/metal ratio (equivalents) at each step of the titration (file example/titration_data/eqs),
- the span of the UV-VIS spectrum (not mandatory: just for plotting, file example/titration_data/span),
- the UV-VIS spectra, one column per step of titration (file example/titration_data/uvs).

For each step of the titration, the program computes the concentrations of the different species in solution given the initial guess of the equilibrium constants. Using the provided UV-VIS spectra (or guessed if the UV-VIS spectrum of a compound is not known), the spectrum is computed at each step of the titration. The program then performs a minimization step on the squared differences between the modeled spectra and the experimental spectra. The iteration is repeated until a stationary point is reached. At this point the program declares convergence and the resulting estimated spectra and equilibrium constants are given.