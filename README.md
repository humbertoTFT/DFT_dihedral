# FFT dihedral fitting

This repository contains the data associated with the manuscript:

"Fast Fourier Transform Enables Automated Parametrization of Complex Dihedral Potentials in All-Atom and Coarse-Grained Force Fields"
submitted to Journal of Chemical Information and Modeling.

## Contents
Main script for AA dihedral FFT-fit
- dihefit_fft.py
  
Molecules files with standard Gromacs format can be found by their respective name in Data folder (in AA or CG section).  
- Molecules structures (.gro)
- Non-fitted topologies for molecules as example (.itp) 
- Worked final topologies with FFT-fitted dihedrals (.itp)
- Parameters  file for AA dihedral fitting (.info)

## Requirements
- Gaussian 16
- Gromacs 2021.x
- Pyhon 3.x:
  - Numpy
  - Matplotlib

## Notes
This repository is provided for academic, non-commercial use.

