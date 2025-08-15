.. flexsweep documentation master file, created by
   sphinx-quickstart on Fri Aug 15 13:09:21 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Flexsweep documentation
=======================

# Flexsweep v2.0

(In development, not recommended for end users).

The second version of [Flexsweep software](https://doi.org/10.1093/molbev/msad139), a versatile tool for detecting selective sweeps. The software trains a convolutional neural network (CNN) to classify genomic loci as sweep or neutral regions. The workflow begins with simulating data under an appropriate demographic model and classify regions as neutral or sweeps, including several selection events regarding sweep strength, age, starting allele frequency (softness), and ending allele frequency (completeness).

The new version simplifies and streamlines the project structure, files, simulations, summary statistics estimation and allows for the easy addition of custom CNN architectures. The software takes advantage of [demes](https://doi.org/10.1093/genetics/iyac131) to simulate custom demography histories and main [scikit-allel](https://scikit-allel.readthedocs.io/) data structures to avoid external software and temporal files. The whole pipeline is parallelized using [joblib](https://joblib.readthedocs.io/en/stable/). 



.. toctree::
   :maxdepth: 2
   :caption: Contents:

