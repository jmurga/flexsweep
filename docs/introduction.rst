Introduction
============

`Flex-sweep <https://doi.org/10.1093/molbev/msad139>`_ is a versatile tool for detecting selective sweeps. It was primarily designed to be flexible in terms of the types, strengths, and ages of sweeps, being particularly useful for scanning the genomes of non-model species. It only requires a single species, being robust to mis-polarisation, requiring only good genome assemblies and phased haplotypes.

This new version simplifies and streamlines the project structure, files, simulations, and summary statistics estimation. Flex-sweep is now able to easily handle the maximum previous simulations set (20,000 training and 2,000 testing for neutral and sweep scenarios) and predict a 1000KG human population in a few hours, moving from HPC to workstation configuration.

Similar to the first version, Flex-sweep works in three main steps: simulation, summary statistics estimation (feature vectors), and training/classification. Once installed, you can access the Command Line Interface to run any module as needed.


Minimum requirements
--------------------
Flex-sweep is now able to run into a workstation, moving from HPC and high storage necessities. The minimum requirements will depend mainly on the species' genome and the number of samples. We tested Flex-sweep on 1000GP data. We run YRI population (n = 108) simulations, summary estimations from simulations, summary statistics from autosomes, training and prediction using the following workstation configuration in a few hours.

* Pop!_OS 22.04
* AMD Ryzen 9 PRO 5945 workstation (24 cores)
* 1TB M.2 NVMe SSD
* 64GB RAM
* NVIDIA GeForce RTX 3080

The summary statistics estimation heavily relies on numpy arrays, so sample size and RAM consumption are the limiting factors of our software.

New features
------------
Now simulations take advantages of `demes <https://doi.org/10.1093/genetics/iyac131>`_ to simulate custom demography histories and main `scikit-allel <https://scikit-allel.readthedocs.io/>`_ data structures to avoid external software and temporal files.

The software now works with VCF data, so it is not needed to convert VCF to hap/map files. When using recombination maps, the software will interpolate genetic positions and recombination rates automatically. We included the polarisation script depending only on the ancestral fasta files and VCF.

The software is now able to estimate summary statistics in custom genomic center ranges as well as window sizes. Features vectors are now estimated in custom regions.

We refactor the entire package, focusing on speed and traceability. The code was refactored and takes advantage of `Numba <https://numba.pydata.org/>`_ as much as possible for all the previous statistics except for nSL and iHS, which rely on scikit-allel functions. All the summary statistic outputs and feature vectors rely on `Polars DataFrames <https://pola.rs/>`_ to avoid the previous huge number of intermediate files and easily inspect outputs while reducing RAM consumption as much as possible.

The new version included optimised versions of `iSAFE <https://doi.org/10.1038/nmeth.4606>`_, `DIND <https://doi.org/10.1371/journal.pgen.1000562>`_, hapDAF-o/s, Sratio, highfreq, lowfreq as well as the custom HAF and H12 as described in `Flex-sweep manuscript <https://doi.org/10.1093/molbev/msad139>`_.

Because we intend to make Flex-sweep as flexible as possible, we included several other summary statistics (note that any other statistic already available in scikit-allel can be used straightforwardly):

* :math:`\Delta\text{-}iHH`: `https://doi.org/10.1126/science.1183863 <https://doi.org/10.1126/science.1183863>`_
* Garud's :math:`H1`, :math:`H12`, :math:`H2/H1`: `https://doi.org/10.1371/journal.pgen.1005004 <https://doi.org/10.1371/journal.pgen.1005004>`_ (numba refactored)
* :math:`\pi`: `https://scikit-allel.readthedocs.io/en/stable/stats/diversity.html#allel.mean_pairwise_difference <https://scikit-allel.readthedocs.io/en/stable/stats/diversity.html#allel.mean_pairwise_difference>`_ (numba refactored)
* :math:`\theta_{W}`: `https://scikit-allel.readthedocs.io/en/stable/stats/diversity.html#allel.watterson_theta <https://scikit-allel.readthedocs.io/en/stable/stats/diversity.html#allel.watterson_theta>`_ (numba refactored)
* Kelly's :math:`Z_{nS}`: `https://doi.org/10.1093/genetics/146.3.1197 <https://doi.org/10.1093/genetics/146.3.1197>`_
* :math:`\omega_{max}`: `https://doi.org/10.1534/genetics.103.025387 <https://doi.org/10.1534/genetics.103.025387>`_
* Fay & Wu's :math:`H` (and :math:`\theta_{H}`): `https://doi.org/10.1534/genetics.106.061432 <https://doi.org/10.1534/genetics.106.061432>`_
* Zeng's :math:`E`: `https://doi.org/10.1534/genetics.106.061432 <https://doi.org/10.1534/genetics.106.061432>`_
* Achaz's :math:`Y`: `https://doi.org/10.1534/genetics.106.061432 <https://doi.org/10.1534/genetics.106.061432>`_
* Fu & Li's :math:`D` and :math:`F`: `https://doi.org/10.1093/genetics/133.3.693 <https://doi.org/10.1093/genetics/133.3.693>`_
* LASSI :math:`T` and :math:`m`: `https://doi.org/10.1093/molbev/msaa115 <https://doi.org/10.1093/molbev/msaa115>`_
* RAiSD :math:`\mu`: `https://doi.org/10.1038/s42003-018-0085-8 <https://doi.org/10.1038/s42003-018-0085-8>`_

Although our current configuration is designed to detect positive selection, we included two other statistics designed to detect balancing selection, which can be used along with other statistics to detect recent balancing selection through CNN (see `BaSE <https://doi.org/10.1111/1755-0998.13379>`_ for further balancing selection analysis using Neural Networks).

* :math:`\beta^{(1)*}_{(std)}`: `https://doi.org/10.1093/gbe/evaa013 <https://doi.org/10.1093/gbe/evaa013>`_
* :math:`NCD1`: `https://doi.org/10.1093/gbe/evy054 <https://doi.org/10.1093/gbe/evy054>`_

The new API allows easy addition of custom CNN architectures, so the user can input custom CNN while training/predicting. Now the CNN class is able to preprocess feature vectors to work not only with the default 2D CNN but also 1D CNN.

To extend for custom configuration, we also included the best rearrangement algorithms from `Zhao, H. and Alachiotis, N. 2025 <https://doi.org/10.1016/j.ymeth.2024.11.003>`_ to work with raw haplotype matrices too. The software included:

* Correlation coefficient sorting
* Derived Allele Frequency sorting
* Occurrence Frequency sorting
* Sub-regions bipartite correlation

We included a rank algorithm to post-process sweep probabilities and ranking for any type of associated genomic element. The function relies on `Pybedtools <https://daler.github.io/pybedtools/>`_ to quickly estimate genomic distances between the genomic region used by Flex-sweep and the genomic element listed in a BED file.

.. A saliency map class to explore which genomic region and statistic are more revelant during training.

Flex-sweep is now able to work with demography mis-specification! We extend our CNN with the Domain-Adaptive model proposed by `Mo, Z. and Siepel A. 2023 <https://doi.org/10.1371/journal.pgen.1011032>`_.
