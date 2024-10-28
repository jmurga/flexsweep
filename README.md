# Flexsweep v2.0

(In development, not recommended for end users. Need to solve RAM issues when performing large number of simulations ~200K. All the statistic are uneficienly saved on Dataframe where most values are nan, was easy to work with).

The second version of [Flexsweep software](https://doi.org/10.1093/molbev/msad139), a versatile tool for detecting selective sweeps with various ages, strengths, starting allele frequencies, and completeness. The software trains a convolutional neural network (CNN) to classify genomic loci as sweep or neutral regions. The workflow begins with simulating data under an appropriate demographic model and regions as neutral or sweeps, including several selection events regarding sweep strength, age, starting allele frequency (softness), and ending allele frequency (completeness).

The new version simplifies and streamlines the project structure, files, simulations, and summary statistics estimation and allows for the easy addition of custom CNN architectures. The software takes advantage of [demes](https://doi.org/10.1093/genetics/iyac131) to simulate custom demography histories and main scikit-alleles formats data structures to avoid external software and temporal files. We included Numba optimized versions of [iSAFE](https://doi.org/10.1038/nmeth.4606), DIND, hapdaf, S ratio and freqs statistics. The software now is also able to run the following statistics:

- $\Delta$ IHH
- $\pi$
- $\theta_{W}$
- Kelly's Zns
- $\omega_{max}$
- Fay & Wu's H
- Zeng's E
- Fu & Li's D and F
- LASSI

Similarly to the first version, Flexsweep works in three main steps: simulation, summary statistics estimation (feature vectors), and training/classification. Once installed, you can access the Command Line Interface to run any module as needed.

`data` folder includes a static-compiled version of [discoal]([https://github.com/kr-colab/discoal](https://doi.org/10.1093/bioinformatics/btw556)), which reduces the virtual memory needed (tested on CentOS, Ubuntu and PopOS!) which is automatically accesed if no other `discoal` binary is provided. It also includes multiple `demes` demography models, including the YRI population history estimated by [Speidel et al. 2019](https://doi.org/10.1038/s41588-019-0484-x).
## Installation
```bash
pip install flexsweep
```

## Tutorial
### Simulation
Running from CLI. By default it only uses 1 thread and simulate $10^4$ neutral and sweep simulation each case. Comma-separated values will draw mutation or recombination rate values from a Uniform distribution while single values will draw mutation or recombination rate values from a Exponential distribution.


```bash
flexsweep --help
```
```
Usage: flexsweep simulator [OPTIONS]

  Run the discoal Simulator

Options:
  --sample_size INTEGER      Sample size for the simulation
                             [required]
  --mutation_rate TEXT       Mutation rate. For two comma-separated
                             values, the first will be used as the
                             lower bound and the second as the upper
                             bound for a uniform distribution. A
                             single value will be treated as the mean
                             for an exponential distribution.
                             [required]
  --recombination_rate TEXT  Mutation rate. For two comma-separated
                             values, the first will be used as the
                             lower bound and the second as the upper
                             bound for a uniform distribution. A
                             single value will be treated as the mean
                             for an exponential distribution.
                             [required]
  --locus_length INTEGER     Length of the locus  [required]
  --demes TEXT               Path to the demes YAML model file
                             [required]
  --output_folder TEXT       Folder where outputs will be saved
                             [required]
  --discoal_path TEXT        Path to the discoal executable
  --num_simulations INTEGER  Number of neutral and sweep simulations
  --nthreads INTEGER         Number of threads for parallelization
  --help                     Show this message and exit.
```

Simulating $10^5$ neutral and sweep scenarios using human mutation rate estimation from [Smith et al. 2019)[https://doi.org/10.1371/journal.pgen.1007254]
```
 flexsweep simulator --sample_size 100 --mutation_rate 5e-9,2e-8 --recombination_rate 1e-8 --locus_length 1200000 --demes data/constant.yaml --output_folder training_eq --num_simulations 100000 --nthreads 100                            
```


