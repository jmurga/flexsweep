# Flexsweep v2.0

(In development, not recommended for end users).

The second version of [Flexsweep software](https://doi.org/10.1093/molbev/msad139), a versatile tool for detecting selective sweeps. The software trains a convolutional neural network (CNN) to classify genomic loci as sweep or neutral regions. The workflow begins with simulating data under an appropriate demographic model and classify regions as neutral or sweeps, including several selection events regarding sweep strength, age, starting allele frequency (softness), and ending allele frequency (completeness).

The new version simplifies and streamlines the project structure, files, simulations, summary statistics estimation and allows for the easy addition of custom CNN architectures. The software takes advantage of [demes](https://doi.org/10.1093/genetics/iyac131) to simulate custom demography histories and main [scikit-allel](https://scikit-allel.readthedocs.io/) data structures to avoid external software and temporal files. The whole pipeline is parallelized using [joblib](https://joblib.readthedocs.io/en/stable/). We included optimized versions of [iSAFE](https://doi.org/10.1038/nmeth.4606), [DIND](https://doi.org/10.1371/journal.pgen.1000562), hapdaf, S ratio, freqs as well as the custom HAF and H12 as described in [Flexsweep manuscript](https://doi.org/10.1093/molbev/msad139). The software now is also able to run the following statistics:

- $\Delta$-IHH: [https://doi.org/10.1126/science.1183863](https://doi.org/10.1126/science.1183863)
- $\pi$: [https://scikit-allel.readthedocs.io/en/stable/stats/diversity.html#allel.mean_pairwise_difference](https://scikit-allel.readthedocs.io/en/stable/stats/diversity.html#allel.mean_pairwise_difference)
- $\theta_{W}$: [https://scikit-allel.readthedocs.io/en/stable/stats/diversity.html#allel.watterson_theta](https://scikit-allel.readthedocs.io/en/stable/stats/diversity.html#allel.watterson_theta)
- Kelly's $Z_{nS}$: [https://doi.org/10.1093/genetics/146.3.1197](https://doi.org/10.1093/genetics/146.3.1197)
- $\omega_{max}$: [https://doi.org/10.1534/genetics.103.025387](https://doi.org/10.1534/genetics.103.025387)
- Fay & Wu's H: [https://doi.org/10.1534/genetics.106.061432](https://doi.org/10.1534/genetics.106.061432)
- Zeng's E: [https://doi.org/10.1534/genetics.106.061432](https://doi.org/10.1534/genetics.106.061432)
- Fu & Li's D and F: [https://doi.org/10.1093/genetics/133.3.693](https://doi.org/10.1093/genetics/133.3.693)
- LASSI $T$ and $m$: [https://doi.org/10.1093/molbev/msaa115](https://doi.org/10.1093/molbev/msaa115)

Similarly to the first version, Flexsweep works in three main steps: simulation, summary statistics estimation (feature vectors), and training/classification. Once installed, you can access the Command Line Interface to run any module as needed.

`data` folder includes a static-compiled version of [discoal](https://doi.org/10.1093/bioinformatics/btw556), which reduces the virtual memory needed (tested on CentOS, Ubuntu and PopOS!). Such binary is automatically accesed if no other `discoal` binary is provided. It also includes multiple `demes` demography models, including the YRI population history estimated by [Speidel et al. 2019](https://doi.org/10.1038/s41588-019-0484-x).

## Installation
```bash
pip install flexsweep
```
~conda install -c bioconda flexsweep~

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
  --time TEXT                Start/end adaptive mutation range timing
  --discoal_path TEXT        Path to the discoal executable
  --num_simulations INTEGER  Number of neutral and sweep simulations
  --nthreads INTEGER         Number of threads for parallelization
  --help                     Show this message and exit.
```

Simulating $10^5$ neutral and sweep scenarios using human mutation rate estimation from [Smith et al. 2019](https://doi.org/10.1371/journal.pgen.1007254)

```bash
flexsweep simulator --sample_size 100 --mutation_rate 5e-9,2e-8 --recombination_rate 1e-8 --locus_length 1200000 --demes data/constant.yaml --output_folder training_eq --num_simulations 100000 --nthreads 100
```

### Feature vectors from simulations
The command will output a parquet file containing the feature vectors input to train Flexsweep CNN as well as the neutral expected values to normalize prediction from empirical data.

```bash
flexsweep fvs-discoal --help
```

```
Usage: flexsweep fvs-discoal [OPTIONS]

  Run the summary statistic estimation from discoal simulation to create CNN
  input feature vectors. Will create two file: a parquet dataframe and a
  pickle dictionary containing neutral expectation and stdev

Options:
  --simulations_path TEXT  Path containing neutral and sweeps discoal
                           simulations.  [required]
  --nthreads INTEGER       Number of threads  [required]
  --help                   Show this message and exit.
```

```bash
flexsweep fvs-discoal --simulation_path training_eq --nthreads 100
```

### Feature vectors from VCF
The command parse a VCF file by sliding window creating each corresponding HaplotypeArray. To parallel reading the software create *a priori* genomic positions ranges so the VCF `contig_name` and `contig_length` must be inputted to avoid reading the entire VCF.

Note that `fvs-discoal` must be run before `fvs-vcf` to properly create the `neutral_bin` data. It outputs a parquet file containing the feature vectors input to train Flexsweep CNN normalized by neutral expectation.

```bash
flexsweep fvs-vcf --help
```

```
Usage: flexsweep fvs-vcf [OPTIONS]

  Run the summary statistic estimation from a VCF file to create CNN input
  feature vectors. Feature vector file will be written within

Options:
  --vcf_path TEXT           VCF file to parse. Must be indexed  [required]
  --neutral_bin TEXT        Neutral bin data from discoal simulations
                            [required]
  --nthreads INTEGER        Number of threads  [required]
  --recombination_map TEXT  Recombination map. Decode CSV format:
                            Chr,Begin,End,cMperMb,cM
  --help                    Show this message and exit.
```

### Training/prediction
Train Flexsweep CNN with the normalized feature vectors to classify neutral and sweep regions.

```bash
flexsweep cnn --help
```

`--mode train` will output a CNN model for later classification in the selected `output_folder`, ROC curve and training history plots. Note that `--mode train` must be executed before `--mode predict` because it will search the CNN model prior to predict execution.

```
Usage: flexsweep cnn [OPTIONS]

  Run the Flexsweep CNN

Options:
  --mode [train|predict]  Mode: 'train' or 'predict'  [required]
  --data TEXT             Path to the training data  [required]
  --output_folder TEXT    Output folder for the CNN model and logs  [required]
  --model TEXT            Input a pretrained model
  --help                  Show this message and exit.
```
