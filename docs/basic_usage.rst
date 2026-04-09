Basic usage
===========

<<<<<<< HEAD
=======
The examples below use the Yoruba (YRI) population from the 1000 Genomes
Project (n = 108), which is one of the populations with higher sample size and
nucleotide diversity. All timing estimates are based on a 24-core AMD Ryzen 9
PRO 5945 workstation with 64 GB RAM and an NVIDIA RTX 3080.

The full workflow — simulations, feature vector estimation for all 22 YRI
autosomes, training, and genome-wide prediction — completes in a few hours on
this configuration.

>>>>>>> ed421eb (pushing to 2.0. dann, recombination stratification normalization, custom stats, center/windows, outlier scan, partial cms, plotting)

Command line
------------

<<<<<<< HEAD
The entire workflow takes ~3 hours, including the feature vector estimation of the 22 YRI autosomes, on a workstation with 24 threads and 64GB of RAM. Most of this time (~80%) is dedicated to estimating autosome feature vectors. On average, the new version can process each human automose in ~6 minutes in such a configuration.

.. code-block:: bash

    # ~23mins
    flexsweep simulator --sample_size 108 --demes yri_spiedel_2019.yaml --output_folder yri_test  --nthreads 24 --num_simulation 12500

.. code-block:: bash

    # ~18mins
    flexsweep fvs-discoal --simulations_path yri_test  --nthreads 24


.. code-block:: bash

    # ~160min
    flexsweep fvs-vcf --vcf_path yri_vcfs --recombination_map decode_sexavg_2019.txt --nthreads 24 --pop yri


.. code-block:: bash

    # ~2min
    flexsweep cnn  --train_data yri_test/fvs.parquet --predict_data yri_vcfs/fvs_yri.parquet --output_folder yri_test

=======
.. code-block:: bash

    # Simulate 22,000 neutral + 22,000 sweep replicates (~70 min)
    flexsweep simulator \
        --sample_size 216 \
        --demes yri_spiedel_2019.yaml \
        --output_folder yri_test \
        --nthreads 24 \
        --num_simulations 220000

.. code-block:: bash

    # Estimate feature vectors from simulations (~50 min)
    flexsweep fvs-discoal \
        --simulations_path yri_test \
        --nthreads 24

.. code-block:: bash

    # Estimate feature vectors from VCF (~27 min, 22 autosomes)
    flexsweep fvs-vcf \
        --vcf_path yri_vcfs \
        --recombination_map decode_sexavg_2019.txt \
        --nthreads 24 \
        --suffix yri

.. code-block:: bash

    # Train and predict (~4 min on RTX 3080)
    flexsweep cnn \
        --train_data yri_test/fvs.parquet \
        --predict_data yri_vcfs/fvs_yri.parquet \
        --output_folder yri_test
>>>>>>> ed421eb (pushing to 2.0. dann, recombination stratification normalization, custom stats, center/windows, outlier scan, partial cms, plotting)


Python interface
----------------

.. code-block:: python

    import flexsweep as fs

<<<<<<< HEAD
    simulator = fs.Simulator(216, fs.DEMES_EXAMPLES['yri'], 'yri_test', num_simulations = int(2e4), nthreads = 24)
=======
    simulator = fs.Simulator(
        216, fs.DEMES_EXAMPLES['yri'], 'yri_test',
        num_simulations=int(2.5e5), nthreads=24
    )
>>>>>>> ed421eb (pushing to 2.0. dann, recombination stratification normalization, custom stats, center/windows, outlier scan, partial cms, plotting)

    # Prior parameters to simulate
    df_params = simulator.create_params()

    # Simulate
    sims_list = simulator.simulate()

<<<<<<< HEAD
    # Estimate fvs to train the CNN
    fvs_sims = fs.summary_statistics(data_dir = "yri_test", nthreads = 24)

    # Estimate fvs to predict
    fvs_vcf = fs.summary_statistics(data_dir = "yri_vcfs", vcf = True, nthreads = 24, recombination_map = fs.DECODE_MAP, population = 'yri')
=======
    # Estimate feature vectors from simulations
    fvs_sims = fs.summary_statistics(data_dir="yri_test", nthreads=24)

    # Estimate feature vectors from VCF
    fvs_vcf = fs.summary_statistics(
        data_dir="yri_vcfs",
        vcf=True,
        nthreads=24,
        recombination_map=fs.DECODE_MAP,
        suffix='yri',
    )
>>>>>>> ed421eb (pushing to 2.0. dann, recombination stratification normalization, custom stats, center/windows, outlier scan, partial cms, plotting)

    # Train and predict
    fs_cnn = fs.CNN(
        train_data="yri_test/fvs.parquet",
        predict_data="yri_vcfs/fvs_yri.parquet",
        output_folder="yri_vcfs",
    )
    fs_cnn.train()
    fs_cnn.predict()

<<<<<<< HEAD
The package includes several default values, such as YRI, CEU and CHB population sizes estimated from `Relate <https://www.nature.com/articles/s41588-019-0484-x>`_ as well as `decode recombination map <https://doi.org/10.1126/science.aau1043>`_. You can easily access it by
=======
The training incorporates early stopping to prevent overfitting and converges
in about 40 epochs for YRI-sized datasets.

The package includes built-in defaults for YRI, CEU, and CHB demographic
models estimated from `Relate <https://www.nature.com/articles/s41588-019-0484-x>`_,
as well as the `deCODE recombination map <https://doi.org/10.1126/science.aau1043>`_.
You can inspect them directly:
>>>>>>> ed421eb (pushing to 2.0. dann, recombination stratification normalization, custom stats, center/windows, outlier scan, partial cms, plotting)

.. code-block:: python

    import flexsweep as fs

<<<<<<< HEAD
    print(fs.simulate_discoal.demes.load(fs.DEMES_EXAMPLES['yri']))

    print(fs.pl.read_csv(
                fs.DECODE_MAP,
                separator="\t",
                comment_prefix="#",
                schema=fs.pl.Schema(
                    [
                        ("chr", fs.pl.String),
                        ("start", fs.pl.Int64),
                        ("end", fs.pl.Int64),
                        ("cm_mb", fs.pl.Float64),
                        ("cm", fs.pl.Float64),
                    ]
                )
            )
        )
=======
    # View the YRI demographic model
    print(fs.simulate_discoal.demes.load(fs.DEMES_EXAMPLES['yri']))

    # View the deCODE recombination map
    print(fs.pl.read_csv(
        fs.DECODE_MAP,
        separator="\t",
        comment_prefix="#",
        schema=fs.pl.Schema([
            ("chr", fs.pl.String),
            ("start", fs.pl.Int64),
            ("end", fs.pl.Int64),
            ("cm_mb", fs.pl.Float64),
            ("cm", fs.pl.Float64),
        ])
    ))
>>>>>>> ed421eb (pushing to 2.0. dann, recombination stratification normalization, custom stats, center/windows, outlier scan, partial cms, plotting)
