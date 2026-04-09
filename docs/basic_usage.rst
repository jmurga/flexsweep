Basic usage
===========

The examples below use the Yoruba (YRI) population from the 1000 Genomes
Project (n = 108), which is one of the populations with higher sample size and
nucleotide diversity. All timing estimates are based on a 24-core AMD Ryzen 9
PRO 5945 workstation with 64 GB RAM and an NVIDIA RTX 3080.

The full workflow — simulations, feature vector estimation for all 22 YRI
autosomes, training, and genome-wide prediction — completes in a ~2 hours on
this configuration.

Command line
------------

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


Python interface
----------------

.. code-block:: python

    import flexsweep as fs

    simulator = fs.Simulator(
        216, fs.DEMES_EXAMPLES['yri'], 'yri_test',
        num_simulations=int(2.5e5), nthreads=24
    )

    # Prior parameters to simulate
    df_params = simulator.create_params()

    # Simulate
    sims_list = simulator.simulate()

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

    # Train and predict
    fs_cnn = fs.CNN(
        train_data="yri_test/fvs.parquet",
        predict_data="yri_vcfs/fvs_yri.parquet",
        output_folder="yri_vcfs",
    )
    fs_cnn.train()
    fs_cnn.predict()

The training incorporates early stopping to prevent overfitting and converges
in about 40 epochs for YRI-sized datasets.

The package includes built-in defaults for YRI, CEU, and CHB demographic
models estimated from `Relate <https://www.nature.com/articles/s41588-019-0484-x>`_,
as well as the `deCODE recombination map <https://doi.org/10.1126/science.aau1043>`_.
You can inspect them directly:

.. code-block:: python

    import flexsweep as fs

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
