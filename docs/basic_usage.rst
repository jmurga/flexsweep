Basic usage
===========


Command line
------------

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



Python interface
----------------

.. code-block:: python

    import flexsweep as fs

    simulator = fs.Simulator(216, fs.DEMES_EXAMPLES['yri'], 'yri_test', num_simulations = int(2e4), nthreads = 24)

    # Prior parameters to simulate
    df_params = simulator.create_params()

    # Simulate
    sims_list = simulator.simulate()

    # Estimate fvs to train the CNN
    fvs_sims = fs.summary_statistics(data_dir = "yri_test", nthreads = 24)

    # Estimate fvs to predict
    fvs_vcf = fs.summary_statistics(data_dir = "yri_vcfs", vcf = True, nthreads = 24, recombination_map = fs.DECODE_MAP, population = 'yri')

    # Train and predict
    fs_cnn = fs.CNN(
        train_data="yri_test/fvs.parquet",
        predict_data="yri_vcfs/fvs_yri.parquet",
        output_folder="yri_vcfs",
    )
    fs_cnn.train()
    fs_cnn.predict()

The package includes several default values, such as YRI, CEU and CHB population sizes estimated from `Relate <https://www.nature.com/articles/s41588-019-0484-x>`_ as well as `decode recombination map <https://doi.org/10.1126/science.aau1043>`_. You can easily access it by

.. code-block:: python

    import flexsweep as fs

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
