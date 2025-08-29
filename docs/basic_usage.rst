Basic usage
===========


Command line
------------

.. code-block:: bash

    flexsweep simulator --sample_size 40 --demes yri_spiedel_2019.yaml --output_folder yri_test  --nthreads 24


.. code-block:: bash

    flexsweep fvs-discoal --simulations_path yri_test  --nthreads 24


.. code-block:: bash

    flexsweep fvs-vcfs --vcf_path yri_vcfs --recombination_map decode_sexavg_2019.txt --nthreads 24


.. code-block:: bash

    flexsweep cnn  --train_data yri_test/fvs.parquet --predict_data yri_vcfs/fvs_yri.parquet --output_folder yri_test


Python interface
----------------

.. code-block:: python

    import flexsweep as fs

    simulator = fs.Simulator(218, fs.DEMES_EXAMPLES['yri'], 'yri_test', num_simulations = int(2e4), nthreads = 24)

    # Prior parameters to simulate
    df_params = simulator.create_params()

    # Simulate
    sims_list = simulator.simulate()

    # Estimate fvs to train the CNN
    fvs_sims = fs.summary_statistics(data_dir = "yri_test", nthreads = 24)

    # Estimate fvs to predict
    # fs.DECODE_MAP
    fvs_vcf = fs.summary_statistics(data_dir = "yri_vcf", vcf = True, nthreads = 24, recombination_map = fs.DECODE_MAP, population = 'yri')



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
