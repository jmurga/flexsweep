
Installation
============

You can install flexsweep and its dependencies using Conda or Pip. Both options will install library requirements automatically.

Conda
-----
We recommend to use install conda from `conda-forge <https://conda-forge.org/download/>`_ which include `mamba <https://mamba.readthedocs.io/en/latest/index.html>`_ by default. In any other case, we recommend to install mamba to speep up conda packages installation.

.. code-block:: bash

    mamba install -c bioconda flexsweep

Pip
---
Ensure you have python>=3.12 and pip installed and then simply run

.. code-block:: bash

    pip install flexsweep

This will install stdpopsim into your local user Python packages (on some systems you will need to use python3 rather than python). Please see the Python package installation tutorial for more details on the various installation options. In particular, we recommend using a virtual environment to improve reproducibility.

It may also be necessary to update your PATH to run the command line interface. See here for details on what this means and how to do it.


Source
------

    uv build
    uv pip install dist/flexsweep-2.0.tar.gz


Testing the Installation
------------------------

You can easily test flexsweep by running the CLI in your terminal

.. code-block:: bash

    flexsweep --help

.. code-block:: console

    Usage: flexsweep [OPTIONS] COMMAND [ARGS]...

      CLI for Simulator and CNN.

    Options:
      --help  Show this message and exit.

    Commands:
      cnn                 Run the Flex-sweep CNN for training or prediction.
      dann                Run the Flex-sweep DANN for training or prediction.
      enrichment          Gene-set sweep enrichment and FDR analysis.
      fvs-discoal         Estimate summary statistics from discoal...
      fvs-vcf             Estimate summary statistics from VCF files and...
      polarize            Polarize VCF using rust est-sfs refactor.
      rank                Rank genomic features by their maximum nearby sweep...
      recombination-bins  Output recombination bins from empirical...
      scan                Standalone outlier scan from a directory of...
      scan-plot           Visualize outlier scan results.
      simulator           Run the discoal simulator with user-specified...
You can also try to import the package into your python enviroment

.. code-block:: bash

    python -c "import flexsweep"



