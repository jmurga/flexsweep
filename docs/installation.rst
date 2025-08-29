
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

You also can build from scratch by cloning the repository and install using poetry and pip

.. code-block:: bash

    git clone https://github.com/jmurga/flexsweep.git
    cd flexswepp
    poetry build
    pip install dist/flexsweep-1.2.tar.gz


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
      cnn          Run the Flexsweep CNN
      fvs-discoal  Run the summary statistic estimation from discoal...
      fvs-vcf      Run the summary statistic estimation from a VCF file to...
      simulator    Run the discoal Simulator

You can also try to import the package into your python enviroment

.. code-block:: bash

    python -c "import flexsweep"



