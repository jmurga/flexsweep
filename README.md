[![Documentation Status](https://readthedocs.org/projects/flexsweep/badge/?version=latest)](https://flexsweep.readthedocs.io/en/latest/?badge=latest)

# Flexsweep v2.0

The second version of [Flexsweep software](https://doi.org/10.1093/molbev/msad139), a versatile tool for detecting selective sweeps. The software trains a convolutional neural network (CNN) to classify genomic loci as sweep or neutral regions. The workflow begins with simulating data under an appropriate demographic model and classify regions as neutral or sweeps, including several selection events regarding sweep strength, age, starting allele frequency (softness), and ending allele frequency (completeness).

Please see [documentation](https://flexsweep.readthedocs.io/en/latest/) for further details.


## Development Setup

This project uses [uv](https://docs.astral.sh/uv/) for dependency management and [maturin](https://www.maturin.rs/) for Rust bindings.

1. **Install uv**: Follow the [official guide](https://docs.astral.sh/uv/getting-started/installation/).
2. **Build for local development**:
   ```bash
   uv run --with maturin --with ziglang maturin develop --zig