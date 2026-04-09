#!/bin/bash
set -euo pipefail

# 1. Build discoal from source
git clone --depth 1 https://github.com/kr-colab/discoal.git discoal-src
cd discoal-src
make discoal
cp discoal "$PREFIX/bin/discoal"
cd ..

# 2. Build polarize (Rust) from bundled src/
cd "$SRC_DIR/flexsweep/src"
cargo build --release
cp target/release/polarize "$PREFIX/bin/polarize"
cd "$SRC_DIR"

# 3. Install Python package
pip install . --no-deps --no-build-isolation -vv
