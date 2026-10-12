.. _haplotype-matrix:

Haplotype matrices
==================

Flex-sweep classifies **feature vectors** of summary statistics (:doc:`feature_vectors`).
Another family of methods trains CNNs directly on the **haplotype matrix**: an image
with one row per haplotype and one column per SNP, where each pixel is the allele a
haplotype carries at a site (e.g. ImaGene, DANN-sweeps by
`Harris et al. 2025 <https://doi.org/10.1101/2025.10.14.682443>`_, TrIdent by
`Arnab et al. 2025 <https://doi.org/10.1093/molbev/msaf094>`_). Flex-sweep 2.0 builds
these images from the same simulations and VCFs as its feature vectors, so the two
representations can be trained and compared on identical data.

This is a separate input path. It does not change the feature vectors or the default
CNN, and ``flexsweep cnn`` does not use it. The module ``flexsweep.hap_matrix``
provides:

* **images** of ``K`` SNPs around the center of each simulated locus or each VCF
  region, in the same region layout as ``fvs-vcf`` (``haplotype_images``);
* **orderings** of rows and columns from the literature, or your own function
  (``sort_haplotypes``), and two pixel codings (``encode_alleles``);
* **disruption utilities** that destroy one property of the matrix (site order, linkage
  disequilibrium or allele frequencies) while keeping others, to test which signal a
  trained CNN relies on (`Tran et al. 2025 <https://doi.org/10.1093/molbev/msaf250>`_);
* the **image CNN architecture** of Harris et al. 2025 (``flexsweep.cnn.hap_image_cnn``).

In our comparison on YRI simulations, no image configuration matched the full feature
vector over the 1.2 Mb region, and the sweep probabilities of the two representations
were only partly correlated. Use the image path as a complement to the feature vectors,
or to work with image-based methods, rather than as a replacement.

Python interface
----------------

The image path is available from Python only. Build images for the simulations and the
VCFs, train a Keras model, and predict:

.. code-block:: python

    import numpy as np
    import tensorflow as tf
    from flexsweep.hap_matrix import haplotype_images
    from flexsweep.cnn import hap_image_cnn

    # Images (N, haplotypes, K), SNP positions (N, K), metadata (one row per image)
    images, positions, meta = haplotype_images("yri_sims", K=201, nthreads=24)
    labels = (meta["model"] != "neutral").to_numpy().astype("float32")

    inp = tf.keras.Input(images.shape[1:] + (1,))
    model = tf.keras.Model(inp, hap_image_cnn(inp))
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-4),
                  loss="binary_crossentropy", metrics=["accuracy"])
    model.fit(images[..., None].astype("float32"), labels, batch_size=32, epochs=100,
              validation_split=0.1, shuffle=True,
              callbacks=[tf.keras.callbacks.EarlyStopping(
                  monitor="val_accuracy", patience=5, restore_best_weights=True)])

    # Same layout for the VCF regions; meta["iter"] holds chr:start-end
    vcf_images, _, vcf_meta = haplotype_images("yri_vcfs", vcf=True, K=201, nthreads=24)
    prob_sweep = model.predict(vcf_images[..., None].astype("float32")).ravel()

Hold out a test set of simulations before ``fit`` when you report accuracy;
``validation_split`` is only used for early stopping.

Building images
---------------

``flexsweep.hap_matrix.haplotype_images`` is the haplotype-matrix counterpart of
``summary_statistics``: it reads the same discoal simulation folder (``neutral/``,
``sweep/``, ``params.txt.gz``) or a directory of bgzipped, indexed VCF/BCF files, and
returns one ``(haplotypes, K)`` 0/1 image per locus. Each image holds the ``K`` contiguous
biallelic SNPs centred on the SNP closest to the locus center (the sweep site in the
simulations); rows are sorted with the functions above (``sort="haplotype_freq"``
default, ``"freq"``, ``"corr"`` or ``None``). Loci with too few SNPs on either side are
zero-padded so the central SNP is always in column ``K // 2``; ``n_snps`` records the real
count. VCF loci slide along each contig exactly like ``fvs-vcf`` (``locus_length``,
``step_vcf``). Common sizes: ``K=201`` (Harris et al. 2025), ``K=499`` (TrIdent).

.. code-block:: python

    from flexsweep.hap_matrix import haplotype_images

    # images (N, 216, 201) uint8, SNP positions (N, 201) with -1 for padding,
    # meta: iter, s, t, f_i, f_t, mu, r, model (sweep subclass or neutral), n_snps
    images, positions, meta = haplotype_images("yri_sims", K=201, nthreads=24, suffix="")
    vcf_images, _, vcf_meta = haplotype_images("yri_vcfs", vcf=True, K=201, nthreads=22)

By default the ``K`` SNPs are contiguous, so the image is narrow where SNP density is
high and its physical width varies from locus to locus. Pass ``span`` to fix the width
instead: the SNPs within ``span / 2`` of the centre are subsampled to ``K`` (evenly, or
with ``thin="random"``), so every image covers the same sequence. This is how
Harris et al. (2025) build theirs, and setting ``span=locus_length`` makes the image
cover the same region that window-based feature vectors tile.

``offset`` goes further and stops assuming the feature of interest is in the middle: the
window centre is then drawn uniformly from ``+/- offset`` bp around the locus centre, with
a draw per locus that is reproducible from ``seed``. In simulations the sweep sits at the
centre, so a model trained with ``offset=0`` only ever sees centred sweeps, while a
genome scan slides windows over everything; training with an offset matches deployment.
It applies to whatever images the call produces, so build training and evaluation sets in
separate calls if you want them to differ.

.. code-block:: python

    # 201 SNPs spread over the whole 1.2 Mb locus instead of the contiguous centre
    wide, positions, meta = haplotype_images("yri_sims", K=201, span=1.2e6, nthreads=24)

    # same, with the sweep anywhere within 300 kb of the image centre
    shifted, _, _ = haplotype_images("yri_sims", K=201, span=1.2e6, offset=3e5, seed=1)

With ``suffix`` set, the arrays and metadata are written to the input folder
(``hap.npy``, ``hap_pos.npy``, ``hap.parquet``). The simulation and VCF images share the
same layout, so a model trained on simulations predicts on ``vcf_images`` directly.

Row and column orderings
------------------------

The order of the haplotypes in a sample is arbitrary, but a convolution sees
neighbouring rows together. Reordering rows (haplotypes) or columns (SNPs) so that
similar haplotypes sit next to each other gives the CNN a consistent picture of a sweep,
in which the haplotypes carrying the selected allele form one block. See
`Zhao et al. 2023 <https://doi.org/10.1093/bioinformatics/btad265>`_ and
`Zhao and Alachiotis 2025 <https://doi.org/10.1016/j.ymeth.2024.11.003>`_ for a
comparison of orderings. All functions below take a binary ``(haplotypes x sites)``
NumPy array.

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Function
     - Description
   * - ``daf_sorting(matrix)``
     - Sort **columns** (SNPs) by descending derived allele frequency. Most
       common derived alleles appear first.
   * - ``freq_sorting(matrix)``
     - Sort **rows** (haplotypes) by descending number of derived alleles
       (Hamming weight). Most derived haplotypes appear first.
   * - ``corr_sorting(matrix)``
     - Sort **rows** by Pearson correlation coefficient with the most
       correlated haplotype. Groups similar haplotypes together.
   * - ``pcc_column_sort_numba(matrix)``
     - Sort **columns** by total PCC score — SNPs most correlated with the
       rest of the matrix appear first.
   * - ``haplotype_freq_sorting(matrix)``
     - Sort **columns** by haplotype frequency (most common haplotype group
       first). Returns ``(reordered, col_order, groups, freqs)``.
   * - ``haplotype_freq_sorting_hamming(matrix)``
     - Same as above but within each frequency group, haplotypes are
       additionally ordered by Hamming distance to the most frequent
       haplotype. Returns ``(reordered, freqs)``.

``daf_sorting``, ``freq_sorting``, ``corr_sorting``, and
``pcc_column_sort_numba`` are Numba-compiled and operate on integer arrays.
The two ``haplotype_freq_sorting`` variants operate on general NumPy arrays.

These functions, the disruption utilities and the image builders live in
``flexsweep.hap_matrix`` (they were in ``flexsweep.utils`` up to version 2.0).
``hap_matrix.sort_haplotypes`` dispatches to them by name, takes **any callable** with the
same signature, and accepts a sequence of rearrangements applied in order (the two-stage
scheme of `Zhao and Alachiotis 2025 <https://doi.org/10.1016/j.ymeth.2024.11.003>`_, e.g.
``("corr", "daf")``). ``hap_matrix.encode_alleles`` codes pixels either as 0/1 derived
alleles (default) or as major -1 / minor +1 with missing 0, the coding of
`Harris et al. 2025 <https://doi.org/10.1101/2025.10.14.682443>`_.

.. code-block:: python

    import numpy as np
    from flexsweep.hap_matrix import (
        daf_sorting, freq_sorting, haplotype_freq_sorting, sort_haplotypes
    )

    # random hap: binary (n_haplotypes × n_sites) array from your VCF window
    hap = np.random.randint(0, 2, (1000, 216), dtype=np.int32)

    # Sort SNPs by DAF, then haplotypes by frequency
    hap_daf = daf_sorting(hap.copy())
    hap_sorted, col_order, groups, freqs = haplotype_freq_sorting(hap_daf)

    # hap_sorted is now ready for a (batch, haplotypes, sites, 1) CNN input as needed

    # the same through the dispatcher, including a two-stage scheme and a custom callable
    hap_zhao = sort_haplotypes(hap.copy(), ("corr", "daf"))
    hap_custom = sort_haplotypes(hap.copy(), lambda m: m[m.sum(1).argsort()])

Disruption utilities
--------------------

These functions return a copy of a ``(haplotypes x sites)`` matrix with one property
destroyed. Comparing a trained model's predictions on original and disrupted images
shows which signal it uses (`Tran et al. 2025 <https://doi.org/10.1093/molbev/msaf250>`_).

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - Function
     - Effect
   * - ``disrupt_genomic_positions(matrix)``
     - Shuffles the columns: removes the order of the sites, keeps each site's
       allele frequency and the haplotypes' allele counts.
   * - ``disrupt_ld(matrix)``
     - Shuffles the site order and the alleles within each site: removes linkage
       disequilibrium, keeps the allele frequency of every site. It shuffles its input
       in place and returns the **transposed** matrix (sites x haplotypes): pass a copy
       and transpose the result back.
   * - ``disrupt_af(matrix)``
     - Shuffles all pixels: removes linkage disequilibrium and per-site allele
       frequencies, keeps the total number of derived alleles.

.. code-block:: python

    from flexsweep.hap_matrix import disrupt_genomic_positions, disrupt_ld

    shuffled = np.stack([disrupt_genomic_positions(img) for img in vcf_images])
    no_ld = np.stack([disrupt_ld(img.copy()).T for img in vcf_images])
    prob_shuffled = model.predict(shuffled[..., None].astype("float32")).ravel()
    prob_no_ld = model.predict(no_ld[..., None].astype("float32")).ravel()

Image architecture
------------------

``flexsweep.cnn.hap_image_cnn`` is the image architecture of DANN-sweeps
(Harris et al. 2025) and, before it, ImaGene: two 3x3 convolutions of 64 channels, each
followed by 2x2 max pooling, then ``Flatten`` and two 128-unit dense layers with dropout
0.5. It flattens the last feature map rather than pooling it globally, so where a
feature sits along the SNP axis is preserved; a global-pooling head would average that
information away. The default Flex-sweep CNN (``CNN.cnn_flexsweep``) is **not** a
drop-in for images: its single pooling stage is sized for the small statistics x
positions grid, and on a 216 x 201 image its dense head alone would hold about 3.5
billion weights.
