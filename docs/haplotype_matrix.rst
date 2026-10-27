.. _haplotype-matrix:

Haplotype matrices
==================

Flex-sweep classifies **feature vectors** of summary statistics (:doc:`feature_vectors`).
Another family of methods trains CNNs directly on the **haplotype matrix**: an image
with one row per haplotype and one column per SNP, where each pixel is the allele a
haplotype carries at a site (e.g. ImaGene, DANN-sweeps by
`Harris et al. 2025 <https://doi.org/10.1101/2025.10.14.682443>`_, TrIdent by
`Arnab et al. 2025 <https://doi.org/10.1093/molbev/msaf094>`_). Flex-sweep 2.0 builds
these images from the same simulations and VCFs as its feature vectors, so both
inputs can be trained on identical data.

This is a separate, advanced input path, available from Python only. It does not change
the feature vectors or the default CNN, and ``flexsweep cnn`` does not use it. Two
functions carry it:

* ``flexsweep.hap_matrix.haplotype_images`` builds one image per simulation or per VCF
  region;
* ``flexsweep.cnn.hap_image_cnn`` is the image CNN of Harris et al. 2025.

.. warning::

   In the Flex-sweep 2.0 paper the image path is a worked example, not a benchmark: one
   demography, one seed, and an image model that was not tuned. We make no claim that
   images or feature vectors are the better input, and we did not evaluate domain
   adaptation on images. Evaluate any image model with simulations of your own
   population and study design.

Example data
------------

All code on this page runs on the output of the example pipeline (:doc:`quickstart`):

.. code-block:: bash

    flexsweep example --output_folder yri_chr22 --nthreads 24 --full

* ``yri_chr22/sims``: 1,000 neutral and 1,000 sweep simulations of a 1.2 Mb locus under
  the YRI demographic model, 100 haplotypes each, sweep at the center (600 kb);
* ``yri_chr22/data/vcfs``: the polarized and masked chr22 VCF of 50 YRI (100
  haplotypes);
* ``yri_chr22/cnn/yri_predictions.txt``: feature-vector CNN predictions for the chr22
  regions.

1,000 simulations per class are enough to run every step, not to train a reliable image
CNN (the paper used 50,000). To simulate more, rerun the simulation and the steps that
use it:

.. code-block:: bash

    flexsweep example --output_folder yri_chr22 --nthreads 24 --full \
        --num_simulations 10000 --steps simulate,fvs-sims,cnn --overwrite

Building images
---------------

.. code-block:: python

    from flexsweep.hap_matrix import haplotype_images

    images, positions, meta = haplotype_images("yri_chr22/sims", K=201, sort=None,
                                               nthreads=24)

``haplotype_images`` reads the simulation folder (``neutral/``, ``sweep/``,
``params.txt.gz``) or, with ``vcf=True``, a directory of bgzipped, indexed VCF/BCF files.
For each locus it keeps the biallelic SNPs, picks ``K`` of them (``K``, ``span``,
``offset`` below), codes the alleles (``encoding``), reorders rows (``sort``) and
returns:

* ``images``: ``(N, haplotypes, K)``; ``uint8`` 0/1 with ``encoding="derived"``
  (ancestral 0, derived 1), ``int8`` -1/+1 with ``encoding="major_minor"`` (major -1,
  minor +1). For Keras: ``images[..., None].astype("float32")``.
* ``positions``: ``(N, K)`` position of the SNP in each column, ``-1`` in padded columns.
* ``meta``: one row per image with ``iter, s, t, f_i, f_t, mu, r, model, n_snps``.
  Simulations: ``iter`` is the replicate within its class and ``model`` the sweep
  subclass or ``neutral``, as in the feature vectors. VCFs: ``iter`` is the region
  ``chr:start-end`` and ``model`` is ``empirical``. ``n_snps`` counts real columns.

VCF regions are 1.2 Mb windows sliding by 10 kb (``locus_length``, ``step_vcf``),
exactly as in ``fvs-vcf``; each image is taken around the region midpoint. With
``suffix`` the three outputs are also saved in the input folder (``hap_{suffix}.npy``,
``hap_pos_{suffix}.npy``, ``hap_{suffix}.parquet``), so they can be reloaded later
without rebuilding.

K, span and offset
^^^^^^^^^^^^^^^^^^

These three arguments decide which SNPs of the 1.2 Mb locus become the ``K`` columns of
the image.

.. code-block:: text

    simulated locus, 1.2 Mb, sweep site x at 600 kb
    |-------------------------------------x-------------------------------------|

    span=None (default): K contiguous SNPs around the SNP nearest 600 kb
                                     [||||x||||]

    span=1.2e6: K SNPs thinned from all SNPs within 600 kb +/- 600 kb
    [|     |     |     |     |     |    x|     |     |     |     |     |     |]

    offset=450e3: the window center is drawn per locus in 600 kb +/- 450 kb
                    [|||||||||]       x

``K`` is the number of SNPs, i.e. the image width in pixels. Harris et al. 2025 use 201,
TrIdent 499. Every image has exactly ``K`` columns; a window with fewer SNPs is
zero-padded on both sides (``positions == -1``) so its center stays in column ``K // 2``.

``span`` sets how much sequence those ``K`` SNPs cover:

* ``span=None`` takes ``K`` **contiguous** SNPs. The image keeps every SNP and the full
  haplotype structure, but its physical width is whatever ``K`` SNPs happen to span and
  varies between loci with SNP density and recombination. With 216 YRI haplotypes, 201
  SNPs spanned a median of 52 kb (interquartile range 38-75 kb); fewer haplotypes give
  fewer SNPs and wider images.
* ``span=<bp>`` keeps the SNPs within ``span / 2`` of the center and subsamples them to
  ``K``, evenly (``thin="even"``) or at random (``thin="random"``, as Harris et al. 2025
  do). Every image then covers the same sequence, but neighbouring columns are no longer
  neighbouring SNPs. ``span=1.2e6`` covers the region that the feature vectors tile.

``offset`` decides where the window sits. With ``offset=0`` every simulated window is
centred on the sweep site, so the model only ever sees a sweep in the middle of the
image. A genome scan sees sweeps anywhere in the window. ``offset=d`` draws the center
of each simulated window uniformly within ``+/- d`` bp of the locus center, in training
and in evaluation. The draw is seeded by ``seed`` and by the locus, so it is
reproducible and does not depend on ``nthreads``. Use it on simulations only; VCF
regions already slide along the chromosome.

Narrow windows make ``offset`` costly. In the paper (201 contiguous SNPs, 52 kb, window
center within +/-450 kb of the sweep site) the image contained the sweep site in 785 of
10,000 sweep regions, and its ROC-AUC fell from 0.963 centred to 0.774 displaced; a
200 kb feature vector displaced the same way fell from 0.964 to 0.914. To see the width
of your own images:

.. code-block:: python

    import numpy as np

    real = positions >= 0
    span_kb = (positions.max(axis=1) - np.where(real, positions, np.inf).min(axis=1)) / 1e3
    print(np.quantile(span_kb, [0.25, 0.5, 0.75]), (meta["n_snps"] < 201).sum())

Other arguments: ``sort`` and ``encoding`` (`Row orderings`_), ``representation="trident"``
with ``K=499`` and ``dim=299`` for the TrIdent image (needs OpenCV), ``iters`` to build
only some replicates (``{"neutral": [...], "sweep": [...]}``, 1-based), and
``nthreads``. A custom ``sort`` function must be defined at module level, not as a
lambda, because images are built in worker processes.

Image architecture
------------------

``flexsweep.cnn.hap_image_cnn(model_input, num_classes=1, filters=64, dense=128,
dropout=0.5)`` is the architecture of DANN-sweeps (Harris et al. 2025), in the ImaGene
lineage. It takes a Keras input tensor of shape ``(haplotypes, K, 1)`` and returns the
output tensor: one sigmoid unit (sweep probability), or a softmax with
``num_classes > 1``.

.. list-table:: Layers for the 100 x 201 images of the example
   :header-rows: 1
   :widths: 40 30 30

   * - Layer
     - Output shape
     - Parameters
   * - ``hap_conv1``: 3x3 convolution, 64 filters, ReLU
     - 98 x 199 x 64
     - 640
   * - ``hap_pool1``: 2x2 max pooling
     - 49 x 99 x 64
     - 0
   * - ``hap_conv2``: 3x3 convolution, 64 filters, ReLU
     - 47 x 97 x 64
     - 36,928
   * - ``hap_pool2``: 2x2 max pooling
     - 23 x 48 x 64
     - 0
   * - ``feature_extractor``: flatten
     - 70,656
     - 0
   * - ``hap_dense1``: dense, ReLU, dropout 0.5
     - 128
     - 9,044,096
   * - ``hap_dense2``: dense, ReLU, dropout 0.5
     - 128
     - 16,512
   * - ``out_dense``: sigmoid
     - 1
     - 129

9.1 million parameters, 99% of them in ``hap_dense1``; the count grows with the image
(about 20.5 million for 216 x 201). The head flattens the last feature map instead of
pooling it globally, so the model keeps where along the SNP axis a feature sits, i.e.
its distance from the window center.

``CNN.train`` and ``flexsweep cnn`` read feature-vector parquet files only, so images are
trained with Keras directly, as below. Any function that maps the input tensor to a
sigmoid output can replace ``hap_image_cnn``. The default Flex-sweep CNN
(``CNN.cnn_flexsweep``) is not one of them: its pooling is sized for the small
statistics x windows grid.

Worked example
--------------

Two image models are trained on ``yri_chr22/sims`` and applied to the chr22 VCF:
**unordered** images (rows as read, alleles 0/1) and **Harris** images (rows grouped by
identical haplotype, most frequent first, alleles major -1 / minor +1, as in Harris et
al. 2025). Both hold 201 contiguous SNPs centred on the sweep site. The last step adds
domain adaptation to the Harris model. Paste the blocks into one Python session in
order; everything is saved under ``yri_chr22/hap`` so you can continue later.

**1. Build and save the images.** Simulations and VCF regions must be built with the
same settings.

.. code-block:: python

    import os
    import numpy as np
    import polars as pl
    import tensorflow as tf
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import train_test_split
    from flexsweep.hap_matrix import haplotype_images
    from flexsweep.cnn import hap_image_cnn

    SIMS, VCFS, OUT = "yri_chr22/sims", "yri_chr22/data/vcfs", "yri_chr22/hap"
    os.makedirs(OUT, exist_ok=True)
    SETTINGS = {
        "unordered": dict(K=201, sort=None),
        "harris": dict(K=201, sort="haplotype_freq", encoding="major_minor"),
    }

    sims, vcf = {}, {}
    for name, kw in SETTINGS.items():
        sims[name] = haplotype_images(SIMS, nthreads=24, suffix=name, **kw)
        vcf[name] = haplotype_images(VCFS, vcf=True, nthreads=24, suffix=name, **kw)

    meta, vcf_meta = sims["unordered"][2], vcf["unordered"][2]
    y = (meta["model"] != "neutral").to_numpy().astype("float32")

The images are now in ``yri_chr22/sims/hap_{name}.npy`` and
``yri_chr22/data/vcfs/hap_{name}.npy`` with their ``hap_pos_*`` and ``.parquet``
companions. In a later session, reload them instead of rebuilding:

.. code-block:: python

    def load(folder, name):
        return (np.load(f"{folder}/hap_{name}.npy"), np.load(f"{folder}/hap_pos_{name}.npy"),
                pl.read_parquet(f"{folder}/hap_{name}.parquet"))

    sims = {name: load(SIMS, name) for name in SETTINGS}
    vcf = {name: load(VCFS, name) for name in SETTINGS}

**2. Split.** 80% of the simulations train, 10% are used for early stopping and 10% are
held out for testing, stratified by class. The same split serves every model.

.. code-block:: python

    idx = np.arange(len(y))
    train, test = train_test_split(idx, test_size=0.1, stratify=y, random_state=1)
    train, val = train_test_split(train, test_size=1 / 9, stratify=y[train], random_state=1)

**3. Train.** ``hap_image_cnn`` with the training settings of the default Flex-sweep CNN
(``CNN.train``): binary cross-entropy, Adam with AMSGrad and a cosine-decay-with-restarts
learning rate starting at 1e-4, batch size 32, early stopping when validation accuracy
has not improved by 1e-4 for 5 epochs, keeping the best weights.

.. code-block:: python

    def train_image_cnn(X, y, train, val, seed=1):
        tf.keras.utils.set_random_seed(seed)
        inp = tf.keras.Input(X.shape[1:])
        model = tf.keras.Model(inp, hap_image_cnn(inp))
        lr = tf.keras.optimizers.schedules.CosineDecayRestarts(1e-4, first_decay_steps=300)
        model.compile(optimizer=tf.keras.optimizers.Adam(lr, epsilon=1e-7, amsgrad=True),
                      loss="binary_crossentropy", metrics=["accuracy"])
        model.fit(X[train], y[train], batch_size=32, epochs=1000, shuffle=True,
                  validation_data=(X[val], y[val]), verbose=2,
                  callbacks=[tf.keras.callbacks.EarlyStopping(
                      monitor="val_accuracy", mode="max", min_delta=1e-4, patience=5,
                      restore_best_weights=True)])
        return model

    def as_input(images):
        return images[..., None].astype("float32")    # (N, 100, 201, 1)

    models = {}
    for name in SETTINGS:
        models[name] = train_image_cnn(as_input(sims[name][0]), y, train, val)
        models[name].save(f"{OUT}/{name}.keras")

A saved model reloads with ``tf.keras.models.load_model(f"{OUT}/{name}.keras")``.

**4. Evaluate on the held-out simulations.** ROC-AUC, and the true positive rate at a 5%
false positive rate with the threshold set on the neutral test simulations:

.. code-block:: python

    def evaluate(p, y_true):
        thr = np.quantile(p[y_true == 0], 0.95)
        return {"roc_auc": roc_auc_score(y_true, p),
                "tpr_at_fpr5": float((p[y_true == 1] > thr).mean())}

    p_test = {name: models[name].predict(as_input(sims[name][0][test]), batch_size=256)
              .ravel() for name in SETTINGS}
    for name, p in p_test.items():
        print(name, evaluate(p, y[test]))

With 100 test simulations per class, differences of a few hundredths of ROC-AUC are
within sampling noise.

**5. Predict the chr22 regions** and compare with the feature-vector CNN of the example.
Regions are matched by start position.

.. code-block:: python

    pred = vcf_meta.select("iter", "n_snps").with_columns(
        start=pl.col("iter").str.extract(r":(\d+)-", 1).cast(pl.Int64),
        **{f"prob_{name}": pl.Series(models[name].predict(as_input(vcf[name][0]),
                                                          batch_size=256).ravel())
           for name in SETTINGS})
    pred.write_csv(f"{OUT}/yri_hap_predictions.tsv", separator="\t")

    fv = (pl.read_csv("yri_chr22/cnn/yri_predictions.txt")
          .select(pl.col("start").cast(pl.Int64), "prob_sweep").sort("start"))
    both = fv.join_asof(pred.sort("start"), on="start", strategy="nearest",
                        tolerance=10_000).drop_nulls()
    probs = ["prob_sweep", "prob_unordered", "prob_harris"]
    print(both.select([pl.corr(a, b, method="spearman").alias(f"{a} ~ {b}")
                       for i, a in enumerate(probs) for b in probs[i + 1:]]))
    print(both.select((pl.col(probs) > 0.5).mean()))     # fraction called sweep

The probabilities are uncalibrated and use no sweep prior. The VCF has no labels, so
agreement between models is not accuracy.

**6. Domain adaptation on Harris images.** The built-in DANN (:doc:`dann`) reads feature
vectors only, but its parts work on images. ``image_dann`` builds the same network around
the image CNN: the convolutional part of ``hap_image_cnn`` is the shared feature
extractor, its dense head is the label head (trained on labelled simulations), and a
domain head separates simulations from VCF regions behind a gradient reversal layer
(``GradReverse``), which pushes the extractor towards features the domain head cannot
separate.

.. code-block:: python

    from flexsweep.cnn import (DAParquetSequence, GradReverse, GRLRamp,
                               masked_bce, masked_binary_accuracy)

    def image_dann(input_shape):
        inp = tf.keras.Input(input_shape)
        x = tf.keras.layers.Conv2D(64, 3, activation="relu")(inp)    # extractor
        x = tf.keras.layers.MaxPooling2D(2)(x)
        x = tf.keras.layers.Conv2D(64, 3, activation="relu")(x)
        x = tf.keras.layers.MaxPooling2D(2)(x)
        feat = tf.keras.layers.Flatten()(x)
        h = feat                                                     # label head
        for _ in range(2):
            h = tf.keras.layers.Dense(128, activation="relu")(h)
            h = tf.keras.layers.Dropout(0.5)(h)
        out_cls = tf.keras.layers.Dense(1, activation="sigmoid", name="classifier")(h)
        grl = GradReverse(lambd=0.0)                                 # domain head
        g = tf.keras.layers.Dense(128, activation="relu")(grl(feat))
        g = tf.keras.layers.Dropout(0.2)(g)
        g = tf.keras.layers.Dense(32, activation="relu")(g)
        g = tf.keras.layers.Dropout(0.1)(g)
        out_dom = tf.keras.layers.Dense(1, activation="sigmoid", name="discriminator")(g)
        return tf.keras.Model(inp, [out_cls, out_dom]), grl

Training follows the defaults of ``flexsweep dann``. The reversal strength rises linearly
from 0 to 1 over 30 epochs (``GRLRamp``). Each step holds 64 labelled simulations for the
label head and 64 images for the domain head, VCF regions and simulations 2:1
(``DAParquetSequence``, the batch generator of the built-in DANN, where -1 marks a label
a head does not use). The domain loss weighs 0.75. Early stopping uses the validation
simulations, since the VCF has no labels, and starts when the ramp ends, so the kept
weights come from a model trained with the domain head active.

.. code-block:: python

    X, X_vcf = as_input(sims["harris"][0]), as_input(vcf["harris"][0])

    tf.keras.utils.set_random_seed(1)
    batches = DAParquetSequence(src_neutral=X[train[y[train] == 0]],
                                src_sweep=X[train[y[train] == 1]], tar_all=X_vcf,
                                batch_size=64, tgt_ratio=2.0, seed=1)
    dann, grl = image_dann(X.shape[1:])
    dann.compile(optimizer=tf.keras.optimizers.Adam(1e-4),
                 loss={"classifier": masked_bce, "discriminator": masked_bce},
                 loss_weights={"classifier": 1.0, "discriminator": 0.75},
                 metrics={"classifier": masked_binary_accuracy,
                          "discriminator": masked_binary_accuracy})
    val_y = {"classifier": y[val], "discriminator": -np.ones(len(val), "float32")}
    dann.fit(batches, epochs=300, validation_data=(X[val], val_y),
             callbacks=[GRLRamp(grl, max_lambda=1.0, epochs=30),
                        tf.keras.callbacks.EarlyStopping(
                            monitor="val_classifier_masked_binary_accuracy", mode="max",
                            patience=20, start_from_epoch=30,
                            restore_best_weights=True)])
    dann.save(f"{OUT}/harris_dann.keras")

    print("harris     ", evaluate(p_test["harris"], y[test]))
    print("harris DANN", evaluate(dann.predict(X[test], batch_size=256)[0].ravel(), y[test]))

    pred = pred.with_columns(
        prob_harris_dann=pl.Series(dann.predict(X_vcf, batch_size=256)[0].ravel()))
    pred.write_csv(f"{OUT}/yri_hap_predictions.tsv", separator="\t")

The validation loss prints as ``nan`` because the domain head has no labels on the
validation simulations; only ``val_classifier_*`` is a measurement. The domain accuracy
is measured on a 2:1 mix of VCF regions and simulations, so a domain head that cannot
tell them apart scores about 0.67, not 0.5. Unlike ``flexsweep dann``, this fits no Platt
scaling and applies no sweep prior. Here the simulations and the VCF share the YRI
demographic model, so the domain shift is small and the DANN should change little;
domain adaptation matters when the simulated demography or genome differs from the data.
Reload the saved DANN with ``tf.keras.models.load_model(f"{OUT}/harris_dann.keras")``
after importing ``flexsweep.cnn``; ``dann.predict(X)[0]`` is the sweep probability.

Row orderings
-------------

The order of the haplotypes in a sample is arbitrary, but a convolution sees
neighbouring rows together. Reordering rows so that similar haplotypes sit next to each
other gives the CNN a consistent picture of a sweep, in which the haplotypes carrying the
selected allele form one block. See
`Zhao et al. 2023 <https://doi.org/10.1093/bioinformatics/btad265>`_ and
`Zhao and Alachiotis 2025 <https://doi.org/10.1016/j.ymeth.2024.11.003>`_ for a
comparison. Pass ``sort=`` to ``haplotype_images``:

.. list-table::
   :header-rows: 1
   :widths: 32 68

   * - ``sort``
     - Ordering
   * - ``"haplotype_freq"`` (default)
     - Rows: identical haplotypes grouped, most frequent group first.
   * - ``"haplotype_freq_hamming"``, ``"row_dist"``
     - Rows: groups also ordered by Hamming distance to the most frequent haplotype
       (``row_dist`` is RowDist of Harris et al. 2025).
   * - ``"center_window_freq"``, ``"center_window_dist"``
     - Rows: grouped by their haplotype over the central 51 SNPs, then ordered within
       each group (CenterWindowFreq and CenterWindowDist of Harris et al. 2025).
   * - ``"freq"``, ``"corr"``
     - Rows: by number of derived alleles, or by r^2 with the most connected haplotype.
   * - ``"daf"``, ``"pcc_cols"``
     - **Columns**: by derived allele frequency, or by summed r^2 with the other SNPs.
       The columns then no longer follow ``positions``.
   * - ``None``
     - No reordering.

``sort`` also takes a sequence applied in order (e.g. ``("corr", "daf")``, Zhao and
Alachiotis 2025) or a function taking and returning a ``(haplotypes, sites)`` array.
Orderings act on the real SNPs only, after they are chosen and coded. ``"freq"``,
``"corr"``, ``"daf"`` and ``"pcc_cols"`` assume 0/1 pixels: use them with
``encoding="derived"``. The haplotype-grouping orderings give the same rows under either
encoding. ``flexsweep.hap_matrix.sort_haplotypes(matrix, sort)`` and
``encode_alleles(matrix, encoding)`` apply the same operations to any
``(haplotypes, sites)`` array.

Disruption utilities
--------------------

To test which signal a trained model uses, destroy one property of the test images and
see how much accuracy drops
(`Tran et al. 2025 <https://doi.org/10.1093/molbev/msaf250>`_). Each function takes one
0/1 ``(haplotypes, sites)`` image:

* ``disrupt_genomic_positions``: permutes the columns; removes the order of the sites,
  keeps each site and its LD with every other site. Unseeded, so it changes on every
  call.
* ``disrupt_ld``: permutes the columns and the alleles within each site; removes LD,
  keeps each site's allele frequency. It modifies its input and returns the transpose:
  call ``disrupt_ld(img.copy()).T``.
* ``disrupt_af``: permutes all pixels; keeps only the total number of derived alleles.

``disrupt_ld`` and ``disrupt_af`` use NumPy's global generator (``np.random.seed``).
Padded columns are permuted like real ones, so use images with ``n_snps == K``. With the
unordered model of the worked example:

.. code-block:: python

    from flexsweep.hap_matrix import disrupt_af, disrupt_genomic_positions, disrupt_ld

    np.random.seed(1)
    imgs = sims["unordered"][0][test]
    for label, f in [("original", lambda m: m), ("site order", disrupt_genomic_positions),
                     ("LD", lambda m: disrupt_ld(m.copy()).T), ("allele freq", disrupt_af)]:
        X_d = as_input(np.stack([f(m) for m in imgs]))
        print(label, evaluate(models["unordered"].predict(X_d, batch_size=256).ravel(),
                              y[test]))
