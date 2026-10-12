.. _cnn:

Training and prediction (CNN)
=============================

The CNN classifies each 1.2 Mb genomic window as **neutral** or **sweep**. It is trained
on labelled feature vectors from simulations (``fvs-discoal``, :doc:`feature_vectors`)
and predicts on feature vectors from VCFs (``fvs-vcf``). The input is not a haplotype
matrix: each window is an image of summary statistics, with one row per statistic and
one column per (window size, center) position, z-scored as described in
:doc:`introduction`. For a model trained on haplotype matrices see
:doc:`haplotype_matrix`; for training with unlabelled empirical data see :doc:`dann`.

Command line and Python interface
---------------------------------

.. code-block:: bash

    # Train on simulations, then predict the empirical windows
    flexsweep cnn \
        --train_data yri_sims/fvs.parquet \
        --predict_data yri_vcfs/fvs_yri.parquet \
        --output_folder yri_cnn \
        --seed 1

    # Reuse the trained model on another dataset
    flexsweep cnn \
        --model yri_cnn/model.keras \
        --predict_data other_vcfs/fvs_other.parquet \
        --output_folder other_cnn

.. code-block:: python

    import flexsweep as fs

    fs_cnn = fs.CNN(
        train_data="yri_sims/fvs.parquet",
        predict_data="yri_vcfs/fvs_yri.parquet",
        output_folder="yri_cnn",
        seed=1,
    )
    fs_cnn.train()                    # writes model.keras and predictions.txt
    df_prediction = fs_cnn.predict()  # writes yri_predictions.txt

    # Reuse a trained model
    fs_other = fs.CNN(
        model="yri_cnn/model.keras",
        predict_data="other_vcfs/fvs_other.parquet",
        output_folder="other_cnn",
    )
    fs_other.predict()

``flexsweep cnn`` picks its mode from the options you pass: ``--train_data`` alone
trains; ``--train_data`` and ``--predict_data`` train and then predict; ``--model``
and ``--predict_data`` only predict. Training splits the simulations into 80%
training, 10% validation (early stopping) and 10% test.

Options
-------

.. list-table::
   :header-rows: 1
   :widths: 22 20 16 42

   * - CLI option
     - Python argument
     - Default
     - Meaning
   * - ``--train_data``
     - ``train_data``
     -
     - Feature-vector parquet from ``fvs-discoal`` (labelled simulations).
   * - ``--predict_data``
     - ``predict_data``
     -
     - Feature-vector parquet from ``fvs-vcf``.
   * - ``--model``
     - ``model``
     -
     - Trained ``model.keras``; skips training.
   * - ``--output_folder``
     - ``output_folder``
     -
     - Folder for the model, plots and predictions (created if missing).
   * - ``--center``
     - ``center``
     - ``0,1.2e6``
     - Start and end of the simulated locus; must match the feature vectors.
   * - ``--windows``
     - ``windows``
     - ``100000``
     - Window size (bp) of the feature vectors to use.
   * - ``--step``
     - ``step``
     - ``100000``
     - Spacing of the window centers (bp).
   * - ``--seed``
     - ``seed``
     - unseeded
     - Seeds Python, NumPy, TensorFlow and the data split. Seeded CPU runs are
       byte-identical; GPU runs can differ in the last digits.

Features are selected by the column names of the parquet (``{stat}_{window}_{center}``),
so feature vectors with several window sizes or non-uniform centers need no extra
settings. The CLI trains with the defaults below; to change them, use Python.

``CNN.train`` arguments, with the defaults used in the paper:

.. list-table::
   :header-rows: 1
   :widths: 22 18 60

   * - Argument
     - Default
     - Description
   * - ``cnn``
     - ``CNN.cnn_flexsweep``
     - Function mapping the input tensor to a sigmoid output
       (:ref:`cnn-custom`).
   * - ``one_dim``
     - ``False``
     - Input layout ``(S, P, 1)`` (2D) or ``(P, S)`` (1D).
   * - ``learning_rate``
     - ``1e-4``
     - Initial value of the cosine-decay-with-restarts schedule.
   * - ``batch_size``
     - ``32``
     -
   * - ``patience``, ``min_delta``
     - ``5``, ``1e-4``
     - Early stopping on validation accuracy.
   * - ``epochs``
     - ``1000``
     - Maximum number of epochs.
   * - ``preprocess``
     - ``False``
     - Keep ``False``: ``fvs-discoal`` and ``fvs-vcf`` already write z-scored features.

Output
------

.. list-table::
   :header-rows: 1
   :widths: 32 68

   * - File
     - Content
   * - ``model.keras``
     - Trained model (training runs).
   * - ``predictions.txt``
     - Predictions on the 10% held-out test simulations (training runs).
   * - ``train_history.svg``, ``roc_curve.svg``, ``auprc.svg``,
       ``confusion_matrix.svg``, ``calibration.svg``
     - Learning curves, and ROC curve, precision-recall curve, confusion matrix and
       calibration curve on the test simulations (training runs).
   * - ``<name>_predictions.txt``
     - Predictions on ``--predict_data``. ``<name>`` is its file name without the
       ``fvs_`` prefix and ``.parquet`` (``fvs_yri.parquet`` gives
       ``yri_predictions.txt``).

``<name>_predictions.txt`` has one row per region: ``chr``, ``start``, ``end``,
``predicted_model`` (``sweep`` if ``prob_sweep >= 0.5``), ``prob_sweep`` and
``prob_neutral``. The simulation parameter columns (``f_i``, ``f_t``, ``s``, ``t``) only
carry values for simulated input. ``prob_sweep`` comes from a classifier trained with
equal numbers of neutral and sweep simulations, so it is not the genome-wide
probability that a region is a sweep; use it to rank regions. :doc:`prediction` shows
how to plot, merge and rank them.

.. _cnn-custom:

Custom architectures
--------------------

The ``CNN`` class accepts any Keras architecture through ``train(cnn=...)``: a function that
maps the input tensor to a sigmoid output. ``CNN`` builds the input tensor from the
``{stat}_{window}_{center}`` columns of the feature-vector parquet, with ``P`` = number of
window x center positions per statistic and ``S`` = number of statistics:

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Model
     - Input shape
   * - 2D (default, ``one_dim=False``)
     - ``(batch, S, P, 1)``: statistics as rows, positions as columns, one channel
   * - 1D (``one_dim=True``)
     - ``(batch, P, S)``: positions as steps, statistics as channels

Positions keep the column order of the parquet, identical for every statistic. The window
and center geometry is read from the column names, so feature vectors computed with
non-uniform centers (``--centers``, e.g. a log-distance or radial scheme around the sweep)
train and predict without passing ``center``/``step``/``windows`` to ``CNN``. ``predict``
detects 1D models from the rank of their input. Note that the default architecture
flattens its convolution output into a dense layer, so its size grows with ``P``; for long
inputs prefer a head that pools along the positions only, as in the 1D example below.

.. code-block:: python

    import flexsweep as fs
    from flexsweep.cnn import *

    def cnn_finer(model_input, num_classes=1):
        """
        Small kernels over statistics (2x1) and positions (1x2), batch normalization
        and dropout. Input: (batch, S, P, 1).
        """

        initializer = tf.keras.initializers.HeNormal()

        x = tf.keras.layers.Conv2D(
            64, (2, 1), padding="same", kernel_initializer=initializer, name="conv_2x1"
        )(model_input)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)

        x = tf.keras.layers.Conv2D(
            64, (1, 2), padding="same", kernel_initializer=initializer, name="conv_1x2"
        )(x)

        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)

        x = tf.keras.layers.Conv2D(
            128, (2, 2), padding="same", kernel_initializer=initializer, name="conv_2x2"
        )(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)

        # Optional deeper conv layer
        x = tf.keras.layers.Conv2D(
            128, (1, 1), padding="same", kernel_initializer=initializer, name="conv_1x1"
        )(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)

        x = tf.keras.layers.Dropout(0.15, name="dropout_1")(x)
        # Flatten keeps which statistic and window each feature comes from; global
        # pooling over the (statistics, positions) map would average them away
        x = tf.keras.layers.Flatten()(x)

        x = tf.keras.layers.Dense(128, activation="relu", name="dense_1")(x)
        x = tf.keras.layers.Dropout(0.2, name="dropout_2")(x)

        x = tf.keras.layers.Dense(32, activation="relu", name="dense_2")(x)
        x = tf.keras.layers.Dropout(0.1, name="dropout_3")(x)

        output = tf.keras.layers.Dense(
            num_classes, activation="sigmoid", name="output"
        )(x)

        return output

    fs_cnn = fs.CNN(
            train_data = "yri_sims/fvs.parquet",
            predict_data = "yri_vcfs/fvs_yri.parquet",
            output_folder = "yri_cnn",
    )
    fs_cnn.train(cnn = cnn_finer)


With ``one_dim=True`` the same feature vectors feed 1D models (Conv1D, recurrent or
attention layers) over genomic positions. **A 1D model has no default: pass your own
architecture.** Example: Conv1D with channel attention (squeeze-and-excitation):

.. code-block:: python

    def cnn_flexsweep_conv1d(model_input, num_classes=1):
        """
        Conv1D over spatial positions (steps) with stats as channels,
        followed by channel-wise (per-stat) attention.
        Expects model_input shape: (batch, P, S)
        """

        x = model_input

        # Conv1D over positions (channels_last): output (batch, positions, filters)
        x = tf.keras.layers.Conv1D(128, 3, padding="same", activation="relu")(x)
        x = tf.keras.layers.Conv1D(256, 2, padding="same", activation="relu")(x)

        # Channel attention (Squeeze-and-Excitation)
        se = tf.keras.layers.GlobalAveragePooling1D()(x)
        se = tf.keras.layers.Dense(256, activation="sigmoid")(se)
        se = tf.keras.layers.Reshape((1, 256))(se)
        x  = tf.keras.layers.Multiply()([x, se])

        # Head
        x = tf.keras.layers.GlobalAveragePooling1D()(x)
        x = tf.keras.layers.Dense(256, activation="relu")(x)
        x = tf.keras.layers.Dropout(0.3)(x)
        x = tf.keras.layers.Dense(128, activation="relu")(x)
        x = tf.keras.layers.Dropout(0.15)(x)
        output = tf.keras.layers.Dense(num_classes, activation="sigmoid")(x)

        return output

    fs_cnn_1d = CNN(
            train_data = "yri_sims/fvs.parquet",
            predict_data = "yri_vcfs/fvs_yri.parquet",
            output_folder = "yri_cnn",
    )
    fs_cnn_1d.train(cnn = cnn_flexsweep_conv1d, one_dim = True)
    fs_cnn_1d.predict()  # 1D layout inferred from the trained model

A bidirectional LSTM over positions needs only a different function:

.. code-block:: python

    def bilstm(model_input):
        x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64))(model_input)
        x = tf.keras.layers.Dense(32, activation="relu")(x)
        return tf.keras.layers.Dense(1, activation="sigmoid")(x)

    fs_cnn_1d.train(cnn = bilstm, one_dim = True)


Hyperparameter search
---------------------

Flex-sweep does not run hyperparameter searches itself; it exposes the training settings as
arguments so that any optimizer (Optuna, KerasTuner, a plain grid) can drive them. The
defaults are the values used in the paper.

``CNN.train``: ``learning_rate`` (1e-4, initial value of the cosine-decay-with-restarts
schedule), ``batch_size`` (32), ``patience`` (5) and ``min_delta`` (1e-4) of early stopping
on validation accuracy, and ``epochs`` (1000, maximum). Architecture choices go through
``cnn=`` (see above).

``CNN.train_da``: see :doc:`dann`.

Select settings on the validation set, which ``train`` records in ``history``; keep the test
split for the final evaluation only. With Optuna:

.. code-block:: python

    import optuna
    import flexsweep as fs

    def objective(trial):
        m = fs.CNN(
            train_data="yri_sims/fvs.parquet",
            output_folder=f"hp/trial_{trial.number}",
            seed=1,
        )
        m.train(
            learning_rate=trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True),
            batch_size=trial.suggest_categorical("batch_size", [32, 64, 128]),
            patience=trial.suggest_int("patience", 3, 10),
        )
        return m.history["val_accuracy"].max()

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=30)
    print(study.best_params)

The same pattern works with a plain loop over a grid. Each trial writes its model, test
predictions and training history to its own ``output_folder``.
