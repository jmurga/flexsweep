import time, os

from . import pl, np

import tensorflow as tf
from tensorflow.keras.utils import Sequence
from tensorflow.keras import layers
from tensorflow.keras.saving import register_keras_serializable

import importlib
import glob
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_curve,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    auc,
)
from itertools import product
from collections import defaultdict
from pybedtools import BedTool
from copy import deepcopy


@register_keras_serializable(package="custom", name="masked_bce_fn")
def masked_bce_fn(y_true, y_pred):
    """
    Binary cross-entropy (BCE) with masking for multi-task domain adaptation.

    This loss behaves like standard BCE **except** that examples with label ``-1``
    are **ignored** (masked) and do not contribute to the loss or gradients.
    It enables mixed minibatches where each sample supervises only the relevant head
    (e.g., classifier vs. domain discriminator) while being ignored by the other.

    Parameters
    ----------
    y_true : tf.Tensor
        Ground-truth labels of shape ``(batch, 1)``. For samples that should be
        ignored by this loss, set the label value to ``-1.0``.
    y_pred : tf.Tensor
        Predicted probabilities of shape ``(batch, 1)``.

    Returns
    -------
    tf.Tensor
        A scalar tensor: the mean BCE over **unmasked** examples.
        If no unmasked examples are present in the batch, returns ``0.0``.

    Notes
    -----
    - Mask sentinel is ``-1.0`` (float). Do not use ``-1`` as a valid class label.
    - This function is used for **both** the classifier head (sweep vs. neutral)
      and the domain discriminator head (source vs. target).
    """

    bce = tf.keras.losses.BinaryCrossentropy(reduction=tf.keras.losses.Reduction.NONE)
    mask = tf.not_equal(y_true, -1.0)
    y_true_m = tf.boolean_mask(y_true, mask)
    y_pred_m = tf.boolean_mask(y_pred, mask)
    return tf.cond(
        tf.size(y_true_m) > 0,
        lambda: tf.reduce_mean(bce(y_true_m, y_pred_m)),
        lambda: tf.constant(0.0, tf.float32),
    )


@register_keras_serializable(package="custom", name="GradReverse")
class GradReverse(tf.keras.layers.Layer):
    """
    Gradient Reversal Layer (GRL) with tunable strength ``λ``.

    Forward pass: identity (returns the input unchanged).
    Backward pass: multiplies the incoming gradient by ``-λ``, which
    *reverses* (and scales) gradients flowing into the shared feature extractor.
    This encourages the extractor to learn **domain-invariant** features when
    the GRL feeds a domain classifier.

    Parameters
    ----------
    lambd : float, default=0.0
        Initial GRL strength ``λ``. The effective gradient multiplier is ``-λ``.
        Can be updated during training (e.g., via :class:`GRLRamp`).
    **kw : Any
        Passed to :class:`tf.keras.layers.Layer`.

    Attributes
    ----------
    lambd : tf.Variable
        Non-trainable scalar variable storing the current ``λ`` value. It can be
        modified by callbacks to schedule warm-up or annealing.

    Notes
    -----
    - Serialization: the layer is Keras-serializable and preserves the initial
      ``λ`` in configs. At runtime, the **variable** value may be updated.
    - Typical schedules **warm up** ``λ`` from 0 → 0.4–1.0 over several epochs.

    References
    ----------
    Ganin & Lempitsky (2015), "Unsupervised Domain Adaptation by
    Backpropagation" (DANN/GRL).
    """
    @staticmethod
    @tf.custom_gradient
    def _grl_with_lambda(x, lambd):
        y = tf.identity(x)

        def grad(dy):
            # grad wrt x is -λ * dy; no grad wrt λ
            return -lambd * dy, tf.zeros_like(lambd)

        return y, grad

    def __init__(self, lambd=0.0, **kw):
        super().__init__(**kw)
        # Keep JSON-safe init value for serialization
        self._lambd_init = float(lambd)
        # Non-trainable so you can control it via callback
        self.lambd = tf.Variable(
            self._lambd_init, trainable=False, dtype=tf.float32, name="grl_lambda"
        )

    def call(self, x):
        # Use the staticmethod custom op
        return GradReverse._grl_with_lambda(x, self.lambd)

    # ---- Keras serialization ----
    def get_config(self):
        cfg = super().get_config()
        cfg.update({"lambd": float(self._lambd_init)})
        return cfg

    @classmethod
    def from_config(cls, cfg):
        return cls(**cfg)


class GRLRamp(tf.keras.callbacks.Callback):
    """
    Linear warm-up schedule for GRL strength ``λ``.

    Increases the GRL factor linearly from 0 to ``max_lambda`` over
    ``epochs`` calls to :meth:`on_epoch_begin`. After warm-up, ``λ`` is held
    constant at ``max_lambda``.

    Parameters
    ----------
    grl_layer : GradReverse
        The GRL layer instance whose ``lambd`` variable will be updated.
    max_lambda : float, default=0.5
        Target value for ``λ`` at the end of the warm-up.
    epochs : int, default=50
        Number of warm-up epochs. If total training epochs exceed this value,
        ``λ`` remains fixed thereafter.

    Notes
    -----
    - Warm-up helps stabilize training by letting the classifier learn a useful
      decision surface **before** strong domain-adversarial pressure is applied.
    - Consider tuning ``max_lambda`` and warm-up length based on how quickly the
      domain accuracy approaches ~0.5 (a sign of domain invariance).
    """
    def __init__(self, grl_layer, max_lambda=0.5, epochs=50):
        """
        epochs = number of ramp epochs (not total training epochs).
        After this many epochs, λ will be held at max_lambda.
        """
        super().__init__()
        self.grl_layer = grl_layer
        self.max_lambda = float(max_lambda)
        self.ramp_epochs = int(max(1, epochs))

    def on_epoch_begin(self, epoch, logs=None):
        # linear warmup 0 → max_lambda over `ramp_epochs`, then hold
        if epoch < self.ramp_epochs:
            t = epoch / max(1, self.ramp_epochs - 1)
            lam = self.max_lambda * t
        else:
            lam = self.max_lambda
        self.grl_layer.lambd.assign(lam)


def se_block_2d(x, reduction=8):
    ch = int(x.shape[-1])
    s = tf.keras.layers.GlobalAveragePooling2D()(x)
    s = tf.keras.layers.Dense(max(4, ch // reduction), activation="relu")(s)
    s = tf.keras.layers.Dense(ch, activation="sigmoid")(s)
    s = tf.keras.layers.Reshape((1, 1, ch))(s)
    return tf.keras.layers.Multiply()([x, s])


class CNN:
    """
    Class to build and train a Convolutional Neural Network (CNN) for Flex-sweep.
    It loads/reshapes Flex-sweep feature vectors, trains, evaluates and predicts, including
    domain-adaptation extension.

    Attributes
    ----------
    train_data : str | pl.DataFrame | None
        Path to training parquet/CSV (or a Polars DataFrame).
    source_data : str | None
        Path to *source* (labeled) parquet for domain adaptation.
    target_data : str | None
        Path to *target/empirical* parquet for domain adaptation (unlabeled).
    predict_data : str | pl.DataFrame | None
        Path/DataFrame with samples to predict (standard supervised path).
    valid_data : Any
        (Reserved) Optional separate validation set path/DF (unused).
    output_folder : str | None
        Directory where models, figures and predictions are written.
    normalize : bool
        If True, apply a Keras `Normalization` layer (fit on train only).
    model : tf.keras.Model | str | None
        A compiled Keras model or a path to a saved model.
    num_stats : int
        Number of per-window statistics used as channels. Default 11.
    center : np.ndarray[int]
        Center coordinates (bp) used to index columns; defaults to 500k..700k step 10k.
    windows : np.ndarray[int]
        Window sizes used to index columns; default [50k, 100k, 200k, 500k, 1M].
    train_split : float
        Fraction of data used for training (rest split equally into val/test).
    gpu : bool
        If False, disable CUDA via `CUDA_VISIBLE_DEVICES=-1`.
    tf : module | None
        TensorFlow module, set by :meth:`check_tf`.
    history : pl.DataFrame | None
        Training history after :meth:`train` / :meth:`train_da`.
    prediction : pl.DataFrame | None
        Latest prediction table produced by :meth:`train` or :meth:`predict*`.
    """

    def __init__(
        self,
        train_data=None,
        source_data=None,
        target_data=None,
        predict_data=None,
        valid_data=None,
        output_folder=None,
        normalize=False,
        model=None,
    ):
        """
        Initialize a CNN runner.

        Parameters
        ----------
        train_data : str | pl.DataFrame | None
            Path to training data (`.parquet`, `.csv[.gz]`) or Polars DataFrame.
        source_data : str | None
            Path to labeled source parquet for domain adaptation.
        target_data : str | None
            Path to unlabeled empirical/target parquet for domain adaptation.
        predict_data : str | pl.DataFrame | None
            Path/DataFrame for inference in :meth:`predict`.
        valid_data : Any, optional
            Reserved for a future explicit validation split (unused).
        output_folder : str | None
            Output directory for artifacts (models, plots, CSVs).
        normalize : bool, default=False
            If True, fit a `Normalization` layer on training features.
        model : tf.keras.Model | str | None
            Prebuilt Keras model or path to a saved model.

        Notes
        -----
        Defaults assume 11 statistics × 5 windows × 21 centers
        organized in column names like: ``{stat}_{window}_{center}``.
        """
        # self.sweep_data = sweep_data
        self.normalize = normalize
        self.train_data = train_data
        self.predict_data = predict_data
        self.test_train_data = None
        self.output_folder = output_folder
        self.output_prediction = "predictions.txt"
        self.num_stats = 11
        self.center = np.arange(5e5, 7e5 + 1e4, 1e4).astype(int)
        self.windows = np.array([50000, 100000, 200000, 500000, 1000000])
        self.train_split = 0.8
        self.prediction = None
        self.history = None
        self.model = model
        self.gpu = True
        self.tf = None
        self.source_data = source_data
        self.target_data = target_data

    def check_tf(self):
        """
        Import TensorFlow (optionally forcing CPU).

        Returns
        -------
        module
            Imported ``tensorflow`` module.

        Notes
        -----
        If ``self.gpu`` is ``False``, the environment variable
        ``CUDA_VISIBLE_DEVICES`` is set to ``-1`` **before** importing TF.
        """
        if self.gpu is False:
            os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        tf = importlib.import_module("tensorflow")
        return tf

    def cnn_flexsweep(self, model_input, num_classes=1):
        """
        Flex-sweep CNN feature extractor + classifier head.

        Parameters
        ----------
        model_input : tf.keras.layers.Input
            Keras input tensor with shape ``(W, C, S)`` where
            ``W=len(self.windows)``, ``C=len(self.center)``, ``S=self.num_stats)``.
        num_classes : int, default=1
            Number of output classes. This method currently returns a single
            sigmoid unit for binary classification.

        Returns
        -------
        tf.Tensor
            Output tensor with shape ``(None, 1)`` and sigmoid activation.

        Notes
        -----
        Architecture uses three parallel Conv2D branches:
        a 3×3 stack, a 2×2 stack with dilation (1,3), and a 2×2 stack with
        dilations (5,1) then (1,5). Their flattened features are concatenated
        and passed through dense layers to a single sigmoid.
        """

        # 3x3 layer
        initializer = tf.keras.initializers.HeNormal()
        layer1 = tf.keras.layers.Conv2D(
            64,
            3,
            padding="same",
            name="convlayer1_1",
            kernel_initializer=initializer,
        )(model_input)
        layer1 = tf.keras.layers.ReLU()(layer1)
        layer1 = tf.keras.layers.Conv2D(
            128,
            3,
            padding="same",
            name="convlayer1_2",
            kernel_initializer=initializer,
        )(layer1)
        layer1 = tf.keras.layers.ReLU()(layer1)
        layer1 = tf.keras.layers.Conv2D(256, 3, padding="same", name="convlayer1_3")(
            layer1
        )
        layer1 = tf.keras.layers.ReLU()(layer1)
        layer1 = tf.keras.layers.MaxPooling2D(
            pool_size=3, name="poollayer1", padding="same"
        )(layer1)
        layer1 = tf.keras.layers.Dropout(0.15, name="droplayer1")(layer1)
        layer1 = tf.keras.layers.Flatten(name="flatlayer1")(layer1)

        # 2x2 layer with 1x3 dilation
        layer2 = tf.keras.layers.Conv2D(
            64,
            2,
            dilation_rate=[1, 3],
            padding="same",
            name="convlayer2_1",
            kernel_initializer=initializer,
        )(model_input)
        layer2 = tf.keras.layers.ReLU()(layer2)
        layer2 = tf.keras.layers.Conv2D(
            128,
            2,
            dilation_rate=[1, 3],
            padding="same",
            name="convlayer2_2",
            kernel_initializer=initializer,
        )(layer2)
        layer2 = tf.keras.layers.ReLU()(layer2)
        layer2 = tf.keras.layers.Conv2D(
            256, 2, dilation_rate=[1, 3], padding="same", name="convlayer2_3"
        )(layer2)
        layer2 = tf.keras.layers.ReLU()(layer2)
        layer2 = tf.keras.layers.MaxPooling2D(pool_size=2, name="poollayer2")(layer2)
        layer2 = tf.keras.layers.Dropout(0.15, name="droplayer2")(layer2)
        layer2 = tf.keras.layers.Flatten(name="flatlayer2")(layer2)

        # 2x2 with 1x5 dilation
        layer3 = tf.keras.layers.Conv2D(
            64,
            2,
            # dilation_rate=[1, 5],
            dilation_rate=[5, 1],
            padding="same",
            name="convlayer4_1",
            kernel_initializer=initializer,
        )(model_input)
        layer3 = tf.keras.layers.ReLU()(layer3)
        layer3 = tf.keras.layers.Conv2D(
            128,
            2,
            dilation_rate=[1, 5],
            padding="same",
            name="convlayer4_2",
            kernel_initializer=initializer,
        )(layer3)
        layer3 = tf.keras.layers.ReLU()(layer3)
        layer3 = tf.keras.layers.Conv2D(
            256, 2, dilation_rate=[1, 5], padding="same", name="convlayer4_3"
        )(layer3)
        layer3 = tf.keras.layers.ReLU()(layer3)
        layer3 = tf.keras.layers.MaxPooling2D(pool_size=2, name="poollayer3")(layer3)
        layer3 = tf.keras.layers.Dropout(0.15, name="droplayer3")(layer3)
        layer3 = tf.keras.layers.Flatten(name="flatlayer3")(layer3)

        # concatenate convolution layers
        concat = tf.keras.layers.concatenate([layer1, layer2, layer3])
        concat = tf.keras.layers.Dense(512, name="512dense", activation="relu")(concat)
        concat = tf.keras.layers.Dropout(0.2, name="dropconcat1")(concat)
        concat = tf.keras.layers.Dense(128, name="last_dense", activation="relu")(
            concat
        )
        concat = tf.keras.layers.Dropout(0.2 / 2, name="dropconcat2")(concat)
        output = tf.keras.layers.Dense(
            1,
            name="out_dense",
            activation="sigmoid",
            kernel_initializer=initializer,
        )(concat)

        return output

    def load_training_data(self, _stats=None, w=None, n=None, one_dim = False):
        """
        Load and reshape training/validation/test tensors from table-format features.

        Parameters
        ----------
        _stats : list[str] | None
            List of statistic base names to include (e.g., ``["ihs","nsl",...]``).
            If None, you must pass an explicit list later in :meth:`train`.
        w : int | list[int] | None
            Restrict to specific window sizes (e.g., 100000 or [50000,100000]).
            Columns are selected by regex suffix ``_{window}``.
        n : int | None
            Optional number of rows to sample from parquet.
        one_dim : bool, default=False
            If True, flatten spatial grid to ``(W*C, S)`` for 1D models.

        Returns
        -------
        tuple
            ``(X_train, X_test, Y_train, Y_test, X_valid, Y_valid)`` with shapes:

            - if ``one_dim`` is False:
              ``X_*`` → ``(N, W, C, S)``, labels are 0/1.
            - if ``one_dim`` is True:
              ``X_*`` → ``(N, W*C, S)``.

        Raises
        ------
        AssertionError
            If ``train_data`` is missing or has an unsupported extension.

        Notes
        -----
        Any ``model`` value not equal to ``"neutral"`` is coerced to ``"sweep"``.
        """

        assert self.train_data is not None, "Please input training data"

        assert (
            "txt" in self.train_data
            or "csv" in self.train_data
            or self.train_data.endswith(".parquet")
        ), "Please save your dataframe as CSV or parquet"

        if isinstance(self.train_data, pl.DataFrame):
            pass
        elif self.train_data.endswith(".gz"):
            tmp = pl.read_csv(self.train_data, separator=",")
        elif self.train_data.endswith(".parquet"):
            tmp = pl.read_parquet(self.train_data)

            if n is not None:
                tmp = tmp.sample(n)

        tmp = tmp.with_columns(
            pl.when((pl.col("model") != "neutral"))
            .then(pl.lit("sweep"))
            .otherwise(pl.lit("neutral"))
            .alias("model")
        )

        if w is not None:
            try:
                self.center = np.array([int(w)])
                tmp = tmp.select(
                    "iter", "s", "t", "f_i", "f_t", "model", f"^*._{int(w)}$"
                )
            except:
                self.center = np.sort(np.array(w).astype(int))
                _tmp = []
                _h = tmp.select("iter", "s", "t", "f_i", "f_t", "model")
                for window in self.center:
                    _tmp.append(tmp.select(f"^*._{int(window)}$"))
                tmp = pl.concat(_tmp, how="horizontal")
                tmp = pl.concat([_h, tmp], how="horizontal")

        sweep_parameters = tmp.filter("model" != "neutral").select(tmp.columns[:5])

        stats = []

        if _stats is not None:
            stats = stats + _stats

        train_stats = []
        for i in stats:
            train_stats.append(tmp.select(pl.col(f"^{i}_[0-9]+_[0-9]+$")))

        train_stats = pl.concat(train_stats, how="horizontal")
        train_stats = pl.concat(
            [tmp.select("model", "iter", "s", "f_i", "f_t", "t"), train_stats],
            how="horizontal",
        )

        y = train_stats.select(
            ((~pl.col("model").str.contains("neutral")).cast(pl.Int8)).alias(
                "neutral_flag"
            )
        )["neutral_flag"].to_numpy()

        test_split = round(1 - self.train_split, 2)

        (
            X_train,
            X_test,
            Y_train,
            y_test,
        ) = train_test_split(train_stats, y, test_size=test_split, shuffle=True)

        X_train = (
            X_train.select(train_stats.columns[6:])
            .to_numpy()
            .reshape(
                X_train.shape[0],
                self.num_stats,
                self.windows.size * self.center.size,
                1,
            )
        )

        X_valid, X_test, Y_valid, Y_test = train_test_split(
            X_test, y_test, test_size=0.5
        )

        X_test_params = X_test.select(X_test.columns[:6])
        X_test = (
            X_test.select(train_stats.columns[6:])
            .to_numpy()
            .reshape(
                X_test.shape[0],
                self.num_stats,
                self.windows.size * self.center.size,
                1,
            )
        )
        X_valid = (
            X_valid.select(train_stats.columns[6:])
            .to_numpy()
            .reshape(
                X_valid.shape[0],
                self.num_stats,
                self.windows.size * self.center.size,
                1,
            )
        )

        # Normalization on training data
        if self.normalize:
            self.stat_norm = tf.keras.layers.Normalization(axis=-1, name="stat_norm")
            self.stat_norm.adapt(X_train)  # learns mean/std from training set only


        # Input stats as channel to improve performance
        # Avoiding changes stats order

        X_train = X_train.reshape(
            X_train.shape[0], self.windows.size, self.center.size, self.num_stats
        )
        X_test = X_test.reshape(
            X_test.shape[0], self.windows.size, self.center.size, self.num_stats
        )
        X_valid = X_valid.reshape(
            X_valid.shape[0], self.windows.size, self.center.size, self.num_stats
        )

        X_test = X_test.reshape(
            X_test.shape[0], self.windows.size, self.center.size, self.num_stats
        )

        if one_dim:

            X_train = X_train.reshape(
                -1, self.windows.size * self.center.size, self.num_stats
            )
            X_valid = X_valid.reshape(
                -1, self.windows.size * self.center.size, self.num_stats
            )
            X_test = X_test.reshape(
                -1, self.windows.size * self.center.size, self.num_stats
            )

        self.test_train_data = [X_test, X_test_params, Y_test]

        return (
            X_train,
            X_test,
            Y_train,
            Y_test,
            X_valid,
            Y_valid,
        )

    def train(self, _iter=1, _stats=None, w=None, cnn=None, one_dim = False):
        """
        Train a CNN on flex-sweep tensors with early stopping and checkpoints.

        Parameters
        ----------
        _iter : int, default=1
            Tag for output naming (kept for backwards compatibility).
        _stats : list[str] | None
            Statistic base names. If None, defaults to the 11 flex-sweep stats.
        w : int | list[int] | None
            Window size(s) to select (see :meth:`load_training_data`).
        cnn : callable | None
            A function mapping a Keras input tensor to an output tensor.
            Defaults to :meth:`cnn_flexsweep`. If ``one_dim=True``, you must
            provide a compatible 1D architecture.
        one_dim : bool, default=False
            If True, uses flattened ``(W*C, S)`` inputs.

        Returns
        -------
        pl.DataFrame
            Predictions on the held-out test set with columns:
            ``['model','f_i','f_t','s','t','predicted_model','prob_sweep','prob_neutral']``.

        Notes
        -----
        - Optimizer: Adam with cosine-restarts schedule.
        - Loss: Binary cross-entropy with label smoothing (0.05).
        - Early stopping monitors validation AUC (restore best weights).
        - Saves ``model.keras`` to ``output_folder`` if provided.
        """


        if one_dim:
            assert cnn is not None, "Please input a 1D CNN architecture"


        # Default stats
        if _stats is None:
            _stats = [
                "ihs",
                "nsl",
                "isafe",
                "hapdaf_o",
                "hapdaf_s",
                "dind",
                "s_ratio",
                "low_freq",
                "high_freq",
                "h12",
                "haf",
            ]

        self.num_stats = len(_stats)

        # Default CNN
        if cnn is None:
            cnn = self.cnn_flexsweep

        (
            X_train,
            X_test,
            Y_train,
            Y_test,
            X_valid,
            Y_valid,
        ) = self.load_training_data(w=w, _stats=_stats, one_dim = one_dim)

        input_shape = (self.center.size, self.windows.size, self.num_stats)

        # put model together
        input_to_model = tf.keras.Input(X_train.shape[1:])
        batch_size = 32

        train_dataset = (
            tf.data.Dataset.from_tensor_slices((X_train, Y_train))
            # tf.data.Dataset.from_tensor_slices(((X_train_ld,X_train_sfs,X_train_div), Y_train))
            .shuffle(10000)
            .batch(batch_size)
            .prefetch(tf.data.AUTOTUNE)
        )

        valid_dataset = (
            tf.data.Dataset.from_tensor_slices((X_valid, Y_valid))
            # tf.data.Dataset.from_tensor_slices(((X_valid_ld,X_valid_sfs,X_valid_div), Y_valid))
            .batch(batch_size).prefetch(tf.data.AUTOTUNE)
        )
        test_dataset = (
            tf.data.Dataset.from_tensor_slices((X_test, Y_test))
            # tf.data.Dataset.from_tensor_slices(((X_test_ld,X_test_sfs,X_test_div), Y_test))
            .batch(batch_size).prefetch(tf.data.AUTOTUNE)
        )

        model = tf.keras.models.Model(
            inputs=[input_to_model], outputs=[cnn(input_to_model)]
        )

        model_path = f"{self.output_folder}/model.keras"

        metrics_measures = [
            tf.keras.metrics.BinaryAccuracy(name="val_accuracy"),
            tf.keras.metrics.Precision(name="val_precision"),
            tf.keras.metrics.AUC(name="roc", curve="ROC"),
        ]

        lr_decayed_fn = tf.keras.optimizers.schedules.CosineDecayRestarts(
            initial_learning_rate=1e-4, first_decay_steps=300
        )
        opt_adam = tf.keras.optimizers.Adam(
            learning_rate=lr_decayed_fn, epsilon=0.0000001, amsgrad=True
        )

        custom_loss = tf.keras.losses.BinaryCrossentropy(label_smoothing=0.05)
        # Keep only one compilation
        model.compile(
            optimizer=opt_adam,
            loss=custom_loss,
            metrics=[
                tf.keras.metrics.BinaryAccuracy(name="accuracy"),
                tf.keras.metrics.AUC(name="auc", curve="ROC"),
                tf.keras.metrics.Precision(name="precision"),
            ],
        )
        earlystop = tf.keras.callbacks.EarlyStopping(
            monitor="val_auc",
            min_delta=0.0001,
            patience=10,
            verbose=2,
            mode="max",
            restore_best_weights=True,
        )

        checkpoint = tf.keras.callbacks.ModelCheckpoint(
            model_path,
            monitor="val_auc",
            verbose=2,
            save_best_only=True,
            mode="max",
        )

        callbacks_list = [checkpoint, earlystop]

        start = time.time()

        history = model.fit(
            train_dataset,
            epochs=1000,
            validation_data=valid_dataset,
            callbacks=callbacks_list,
        )

        val_score = model.evaluate(
            valid_dataset,
            batch_size=32,
            steps=len(Y_valid) // 32,
        )
        test_score = model.evaluate(
            test_dataset,
            batch_size=32,
            steps=len(Y_test) // 32,
        )

        train_score = model.evaluate(
            train_dataset,
            batch_size=32,
            steps=len(Y_train) // 32,
        )
        self.model = model

        df_history = pl.DataFrame(history.history)
        self.history = df_history

        print(
            "Training and testing model took {} seconds".format(
                round(time.time() - start, 3)
            )
        )

        if self.output_folder is not None:
            model.save(model_path)

        # ROC curves and confusion matrix
        _output_prediction = self.output_folder + "/" + self.output_prediction

        test_X, test_X_params, test_Y = deepcopy(self.test_train_data)

        # test_X = test_X.reshape(
        #     test_X.shape[0], self.windows.size, self.center.size, self.num_stats
        # )
        preds = model.predict(test_X)

        preds = np.column_stack([1 - preds, preds])
        predictions = np.argmax(preds, axis=1)
        prediction_dict = {
            0: "neutral",
            1: "sweep",
        }
        predictions_class = np.vectorize(prediction_dict.get)(predictions)
        df_prediction = pl.concat(
            [
                test_X_params.select("model", "f_i", "f_t", "s", "t"),
                pl.DataFrame(
                    {
                        "predicted_model": predictions_class,
                        "prob_sweep": preds[:, 1],
                        "prob_neutral": preds[:, 0],
                    }
                ),
            ],
            how="horizontal",
        )

        y_true = (test_X_params["model"] != "neutral").to_numpy().astype(int)

        fpr, tpr, thresh = roc_curve(y_true, preds[:, 1])

        # Pick FPR threshold, FPR <= max_fpr, maximize TPR
        max_fpr = 0.9
        valid = np.where(fpr <= max_fpr)[0]
        if len(valid) > 0:
            idx = valid[np.argmax(tpr[valid])]
        else:
            idx = np.argmin(fpr)
        best_thresh = thresh[idx]

        prediction_dict = {
            0: "neutral",
            1: "sweep",
        }
        predictions_class = np.vectorize(prediction_dict.get)(predictions)
        # predictions_class = np.where(preds[:, 1] >= best_thresh, "sweep", "neutral")

        df_prediction = pl.concat(
            [
                test_X_params.select("model", "f_i", "f_t", "s", "t"),
                pl.DataFrame(
                    {
                        "predicted_model": predictions_class,
                        "prob_sweep": preds[:, 1],
                        "prob_neutral": preds[:, 0],
                    }
                ),
            ],
            how="horizontal",
        )

        self.prediction = df_prediction
        self.roc_curve()

        if self.output_folder is not None:
            df_prediction.write_csv(_output_prediction)

        return df_prediction

    def predict(self, _stats=None, w=None, simulations = False, _iter=1):
        """
        Predict on a feature table using a trained model.

        Parameters
        ----------
        _stats : list[str] | None
            Statistic base names to include; defaults to the 11 flex-sweep stats.
        w : int | list[int] | None
            Window size(s) to select.
        simulations : bool, default=False
            Reserved flag; has no effect here.
        _iter : int, default=1
            Tag for output naming (unused).

        Returns
        -------
        pl.DataFrame
            Sorted predictions per region with columns:
            ``['chr','start','end','f_i','f_t','s','t','predicted_model','prob_sweep','prob_neutral']``.

        Raises
        ------
        AssertionError
            If ``self.model`` is not set or ``predict_data`` is missing.

        Notes
        -----
        If ``self.model`` is a string path, it is loaded via
        ``tf.keras.models.load_model``.
        """

        if _stats is None:
            _stats = ["ihs", "nsl", "isafe","hapdaf_o", "hapdaf_s","dind", "s_ratio", "low_freq", "high_freq",'h12','haf']

        self.num_stats = len(_stats)

        assert self.model is not None, "Please input the CNN trained model"

        if isinstance(self.model, str):
            model = tf.keras.models.load_model(self.model)
        else:
            model = self.model

        # import data to predict
        assert self.predict_data is not None, "Please input training data"
        assert (
            isinstance(self.predict_data, pl.DataFrame)
            or "txt" in self.predict_data
            or "csv" in self.predict_data
            or self.predict_data.endswith(".parquet")
        ), "Please input a parquet pl.DataFrame"


        df_test = pl.read_parquet(self.predict_data)

        df_test = df_test.with_columns(
            pl.when((pl.col("model") != "neutral"))
            .then(pl.lit("sweep"))
            .otherwise(pl.lit("neutral"))
            .alias("model")
        )
        regions = df_test["iter"].to_numpy()

        stats = []
        if _stats is not None:
            stats = stats + _stats
        test_stats = []

        for i in stats:
            test_stats.append(df_test.select(pl.col(f"^{i}_[0-9]+_[0-9]+$")))

        X_test = pl.concat(test_stats, how="horizontal")

        if w is not None:
            try:
                self.center = np.array([int(w)])
                X_test = X_test.select(f"^*._{int(w)}$")
            except:
                self.center = np.sort(np.array(w).astype(int))
                _X_test = []
                for window in self.center:
                    _X_test.append(X_test.select(f"^*._{int(window)}$"))
                X_test = pl.concat(_X_test, how="horizontal")

        test_X_params = df_test.select("model", "iter", "s", "f_i", "f_t", "t")



        test_X = X_test.to_numpy().reshape(
            X_test.shape[0], self.windows.size, self.center.size, self.num_stats
        )

        preds = model.predict(test_X)

        preds = np.column_stack([1 - preds, preds])
        predictions = np.argmax(preds, axis=1)
        prediction_dict = {
            0: "neutral",
            1: "sweep",
        }
        predictions_class = np.vectorize(prediction_dict.get)(predictions)
        df_prediction = pl.concat(
            [
                test_X_params.select("model", "f_i", "f_t", "s", "t"),
                pl.DataFrame(
                    {
                        "predicted_model": predictions_class,
                        "prob_sweep": preds[:, 1],
                        "prob_neutral": preds[:, 0],
                    }
                ),
            ],
            how="horizontal",
        )

        # Same folder custom fvs name based on input VCF.
        _output_prediction = f"{self.output_folder}/{os.path.basename(self.predict_data).replace("fvs_", "").replace(".parquet", "_predictions.txt")}"

        df_prediction = df_prediction.with_columns(pl.Series("region", regions))
        chr_start_end = np.array(
            [item.replace(":", "-").split("-") for item in regions]
        )

        df_prediction = df_prediction.with_columns(
            pl.Series("chr", chr_start_end[:, 0]),
            pl.Series("start", chr_start_end[:, 1], dtype=pl.Int64),
            pl.Series("end", chr_start_end[:, 2], dtype=pl.Int64),
            pl.Series(
                "nchr",
                pl.Series(chr_start_end[:, 0]).str.replace("chr", "").cast(int),
            ),
        )
        df_prediction = df_prediction.sort("nchr", "start").select(
            pl.exclude("region", "iter", "model", "nchr")
        )

        if self.output_folder is not None:
            df_prediction.write_csv(_output_prediction)

        df_prediction = df_prediction.select(['chr', 'start', 'end', 'f_i','f_t', 's', 't', 'predicted_model', 'prob_sweep', 'prob_neutral'])

        return df_prediction

    def roc_curve(self, _iter=1):
        """
        Build ROC curve, confusion matrix and training-history plots.

        Parameters
        ----------
        _iter : int, default=1
            Tag for output naming (kept for compatibility).

        Returns
        -------
        tuple[matplotlib.figure.Figure, matplotlib.figure.Figure]
            ``(plot_roc, plot_history)`` figures. Confusion matrix is also saved
            to ``confusion_matrix.svg`` when ``output_folder`` is set.

        Notes
        -----
        - AUC is computed treating ``'sweep'`` as the positive class.
        - The method expects :attr:`prediction` to contain the latest
          predictions including ``prob_sweep``.
        """

        import matplotlib.pyplot as plt

        if isinstance(self.prediction, str):
            pred_data = pl.read_csv(self.prediction)
        else:
            pred_data = self.prediction

        # Create confusion dataframe
        confusion_data = pred_data.group_by(["model", "predicted_model"]).agg(
            pl.len().alias("n")
        )

        expected_combinations = pl.DataFrame(
            {
                "model": ["sweep", "sweep", "neutral", "neutral"],
                "predicted_model": ["sweep", "neutral", "neutral", "sweep"],
            }
        )

        confusion_data = expected_combinations.join(
            confusion_data, on=["model", "predicted_model"], how="left"
        ).fill_null(
            0
        )  # Fill missing values with 0
        # Adding the "true_false" column
        confusion_data = confusion_data.with_columns(
            pl.when(
                (pl.col("model") == pl.col("predicted_model"))
                & (pl.col("model") == "neutral")
            )
            .then(pl.lit("true_negative"))  # Explicit literal for Polars
            .when(
                (pl.col("model") == pl.col("predicted_model"))
                & (pl.col("model") == "sweep")
            )
            .then(pl.lit("true_positive"))  # Explicit literal for Polars
            .when(
                (pl.col("model") != pl.col("predicted_model"))
                & (pl.col("model") == "neutral")
            )
            .then(pl.lit("false_positive"))  # Explicit literal for Polars
            .otherwise(pl.lit("false_negative"))  # Explicit literal for Polars
            .alias("true_false")
        )

        confusion_pivot = confusion_data.pivot(
            values="n", index=None, on="true_false", aggregate_function="sum"
        ).fill_null(0)

        # Copying the pivoted data (optional as Polars is immutable)
        rate_data = confusion_pivot.select(
            ["false_negative", "false_positive", "true_negative", "true_positive"]
        ).sum()

        # Compute the required row sums for normalization
        required_cols = [
            "false_negative",
            "false_positive",
            "true_negative",
            "true_positive",
        ]
        for col in required_cols:
            if col not in rate_data.columns:
                rate_data[col] = 0

        # Calculate row sums for normalization
        rate_data = rate_data.with_columns(
            (pl.col("false_negative") + pl.col("true_positive")).alias("sum_fn_tp")
        )
        rate_data = rate_data.with_columns(
            (pl.col("false_positive") + pl.col("true_negative")).alias("sum_fp_tn")
        )

        # Compute normalized rates
        rate_data = rate_data.with_columns(
            (pl.col("false_negative") / pl.col("sum_fn_tp")).alias("false_negative"),
            (pl.col("false_positive") / pl.col("sum_fp_tn")).alias("false_positive"),
            (pl.col("true_negative") / pl.col("sum_fp_tn")).alias("true_negative"),
            (pl.col("true_positive") / pl.col("sum_fn_tp")).alias("true_positive"),
        )

        # Replace NaN values with 0
        rate_data = rate_data.with_columns(
            [
                pl.col("false_negative").fill_null(0).alias("false_negative"),
                pl.col("false_positive").fill_null(0).alias("false_positive"),
                pl.col("true_negative").fill_null(0).alias("true_negative"),
                pl.col("true_positive").fill_null(0).alias("true_positive"),
            ]
        )

        # Calculate accuracy and precision
        rate_data = rate_data.with_columns(
            [
                (
                    (pl.col("true_positive") + pl.col("true_negative"))
                    / (
                        pl.col("true_positive")
                        + pl.col("true_negative")
                        + pl.col("false_positive")
                        + pl.col("false_negative")
                    )
                ).alias("accuracy"),
                (
                    pl.col("true_positive")
                    / (pl.col("true_positive") + pl.col("false_positive"))
                )
                .fill_null(0)
                .alias("precision"),
            ]
        )

        # Compute ROC AUC and prepare roc_data. Set 'sweep' as the positive class
        pred_rate_auc_data = pred_data.clone().with_columns(
            pl.col("model").cast(pl.Categorical).alias("model")
        )

        # Calculate ROC AUC
        roc_auc_value = roc_auc_score(
            (pred_rate_auc_data["model"] == "sweep").cast(int),
            pred_rate_auc_data["prob_sweep"].cast(float),
        )

        # Create roc_data DataFrame
        roc_data = pl.DataFrame({"AUC": [roc_auc_value]})

        rate_roc_data = pl.concat([pl.DataFrame(rate_data), roc_data], how="horizontal")

        first_row_values = rate_roc_data.row(0)

        pred_rate_auc_data = pred_rate_auc_data.with_columns(
            [
                pl.lit(value).alias(col)
                for col, value in zip(rate_roc_data.columns, first_row_values)
            ]
        )

        # Compute ROC curve using sklearn
        fpr, tpr, thresholds = roc_curve(
            (pred_rate_auc_data["model"] == "sweep").cast(int),
            pred_rate_auc_data["prob_sweep"].cast(float),
        )

        roc_df = pl.DataFrame({"false_positive_rate": fpr, "sensitivity": tpr})

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(
            roc_df["false_positive_rate"],
            roc_df["sensitivity"],
            color="orange",
            linewidth=2,
            label="ROC Curve",
        )
        ax.plot(
            [0, 1], [0, 1], color="grey", linestyle="--"
        )  # Diagonal line for reference
        ax.set_xlabel("false positive rate")
        ax.set_ylabel("power")
        ax.set_title("ROC Curve")
        ax.axis("equal")  # Equivalent to coord_equal in ggplot
        ax.grid(True, which="both", linestyle="--", linewidth=0.5)
        ax.legend()
        fig.tight_layout()
        plot_roc = fig

        ## History
        # Load and preprocess the data
        history_data = self.history
        h = history_data.select(
            ["loss", "val_loss", "accuracy", "val_accuracy"]
        ).clone()

        h = h.with_columns((pl.arange(0, h.height) + 1).alias("epoch"))

        h_melted = h.unpivot(
            index=["epoch"],
            on=["loss", "val_loss", "accuracy", "val_accuracy"],
            variable_name="metric_name",
            value_name="metric_val",
        )

        line_styles = {
            "loss": "-",
            "val_loss": "--",
            "accuracy": "-",
            "val_accuracy": "--",
        }
        colors = {
            "loss": "orange",
            "val_loss": "orange",
            "accuracy": "blue",
            "val_accuracy": "blue",
        }

        fig, ax = plt.subplots(figsize=(10, 6))

        for group_name, group_df in h_melted.group_by("metric_name"):
            ax.plot(
                group_df["epoch"].to_numpy(),
                group_df["metric_val"].to_numpy(),
                label=group_name[0],
                linestyle=line_styles[group_name[0]],
                color=colors[group_name[0]],
                linewidth=2,
            )
        ax.set_title("History")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Value")
        ax.tick_params(axis="both", labelsize=10)
        ax.grid(True)
        ax.legend(title="", loc="upper right")

        plot_history = fig

        ##############
        cm = confusion_matrix(
            pred_data["model"],
            pred_data["predicted_model"],
            labels=["neutral", "sweep"],
            normalize="true",
        )
        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm, display_labels=["neutral", "sweep"]
        )
        cm_plot = disp.plot(cmap="Blues")

        print(cm)

        if self.output_folder is not None:
            plt.savefig(self.output_folder + "/confusion_matrix.svg")
            plt.close()
            plot_roc.savefig(self.output_folder + f"/roc_curve.svg")
            plot_history.savefig(
                # self.output_folder + f"/train_history_{ihs}_{_iter}.svg"
                self.output_folder
                + f"/train_history.svg"
            )
        return plot_roc, plot_history

    def _select_stats_matrix_like_old(self, df: pl.DataFrame, stats: list[str], w=None):
        # Standardize model: anything not 'neutral' -> 'sweep'
        df = df.with_columns(
            pl.when(pl.col("model") != "neutral")
            .then(pl.lit("sweep"))
            .otherwise(pl.lit("neutral"))
            .alias("model")
        )

        # (optional) keep only requested center(s) like your old code
        if w is not None:
            try:
                self.center = np.array([int(w)])
                df = df.select(
                    "iter", "s", "t", "f_i", "f_t", "model", f"^*._{int(w)}$"
                )
            except:
                self.center = np.sort(np.array(w).astype(int))
                _tmp = []
                _h = df.select("iter", "s", "t", "f_i", "f_t", "model")
                for ww in self.center:
                    _tmp.append(df.select(f"^*._{int(ww)}$"))
                df = pl.concat(_tmp, how="horizontal")
                df = pl.concat([_h, df], how="horizontal")

        # Your original regex selection per stat
        blocks = []
        for s in stats:
            blk = df.select(pl.col(f"^{s}_[0-9]+_[0-9]+$"))

            # sort columns deterministically by (window, center)
            cols = blk.columns

            def key(col):
                base, a, b = col.rsplit("_", 2)  # safe for stat names with underscores
                a = int(a)
                b = int(b)
                # try to infer which token is window vs center
                if (a in set(self.windows.tolist())) and (
                    b in set(self.center.tolist())
                ):
                    wv, cv = a, b
                elif (a in set(self.center.tolist())) and (
                    b in set(self.windows.tolist())
                ):
                    wv, cv = b, a
                else:
                    # fallback: assume {stat}_{center}_{window}
                    cv, wv = a, b
                return (wv, cv)

            cols_sorted = sorted(cols, key=key)
            blocks.append(blk.select(cols_sorted))

        X = pl.concat(blocks, how="horizontal")
        y = (df["model"] != "neutral").cast(pl.Int8).to_numpy().astype(np.float32)
        params = df.select("iter", "s", "t", "f_i", "f_t", "model")

        # reshape to (N, W, C, S)
        N = df.height
        X = (
            X.to_numpy()
            .reshape(N, self.windows.size, self.center.size, len(stats))
            .astype(np.float32)
        )

        return X, y, params

    def load_da_data_sims(
        self,
        _stats=None,
        w=None,
        n_src=None,
        n_tgt=None,
        test_size=0.20,
        val_size=0.10,
    ):
        assert (
            self.source_data is not None and self.target_data is not None
        ), "Set source_data and target_data"
        assert self.source_data.endswith(".parquet") and self.target_data.endswith(
            ".parquet"
        )

        src_df = pl.read_parquet(self.source_data)
        tgt_df = pl.read_parquet(self.target_data)

        if n_src is not None:
            src_df = src_df.sample(n_src)
        if n_tgt is not None:
            tgt_df = tgt_df.sample(n_tgt)

        # stats comes from caller (like your old loader)
        stats = []
        if _stats is not None:
            stats = stats + _stats

        # Build tensors exactly like your old code
        X_src, y_src, _ = self._select_stats_matrix_like_old(src_df, stats, w=w)
        X_tgt, y_tgt, tgt_params = self._select_stats_matrix_like_old(
            tgt_df, stats, w=w
        )

        # Split target into train/val/test
        from sklearn.model_selection import train_test_split

        X_t_tr, X_t_te, y_t_tr, y_t_te, p_tr, p_te = train_test_split(
            X_tgt,
            y_tgt,
            tgt_params,
            test_size=test_size,
            stratify=y_tgt,
        )
        val_frac = val_size / (1.0 - test_size)
        X_t_tr, X_t_va, y_t_tr, y_t_va, p_tr, p_va = train_test_split(
            X_t_tr, y_t_tr, p_tr, test_size=val_frac, stratify=y_t_tr
        )

        # keep for training/prediction
        self.da_data = {
            "stats": stats,
            "X_src": X_src,
            "y_src": y_src,
            "X_tgt_train": X_t_tr,
            "y_tgt_train": y_t_tr,
            "X_tgt_val": X_t_va,
            "y_tgt_val": y_t_va,
            "tgt_val_params": p_va,
            "X_tgt_test": X_t_te,
            "y_tgt_test": y_t_te,
            "tgt_test_params": p_te,
        }
        # set test_data so predict_da mimics your predict()
        self.test_data = [X_t_te, p_te, y_t_te]
        return self.da_data

    def train_da_sims(self, _stats=None, w=None, batch_size=32, epochs=100):
        self.num_stats = len(_stats)

        if not hasattr(self, "da_data") or self.da_data is None:
            self.load_da_data_sims(_stats=_stats, w=w)

        da = self.da_data

        if self.normalize:
            X_adapt = np.concatenate([da["X_src"], da["X_tgt_train"]], axis=0).astype(
                np.float32
            )
            self.stat_norm_da = tf.keras.layers.Normalization(
                axis=-1, name="stat_norm_da"
            )
            self.stat_norm_da.adapt(X_adapt)  # train-only union

        # generator: builds 2B batches with masking exactly like Siepel
        gen = DAParquetSequence_sims(
            da["X_src"],
            da["y_src"],
            da["X_tgt_train"],
            da["y_tgt_train"],
            batch_size=batch_size,
            shuffle=True,
        )

        input_shape = (self.windows.size, self.center.size, len(da["stats"]))
        model = self.build_grl_model(input_shape)

        # small mixed-domain validation (so discriminator metrics aren't 0/NaN)
        k = min(2000, len(da["X_src"]))
        idx_src = np.random.RandomState().choice(
            len(da["X_src"]), size=k, replace=False
        )
        Xv = np.concatenate([da["X_src"][idx_src], da["X_tgt_val"]], axis=0)
        yv_cls = np.concatenate(
            [-np.ones((k, 1), np.float32), da["y_tgt_val"][:, None].astype(np.float32)],
            axis=0,
        )
        yv_dom = np.concatenate(
            [
                np.zeros((k, 1), np.float32),
                np.ones((da["X_tgt_val"].shape[0], 1), np.float32),
            ],
            axis=0,
        )

        lr_sched = tf.keras.optimizers.schedules.CosineDecayRestarts(
            initial_learning_rate=5e-5, first_decay_steps=300
        )
        opt = tf.keras.optimizers.Adam(
            learning_rate=lr_sched, epsilon=1e-7, amsgrad=True
        )

        # compile with masked BCE (Siepel semantics)
        model.compile(
            optimizer=opt,
            loss={"classifier": masked_bce_fn, "discriminator": masked_bce_fn},
            # loss_weights={"classifier": 1.0, "discriminator": 1.0},
            loss_weights={"classifier": 1.0, "discriminator": 0.5},
            metrics={
                "classifier": [
                    tf.keras.metrics.AUC(name="auc"),
                    tf.keras.metrics.BinaryAccuracy(name="accuracy"),
                ],
                "discriminator": [tf.keras.metrics.BinaryAccuracy(name="accuracy")],
            },
        )

        ckpt_path = (
            f"{self.output_folder}/model_da.keras" if self.output_folder else None
        )
        callbacks = []
        callbacks.append(
            tf.keras.callbacks.EarlyStopping(
                monitor="val_classifier_auc",
                mode="max",
                patience=20,
                min_delta=1e-4,
                restore_best_weights=True,
            )
        )

        # warmup = 10
        # ramp_to = int(0.4 * epochs)  # reach max by 40% of training
        # callbacks.append(
        #     GRLRamp(self.grl, max_lambda=1, epochs=max(1, ramp_to - warmup))
        # )

        if ckpt_path:
            callbacks.append(
                tf.keras.callbacks.ModelCheckpoint(
                    ckpt_path,
                    monitor="val_classifier_auc",
                    mode="max",
                    save_best_only=True,
                    verbose=2,
                )
            )

        # === fit ===
        history = model.fit(
            gen,
            epochs=epochs,
            steps_per_epoch=len(gen),
            validation_data=(Xv, {"classifier": yv_cls, "discriminator": yv_dom}),
            callbacks=callbacks,
            verbose=2,
        )

        # store history with the names your roc_curve() expects
        hh = history.history
        self.history = pl.DataFrame(
            {
                "loss": hh.get("loss", []),
                "val_loss": hh.get("val_loss", []),
                "accuracy": hh.get("classifier_accuracy", []),
                "val_accuracy": hh.get("val_classifier_accuracy", []),
                "auc": hh.get("classifier_auc", []),
                "val_auc": hh.get("val_classifier_auc", []),
            }
        )
        self.model = model
        if self.output_folder:
            model.save(f"{self.output_folder}/model_da.keras")
        return model

    def predict_da_sims(self, _stats=None):
        assert self.model is not None, "Call train_da_sims() first"
        assert isinstance(self.test_data, (list, tuple)) and len(self.test_data) == 3

        self.num_stats = len(_stats)

        X_test, test_params, Y_test = self.test_data
        out = self.model.predict(X_test, verbose=0, batch_size=32)
        cls = out[0] if isinstance(out, (list, tuple)) else out
        p = cls.ravel().astype(np.float32)

        df_pred = pl.concat(
            [
                test_params,
                pl.DataFrame(
                    {
                        "predicted_model": np.where(p >= 0.5, "sweep", "neutral"),
                        "prob_sweep": p,
                        "prob_neutral": 1.0 - p,
                    }
                ),
            ],
            how="horizontal",
        )

        # overwrite 'model' with clean neutral/sweep from numeric Y for ROC/CM
        df_pred = df_pred.drop("model").with_columns(
            pl.Series("model", np.where(Y_test == 1, "sweep", "neutral"))
        )

        self.prediction = df_pred
        self.roc_curve()
        if self.output_folder:
            df_pred.write_csv(self.output_folder + "/predictions_da.txt")
        return df_pred

    def feature_extractor(self, model_input):
        """
        Shared 2D-CNN feature extractor (three branches).

        Parameters
        ----------
        model_input : tf.keras.layers.Input
            Input tensor of shape ``(W, C, S)``.

        Returns
        -------
        tf.Tensor
            Flattened concatenated features from the three branches.

        See Also
        --------
        cnn_flexsweep : Similar branch structure followed by classification head.
        """
        He = tf.keras.initializers.HeNormal()

        # --- Branch 1: 3x3 convs ---
        b1 = tf.keras.layers.Conv2D(
            64, 3, padding="same", kernel_initializer=He, name="fx_b1_c1"
        )(model_input)
        b1 = tf.keras.layers.ReLU()(b1)
        b1 = tf.keras.layers.Conv2D(
            128, 3, padding="same", kernel_initializer=He, name="fx_b1_c2"
        )(b1)
        b1 = tf.keras.layers.ReLU()(b1)
        b1 = tf.keras.layers.Conv2D(
            256, 3, padding="same", kernel_initializer=He, name="fx_b1_c3"
        )(b1)
        b1 = tf.keras.layers.ReLU()(b1)
        b1 = tf.keras.layers.MaxPooling2D(
            pool_size=3, padding="same", name="fx_b1_pool"
        )(b1)
        b1 = tf.keras.layers.Dropout(0.15, name="fx_b1_drop")(b1)
        b1 = tf.keras.layers.Flatten(name="fx_b1_flat")(b1)

        # --- Branch 2: 2x2 convs with dilation (1,3) ---
        b2 = tf.keras.layers.Conv2D(
            64,
            2,
            dilation_rate=[1, 3],
            padding="same",
            kernel_initializer=He,
            name="fx_b2_c1",
        )(model_input)
        b2 = tf.keras.layers.ReLU()(b2)
        b2 = tf.keras.layers.Conv2D(
            128,
            2,
            dilation_rate=[1, 3],
            padding="same",
            kernel_initializer=He,
            name="fx_b2_c2",
        )(b2)
        b2 = tf.keras.layers.ReLU()(b2)
        b2 = tf.keras.layers.Conv2D(
            256,
            2,
            dilation_rate=[1, 3],
            padding="same",
            kernel_initializer=He,
            name="fx_b2_c3",
        )(b2)
        b2 = tf.keras.layers.ReLU()(b2)
        b2 = tf.keras.layers.MaxPooling2D(pool_size=2, name="fx_b2_pool")(b2)
        b2 = tf.keras.layers.Dropout(0.15, name="fx_b2_drop")(b2)
        b2 = tf.keras.layers.Flatten(name="fx_b2_flat")(b2)

        # --- Branch 3: 2x2 convs with dilation (5,1) then (1,5) ---
        b3 = tf.keras.layers.Conv2D(
            64,
            2,
            dilation_rate=[5, 1],
            padding="same",
            kernel_initializer=He,
            name="fx_b3_c1",
        )(model_input)
        b3 = tf.keras.layers.ReLU()(b3)
        b3 = tf.keras.layers.Conv2D(
            128,
            2,
            dilation_rate=[1, 5],
            padding="same",
            kernel_initializer=He,
            name="fx_b3_c2",
        )(b3)
        b3 = tf.keras.layers.ReLU()(b3)
        b3 = tf.keras.layers.Conv2D(
            256,
            2,
            dilation_rate=[1, 5],
            padding="same",
            kernel_initializer=He,
            name="fx_b3_c3",
        )(b3)
        b3 = tf.keras.layers.ReLU()(b3)
        b3 = tf.keras.layers.MaxPooling2D(pool_size=2, name="fx_b3_pool")(b3)
        b3 = tf.keras.layers.Dropout(0.15, name="fx_b3_drop")(b3)
        b3 = tf.keras.layers.Flatten(name="fx_b3_flat")(b3)

        feat = tf.keras.layers.Concatenate(name="fx_concat")(
            [b1, b2, b3]
        )  # shared representation

        return feat

    def build_grl_model(self, input_shape):
        """
        Build a two-head domain-adversarial CNN with a Gradient Reversal Layer.

        Architecture
        ------------
        - **Shared feature extractor**: :meth:`feature_extractor` over inputs shaped
          ``(W, C, S)`` (windows × centers × statistics), channels-last.
        - **Classifier head** (task): 2 dense layers + sigmoid output named
          ``"classifier"`` (sweep vs. neutral, BCE).
        - **Domain head**: GRL → 2 dense layers + sigmoid output named
          ``"discriminator"`` (source=0 vs. target=1, BCE).

        Parameters
        ----------
        input_shape : tuple[int, int, int]
            ``(W, C, S)`` defining windows, centers, and number of stats (channels).

        Returns
        -------
        tf.keras.Model
            Uncompiled Keras model with two outputs:
            ``[classifier(sigmoid), discriminator(sigmoid)]``.

        Notes
        -----
        - The GRL instance is stored at ``self.grl`` so a callback (e.g., :class:`GRLRamp`)
          can update its strength during training.
        - Compilation (optimizer, losses, metrics) is performed in
          :meth:`train_da_empirical`.
        """
        inp = tf.keras.Input(shape=input_shape)  # (W, C, S), channels-last

        x_in = (
            self.stat_norm_da(inp)
            if hasattr(self, "stat_norm_da") and self.stat_norm_da is not None
            else inp
        )

        feat = self.feature_extractor(x_in)

        # classifier head
        h = tf.keras.layers.Dense(128, activation="relu")(feat)
        h = tf.keras.layers.Dropout(0.20)(h)
        h = tf.keras.layers.Dense(32, activation="relu")(h)
        h = tf.keras.layers.Dropout(0.10)(h)
        out_cls = tf.keras.layers.Dense(1, activation="sigmoid", name="classifier")(h)

        # domain head via GRL (store the layer for ramping)
        self.grl = GradReverse(lambd=0.0)
        g = self.grl(feat)
        g = tf.keras.layers.Dense(128, activation="relu")(g)
        g = tf.keras.layers.Dropout(0.20)(g)
        g = tf.keras.layers.Dense(32, activation="relu")(g)
        out_dom = tf.keras.layers.Dense(1, activation="sigmoid", name="discriminator")(
            g
        )

        model = tf.keras.Model(inputs=inp, outputs=[out_cls, out_dom])

        return model

    def load_da_data(
        self, _stats=None, w=None, n_src=None, src_val_frac=0.10
    ):
        """
        Prepare labeled **source** and unlabeled **target** tensors for DA training.

        Source (simulated) data are split into train/validation. Target (empirical)
        data are used for the domain discriminator and later inference. When
        ``self.normalize`` is True, a `Normalization` layer is adapted on the union
        of source-train and target features.

        Parameters
        ----------
        _stats : list[str] | None
            Statistic base names to include (e.g. ``["ihs","nsl",...]``). If None,
            pass explicitly when calling.
        w : int | list[int] | None
            Restrict to one or more window sizes. Columns are matched via suffix.
        n_src : int | None
            Optional number of source rows to sample for faster tests.
        src_val_frac : float, default=0.10
            Fraction of the source set reserved for validation.

        Returns
        -------
        dict
            A mapping with keys:
            - ``"stats"`` : list[str], the stats actually used
            - ``"X_src_tr"`` : np.ndarray, shape ``(Ns_tr, W, C, S)``
            - ``"y_src_tr"`` : np.ndarray, shape ``(Ns_tr,)`` (0/1)
            - ``"X_src_val"`` : np.ndarray, shape ``(Ns_val, W, C, S)``
            - ``"y_src_val"`` : np.ndarray, shape ``(Ns_val,)`` (0/1)
            - ``"X_tgt"`` : np.ndarray, shape ``(Nt, W, C, S)``
            - ``"tgt_params"`` : pl.DataFrame with region metadata

        Notes
        -----
        - If the target parquet has no ``model`` column, a dummy ``"neutral"`` is
          injected for column selection consistency; target labels are **not** used.
        - The normalization layer (if enabled) is stored as ``self.stat_norm_da`` and
          later applied in :meth:`build_grl_model`.
        """

        # Source (labeled): split into train/val (for early stopping)
        src_df = pl.read_parquet(self.source_data)
        n_simulations = src_df.shape[0]
        if n_src is not None:
            src_df = src_df.sample(n_src, shuffle=True)
            n_simulations = n_src

        stats = []
        if _stats is not None:
            stats = stats + _stats

        X_src, y_src, _src_params = self._select_stats_matrix_like_old(
            src_df, stats, w=w
        )

        Xs_tr, Xs_va, ys_tr, ys_va = train_test_split(
            X_src, y_src, test_size=src_val_frac, stratify=y_src
        )

        # Target (empirical empirical): no split required, use all for domain training and later prediction
        tgt_df = pl.read_parquet(self.target_data).sample(n_simulations)
        X_tgt, _yt_placeholder, tgt_params = self._select_stats_matrix_like_old(
            # if empirical files have no "model", you can inject a dummy column before calling:
            tgt_df.with_columns(pl.lit("neutral").alias("model"))
            if "model" not in tgt_df.columns
            else tgt_df,
            stats,
            w=w,
        )

        # normalization (fit on train-only union)
        if getattr(self, "normalize", False):
            tf_ = self.check_tf()
            self.stat_norm_da = tf_.keras.layers.Normalization(
                axis=-1, name="stat_norm_da"
            )
            self.stat_norm_da.adapt(
                np.concatenate([Xs_tr, X_tgt], axis=0).astype(np.float32)
            )

        self.da_data = {
            "stats": stats,
            "X_src_tr": Xs_tr,
            "y_src_tr": ys_tr,
            "X_src_val": Xs_va,
            "y_src_val": ys_va,
            "X_tgt": X_tgt,
            "tgt_params": tgt_params,
        }
        # for predict_da() later

        return self.da_data

    def train_da(
        self, _stats=None, w=None, batch_size=32, epochs=200
    ):
        """
        Train the domain-adversarial model (GRL) using empirical target batches.

        Uses a custom generator that interleaves: 1. **Source** (simulated) samples with true class labels for the classifier head, and 2. Mixed **source/target** samples for the domain head (source=0, target=1), while masking the irrelevant head per sample.

        Parameters
        ----------
        _stats : list[str] | None
            Statistic base names to include.
        w : int | list[int] | None
            Window size(s) to select.
        batch_size : int, default=32
            Number of **classifier** samples (A) per step.
            The discriminator receives an additional ``batch_size`` examples per step
            split between source and target (see generator notes).
        epochs : int, default=200
            Number of training epochs.


        Returns
        -------
        tf.keras.Model
            The trained DA model (also saved to ``model_da.keras`` when
            ``output_folder`` is provided).

        Training Details
        ----------------
        - **Compilation**: optimizer Adam (CosineDecayRestarts LR), losses:
          ``masked_bce_fn`` for both ``"classifier"`` and ``"discriminator"``,
          loss weights 1.0/1.0.
        - **Metrics**:
          - Classifier: ``AUC`` (ROC), ``BinaryAccuracy``.
          - Discriminator: ``BinaryAccuracy`` (domain acc).
        - **Validation**: source-only validation set (clean labels) with
          domain labels fixed to 0. Early stopping monitors ``val_classifier_auc``.
        - **GRL schedule**: :class:`GRLRamp` warms ``λ`` to ``max_lambda`` over
          ~80% of epochs, then holds constant.

        Notes
        -----
        - Healthy domain alignment typically drives domain accuracy towards ≈0.5.
          If it stays ≫0.5, consider increasing GRL strength; if classifier AUC
          stagnates, reduce it or decrease target ratio in the generator.
        """

        tf_ = self.check_tf()
        if _stats is None:
            _stats = ["ihs", "nsl", "isafe","hapdaf_o", "hapdaf_s","dind", "s_ratio", "low_freq", "high_freq",'h12','haf']

        self.test_data = self.target_data

        if not hasattr(self, "da_data") or self.da_data is None:
            self.load_da_data(_stats=_stats, w=w)

        d = self.da_data

        # XYseq empirical target
        gen = DAParquetSequence(
            d["X_src_tr"], d["y_src_tr"], d["X_tgt"], batch_size=batch_size, tgt_ratio=1
        )

        input_shape = (self.windows.size, self.center.size, len(d["stats"]))
        model = self.build_grl_model(input_shape)

        # optimizer + compile (masked losses as needed)
        opt = tf_.keras.optimizers.Adam(
            learning_rate=tf_.keras.optimizers.schedules.CosineDecayRestarts(5e-5, 300),
            epsilon=1e-7,
            amsgrad=True,
        )
        model.compile(
            optimizer=opt,
            loss={"classifier": masked_bce_fn, "discriminator": masked_bce_fn},
            loss_weights={"classifier": 1.0, "discriminator": 1.0},
            metrics={
                "classifier": [
                    tf_.keras.metrics.AUC(name="auc"),
                    tf_.keras.metrics.AUC(curve="PR", name="auc_pr"),
                    tf_.keras.metrics.BinaryAccuracy(name="accuracy"),
                ],
                "discriminator": [tf_.keras.metrics.BinaryAccuracy(name="accuracy")],
            },
        )

        # validation set: source-only for clean auc
        Xv = d["X_src_val"]
        yv_cls = d["y_src_val"][:, None].astype(
            np.float32
        )
        # real labels for classifier
        # domain=0 (source)
        yv_dom = np.zeros((Xv.shape[0], 1), np.float32)

        callbacks = [
            # GRL λ-ramp: rise for ~80% epochs then hold (robust for stronger shifts)
            GRLRamp(self.grl, max_lambda=0.4, epochs=int(0.8 * epochs)),
            tf_.keras.callbacks.EarlyStopping(
                monitor="val_classifier_auc",
                mode="max",
                patience=25,
                min_delta=1e-4,
                restore_best_weights=True,
            ),
        ]
        if self.output_folder:
            callbacks.append(
                tf_.keras.callbacks.ModelCheckpoint(
                    f"{self.output_folder}/model_da.keras",
                    monitor="val_classifier_auc",
                    mode="max",
                    save_best_only=True,
                    verbose=1,
                )
            )

        hist = model.fit(
            gen,
            epochs=epochs,
            steps_per_epoch=len(gen),
            validation_data=(Xv, {"classifier": yv_cls, "discriminator": yv_dom}),
            callbacks=callbacks,
            verbose=2,
        )

        # store history in your usual format
        hh = hist.history
        self.history = pl.DataFrame(
            {
                "loss": hh.get("loss", []),
                "val_loss": hh.get("val_loss", []),
                "accuracy": hh.get("classifier_accuracy", []),
                "val_accuracy": hh.get("val_classifier_accuracy", []),
                "auc": hh.get("classifier_auc", []),
                "val_auc": hh.get("val_classifier_auc", []),
            }
        )
        self.model = model
        if self.output_folder:
            model.save(f"{self.output_folder}/model_da.keras")

        return model

    def predict_da(self, _stats=None):
        """
        Predict sweep probabilities on empirical (target) data using a DA model.

        Loads a trained two-head model and returns per-region predictions from the
        **classifier** head (sweep vs. neutral). The domain head is unused at inference.

        Parameters
        ----------
        _stats : list[str] | None
            Statistic base names to include (must match training).

        Returns
        -------
        pl.DataFrame
            Table with per-region predictions and metadata, including:
            ``['chr','start','end','f_i','f_t','s','t','predicted_model',
            'prob_sweep','prob_neutral']`` sorted by chromosome and start.

        Raises
        ------
        AssertionError
            If no model is loaded or the test data path is invalid.

        Notes
        -----
        - Expects the same (W, C, S) layout used in training.
        - Output ``prob_sweep`` is the classifier sigmoid; ``prob_neutral=1-prob_sweep``.
        """
        assert self.model is not None, "Call train_da() first"


        tf_ = self.check_tf()
        if _stats is None:
            _stats = ["ihs", "nsl", "isafe","hapdaf_o", "hapdaf_s","dind", "s_ratio", "low_freq", "high_freq",'h12','haf']


        if isinstance(self.model, str):
            model = tf.keras.models.load_model(
                self.model,
                safe_mode=True,
            )
        else:
            model = self.model

        assert (
            isinstance(self.test_data, pl.DataFrame)
            or "txt" in self.test_data
            or "csv" in self.test_data
            or self.test_data.endswith(".parquet")
        ), "Please input a pl.DataFrame or save it as CSV or parquet"
        try:
            df_test = pl.read_parquet(self.test_data)
            if "test" in self.test_data:
                df_test = df_test.sample(
                    with_replacement=False, fraction=1.0, shuffle=True
                )
        except:
            df_test = pl.read_csv(self.test_data, separator=",")

        df_test = df_test.with_columns(
            pl.when((pl.col("model") != "neutral"))
            .then(pl.lit("sweep"))
            .otherwise(pl.lit("neutral"))
            .alias("model")
        )
        regions = df_test["iter"].to_numpy()

        stats = []
        if _stats is not None:
            stats = stats + _stats
        test_stats = []

        for i in stats:
            test_stats.append(df_test.select(pl.col(f"^{i}_[0-9]+_[0-9]+$")))

        X_test = pl.concat(test_stats, how="horizontal")

        test_X_params = df_test.select("model", "iter", "s", "f_i", "f_t", "t")

        test_X = (
            X_test.select(X_test)
            .to_numpy()
            .reshape(
                X_test.shape[0],
                self.num_stats,
                self.windows.size * self.center.size,
                1,
            )
        )



        test_X = test_X.reshape(
            test_X.shape[0], self.windows.size, self.center.size, self.num_stats
        )


        # Same folder custom fvs name based on input VCF.
        _output_prediction = (
            self.output_folder
            + "/"
            + (
                os.path.basename(self.test_data)
                .replace("fvs_", "")
                .replace(".parquet", "_da_predictions.txt")
            )
        )

        out = model.predict(test_X, batch_size=64)
        # two heads → [classifier_probs, discriminator_probs]
        cls = out[0] if isinstance(out, (list, tuple)) else out
        p = cls.ravel().astype(np.float32)

        df_pred = pl.concat(
            [
                test_X_params,
                pl.DataFrame(
                    {
                        "predicted_model": np.where(p >= 0.5, "sweep", "neutral"),
                        "prob_sweep": p,
                        "prob_neutral": 1.0 - p,
                    }
                ),
            ],
            how="horizontal",
        )

        df_prediction = df_pred.with_columns(pl.Series("region", regions))
        chr_start_end = np.array(
            [item.replace(":", "-").split("-") for item in regions]
        )

        df_prediction = df_prediction.with_columns(
            pl.Series("chr", chr_start_end[:, 0]),
            pl.Series("start", chr_start_end[:, 1], dtype=pl.Int64),
            pl.Series("end", chr_start_end[:, 2], dtype=pl.Int64),
            pl.Series(
                "nchr",
                pl.Series(chr_start_end[:, 0]).str.replace("chr", "").cast(int),
            ),
        )
        df_prediction = df_prediction.sort("nchr", "start").select(['chr', 'start', 'end', 'f_i','f_t', 's', 't', 'predicted_model', 'prob_sweep', 'prob_neutral']
        )

        self.prediction = df_prediction
        if self.output_folder:
            df_prediction.write_csv(_output_prediction)

        return df_prediction


class DAParquetSequence_sims(Sequence):
    def __init__(
        self, X_src, y_src, X_tgt, y_tgt, batch_size=32, shuffle=True
    ):
        assert X_src.shape[1:] == X_tgt.shape[1:], "Source/Target shapes must match"
        self.Xs, self.ys = X_src, y_src.astype(np.int32)
        self.Xt, self.yt = X_tgt, y_tgt.astype(np.int32)
        self.B = batch_size
        self.shuffle = shuffle
        self.rng = np.random.RandomState()
        self._reset_epoch()
        # limit batches by pools as Siepel does
        self.n_batches = int(
            np.floor(min(len(self.src_pool_cls), len(self.tgt_pool_dis)) / self.B)
        )

    def _reset_epoch(self):
        self.src_pool_cls = self.rng.permutation(
            len(self.ys)
        )  # for classifier (source)
        self.src_pool_dis = self.rng.permutation(
            len(self.ys)
        )  # for discriminator (source)
        self.tgt_pool_dis = self.rng.permutation(
            len(self.yt)
        )  # for discriminator (target))

    def __len__(self):
        return self.n_batches

    def __getitem__(self, idx):
        # A) B source for classifier (0/1), domain masked
        idxA = self.src_pool_cls[idx * self.B : (idx + 1) * self.B]
        XA = self.Xs[idxA]
        yA_cls = self.ys[idxA].astype(np.float32).reshape(-1, 1)
        yA_dom = -np.ones((XA.shape[0], 1), np.float32)

        # B) B/2 source for discriminator (domain=0), classifier masked
        half = self.B // 2
        idxB = self.src_pool_dis[idx * half : (idx + 1) * half]
        XB = self.Xs[idxB]
        yB_cls = -np.ones((XB.shape[0], 1), np.float32)
        yB_dom = np.zeros((XB.shape[0], 1), np.float32)

        # C) B/2 target for discriminator (domain=1), classifier masked
        idxC = self.tgt_pool_dis[idx * half : (idx + 1) * half]
        XC = self.Xt[idxC]
        yC_cls = -np.ones((XC.shape[0], 1), np.float32)
        yC_dom = np.ones((XC.shape[0], 1), np.float32)

        X = np.concatenate([XA, XB, XC], axis=0)
        y_cls = np.concatenate([yA_cls, yB_cls, yC_cls], axis=0)
        y_dom = np.concatenate([yA_dom, yB_dom, yC_dom], axis=0)

        assert X.shape[0] == 2 * self.B
        return X, {"classifier": y_cls, "discriminator": y_dom}

    def on_epoch_end(self):
        if self.shuffle:
            self._reset_epoch()


class DAParquetSequence(Sequence):
    """
    Data generator for domain-adversarial training with masked multi-task labels.

    Each step yields a mixed minibatch that contains:
      A) ``B`` **source** samples for the classifier (with true labels 0/1),
         masked for the domain head (label = -1).
      B) ``half_src`` **source** samples for the domain discriminator (domain=0),
         masked for the classifier (label = -1).
      C) ``half_tgt`` **target** samples for the domain discriminator (domain=1),
         masked for the classifier (label = -1).

    The ratio ``half_src : half_tgt`` is controlled by ``tgt_ratio``. With
    ``tgt_ratio=1``, the domain minibatch is balanced (50/50). The total number
    of examples returned per step is ``B + half_src + half_tgt``.

    Parameters
    ----------
    X_src : np.ndarray
        Source feature tensor of shape ``(Ns, W, C, S)``.
    y_src : np.ndarray
        Source class labels of shape ``(Ns,)`` with values in ``{0,1}``.
    X_tgt : np.ndarray
        Target feature tensor of shape ``(Nt, W, C, S)`` (unlabeled).
    batch_size : int, default=32
        Number of **classifier** (A) examples per step (``B``).
    shuffle : bool, default=True
        If True, reshuffles source/target pools at each epoch.
    tgt_ratio : int, default=2
        Domain-target ratio relative to source in the discriminator part.
        For example, with ``B=32`` and ``tgt_ratio=2``, the discriminator part
        will allocate roughly 10 source vs. 22 target samples (numbers depend
        on integer division).

    Methods
    -------
    __len__()
        Number of steps per epoch (limited by the smallest pool).
    __getitem__(idx)
        Return a tuple ``(X, {"classifier": y_cls, "discriminator": y_dom})`` where:
          - ``X`` has shape ``(B + half_src + half_tgt, W, C, S)``,
          - ``y_cls`` is shape-matched with labels ``{0,1,-1}`` (``-1`` masked),
          - ``y_dom`` is shape-matched with labels ``{0,1,-1}`` (``-1`` masked).
    on_epoch_end()
        Shuffle pools at epoch boundaries when ``shuffle=True``.

    Notes
    -----
    - Mask sentinel is ``-1.0`` for both heads; see :func:`masked_bce_fn`.
    - Ensures ``X_src`` and ``X_tgt`` have identical spatial shapes.
    - For exact parity with balanced domain batches, set ``tgt_ratio=1``.
    """

    def __init__(
        self, X_src, y_src, X_tgt, batch_size=32, shuffle=True, tgt_ratio=2
    ):
        # y_src is required (0/1); target is unlabeled
        assert X_src.shape[1:] == X_tgt.shape[1:], "Source/Target shapes must match"
        self.Xs, self.ys = X_src, y_src.astype(np.int32)
        self.Xt = X_tgt
        self.B = batch_size
        self.shuffle = shuffle
        self.rng = np.random.RandomState()
        self.tgt_ratio = int(max(1, tgt_ratio))

        self._reset_epoch()
        # discriminator pool sizes: give more target to the discriminator
        half_dis = self.B  # total discriminator examples each step
        self.half_src = max(1, half_dis // (1 + self.tgt_ratio))
        self.half_tgt = half_dis - self.half_src
        self.n_batches = int(
            np.floor(min(len(self.src_pool_cls), len(self.tgt_pool_dis)) / self.B)
        )

    def _reset_epoch(self):
        self.src_pool_cls = self.rng.permutation(len(self.ys))  # source → classifier
        self.src_pool_dis = self.rng.permutation(
            len(self.ys)
        )  # source → discriminator (domain=0)
        self.tgt_pool_dis = self.rng.permutation(
            len(self.Xt)
        )  # target → discriminator (domain=1)

    def __len__(self):
        return self.n_batches

    def __getitem__(self, idx):
        # A) B source for classifier (labels present), domain masked
        idxA = self.src_pool_cls[idx * self.B : (idx + 1) * self.B]
        XA = self.Xs[idxA]
        yA_cls = self.ys[idxA].astype(np.float32).reshape(-1, 1)
        yA_dom = -np.ones((XA.shape[0], 1), np.float32)  # mask domain loss for A

        # B) discriminator: source chunk (domain=0), classifier masked
        idxB = self.src_pool_dis[idx * self.half_src : (idx + 1) * self.half_src]
        XB = self.Xs[idxB]
        yB_cls = -np.ones((XB.shape[0], 1), np.float32)  # mask classifier
        yB_dom = np.zeros((XB.shape[0], 1), np.float32)  # domain=0

        # C) discriminator: target chunk (domain=1), classifier masked
        idxC = self.tgt_pool_dis[idx * self.half_tgt : (idx + 1) * self.half_tgt]
        XC = self.Xt[idxC]
        yC_cls = -np.ones((XC.shape[0], 1), np.float32)  # mask classifier
        yC_dom = np.ones((XC.shape[0], 1), np.float32)  # domain=1

        X = np.concatenate([XA, XB, XC], axis=0)
        y_cls = np.concatenate([yA_cls, yB_cls, yC_cls], axis=0)
        y_dom = np.concatenate([yA_dom, yB_dom, yC_dom], axis=0)
        return X, {"classifier": y_cls, "discriminator": y_dom}

    def on_epoch_end(self):
        if self.shuffle:
            self._reset_epoch()


def rank_probabilities(data_dir, feature_coordinates, pop, include_xy=False):
    """
    Rank genomic features by their maximum nearby sweep probability.

    The function scans per-chromosome prediction files inside ``data_dir``
    (matching ``*_predictions.txt``), associates each feature in
    ``feature_coordinates`` to the **closest** prediction window using
    BEDTools ``closest``, and then ranks features by their **maximum**
    associated sweep probability across all windows on the same chromosome.

    Parameters
    ----------
    data_dir : str
        Directory containing per-chromosome prediction files produced by the
        CNN pipeline. Each file must have at least the columns:
        ``chr, start, end, prob_sweep``. File names are assumed to include a
        token like ``chr{N}`` (e.g., ``chr7``) so the chromosome can be inferred.
    feature_coordinates : str
        Path to a BED-like, tab-separated file with **no header** and columns:
        ``chr, start, end, feature_id, strand``. Coordinates are interpreted as
        0-based half-open for BED operations (as per pybedtools conventions).
    pop : str
        Label used to name the output rank file:
        ``{data_dir}/{pop}_predictions_ranks.txt``.
    include_xy : bool, default=False
        If ``True``, X and Y features are included **provided** both the BED and
        predictions contain those chromosomes with compatible naming
        (e.g., ``chrX``/``chrY`` in predictions; ``X``/``Y`` or numeric codes in the BED).
        If ``False``, only autosomes (1–22) are processed.

    Returns
    -------
    tuple[pl.DataFrame, int]
        - **w_closest_rank** : Polars DataFrame with one row per feature and columns:
          ``['gene_id', 'rank', 'prob_sweep', 'chr', 'start', 'end', 'iter']`` where
          ``rank`` is an ordinal rank (1 = highest probability), ``prob_sweep`` is the
          maximum sweep probability linked to the feature, and ``iter`` is the window
          identifier of that maximum (from the predictions).
        - **n_rank_max** : int
          Number of features tied for the top ``prob_sweep`` value.

    Writes a TSV/CSV to:
    ``{data_dir}/{pop}_predictions_ranks.txt``

    Notes
    -----
    - The function computes per-feature **midpoints** to create a 1-bp interval
      and then uses BEDTools ``closest`` to link each feature to the nearest
      prediction window on the same chromosome. Among all linked windows for a
      feature, the **maximum** ``prob_sweep`` is retained.
    - Chromosome handling:
        * Autosomes are filtered as 1–22. Ensure prediction files use names like
          ``chr1``…``chr22``; the code strips the ``chr`` prefix internally.
        * If ``include_xy=True``, the BED and predictions must consistently
          represent sex chromosomes (e.g., BED has ``X``/``Y`` and predictions
          contain ``chrX``/``chrY``); otherwise those records may be excluded.
    - Dependencies: requires **pybedtools** and an installed BEDTools binary.
    - Performance: for large genomes, reading all per-chromosome predictions and
      performing closest matches can be memory-intensive. Consider splitting
      by chromosome or streaming if your datasets are very large.

    """
    # data_dir = "/labstorage/jmurgamoreno/bchak/mno/mno_260325/"
    # Always filtering
    # feature_coordinates = "/labstorage/jmurgamoreno/bchak/ensembl_gene_coords_v109.bed"
    df_genes = (
        pl.read_csv(
            feature_coordinates,
            has_header=False,
            separator="\t",
            schema={
                "chr": pl.Utf8,
                "start": pl.Int32,
                "end": pl.Int32,
                "gene_id": pl.Utf8,
                "strand": pl.Utf8,
            },
        )
        .with_columns(
            (((pl.col("start") + pl.col("end")) / 2)).alias("center_1").cast(pl.Int64),
            (((pl.col("start") + pl.col("end")) / 2) + 1)
            .alias("center_2")
            .cast(pl.Int64),
        )
        .select(pl.exclude(["start", "end"]))
        .rename({"center_1": "start", "center_2": "end"})
        .select("chr", "start", "end", "strand", "gene_id")
        .filter(pl.col("chr").is_in(np.arange(1, 23).astype(str)))
        .with_columns(pl.col("chr").cast(pl.Int64))
        .sort("chr", "start")
    )

    if include_xy:
        df_xy = df_genes.filter(pl.col("chr").is_in(["X", "Y"])).sort("chr", "start")
        df_genes_filtered = pl.concat(
            [df_genes_filtered.with_columns(pl.col("chr").cast(pl.Utf8)), df_xy]
        )

    df_pred = []
    df_genes_filtered = []
    for i in glob.glob(f"{data_dir}/*_predictions.txt"):
        chrom = i.split("chr")[-1].split("_")[0]

        df = df_genes.filter(pl.col("chr") == int(chrom)).sort("start")
        tmp_pred = (
            pl.read_csv(i)
            .select("chr", "start", "end", "prob_sweep")
            .with_columns(
                pl.col("chr").str.replace("chr", "").cast(pl.Int64),
                (((pl.col("start") + pl.col("end")) / 2))
                .alias("center_1")
                .cast(pl.Int64),
                (((pl.col("start") + pl.col("end")) / 2) + 1)
                .alias("center_2")
                .cast(pl.Int64),
                (
                    pl.col("chr")
                    + ":"
                    + pl.col("start").cast(str)
                    + "-"
                    + pl.col("end").cast(str)
                ).alias("iter"),
            )
            .select(pl.exclude(["start", "end"]))
            .rename({"center_1": "start", "center_2": "end"})
            .select("chr", "start", "end", "prob_sweep", "iter")
        )

        df_pred.append(tmp_pred)
        df_genes_filtered.append(df)
    df_pred = pl.concat(df_pred).sort("chr", "start")
    df_genes_filtered = pl.concat(df_genes_filtered).sort("chr", "start")

    gene_bed = BedTool.from_dataframe(df_genes_filtered.to_pandas())
    pred_bed = BedTool.from_dataframe(df_pred.to_pandas())
    w_closest_bed = gene_bed.closest(pred_bed, d=True).to_dataframe(
        disable_auto_names=True, header=None
    )

    w_closest = (
        pl.DataFrame(
            w_closest_bed,
            schema=[
                "chr_gene",
                "start_gene",
                "end_gene",
                "strand",
                "gene_id",
                "chr",
                "start",
                "end",
                "prob_sweep",
                "iter",
                "d",
            ],
        )
        .group_by(["chr", "gene_id"])
        .agg(
            prob_sweep=pl.col("prob_sweep").max(),
            start=pl.col("start_gene")
            .filter(pl.col("prob_sweep") == pl.col("prob_sweep").max())
            .first(),
            end=pl.col("end_gene")
            .filter(pl.col("prob_sweep") == pl.col("prob_sweep").max())
            .first(),
            iter=pl.col("iter")
            .filter(pl.col("prob_sweep") == pl.col("prob_sweep").max())
            .first(),
        )
        .sort("chr", "start")
    )

    w_closest_rank = (
        w_closest.with_columns(
            pl.col("prob_sweep")
            .rank(method="ordinal", descending=True)
            .alias("rank")
            .cast(pl.Int64)
        )
        .select("gene_id", "rank", "prob_sweep", "chr", "start", "end", "iter")
        .sort("rank")
    )

    n_rank_max = w_closest_rank.filter(
        pl.col("prob_sweep") == pl.col("prob_sweep").max()
    ).shape[0]

    w_closest_rank.write_csv(f"{data_dir}/{pop}_predictions_ranks.txt")
    return w_closest_rank, n_rank_max


def roc_curve_rep(data_dir):
    """
    Generates and plots ROC curves along with a history plot of model metrics.

    Returns:
        tuple: A tuple containing:
            - plot_roc (Figure): The ROC curve plot.
            - plot_history (Figure): The history plot of model metrics (loss, validation loss, accuracy, validation accuracy).

    Example:
        roc_plot, history_plot = model.roc_curves()
        plt.show(roc_plot)
        plt.show(history_plot)
    """
    import matplotlib.pyplot as plt
    import glob

    roc_df = []
    for k, i in enumerate(glob.glob(f"{data_dir}/predictions*.txt")):
        pred_data = pl.read_csv(i)
        pred_data = pred_data.with_columns(pl.lit(k).alias("iter"))
        # Create confusion dataframe
        confusion_data = pred_data.group_by(["model", "predicted_model"]).agg(
            pl.len().alias("n")
        )

        expected_combinations = pl.DataFrame(
            {
                "model": ["sweep", "sweep", "neutral", "neutral"],
                "predicted_model": ["sweep", "neutral", "neutral", "sweep"],
            }
        )

        confusion_data = expected_combinations.join(
            confusion_data, on=["model", "predicted_model"], how="left"
        ).fill_null(
            0
        )  # Fill missing values with 0
        # Adding the "true_false" column
        confusion_data = confusion_data.with_columns(
            pl.when(
                (pl.col("model") == pl.col("predicted_model"))
                & (pl.col("model") == "neutral")
            )
            .then(pl.lit("true_negative"))  # Explicit literal for Polars
            .when(
                (pl.col("model") == pl.col("predicted_model"))
                & (pl.col("model") == "sweep")
            )
            .then(pl.lit("true_positive"))  # Explicit literal for Polars
            .when(
                (pl.col("model") != pl.col("predicted_model"))
                & (pl.col("model") == "neutral")
            )
            .then(pl.lit("false_positive"))  # Explicit literal for Polars
            .otherwise(pl.lit("false_negative"))  # Explicit literal for Polars
            .alias("true_false")
        )

        confusion_pivot = confusion_data.pivot(
            values="n", index=None, on="true_false", aggregate_function="sum"
        ).fill_null(0)

        # Copying the pivoted data (optional as Polars is immutable)
        rate_data = confusion_pivot.select(
            ["false_negative", "false_positive", "true_negative", "true_positive"]
        ).sum()

        # Compute the required row sums for normalization
        required_cols = [
            "false_negative",
            "false_positive",
            "true_negative",
            "true_positive",
        ]
        for col in required_cols:
            if col not in rate_data.columns:
                rate_data[col] = 0

        # Calculate row sums for normalization
        rate_data = rate_data.with_columns(
            (pl.col("false_negative") + pl.col("true_positive")).alias("sum_fn_tp")
        )
        rate_data = rate_data.with_columns(
            (pl.col("false_positive") + pl.col("true_negative")).alias("sum_fp_tn")
        )

        # Compute normalized rates
        rate_data = rate_data.with_columns(
            (pl.col("false_negative") / pl.col("sum_fn_tp")).alias("false_negative"),
            (pl.col("false_positive") / pl.col("sum_fp_tn")).alias("false_positive"),
            (pl.col("true_negative") / pl.col("sum_fp_tn")).alias("true_negative"),
            (pl.col("true_positive") / pl.col("sum_fn_tp")).alias("true_positive"),
        )

        # Replace NaN values with 0
        rate_data = rate_data.with_columns(
            [
                pl.col("false_negative").fill_null(0).alias("false_negative"),
                pl.col("false_positive").fill_null(0).alias("false_positive"),
                pl.col("true_negative").fill_null(0).alias("true_negative"),
                pl.col("true_positive").fill_null(0).alias("true_positive"),
            ]
        )

        # Calculate accuracy and precision
        rate_data = rate_data.with_columns(
            [
                (
                    (pl.col("true_positive") + pl.col("true_negative"))
                    / (
                        pl.col("true_positive")
                        + pl.col("true_negative")
                        + pl.col("false_positive")
                        + pl.col("false_negative")
                    )
                ).alias("accuracy"),
                (
                    pl.col("true_positive")
                    / (pl.col("true_positive") + pl.col("false_positive"))
                )
                .fill_null(0)
                .alias("precision"),
            ]
        )

        # Compute ROC AUC and prepare roc_data. Set 'sweep' as the positive class
        pred_rate_auc_data = pred_data.clone().with_columns(
            pl.col("model").cast(pl.Categorical).alias("model")
        )

        # Calculate ROC AUC
        roc_auc_value = roc_auc_score(
            (pred_rate_auc_data["model"] == "sweep").cast(int),
            pred_rate_auc_data["prob_sweep"].cast(float),
        )

        # Create roc_data DataFrame
        roc_data = pl.DataFrame({"AUC": [roc_auc_value]})

        rate_roc_data = pl.concat([pl.DataFrame(rate_data), roc_data], how="horizontal")

        first_row_values = rate_roc_data.row(0)

        pred_rate_auc_data = pred_rate_auc_data.with_columns(
            [
                pl.lit(value).alias(col)
                for col, value in zip(rate_roc_data.columns, first_row_values)
            ]
        )

        # Compute ROC curve using sklearn
        fpr, tpr, thresholds = roc_curve(
            (pred_rate_auc_data["model"] == "sweep").cast(int),
            pred_rate_auc_data["prob_sweep"].cast(float),
        )

        roc_df.append(
            pl.DataFrame({"iter": k, "false_positive_rate": fpr, "sensitivity": tpr})
        )

    ###########

    roc_df = pl.concat(roc_df)

    # Group by 'iter' and plot each group
    fig, ax = plt.subplots(figsize=(10, 6))

    # Loop through each unique 'iter' value
    for i, (iteration, group) in enumerate(roc_df.group_by("iter")):
        ax.plot(
            group["false_positive_rate"],
            group["sensitivity"],
            linewidth=1.5,
            label=f"ROC Curve {iteration}",
            alpha=0.7,
        )

    # Add diagonal reference line
    ax.plot([0, 1], [0, 1], color="grey", linestyle="--", linewidth=1)

    # Labeling and formatting
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("Power")
    ax.axis("equal")
    ax.grid(True, which="both", linestyle="--", linewidth=0.5)
    fig.tight_layout()

    fig.savefig(f"{data_dir}/roc_curve.svg")

    return fig
