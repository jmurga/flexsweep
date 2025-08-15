import time, os, sys

from . import pl, np

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
from tqdm import tqdm
from pybedtools import BedTool
from copy import deepcopy

# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


# # Gradient Reversal Layer (GRL)
# @tf.custom_gradient
# def grad_reverse(x):
#     y = tf.identity(x)

#     def custom_grad(dy):
#         return -dy  # reverse the gradient sign

#     return y, custom_grad


# class GradReverse(Layer):
#     def call(self, x):
#         re

import tensorflow as tf
from tensorflow.keras.layers import (
    Conv2D,
    SeparableConv2D,
    MaxPooling2D,
    GlobalAveragePooling2D,
    Dropout,
    Flatten,
    Dense,
    BatchNormalization,
    Activation,
    concatenate,
)


def focal_loss(alpha=0.25, gamma=2.0):
    def loss_fn(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)

        # Clip predictions to prevent log(0) error
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)

        # Compute cross-entropy loss
        ce_loss = -(
            y_true * tf.math.log(y_pred) + (1 - y_true) * tf.math.log(1 - y_pred)
        )

        # Compute modulating factor
        p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        modulating_factor = tf.pow(1.0 - p_t, gamma)

        # Compute alpha weighting factor
        alpha_factor = y_true * alpha + (1 - y_true) * (1 - alpha)

        # Compute focal loss
        loss = alpha_factor * modulating_factor * ce_loss

        return tf.reduce_mean(loss)

    return loss_fn


def focal_loss(gamma=2.0, alpha=0.25):
    def loss(y_true, y_pred):
        bce = tf.keras.backend.binary_crossentropy(y_true, y_pred)
        pt = tf.where(tf.equal(y_true, 1), y_pred, 1 - y_pred)
        return alpha * tf.pow(1.0 - pt, gamma) * bce

    return loss


def se_block_1d(x, reduction=8, name=None):
    """Squeeze‐and‐Excitation for 1D features."""
    filters = x.shape[-1]
    se = GlobalAveragePooling1D(name=f"{name}_se_gap")(x)
    se = Dense(filters // reduction, activation="relu", name=f"{name}_se_dense1")(se)
    se = Dense(filters, activation="sigmoid", name=f"{name}_se_dense2")(se)
    # expand dims to multiply back onto the sequence
    se = tf.expand_dims(se, axis=1)
    return Multiply(name=f"{name}_se_excite")([x, se])


def spatial_attention(input_tensor):
    avg_pool = tf.reduce_mean(input_tensor, axis=-1, keepdims=True)
    max_pool = tf.reduce_max(input_tensor, axis=-1, keepdims=True)
    concat = tf.concat([avg_pool, max_pool], axis=-1)
    attention = Conv2D(1, kernel_size=7, padding="same", activation="sigmoid")(concat)
    return input_tensor * attention


class CNN:
    """
    A class for building and training a Convolutional Neural Network (CNN) for classification tasks using flex-sweep input statistics.
    Attributes:
        train_data (str or pl.DataFrame): Path to the training data file or a pandas DataFrame containing the training data.
        test_data (pl.DataFrame): DataFrame containing test data after being loaded from a file.
        output_folder (str): Directory where the trained model and history will be saved.
        num_stats (int): Number of statistics/features in the training data. Default is 11.
        center (np.ndarray): Array defining the center positions for processing. Default ranges from 500000 to 700000.
        windows (np.ndarray): Array defining the window sizes for the CNN. Default values are [50000, 100000, 200000, 500000, 1000000].
        train_split (float): Fraction of the training data used for training. Default is 0.8.
        model (tf.keras.Model): Keras model instance for the CNN.
        gpu (bool): Indicates whether to use GPU for training. Default is True.
    """

    def __init__(
        self,
        train_data=None,
        test_data=None,
        valid_data=None,
        output_folder=None,
        model=None,
    ):
        """
        Initializes the CNN class with training data and output folder.

        Args:
            train_data (str or pl.DataFrame): Path to the training data file or a pandas DataFrame containing the training data.
            output_folder (str): Directory to save the trained model and history.
        """
        # self.sweep_data = sweep_data
        self.train_data = train_data
        self.test_data = test_data
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
        self.seed = None
        self.misspecification = False
        self.channel = 11
        self.split = True
        self.test_data_splitted = None
        self.lstm = False

    def check_tf(self):
        """
        Checks and imports the TensorFlow library.

        Returns:
            tf.Module: The TensorFlow module.
        """
        if self.gpu is False:
            os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        tf = importlib.import_module("tensorflow")
        return tf

    def cnn_flexsweep(self, model_input, num_classes=1):
        """
        Flex-sweep CNN architecture with multiple convolutional and pooling layers.

        Args:
            input_shape (tuple): Shape of the input data, e.g., (224, 224, 3). Default Flex-sweep input statistics, windows and centers
            num_classes (int): Number of output classes in the classification problem. Default: Flex-sweep binary classification

        Returns:
            Model: A Keras model instance representing the Flex-sweep CNN architecture.
        """
        tf = self.check_tf()
        # 3x3 layer
        initializer = tf.keras.initializers.HeNormal()
        layer1 = tf.keras.layers.Conv2D(
            64,
            3,
            padding="same",
            name="convlayer1_1",
            kernel_initializer=initializer,
        )(model_input)
        layer1 = tf.keras.layers.ReLU(negative_slope=0)(layer1)
        layer1 = tf.keras.layers.Conv2D(
            128,
            3,
            padding="same",
            name="convlayer1_2",
            kernel_initializer=initializer,
        )(layer1)
        layer1 = tf.keras.layers.ReLU(negative_slope=0)(layer1)
        layer1 = tf.keras.layers.Conv2D(256, 3, padding="same", name="convlayer1_3")(
            layer1
        )
        layer1 = tf.keras.layers.ReLU(negative_slope=0)(layer1)
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
        layer2 = tf.keras.layers.ReLU(negative_slope=0)(layer2)
        layer2 = tf.keras.layers.Conv2D(
            128,
            2,
            dilation_rate=[1, 3],
            padding="same",
            name="convlayer2_2",
            kernel_initializer=initializer,
        )(layer2)
        layer2 = tf.keras.layers.ReLU(negative_slope=0)(layer2)
        layer2 = tf.keras.layers.Conv2D(
            256, 2, dilation_rate=[1, 3], padding="same", name="convlayer2_3"
        )(layer2)
        layer2 = tf.keras.layers.ReLU(negative_slope=0)(layer2)
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
        layer3 = tf.keras.layers.ReLU(negative_slope=0)(layer3)
        layer3 = tf.keras.layers.Conv2D(
            128,
            2,
            dilation_rate=[1, 5],
            padding="same",
            name="convlayer4_2",
            kernel_initializer=initializer,
        )(layer3)
        layer3 = tf.keras.layers.ReLU(negative_slope=0)(layer3)
        layer3 = tf.keras.layers.Conv2D(
            256, 2, dilation_rate=[1, 5], padding="same", name="convlayer4_3"
        )(layer3)
        layer3 = tf.keras.layers.ReLU(negative_slope=0)(layer3)
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

    def cnn_flexsweep_extra(self, model_input, num_classes=1):
        tf_ = self.check_tf()

        activation = "swish"
        use_batch_norm = True
        use_attention = True
        num_branches = 4

        # Branch 1 params
        b1_num_layers = 4
        b1_filters = 64
        b1_initializer = tf_.keras.initializers.GlorotUniform()
        b1_conv_type = "separable_conv2d"
        b1_pool_type = "max"
        b1_pool_size = 3
        b1_dropout = 0.1011023242666608

        # Branch 2 params
        b2_num_layers = 2
        b2_filters = 96
        b2_initializer = tf_.keras.initializers.HeNormal()
        b2_conv_type = "conv2d"
        b2_pool_type = "adaptive"  # interpreted as GlobalAveragePooling2D
        b2_pool_size = 2
        b2_dropout = 0.47788637883972374

        # Branch 3 params
        b3_num_layers = 3
        b3_filters = 32
        b3_initializer = tf_.keras.initializers.LecunNormal()
        b3_conv_type = "separable_conv2d"
        b3_pool_type = "max"
        b3_pool_size = 3
        b3_dropout = 0.10524609379806012

        # Branch 4 params
        b4_num_layers = (
            3  # inferred same as branch3 since not specified (safe assumption)
        )
        b4_filters = 96
        b4_initializer = tf_.keras.initializers.LecunNormal()
        b4_conv_type = "separable_conv2d"
        b4_pool_type = "max"
        b4_pool_size = 2
        b4_dropout = 0.36750076181366254

        # Dense layers params
        dense_units = [640, 448, 448]
        dense_dropouts = [0.26072424838545916, 0.4797966686977233, 0.40069477982811774]
        dense_initializer = tf_.keras.initializers.HeNormal()
        dense_batch_norm = True

        output_initializer = tf_.keras.initializers.GlorotUniform()

        def build_branch(
            x,
            num_layers,
            filters,
            initializer,
            conv_type,
            pool_type,
            pool_size,
            dropout_rate,
            branch_name,
        ):
            for layer_idx in range(num_layers):
                if conv_type == "separable_conv2d":
                    x = SeparableConv2D(
                        filters,
                        kernel_size=3,
                        padding="same",
                        depthwise_initializer=initializer,
                        pointwise_initializer=initializer,
                        name=f"{branch_name}_conv{layer_idx+1}",
                    )(x)
                else:
                    x = Conv2D(
                        filters,
                        kernel_size=3,
                        padding="same",
                        kernel_initializer=initializer,
                        name=f"{branch_name}_conv{layer_idx+1}",
                    )(x)

                if use_batch_norm:
                    x = BatchNormalization(name=f"{branch_name}_bn{layer_idx+1}")(x)
                x = Activation(activation, name=f"{branch_name}_act{layer_idx+1}")(x)

            if pool_type == "max":
                x = MaxPooling2D(pool_size=pool_size, name=f"{branch_name}_pool")(x)
            elif pool_type == "adaptive":
                x = GlobalAveragePooling2D(name=f"{branch_name}_global_pool")(x)

            if pool_type != "adaptive":
                x = Flatten(name=f"{branch_name}_flatten")(x)

            if dropout_rate > 0:
                x = Dropout(dropout_rate, name=f"{branch_name}_dropout")(x)

            if use_attention and pool_type != "adaptive":
                x = spatial_attention(x)

            return x

        # Build branches
        branch1 = build_branch(
            model_input,
            b1_num_layers,
            b1_filters,
            b1_initializer,
            b1_conv_type,
            b1_pool_type,
            b1_pool_size,
            b1_dropout,
            "branch1",
        )
        branch2 = build_branch(
            model_input,
            b2_num_layers,
            b2_filters,
            b2_initializer,
            b2_conv_type,
            b2_pool_type,
            b2_pool_size,
            b2_dropout,
            "branch2",
        )
        branch3 = build_branch(
            model_input,
            b3_num_layers,
            b3_filters,
            b3_initializer,
            b3_conv_type,
            b3_pool_type,
            b3_pool_size,
            b3_dropout,
            "branch3",
        )
        branch4 = build_branch(
            model_input,
            b4_num_layers,
            b4_filters,
            b4_initializer,
            b4_conv_type,
            b4_pool_type,
            b4_pool_size,
            b4_dropout,
            "branch4",
        )

        concat = concatenate(
            [branch1, branch2, branch3, branch4], name="concat_branches"
        )

        x = concat
        for i, (units, dropout_rate) in enumerate(zip(dense_units, dense_dropouts)):
            x = Dense(units, kernel_initializer=dense_initializer, name=f"dense_{i+1}")(
                x
            )
            if dense_batch_norm:
                x = BatchNormalization(name=f"dense_bn_{i+1}")(x)
            x = Activation(activation, name=f"dense_act_{i+1}")(x)
            x = Dropout(dropout_rate, name=f"dense_dropout_{i+1}")(x)

        output_units = 1 if num_classes == 1 else num_classes
        output_activation = "sigmoid" if num_classes == 1 else "softmax"

        output = Dense(
            output_units,
            activation=output_activation,
            kernel_initializer=output_initializer,
            name="output",
        )(x)

        return output

    def cnn_simple(self, model_input, num_classes=1):
        """
        Ultra-conservative CNN that respects your tiny spatial dimensions
        Works for both (21,5,11) and (5,21,11)
        """
        initializer = tf.keras.initializers.HeNormal()

        # Only use tiny kernels
        x = tf.keras.layers.Conv2D(
            64,
            (2, 1),  # Smallest possible center/window pattern
            padding="same",
            name="conv_2x1",
            kernel_initializer=initializer,
        )(model_input)
        x = tf.keras.layers.ReLU()(x)

        x = tf.keras.layers.Conv2D(
            64,
            (1, 2),  # Smallest possible window/center pattern
            padding="same",
            name="conv_1x2",
            kernel_initializer=initializer,
        )(x)
        x = tf.keras.layers.ReLU()(x)

        x = tf.keras.layers.Conv2D(
            128,
            (2, 2),  # Tiny mixed interactions
            padding="same",
            name="conv_2x2",
            kernel_initializer=initializer,
        )(x)
        x = tf.keras.layers.ReLU()(x)

        # Very gentle pooling
        x = tf.keras.layers.MaxPooling2D(pool_size=(2, 1), name="gentle_pool")(x)

        x = tf.keras.layers.Dropout(0.2, name="dropout_1")(x)
        x = tf.keras.layers.Flatten(name="flatten")(x)

        # Dense layers
        x = tf.keras.layers.Dense(256, activation="relu", name="dense_1")(x)
        x = tf.keras.layers.Dropout(0.3, name="dropout_2")(x)

        x = tf.keras.layers.Dense(64, activation="relu", name="dense_2")(x)
        x = tf.keras.layers.Dropout(0.2, name="dropout_3")(x)

        output = tf.keras.layers.Dense(
            num_classes, activation="sigmoid", name="output"
        )(x)

        return output

    def cnn_simple_opt(self, model_input, num_classes=1):
        """
        Improved CNN for popgen summary statistics with tiny spatial dimensions.
        Includes BN, global pooling, and tuned dropout to help generalization and recall.
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

        x = tf.keras.layers.MaxPooling2D(pool_size=(2, 1), name="gentle_pool")(x)
        x = tf.keras.layers.Dropout(0.15, name="dropout_1")(x)

        # Swap Flatten → GlobalAveragePooling2D
        x = tf.keras.layers.GlobalAveragePooling2D(name="global_avg_pool")(x)

        x = tf.keras.layers.Dense(128, activation="relu", name="dense_1")(x)
        x = tf.keras.layers.Dropout(0.2, name="dropout_2")(x)

        x = tf.keras.layers.Dense(32, activation="relu", name="dense_2")(x)
        x = tf.keras.layers.Dropout(0.1, name="dropout_3")(x)

        output = tf.keras.layers.Dense(
            num_classes, activation="sigmoid", name="output"
        )(x)

        return output

    def cnn_flexsweep_2d(self, model_input, num_classes=1):
        """
        Build a CNN as described in the provided JSON, following the explicit, layered cnn_flexsweep style.
        All parameters and layer names are from your config.
        """
        tf = self.check_tf()

        # --- Main branch ---
        x = tf.keras.layers.Dropout(0.1, name="dropout")(model_input)
        x = tf.keras.layers.Flatten(name="flatten")(x)
        x = tf.keras.layers.Dense(256, activation="relu", name="dense_2")(x)
        x = tf.keras.layers.Dropout(0.2, name="dropout_1")(x)
        x = tf.keras.layers.Dense(256, activation="relu", name="dense_3")(x)
        x = tf.keras.layers.Dropout(0.2, name="dropout_2")(x)
        out = tf.keras.layers.Dense(num_classes, activation="sigmoid", name="dense_4")(
            x
        )

        return out

    def cnn_flexsweep_1d(self, model_input, num_classes=1):
        """
        1D Flex-sweep CNN architecture with multiple convolutional and pooling layers.

        Args:
            model_input: Keras Input tensor with shape (windows, stats)
            num_classes (int): Number of output classes. Default: binary.
        Returns:
            output tensor of shape (num_classes,)
        """
        # Branch 1: kernel size 3, dilation 1
        initializer = tf.keras.initializers.HeNormal()

        b1 = tf.keras.layers.Conv1D(
            64, 3, padding="same", kernel_initializer=initializer, name="b1_conv1"
        )(model_input)
        b1 = tf.keras.layers.ReLU()(b1)
        b1 = tf.keras.layers.Conv1D(
            128, 3, padding="same", kernel_initializer=initializer, name="b1_conv2"
        )(b1)
        b1 = tf.keras.layers.ReLU()(b1)
        b1 = tf.keras.layers.Conv1D(256, 3, padding="same", name="b1_conv3")(b1)
        b1 = tf.keras.layers.ReLU()(b1)
        b1 = tf.keras.layers.MaxPooling1D(pool_size=2, padding="same", name="b1_pool")(
            b1
        )
        b1 = tf.keras.layers.Dropout(0.15, name="b1_drop")(b1)
        b1 = tf.keras.layers.Flatten(name="b1_flat")(b1)

        # Branch 2: kernel size 2, dilation rate 3
        b2 = tf.keras.layers.Conv1D(
            64,
            2,
            dilation_rate=3,
            padding="same",
            kernel_initializer=initializer,
            name="b2_conv1",
        )(model_input)
        b2 = tf.keras.layers.ReLU()(b2)
        b2 = tf.keras.layers.Conv1D(
            128,
            2,
            dilation_rate=3,
            padding="same",
            kernel_initializer=initializer,
            name="b2_conv2",
        )(b2)
        b2 = tf.keras.layers.ReLU()(b2)
        b2 = tf.keras.layers.Conv1D(
            256, 2, dilation_rate=3, padding="same", name="b2_conv3"
        )(b2)
        b2 = tf.keras.layers.ReLU()(b2)
        b2 = tf.keras.layers.MaxPooling1D(pool_size=2, name="b2_pool")(b2)
        b2 = tf.keras.layers.Dropout(0.15, name="b2_drop")(b2)
        b2 = tf.keras.layers.Flatten(name="b2_flat")(b2)

        # Branch 3: kernel size 2, dilation rate 5
        b3 = tf.keras.layers.Conv1D(
            64,
            2,
            dilation_rate=5,
            padding="same",
            kernel_initializer=initializer,
            name="b3_conv1",
        )(model_input)
        b3 = tf.keras.layers.ReLU()(b3)
        b3 = tf.keras.layers.Conv1D(
            128,
            2,
            dilation_rate=5,
            padding="same",
            kernel_initializer=initializer,
            name="b3_conv2",
        )(b3)
        b3 = tf.keras.layers.ReLU()(b3)
        b3 = tf.keras.layers.Conv1D(
            256, 2, dilation_rate=5, padding="same", name="b3_conv3"
        )(b3)
        b3 = tf.keras.layers.ReLU()(b3)
        b3 = tf.keras.layers.MaxPooling1D(pool_size=2, name="b3_pool")(b3)
        b3 = tf.keras.layers.Dropout(0.15, name="b3_drop")(b3)
        b3 = tf.keras.layers.Flatten(name="b3_flat")(b3)

        # Concatenate branches
        x = tf.keras.layers.concatenate([b1, b2, b3], name="concat")
        x = tf.keras.layers.Dense(512, activation="relu", name="dense1")(x)
        x = tf.keras.layers.Dropout(0.2, name="drop1")(x)
        x = tf.keras.layers.Dense(128, activation="relu", name="dense2")(x)
        x = tf.keras.layers.Dropout(0.1, name="drop2")(x)
        output = tf.keras.layers.Dense(
            num_classes,
            activation="sigmoid" if num_classes == 1 else "softmax",
            name="out",
        )(x)
        return output

    def cnn_flexsweep_2d_hapmix(self, stats_input, haplo_input, num_classes=1):
        """
        Hybrid CNN + LSTM model for combining summary statistics and haplotype matrix signals.

        Args:
            stats_input (Keras.Input): Input layer for summary statistics of shape (21, 5, 11)
            haplo_input (Keras.Input): Input layer for haplotype matrix of shape (n_haplotypes, 512)

        Returns:
            tf.keras.Model: Compiled Keras model
        """
        tf = self.check_tf()

        # ---- 2D CNN branch for summary statistics ----
        x = tf.keras.layers.Conv2D(64, (3, 3), padding="same", activation="relu")(
            stats_input
        )
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dropout(0.1, name="dropout")(x)
        x = tf.keras.layers.Flatten(name="flatten")(x)
        x = tf.keras.layers.Dense(256, activation="relu", name="dense_2")(x)
        x = tf.keras.layers.Dropout(0.2, name="dropout_1")(x)
        x = tf.keras.layers.Dense(256, activation="relu", name="dense_3")(x)
        x = tf.keras.layers.Dropout(0.2, name="dropout_2")(x)

        # ---- LSTM branch for haplotype matrix ----
        # Expected shape: (n_haplotypes, 512), where 512 is SNPs and each row is a haplotype
        h = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(64, return_sequences=True)
        )(haplo_input)
        h = tf.keras.layers.LayerNormalization()(h)  # Normalization after first LSTM
        h = tf.keras.layers.Dropout(0.3)(h)  # Dropout after first LSTM

        h = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32))(h)
        h = tf.keras.layers.LayerNormalization()(h)  # Normalization after second LSTM
        h = tf.keras.layers.Dropout(0.2)(h)  # Dropout after second LSTM

        # ---- Combine branches ----
        merged = tf.keras.layers.Concatenate()([x, h])
        z = tf.keras.layers.Dense(256, activation="relu")(merged)
        z = tf.keras.layers.Dropout(0.4)(z)
        z = tf.keras.layers.Dense(128, activation="relu")(z)
        z = tf.keras.layers.Dropout(0.2)(z)
        output = tf.keras.layers.Dense(1, activation="sigmoid")(z)

        return output

    def cnn_flexsweep_2d_1d(self, stats_input, haplo_input, num_classes=1):
        """
        Hybrid CNN + LSTM model for combining summary statistics and haplotype matrix signals.

        Args:
            stats_input (Keras.Input): Input layer for summary statistics of shape (21, 5, 11)
            haplo_input (Keras.Input): Input layer for haplotype matrix of shape (n_haplotypes, 512)

        Returns:
            tf.keras.Model: Compiled Keras model
        """
        tf = self.check_tf()

        # ---- 2D CNN branch for summary statistics ----
        # x = tf.keras.layers.Conv2D(64, (3, 3), padding="same", activation="relu")(stats_input)
        # x = tf.keras.layers.BatchNormalization()(x)
        # x = tf.keras.layers.Dropout(0.1, name="dropout")(x)
        # x = tf.keras.layers.Flatten(name="flatten")(x)
        # x = tf.keras.layers.Dense(256, activation="relu", name="dense_2")(x)
        # x = tf.keras.layers.Dropout(0.2, name="dropout_1")(x)
        # x = tf.keras.layers.Dense(256, activation="relu", name="dense_3")(x)
        # x = tf.keras.layers.Dropout(0.2, name="dropout_2")(x)

        # # ---- 1D CNN branch for haplotype matrix ----
        h = tf.keras.layers.Conv1D(
            64, kernel_size=5, activation="relu", padding="same"
        )(haplo_input)
        h = tf.keras.layers.BatchNormalization()(h)
        h = tf.keras.layers.Conv1D(
            128, kernel_size=5, activation="relu", padding="same"
        )(h)
        h = tf.keras.layers.BatchNormalization()(h)
        h = tf.keras.layers.Conv1D(
            128, kernel_size=3, activation="relu", padding="same"
        )(h)
        h = tf.keras.layers.BatchNormalization()(h)
        h = tf.keras.layers.Conv1D(
            256, kernel_size=3, activation="relu", padding="same"
        )(h)
        h = tf.keras.layers.BatchNormalization()(h)
        h = tf.keras.layers.GlobalAveragePooling1D()(h)

        # # ---- Combine branches ----
        # merged = tf.keras.layers.Concatenate()([x, h])
        h = tf.keras.layers.Dense(256, activation="relu")(merged)
        h = tf.keras.layers.Dropout(0.4)(h)
        h = tf.keras.layers.Dense(128, activation="relu")(h)
        h = tf.keras.layers.Dropout(0.2)(h)
        output = tf.keras.layers.Dense(1, activation="sigmoid")(h)

        return output

    def cnn_flexsweep_subtle(self, model_input):
        tf = self.check_tf()

        # Capture input shape directly
        input_shape = model_input.shape[1:]

        # Branch 1: Spatial convs (standard + dilation)
        b1 = tf.keras.layers.SeparableConv2D(
            128, (3, 3), padding="same", dilation_rate=(1, 1)
        )(model_input)
        b1 = tf.keras.layers.BatchNormalization()(b1)
        b1 = tf.keras.layers.ReLU()(b1)
        b1 = tf.keras.layers.SeparableConv2D(
            256, (3, 3), padding="same", dilation_rate=(1, 3)
        )(b1)
        b1 = tf.keras.layers.BatchNormalization()(b1)
        b1 = tf.keras.layers.ReLU()(b1)
        b1 = tf.keras.layers.GlobalAveragePooling2D()(b1)

        # Branch 2: Wider dilation for decay patterns
        b2 = tf.keras.layers.Conv2D(128, (3, 3), padding="same", dilation_rate=(3, 1))(
            model_input
        )
        b2 = tf.keras.layers.BatchNormalization()(b2)
        b2 = tf.keras.layers.ELU()(b2)
        b2 = tf.keras.layers.Conv2D(256, (3, 3), padding="same", dilation_rate=(5, 1))(
            b2
        )
        b2 = tf.keras.layers.BatchNormalization()(b2)
        b2 = tf.keras.layers.ELU()(b2)
        b2 = tf.keras.layers.GlobalMaxPooling2D()(b2)

        # Branch 3: 1D convolution over center axis (simulates decay)
        reshape_1d = tf.keras.layers.Reshape(
            (input_shape[0], input_shape[1] * input_shape[2])
        )(model_input)
        b3 = tf.keras.layers.Conv1D(
            64, kernel_size=3, padding="same", activation="relu"
        )(reshape_1d)
        b3 = tf.keras.layers.GlobalAveragePooling1D()(b3)

        # Branch 4: BiLSTM over flattened spatial stats (center-window plane)
        xflat = tf.keras.layers.Reshape(
            (input_shape[0] * input_shape[1], input_shape[2])
        )(model_input)
        b4 = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(64, return_sequences=False)
        )(xflat)

        # Merge all branches
        merged = tf.keras.layers.Concatenate()([b1, b2, b3, b4])
        x = tf.keras.layers.Dense(256, activation="relu")(merged)
        x = tf.keras.layers.Dropout(0.4)(x)
        x = tf.keras.layers.Dense(128, activation="relu")(x)
        x = tf.keras.layers.Dropout(0.2)(x)
        output = tf.keras.layers.Dense(1, activation="sigmoid")(x)

        return output

    def cnn_flexsweep_with_DANN(self, model_input, num_classes=2):
        # Your original CNN conv layers from cnn_flexsweep
        # (abbreviated here for clarity, you’d copy your layers)
        # ... layer1, layer2, layer3 as before ...

        # Example: just use your concatenated feature as 'features'
        # For demo: suppose 'concat' is the combined features after convs
        layer1 = tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu")(
            model_input
        )
        layer1 = tf.keras.layers.Conv2D(128, 3, padding="same", activation="relu")(
            layer1
        )
        layer1 = tf.keras.layers.Conv2D(256, 3, padding="same", activation="relu")(
            layer1
        )
        layer1 = tf.keras.layers.MaxPooling2D(pool_size=3, padding="same")(layer1)
        layer1 = tf.keras.layers.Dropout(0.15)(layer1)
        layer1 = tf.keras.layers.Flatten()(layer1)

        layer2 = tf.keras.layers.Conv2D(
            64, 2, dilation_rate=[1, 3], padding="same", activation="relu"
        )(model_input)
        layer2 = tf.keras.layers.Conv2D(
            128, 2, dilation_rate=[1, 3], padding="same", activation="relu"
        )(layer2)
        layer2 = tf.keras.layers.Conv2D(
            256, 2, dilation_rate=[1, 3], padding="same", activation="relu"
        )(layer2)
        layer2 = tf.keras.layers.MaxPooling2D(pool_size=2)(layer2)
        layer2 = tf.keras.layers.Dropout(0.15)(layer2)
        layer2 = tf.keras.layers.Flatten()(layer2)

        layer3 = tf.keras.layers.Conv2D(
            64, 2, dilation_rate=[1, 5], padding="same", activation="relu"
        )(model_input)
        layer3 = tf.keras.layers.Conv2D(
            128, 2, dilation_rate=[1, 5], padding="same", activation="relu"
        )(layer3)
        layer3 = tf.keras.layers.Conv2D(
            256, 2, dilation_rate=[1, 5], padding="same", activation="relu"
        )(layer3)
        layer3 = tf.keras.layers.MaxPooling2D(pool_size=2)(layer3)
        layer3 = tf.keras.layers.Dropout(0.15)(layer3)
        layer3 = tf.keras.layers.GlobalAveragePooling2D()(layer3)

        concat = tf.keras.layers.concatenate([layer1, layer2, layer3])

        # Classification head (same as before)
        class_dense = tf.keras.layers.Dense(512, activation="relu")(concat)
        class_dense = tf.keras.layers.Dropout(0.2)(class_dense)
        class_dense = tf.keras.layers.Dense(128, activation="relu")(class_dense)
        class_dense = tf.keras.layers.Dropout(0.1)(class_dense)
        out_class = tf.keras.layers.Dense(1, activation="sigmoid", name="classifier")(
            class_dense
        )

        # Domain discriminator branch with GRL
        grl_features = GradReverse()(concat)
        disc_dense = tf.keras.layers.Dense(512, activation="relu")(grl_features)
        disc_dense = tf.keras.layers.Dropout(0.2)(disc_dense)
        disc_dense = tf.keras.layers.Dense(128, activation="relu")(disc_dense)
        disc_dense = tf.keras.layers.Dropout(0.1)(disc_dense)
        out_domain = tf.keras.layers.Dense(
            1, activation="sigmoid", name="discriminator"
        )(disc_dense)

        # Build the model with two outputs
        model = Model(inputs=model_input, outputs=[out_class, out_domain])

        return model

    def cnn_flexsweep_conv1d(self, model_input, num_classes=2):
        """
        Option 1: Conv1D over spatial positions with stats as channels,
        followed by channel (stat)-wise attention.
        """
        tf = self.check_tf()

        # Reshape input from (stats, positions, 1) to (positions, stats)
        x = tf.keras.layers.Reshape((model_input.shape[2], model_input.shape[1]))(
            model_input
        )

        # Conv1D layers over spatial dimension
        x = tf.keras.layers.Conv1D(128, 3, padding="same", activation="relu")(x)
        x = tf.keras.layers.Conv1D(256, 2, padding="same", activation="relu")(x)

        # Attention across stats: compute weights per stat and apply
        # shape manipulation to get (batch, stats, positions)
        attn = tf.keras.layers.Permute((2, 1))(x)
        attn = tf.keras.layers.Dense(attn.shape[-1], activation="softmax")(attn)
        attn = tf.keras.layers.Permute((1, 2))(
            attn
        )  # back to (batch, positions, stats)
        x = tf.keras.layers.Multiply()([x, attn])

        x = tf.keras.layers.GlobalAveragePooling1D()(x)
        x = tf.keras.layers.Dense(256, activation="relu")(x)
        x = tf.keras.layers.Dropout(0.3)(x)
        x = tf.keras.layers.Dense(128, activation="relu")(x)
        x = tf.keras.layers.Dropout(0.15)(x)
        output = tf.keras.layers.Dense(1, activation="sigmoid")(x)

        return output

    def load_training_data_mat(self, _stats=None, w=None, n=None):
        """
        Loads training data from specified files and preprocesses it for training.

        Returns:
            tuple: Contains the training and validation datasets:
                - X_train (np.ndarray): Input features for training.
                - X_test (np.ndarray): Input features for testing.
                - Y_train (np.ndarray): One-hot encoded labels for training.
                - Y_test (np.ndarray): One-hot encoded labels for testing.
                - X_valid (np.ndarray): Input features for validation.
                - Y_valid (np.ndarray): One-hot encoded labels for validation.
        """
        tf = self.check_tf()

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
            tmp_hap = np.load("haplotype_matrices.npz")

            mask_neutral = (
                tmp.filter(pl.col("model") == "neutral")["iter"].to_numpy().flatten()
                - 1
            )
            mask_sweep = (
                tmp.filter(pl.col("model") != "neutral")["iter"].to_numpy().flatten()
                - 1
            )

            hap_neutral = tmp_hap["neutral"][mask_neutral]
            hap_sweep = tmp_hap["sweep"][mask_sweep]
            hap_mat = np.concatenate([hap_neutral, hap_sweep])

            if n is not None:
                tmp = tmp.sample(n)
        if self.num_stats <= 21:
            tmp = tmp.select([col for col in tmp.columns if "flip" not in col])

        tmp = tmp.with_columns(
            pl.when((pl.col("model") != "neutral"))
            .then(pl.lit("sweep"))
            .otherwise(pl.lit("neutral"))
            .alias("model")
        )
        # tmp = tmp.filter(
        #     ((pl.col("f_t") >= 0.4) & (pl.col("model") == "sweep"))
        #     | (pl.col("model") == "neutral")
        # )

        if w is not None:
            self.center = np.array([int(w)])
            tmp = tmp.select("iter", "s", "t", "f_i", "f_t", "model", f"^*._{int(w)}$")

        sweep_parameters = tmp.filter("model" != "neutral").select(tmp.columns[:5])

        # stats = ["iter","model","dind","haf","hapdaf_o","isafe","high_freq","hapdaf_s","nsl","s_ratio","low_freq","ihs","h12",]

        stats = ["iter", "model"]
        # ["ihs", "nsl", "isafe", "haf", "h12","hapdaf_o", "hapdaf_s","dind", "s_ratio", "low_freq", "high_freq"]

        # if self.seed is not None:
        #     np.random.seed(self.seed)
        #     _stats = np.random.choice(_stats, self.num_stats)
        if _stats is not None:
            stats = stats + _stats

        train_stats = []
        for i in stats:
            train_stats.append(tmp.select(pl.col("^.*" + i + ".*$")))
        train_stats = pl.concat(train_stats, how="horizontal").select(
            pl.exclude("^.*flip.*$"),
        )
        train_stats = pl.concat(
            [tmp.select("s", "f_i", "f_t", "t"), train_stats], how="horizontal"
        )

        # old_partial_mask = ((train_stats['model'] != 'neutral') & ((train_stats['t'] > 2000) & (train_stats['f_t'] < 1 ))).to_numpy()
        # train_stats = train_stats.filter((pl.col('model')=='neutral') | (old_partial_mask))

        y = train_stats.select(
            ((~pl.col("model").str.contains("neutral")).cast(pl.Int8)).alias(
                "neutral_flag"
            )
        )["neutral_flag"].to_numpy()

        test_split = round(1 - self.train_split, 2)

        if self.misspecification:
            stratify_key = np.char.add(
                tmp["demo"].to_numpy().astype(str), np.char.add("_", y.astype(str))
            )
            X_train, X_test, y_train, y_test = train_test_split(
                train_stats,
                y,
                test_size=test_split,
                shuffle=True,
                stratify=stratify_key,
            )
        else:
            (
                X_train,
                X_test,
                X_train_hap,
                X_test_hap,
                Y_train,
                y_test,
            ) = train_test_split(
                train_stats, hap_mat, y, test_size=test_split, shuffle=True
            )

        # old_partial_mask_train = old_partial_mask_train.astype(int)
        # old_partial_mask_train[old_partial_mask_train > 0] = 3
        # old_partial_mask_train[old_partial_mask_train == 0] = 1

        # self.weights = old_partial_mask_train
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

        # Y_train = y_train
        # Y_train = tf.keras.utils.to_categorical(y_train, 2)
        # Y_test = tf.keras.utils.to_categorical(y_test, 2)

        X_valid, X_test, X_test_hap, X_valid_hap, Y_valid, Y_test = train_test_split(
            X_test, X_test_hap, y_test, test_size=0.5
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

        # Y_train = np.argmax(Y_train, axis=1)
        # Y_test = np.argmax(Y_test, axis=1)
        # Y_valid = np.argmax(Y_valid, axis=1)

        self.test_data = [X_test, X_test_hap, X_test_params, Y_test]

        return (
            X_train,
            X_test,
            Y_train,
            Y_test,
            X_valid,
            Y_valid,
            X_train_hap,
            X_test_hap,
            X_valid_hap,
        )

    def load_training_data(self, _stats=None, w=None, n=None):
        """
        Loads training data from specified files and preprocesses it for training.

        Returns:
            tuple: Contains the training and validation datasets:
                - X_train (np.ndarray): Input features for training.
                - X_test (np.ndarray): Input features for testing.
                - Y_train (np.ndarray): One-hot encoded labels for training.
                - Y_test (np.ndarray): One-hot encoded labels for testing.
                - X_valid (np.ndarray): Input features for validation.
                - Y_valid (np.ndarray): One-hot encoded labels for validation.
        """
        tf = self.check_tf()

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
        # if self.num_stats <= 21:
        #     tmp = tmp.select([col for col in tmp.columns if "flip" not in col])

        tmp = tmp.with_columns(
            pl.when((pl.col("model") != "neutral"))
            .then(pl.lit("sweep"))
            .otherwise(pl.lit("neutral"))
            .alias("model")
        )
        # tmp = tmp.filter(
        #     (pl.col("model") == "neutral")
        #     | (pl.col("t") <= 2000)
        #     | ((pl.col("t") > 2000) & (pl.col("f_t") >= 0.5))
        # )

        if w is not None:
            self.center = np.array([int(w)])
            tmp = tmp.select("iter", "s", "t", "f_i", "f_t", "model", f"^*._{int(w)}$")

        sweep_parameters = tmp.filter("model" != "neutral").select(tmp.columns[:5])

        stats = [
            # "dind",
            # "hapdaf_o",
            # "isafe",
            # "high_freq",
            # "hapdaf_s",
            # "nsl",
            # "s_ratio",
            # "low_freq",
            # "ihs",
            # "h12",
            # "haf"
        ]
        # ld_stats = ["ihs", "nsl", "isafe", "haf", "h12","omega_max","zns",'mu_ld']
        # sfs_stats = ["hapdaf_o", "hapdaf_s",'fay_wu_h','mu_sfs']
        # div_stats = ["dind", "s_ratio", "low_freq", "high_freq","pi",'mu_var']
        # stats = ['model','iter'] + ld_stats + sfs_stats + div_stats
        # stats = ["model","iter","ihs", "nsl", "isafe", "haf", "h12","hapdaf_o", "hapdaf_s","dind", "s_ratio", "low_freq", "high_freq"]

        if _stats is not None:
            stats = stats + _stats

        train_stats = []
        for i in stats:
            # train_stats.append(tmp.select(pl.col("^.*" + i + ".*$")))
            train_stats.append(tmp.select(pl.col(f"^{i}_[0-9]+_[0-9]+$")))

        # train_stats = pl.concat(train_stats, how="horizontal").select(
        #     pl.exclude("^.*flip.*$"),
        # )
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

        if self.misspecification:
            stratify_key = np.char.add(
                tmp["demo"].to_numpy().astype(str), np.char.add("_", y.astype(str))
            )
            X_train, X_test, y_train, y_test = train_test_split(
                train_stats,
                y,
                test_size=test_split,
                shuffle=True,
                stratify=stratify_key,
            )
        else:
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

        # Y_train = np.argmax(Y_train, axis=1)
        # Y_test = np.argmax(Y_test, axis=1)
        # Y_valid = np.argmax(Y_valid, axis=1)

        self.test_data = [X_test, X_test_params, Y_test]

        return (
            X_train,
            X_test,
            Y_train,
            Y_test,
            X_valid,
            Y_valid,
        )

    def load_training_data_stat(self, full_stats=False, w=None, n=None):
        """
        Loads training data from specified files and preprocesses it for training.

        Returns:
            tuple: Contains the training and validation datasets:
                - X_train (np.ndarray): Input features for training.
                - X_test (np.ndarray): Input features for testing.
                - Y_train (np.ndarray): One-hot encoded labels for training.
                - Y_test (np.ndarray): One-hot encoded labels for testing.
                - X_valid (np.ndarray): Input features for validation.
                - Y_valid (np.ndarray): One-hot encoded labels for validation.
        """
        tf = self.check_tf()

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
        if self.num_stats <= 21:
            tmp = tmp.select([col for col in tmp.columns if "flip" not in col])

        tmp = tmp.with_columns(
            pl.when((pl.col("model") != "neutral"))
            .then(pl.lit("sweep"))
            .otherwise(pl.lit("neutral"))
            .alias("model")
        )

        if w is not None:
            self.center = np.array([int(w)])
            tmp = tmp.select("iter", "s", "t", "f_i", "f_t", "model", f"^*._{int(w)}$")

        sweep_parameters = tmp.filter("model" != "neutral").select(tmp.columns[:5])
        stats = ["mode", "iter"]

        if full_stats:
            ld_stats = [
                "ihs",
                "nsl",
                "isafe",
                "haf",
                "h12",
                "omega_max",
                "zns",
                "mu_ld",
            ]
            sfs_stats = ["hapdaf_o", "hapdaf_s", "fay_wu_h", "mu_sfs"]
            div_stats = ["dind", "s_ratio", "low_freq", "high_freq", "pi", "mu_var"]
        else:
            ld_stats = ["ihs", "nsl", "isafe", "haf", "h12"]
            sfs_stats = ["hapdaf_o", "hapdaf_s"]
            div_stats = ["dind", "s_ratio", "low_freq", "high_freq"]

        train_ld = []
        train_sfs = []
        train_div = []
        for i in ld_stats:
            train_ld.append(tmp.select(pl.col("^.*" + i + ".*$")))
        for i in sfs_stats:
            train_sfs.append(tmp.select(pl.col("^.*" + i + ".*$")))
        for i in div_stats:
            train_div.append(tmp.select(pl.col("^.*" + i + ".*$")))

        train_ld = pl.concat(train_ld, how="horizontal").select(
            pl.exclude("^.*flip.*$"),
        )
        train_sfs = pl.concat(train_sfs, how="horizontal").select(
            pl.exclude("^.*flip.*$"),
        )
        train_div = pl.concat(train_div, how="horizontal").select(
            pl.exclude("^.*flip.*$"),
        )

        y = tmp.select(
            ((~pl.col("model").str.contains("neutral")).cast(pl.Int8)).alias(
                "neutral_flag"
            )
        )["neutral_flag"].to_numpy()

        test_split = round(1 - self.train_split, 2)

        if self.misspecification:
            stratify_key = np.char.add(
                tmp["demo"].to_numpy().astype(str), np.char.add("_", y.astype(str))
            )
            X_train, X_test, y_train, y_test = train_test_split(
                train_stats,
                y,
                test_size=test_split,
                shuffle=True,
                stratify=stratify_key,
            )
        else:
            (
                X_train_ld,
                X_test_ld,
                X_train_sfs,
                X_test_sfs,
                X_train_div,
                X_test_div,
                Y_train,
                y_test,
            ) = train_test_split(
                train_ld, train_sfs, train_div, y, test_size=test_split, shuffle=True
            )

        X_train_ld = X_train_ld.to_numpy().reshape(
            X_train_ld.shape[0],
            len(ld_stats),
            self.windows.size * self.center.size,
            1,
        )
        X_train_sfs = X_train_sfs.to_numpy().reshape(
            X_train_sfs.shape[0],
            len(sfs_stats),
            self.windows.size * self.center.size,
            1,
        )
        X_train_div = X_train_div.to_numpy().reshape(
            X_train_div.shape[0],
            len(div_stats),
            self.windows.size * self.center.size,
            1,
        )

        (
            X_valid_ld,
            X_test_ld,
            X_valid_sfs,
            X_test_sfs,
            X_valid_div,
            X_test_div,
            Y_valid,
            Y_test,
        ) = train_test_split(X_test_ld, X_test_sfs, X_test_div, y_test, test_size=0.5)

        X_test_ld = X_test_ld.to_numpy().reshape(
            X_test_ld.shape[0],
            len(ld_stats),
            self.windows.size * self.center.size,
            1,
        )
        X_valid_ld = X_valid_ld.to_numpy().reshape(
            X_valid_ld.shape[0],
            len(ld_stats),
            self.windows.size * self.center.size,
            1,
        )
        X_test_sfs = X_test_sfs.to_numpy().reshape(
            X_test_sfs.shape[0],
            len(sfs_stats),
            self.windows.size * self.center.size,
            1,
        )
        X_valid_sfs = X_valid_sfs.to_numpy().reshape(
            X_valid_sfs.shape[0],
            len(sfs_stats),
            self.windows.size * self.center.size,
            1,
        )
        X_test_div = X_test_div.to_numpy().reshape(
            X_test_div.shape[0],
            len(div_stats),
            self.windows.size * self.center.size,
            1,
        )
        X_valid_div = X_valid_div.to_numpy().reshape(
            X_valid_div.shape[0],
            len(div_stats),
            self.windows.size * self.center.size,
            1,
        )

        X_train_ld = X_train_ld.reshape(
            X_train_ld.shape[0], self.windows.size, self.center.size, len(ld_stats)
        )
        X_test_ld = X_test_ld.reshape(
            X_test_ld.shape[0], self.windows.size, self.center.size, len(ld_stats)
        )
        X_valid_ld = X_valid_ld.reshape(
            X_valid_ld.shape[0], self.windows.size, self.center.size, len(ld_stats)
        )
        X_train_sfs = X_train_sfs.reshape(
            X_train_sfs.shape[0], self.windows.size, self.center.size, len(sfs_stats)
        )
        X_test_sfs = X_test_sfs.reshape(
            X_test_sfs.shape[0], self.windows.size, self.center.size, len(sfs_stats)
        )
        X_valid_sfs = X_valid_sfs.reshape(
            X_valid_sfs.shape[0], self.windows.size, self.center.size, len(sfs_stats)
        )
        X_train_div = X_train_div.reshape(
            X_train_div.shape[0], self.windows.size, self.center.size, len(div_stats)
        )
        X_test_div = X_test_div.reshape(
            X_test_div.shape[0], self.windows.size, self.center.size, len(div_stats)
        )
        X_valid_div = X_valid_div.reshape(
            X_valid_div.shape[0], self.windows.size, self.center.size, len(div_stats)
        )

        return (
            X_train_ld,
            X_test_ld,
            X_valid_ld,
            X_train_sfs,
            X_test_sfs,
            X_valid_sfs,
            X_train_div,
            X_test_div,
            X_valid_div,
        )

    def train(self, _iter=1, _stats=None, w=None, cnn=None, input_data=None):
        """
        Trains the CNN model on the training data.

        This method preprocesses the data, sets up data augmentation, defines the model architecture,
        compiles the model, and fits it to the training data while saving the best model and training history.
        """
        tf = self.check_tf()

        if cnn is None:
            cnn = self.cnn_flexsweep

        if self.split is True:
            (
                X_train,
                X_test,
                Y_train,
                Y_test,
                X_valid,
                Y_valid,
            ) = self.load_training_data(w=w, _stats=_stats)
            # (X_train_ld,X_test_ld,X_valid_ld,X_train_sfs,X_test_sfs,X_valid_sfs,X_train_div,X_test_div,X_valid_div) = self.load_training_data_stat()
            self.test_data_splitted = (
                X_train,
                X_test,
                Y_train,
                Y_test,
                X_valid,
                Y_valid,
                # X_train_hap,
                # X_test_hap,
                # X_valid_hap,
            )

        else:
            (
                X_train,
                X_test,
                Y_train,
                Y_test,
                X_valid,
                Y_valid,
                # X_train_hap,
                # X_test_hap,
                # X_valid_hap,
            ) = self.test_data_splitted

        if self.channel == 5:
            X_train = X_train.reshape(
                X_train.shape[0], self.num_stats, self.center.size, self.windows.size
            )
            X_test = X_test.reshape(
                X_test.shape[0], self.num_stats, self.center.size, self.windows.size
            )
            X_valid = X_valid.reshape(
                X_valid.shape[0], self.num_stats, self.center.size, self.windows.size
            )
            input_shape = (self.num_stats, self.center.size, self.windows.size)
        elif self.channel == 11:
            X_train = X_train.reshape(
                X_train.shape[0], self.windows.size, self.center.size, self.num_stats
            )
            X_test = X_test.reshape(
                X_test.shape[0], self.windows.size, self.center.size, self.num_stats
            )
            X_valid = X_valid.reshape(
                X_valid.shape[0], self.windows.size, self.center.size, self.num_stats
            )
            input_shape = (self.center.size, self.windows.size, self.num_stats)
        elif self.channel == 21:
            X_train = X_train.reshape(
                X_train.shape[0], self.windows.size, self.num_stats, self.center.size
            )
            X_test = X_test.reshape(
                X_test.shape[0], self.windows.size, self.num_stats, self.center.size
            )
            X_valid = X_valid.reshape(
                X_valid.shape[0], self.windows.size, self.num_stats, self.center.size
            )
            input_shape = X_train.shape[1:]

        # put model together
        input_to_model = tf.keras.Input(X_train.shape[1:])
        # input_ld = X_train_ld.shape[1:]
        # input_sfs = X_train_sfs.shape[1:]
        # input_div = X_train_div.shape[1:]
        batch_size = 32

        # Instantiate your augmentation pipeline
        # aug_pipeline = AugmentationPipeline(
        #     rotation_prob=0.1,
        #     flip_prob=0.3,
        #     # shuffle_centers_prob=0.5,
        #     # shuffle_windows_prob=0.3,
        #     # noise_factor=0.02,
        #     # dropout_centers_prob=0.15,
        #     # dropout_windows_prob=0.15,
        #     # statistic_dropout_prob=0.1,
        #     # swap_centers_prob=0.3
        # )

        train_dataset = (
            tf.data.Dataset.from_tensor_slices((X_train, Y_train))
            # tf.data.Dataset.from_tensor_slices(((X_train_ld,X_train_sfs,X_train_div), Y_train))
            .shuffle(10000)
            .batch(batch_size)
            .prefetch(tf.data.AUTOTUNE)
        )

        # train_dataset = aug_pipeline.create_augmented_dataset(raw_train_dataset, batch_size=batch_size)
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

        # model = cnn(input_ld,input_sfs,input_div)

        model_path = f"{self.output_folder}/model5.keras"

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

        # model.compile(loss=focal_loss(gamma=2.0, alpha=0.5))
        model.compile(
            optimizer=opt_adam,
            loss=custom_loss,
            # loss = focal_loss(gamma=2.0, alpha=0.5),
            metrics=[
                tf.keras.metrics.BinaryAccuracy(name="accuracy"),
                tf.keras.metrics.AUC(name="auc", curve="ROC"),
                tf.keras.metrics.Precision(name="precision"),
            ],
        )
        earlystop = tf.keras.callbacks.EarlyStopping(
            monitor="val_auc",
            min_delta=0.0001,
            patience=5,
            verbose=2,
            mode="max",
            restore_best_weights=True,
        )

        # monitor="val_auc","val_prc","sweep_recall",
        checkpoint = tf.keras.callbacks.ModelCheckpoint(
            model_path,
            monitor="val_auc",
            verbose=2,
            save_best_only=True,
            mode="max",
        )

        callbacks_list = [checkpoint, earlystop]

        start = time.time()

        # class_weight = {0: 1.0, 1: 1.2}
        # model.fit(..., )

        history = model.fit(
            train_dataset,
            # class_weight=class_weight,
            epochs=100,
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

        # return self.model.evaluate(X_valid, Y_valid, verbose=0)[3]

    def predict(self, _stats=None, input_data=None, _iter=1):
        """
        Makes predictions on the test data using the trained CNN model.

        This method loads test data, processes it, and applies the trained model to generate predictions.

        Raises:
            AssertionError: If the model has not been trained or loaded.
        """
        tf = self.check_tf()

        assert self.model is not None, "Please input the CNN trained model"

        assert self.test_data is not None, "Please input training data"

        # import data to predict
        _output_prediction = self.output_folder + "/" + self.output_prediction
        if isinstance(self.test_data, str):
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
            # if self.num_stats < 17:
            #     df_test = df_test.select(
            #         [col for col in df_test.columns if "flip" not in col]
            #     )

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

            test_Y = df_test.select(
                ((~pl.col("model").str.contains("neutral")).cast(pl.Int8)).alias(
                    "neutral_flag"
                )
            )["neutral_flag"].to_numpy()

            # Same folder custom fvs name based on input VCF.
            self.output_prediction = (
                os.path.basename(self.test_data)
                .replace("fvs_", "")
                .replace(".parquet", "_predictions.txt")
            )

        else:
            try:
                # test_X, test_X_hap, test_X_params, test_Y = deepcopy(self.test_data)
                (X_test_ld, X_test_sfs, X_test_div, Y_test) = deepcopy(self.test_data)
            except:
                test_X, test_X_params, test_Y = deepcopy(self.test_data)

        if self.channel == 5:
            test_X = test_X.reshape(
                test_X.shape[0], self.num_stats, self.center.size, self.windows.size
            )
        elif self.channel == 11:
            test_X = test_X.reshape(
                test_X.shape[0], self.windows.size, self.center.size, self.num_stats
            )
        elif self.channel == 21:
            test_X = test_X.reshape(
                test_X.shape[0], self.windows.size, self.num_stats, self.center.size
            )

        # batch size, image width, image height,number of channels
        if isinstance(self.model, str):
            model = tf.keras.models.load_model(self.model)
        else:
            model = self.model

        # make predictions
        # try:
        if self.lstm:
            preds = model.predict((test_X, test_X_hap))
        else:
            try:
                preds = model.predict((X_test_ld, X_test_sfs, X_test_div))
                # preds = model.predict(test_dataset)
            except:
                preds = model.predict(test_X)

            # except:
            #     preds = model.predict(test_X.reshape(-1, 11, 21, 5, 1))

        preds = np.column_stack([1 - preds, preds])
        predictions = np.argmax(preds, axis=1)

        y_true = (test_X_params["model"] != "neutral").to_numpy().astype(int)
        # y_true = Y_test
        fpr, tpr, thresh = roc_curve(y_true, preds[:, 1])

        # 2) Pick threshold: FPR <= max_fpr, maximize TPR
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
        # predictions_class = np.where(probs >= best_thresh, "sweep", "neutral")

        df_prediction = pl.concat(
            [
                # pl.DataFrame(np.column_stack([np.vectorize(prediction_dict.get)(y_true),np.zeros((preds.shape[0],4))]),schema = ['model','f_i','f_t','s','t']),
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

        # df_prediction = df_prediction.with_columns(pl.lit(np.vectorize(prediction_dict.get)(y_true)).alias('model'))
        if isinstance(self.test_data, str) and "test" not in self.test_data:
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

        self.prediction = df_prediction
        self.roc_curve()

        if self.output_folder is not None:
            df_prediction.write_csv(_output_prediction)

        return df_prediction

    def train_1d(self, _iter=1, _stats=None, w=None, cnn=None, input_data=None):
        """
        Trains the CNN model on the training data.

        This method preprocesses the data, sets up data augmentation, defines the model architecture,
        compiles the model, and fits it to the training data while saving the best model and training history.
        """
        tf = self.check_tf()

        if cnn is None:
            cnn = self.cnn_flexsweep_1d

        if input_data is not None:
            (
                X_train,
                Y_train,
                X_test,
                Y_test,
                X_valid,
                Y_valid,
                X_test_params,
            ) = input_data
            self.test_data = [X_test, X_test_params, Y_test]
        else:
            if self.split is True:
                (
                    X_train,
                    X_test,
                    Y_train,
                    Y_test,
                    X_valid,
                    Y_valid,
                ) = self.load_training_data(w=w, _stats=_stats, n=None)
                self.test_data_splitted = (
                    X_train,
                    X_test,
                    Y_train,
                    Y_test,
                    X_valid,
                    Y_valid,
                )
            else:
                (
                    X_train,
                    X_test,
                    Y_train,
                    Y_test,
                    X_valid,
                    Y_valid,
                ) = self.test_data_splitted

        X_train = X_train.reshape(
            -1, self.windows.size * self.center.size, self.num_stats
        )
        X_valid = X_valid.reshape(
            -1, self.windows.size * self.center.size, self.num_stats
        )
        X_test = X_test.reshape(
            -1, self.windows.size * self.center.size, self.num_stats
        )

        input_shape = X_train.shape[1:]

        # put model together
        input_to_model = tf.keras.Input(X_train.shape[1:])
        batch_size = 32

        if self.lstm:
            input_hap_to_model = tf.keras.Input(X_hap_train.shape[1:])

            train_dataset = (
                tf.data.Dataset.from_tensor_slices(((X_train, X_train_hap), Y_train))
                .shuffle(10000)
                .batch(batch_size)
                .prefetch(tf.data.AUTOTUNE)
            )
            valid_dataset = (
                tf.data.Dataset.from_tensor_slices(((X_valid, X_valid_hap), Y_valid))
                .batch(batch_size)
                .prefetch(tf.data.AUTOTUNE)
            )
            test_dataset = (
                tf.data.Dataset.from_tensor_slices(((X_test, X_test_hap), Y_valid))
                .batch(batch_size)
                .prefetch(tf.data.AUTOTUNE)
            )
            model = tf.keras.models.Model(
                inputs=[input_to_model, input_hap_to_model],
                outputs=[cnn(input_to_model, input_hap_to_model)],
            )

        else:
            train_dataset = (
                tf.data.Dataset.from_tensor_slices((X_train, Y_train))
                .shuffle(10000)
                .batch(batch_size)
                .prefetch(tf.data.AUTOTUNE)
            )
            valid_dataset = (
                tf.data.Dataset.from_tensor_slices((X_valid, Y_valid))
                .batch(batch_size)
                .prefetch(tf.data.AUTOTUNE)
            )
            test_dataset = (
                tf.data.Dataset.from_tensor_slices((X_test, Y_test))
                .batch(batch_size)
                .prefetch(tf.data.AUTOTUNE)
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
            initial_learning_rate=0.001, first_decay_steps=300
        )
        opt_adam = tf.keras.optimizers.Adam(
            learning_rate=lr_decayed_fn, epsilon=0.0000001, amsgrad=True
        )

        custom_loss = tf.keras.losses.BinaryCrossentropy(label_smoothing=0.0)
        # Keep only one compilation
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.00010056801751838127),
            loss=tf.keras.losses.BinaryCrossentropy(label_smoothing=0.0),
            metrics=[
                tf.keras.metrics.BinaryAccuracy(name="accuracy"),
                tf.keras.metrics.AUC(name="auc", curve="ROC"),
                tf.keras.metrics.Precision(name="precision"),
            ],
        )
        earlystop = tf.keras.callbacks.EarlyStopping(
            monitor="val_auc",
            min_delta=0.001,
            patience=10,
            verbose=1,
            mode="max",
            restore_best_weights=True,
        )

        # monitor="val_accuracy","val_prc","sweep_recall",
        checkpoint = tf.keras.callbacks.ModelCheckpoint(
            model_path,
            monitor="val_auc",
            verbose=1,
            save_best_only=True,
            mode="max",
        )

        callbacks_list = [checkpoint, earlystop]

        start = time.time()

        history = model.fit(
            train_dataset,
            epochs=100,
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

        return self.model.evaluate(X_valid, Y_valid, verbose=0)[3]

    def predict_1d(self, _stats=None, _iter=1):
        """
        Makes predictions on the test data using the trained CNN model.

        This method loads test data, processes it, and applies the trained model to generate predictions.

        Raises:
            AssertionError: If the model has not been trained or loaded.
        """
        tf = self.check_tf()

        assert self.model is not None, "Please input the CNN trained model"

        assert self.test_data is not None, "Please input training data"

        # import data to predict
        _output_prediction = self.output_folder + "/" + self.output_prediction
        if isinstance(self.test_data, str):
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
            if self.num_stats < 17:
                df_test = df_test.select(
                    [col for col in df_test.columns if "flip" not in col]
                )

            df_test = df_test.with_columns(
                pl.when((pl.col("model") != "neutral"))
                .then(pl.lit("sweep"))
                .otherwise(pl.lit("neutral"))
                .alias("model")
            )
            regions = df_test["iter"].to_numpy()

            stats = ["iter", "model"]
            if _stats is not None:
                stats = stats + _stats
            test_stats = []
            for i in stats:
                test_stats.append(df_test.select(pl.col("^.*" + i + ".*$")))
            X_test = pl.concat(test_stats, how="horizontal").select(
                pl.exclude("^.*flip.*$"),
            )

            test_X_params = df_test.select("model", "iter", "s", "f_i", "f_t", "t")

            test_X = (
                X_test.select(X_test.columns[2:])
                .to_numpy()
                .reshape(
                    X_test.shape[0],
                    self.num_stats,
                    self.windows.size * self.center.size,
                    1,
                )
            )

            test_Y = df_test.select(
                ((~pl.col("model").str.contains("neutral")).cast(pl.Int8)).alias(
                    "neutral_flag"
                )
            )["neutral_flag"].to_numpy()

            # Same folder custom fvs name based on input VCF.
            self.output_prediction = (
                os.path.basename(self.test_data)
                .replace("fvs_", "")
                .replace(".parquet", "_predictions.txt")
            )

        else:
            try:
                # test_X, test_X_hap, test_X_params, test_Y = deepcopy(self.test_data)
                (X_test_ld, X_test_sfs, X_test_div, Y_test) = deepcopy(self.test_data)
            except:
                test_X, test_X_params, test_Y = deepcopy(self.test_data)

        test_X = test_X.reshape(
            -1, self.windows.size * self.center.size, self.num_stats
        )

        # batch size, image width, image height,number of channels
        if isinstance(self.model, str):
            model = tf.keras.models.load_model(self.model)
        else:
            model = self.model

        # make predictions
        # try:
        if self.lstm:
            preds = model.predict((test_X, test_X_hap))
        else:
            try:
                preds = model.predict((X_test_ld, X_test_sfs, X_test_div))
            except:
                preds = model.predict(test_X)
            # except:
            #     preds = model.predict(test_X.reshape(-1, 11, 21, 5, 1))
        preds = np.column_stack([1 - preds, preds])
        predictions = np.argmax(preds, axis=1)

        y_true = (test_X_params["model"] == "sweep").to_numpy().astype(int)
        # y_true = Y_test
        fpr, tpr, thresh = roc_curve(y_true, preds[:, 1])

        # 2) Pick threshold: FPR <= max_fpr, maximize TPR
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
        # predictions_class = np.where(probs >= best_thresh, "sweep", "neutral")

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

        # df_prediction = df_prediction.with_columns(pl.lit(np.vectorize(prediction_dict.get)(y_true)).alias('model'))
        if isinstance(self.test_data, str) and "test" not in self.test_data:
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

        self.prediction = df_prediction
        self.roc_curve()

        if self.output_folder is not None:
            df_prediction.write_csv(_output_prediction)

        return df_prediction

    def train_grl(self, _iter=1, cnn=None):
        """
        Trains the CNN model on the training data.

        This method preprocesses the data, sets up data augmentation, defines the model architecture,
        compiles the model, and fits it to the training data while saving the best model and training history.
        """
        tf = self.check_tf()

        if cnn is None:
            cnn = self.cnn_flexsweep

        (X_train, X_test, Y_train, Y_test, X_valid, Y_valid) = self.load_training_data()
        E_train = self.load_predict_data(20000)

        num_empirical = E_train.shape[0]
        dummy_labels = np.zeros((num_empirical, 2))  # or np.ones if you prefer

        # Combine X and Y
        X_combined = np.concatenate([X_train, E_train], axis=0)
        Y_combined = np.concatenate([Y_train, dummy_labels], axis=0)

        # Domain labels: 0 for simulation, 1 for empirical
        domain_labels = np.concatenate(
            [
                np.zeros((X_train.shape[0], 2)),  # e.g., [1, 0] → simulated
                np.ones((E_train.shape[0], 2)),  # e.g., [0, 1] → empirical
            ],
            axis=0,
        )

        # Create sample weights to ignore classifier loss on empirical data
        classifier_sample_weight = np.concatenate(
            [
                np.ones((X_train.shape[0],)),  # supervised
                np.zeros(
                    (E_train.shape[0],)
                ),  # unsupervised (ignored in classifier loss)
            ],
            axis=0,
        )

        # Domain task is supervised across all
        domain_sample_weight = np.ones((X_combined.shape[0],))

        # put model together
        input_to_model = tf.keras.Input(X_train.shape[1:])
        model = tf.keras.models.Model(
            inputs=[input_to_model], outputs=[cnn(input_to_model)]
        )
        # model_path = f"{self.output_folder}/model_{ihs}_{_iter}.keras"
        model_path = f"{self.output_folder}/model.keras"
        # weights_path = f"{self.output_folder}/model_{ihs}_weights.hdf5"

        metrics_measures = [
            # tf.keras.metrics.TruePositives(name="tp"),
            # tf.keras.metrics.FalsePositives(name="fp"),
            # tf.keras.metrics.TrueNegatives(name="tn"),
            # tf.keras.metrics.FalseNegatives(name="fn"),
            tf.keras.metrics.BinaryAccuracy(name="accuracy"),
            tf.keras.metrics.AUC(name="auc"),
            tf.keras.metrics.AUC(name="prc", curve="PR"),  # precision-recall curve
            tf.keras.metrics.Precision(name="precision"),
            # tf.keras.metrics.Recall(name="recall"),
        ]

        lr_decayed_fn = tf.keras.optimizers.schedules.CosineDecayRestarts(
            initial_learning_rate=0.001, first_decay_steps=300
        )

        opt_adam = tf.keras.optimizers.Adam(
            learning_rate=lr_decayed_fn, epsilon=0.0000001, amsgrad=True
        )

        earlystop = tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            min_delta=0.001,
            patience=10,
            verbose=1,
            mode="max",
            restore_best_weights=True,
        )

        checkpoint = tf.keras.callbacks.ModelCheckpoint(
            model_path,
            monitor="val_accuracy",
            verbose=1,
            save_best_only=True,
            mode="max",
        )
        # callbacks_list = [checkpoint]
        callbacks_list = [checkpoint, earlystop]

        start = time.time()

        input_shape = (11, 105, 1)
        model_input = tf.keras.Input(shape=input_shape)
        model = cnn_flexsweep_with_DANN(model_input)

        model.compile(
            optimizer=tf.keras.optimizers.Adam(1e-3),
            loss={
                "classifier": "binary_crossentropy",
                "discriminator": "categorical_crossentropy",
            },
            metrics={
                "classifier": [tf.keras.metrics.AUC(name="auc")],
                "discriminator": [tf.keras.metrics.BinaryAccuracy()],
            },
        )

        # Fit model with separate sample weights
        history = model.fit(
            x=X_combined,
            y={"classifier": Y_combined, "discriminator": domain_labels},
            sample_weight={
                "classifier": classifier_sample_weight,
                "discriminator": domain_sample_weight,
            },
            validation_data=(
                X_valid,
                {"classifier": Y_valid, "discriminator": np.zeros_like(Y_valid)},
            ),
            epochs=100,
            batch_size=32,
            callbacks=callbacks_list,
        )

        val_score = model.evaluate(
            validation_gen.flow(X_valid, Y_valid, batch_size=32),
            batch_size=32,
            steps=len(Y_valid) // 32,
        )
        test_score = model.evaluate(
            test_gen.flow(X_test, Y_test, batch_size=32),
            batch_size=32,
            steps=len(Y_test) // 32,
        )

        train_score = model.evaluate(
            datagen.flow(X_train, Y_train, batch_size=32),
            batch_size=32,
            steps=len(Y_train) // 32,
        )
        self.model = model
        print(
            "Training and testing model took {} seconds".format(
                round(time.time() - start, 3)
            )
        )

        df_history = pl.DataFrame(history.history)
        self.history = df_history

        if self.output_folder is not None:
            model.save(model_path)

        return self.model.evaluate(X_valid, Y_valid, verbose=0)[3]

    def load_predict_data(self):
        try:
            df_test = pl.read_parquet(self.test_data)
        except:
            df_test = pl.read_csv(self.test_data, separator=",")
        if self.num_stats < 17:
            df_test = df_test.select(
                [col for col in df_test.columns if "flip" not in col]
            )

        regions = df_test["iter"].to_numpy()

        test_X = df_test.select(
            pl.exclude("iter", "s", "f_i", "f_t", "t", "model")
        ).to_numpy()
        test_X_params = df_test.select("s", "f_i", "f_t", "t", "model", "iter")
        test_X = test_X.reshape(
            test_X.shape[0],
            self.num_stats,
            self.windows.size * self.center.size,
            1,
        )
        return test_X

    def roc_curve(self, _iter=1):
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

    def load_flat_training_data(self, _stats=None):
        """
        Loads training data and flattens the features for RandomForest.
        Returns flat (n_samples, n_features) arrays for X.
        """
        (
            X_train,
            X_test,
            Y_train,
            Y_test,
            X_valid,
            Y_valid,
            # X_train_hap,
            # X_test_hap,
            # X_valid_hap,
        ) = self.load_training_data(w=None, _stats=_stats)

        # Flatten 3D (samples, stats, windows) into 2D (samples, features)
        X_train = X_train.reshape((X_train.shape[0], -1))
        X_test = X_test.reshape((X_test.shape[0], -1))
        X_valid = X_valid.reshape((X_valid.shape[0], -1))

        # # Convert from one-hot to class integers
        # Y_train = np.argmax(Y_train, axis=1)
        # Y_test = np.argmax(Y_test, axis=1)
        # Y_valid = np.argmax(Y_valid, axis=1)

        return X_train, X_test, Y_train, Y_test, X_valid, Y_valid

    def train_random_forest(
        self,
        n_estimators=300,
        criterion="entropy",
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=3,
        max_features="sqrt",
        class_weight="balanced",
        n_jobs=-1,
        _stats=None,
    ):
        """
        Trains a RandomForest classifier on the training data, prints evaluation metrics,
        and plots the ROC curve if this is a binary classification.
        """
        import matplotlib.pyplot as plt
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import accuracy_score, classification_report

        (
            X_train,
            X_test,
            Y_train,
            Y_test,
            X_valid,
            Y_valid,
        ) = self.load_flat_training_data(_stats=_stats)

        print(f"Training RandomForest with {X_train.shape[1]} features...")

        clf = RandomForestClassifier(
            n_estimators=n_estimators,
            criterion=criterion,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            class_weight=class_weight,
            n_jobs=n_jobs,
        )
        clf.fit(X_train, Y_train)
        self.rf_model = clf  # save for later

        # 3) Evaluate on train / valid / test
        for name, X, Y in [
            ("Train", X_train, Y_train),
            ("Validation", X_valid, Y_valid),
            ("Test", X_test, Y_test),
        ]:
            y_pred = clf.predict(X)
            print(f"\n{name} accuracy: {accuracy_score(Y, y_pred):.4f}")
            print(classification_report(Y, y_pred))
            if clf.n_classes_ == 2:
                y_prob = clf.predict_proba(X)[:, 1]
                print(f"{name} ROC AUC: {roc_auc_score(Y, y_prob):.4f}")

        # 4) Plot ROC curve for test set if binary
        if clf.n_classes_ == 2:
            y_prob = clf.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(Y_test, y_prob)
            roc_auc = auc(fpr, tpr)

            plt.figure()
            plt.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.2f})")
            plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title("Random Forest ROC Curve")
            plt.legend(loc="lower right")
            plt.show()

        y_pred_test = clf.predict(X_test)
        prediction_dict = {0: "sweep", 1: "neutral"}

        pred_data = np.vectorize(prediction_dict.get)(y_pred_test)
        y_model = np.vectorize(prediction_dict.get)(Y_test)
        # Generate confusion matrix, normalized by true labels (rows sum to 1)
        cm = confusion_matrix(
            y_model,
            pred_data,
            labels=["neutral", "sweep"],
            normalize="true",
        )

        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm, display_labels=["neutral", "sweep"]
        )
        fig, ax = plt.subplots(figsize=(4, 4))
        cm_plot = disp.plot(cmap="Blues", ax=ax, colorbar=False)
        plt.show()

        return clf

    def train_xgboost(self, _stats=None):
        """
        Trains an XGBoost classifier on the training data and evaluates performance.
        """
        # Load preprocessed flattened data
        from sklearn.metrics import accuracy_score, classification_report
        import xgboost as xgb
        import matplotlib.pyplot as plt

        (
            X_train,
            X_test,
            Y_train,
            Y_test,
            X_valid,
            Y_valid,
        ) = self.load_flat_training_data(_stats=_stats)

        print(f"Training XGBoost with {X_train.shape[1]} features...")

        # Aggressive high-performance parameters
        params = {
            "n_estimators": 5000,  # more trees for better fit, with early stopping
            "learning_rate": 0.005,  # smaller LR for smoother convergence
            "max_depth": 10,  # deeper trees to capture complexity
            "subsample": 0.8,  # slightly higher sample fraction to reduce overfitting
            "colsample_bytree": 0.8,  # more features per tree to improve performance
            "gamma": 1.5,  # reduce gamma slightly for more splits
            "min_child_weight": 3,  # allow smaller leaves for detail
            "reg_alpha": 4,  # stronger L1 regularization
            "reg_lambda": 1.0,  # stronger L2 regularization
            "scale_pos_weight": 1,
            "use_label_encoder": False,
            "eval_metric": "auc",
            "tree_method": "hist",
            "n_jobs": 20,
            "tree_method": "hist",
            "device": "cuda",
        }
        clf = xgb.XGBClassifier(**params, early_stopping_rounds=30)

        # Fit with early stopping
        clf.fit(
            X_train,
            Y_train,
            eval_set=[(X_valid, Y_valid)],
            verbose=True,
        )

        self.xgb_model = clf  # Save model

        # Evaluation and ROC
        for name, X, Y in [
            ("Train", X_train, Y_train),
            # ("Validation", X_valid, Y_valid),
            # ("Test", X_test, Y_test),
        ]:
            y_pred = clf.predict(X)
            y_prob = clf.predict_proba(X)[:, 1]

            print(f"\n{name} Accuracy: {accuracy_score(Y, y_pred):.4f}")
            print(f"{name} ROC AUC: {roc_auc_score(Y, y_prob):.4f}")
            print(classification_report(Y, y_pred))

            # Plot ROC on test set
            y_prob_test = clf.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(Y_test, y_prob_test)
            roc_auc = auc(fpr, tpr)

            plt.figure()
            plt.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.2f})")
            plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title("XGBoost ROC Curve")
            plt.legend(loc="lower right")
            plt.grid(True)
            plt.tight_layout()
            plt.show()

        y_pred_test = clf.predict(X_test)
        prediction_dict = {0: "sweep", 1: "neutral"}

        pred_data = np.vectorize(prediction_dict.get)(y_pred_test)
        y_model = np.vectorize(prediction_dict.get)(Y_test)
        # Generate confusion matrix, normalized by true labels (rows sum to 1)
        cm = confusion_matrix(
            y_model,
            pred_data,
            labels=["neutral", "sweep"],
            normalize="true",
        )

        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm, display_labels=["neutral", "sweep"]
        )
        fig, ax = plt.subplots(figsize=(4, 4))
        cm_plot = disp.plot(cmap="Blues", ax=ax, colorbar=False)
        plt.show()

        return clf

    def train_catboost(self, _stats=None):
        """
        Trains a CatBoost classifier on the training data and evaluates performance.
        """
        # Load preprocessed flattened data
        import matplotlib.pyplot as plt
        from catboost import CatBoostClassifier, Pool
        from sklearn.metrics import accuracy_score, classification_report

        (
            X_train,
            X_test,
            Y_train,
            Y_test,
            X_valid,
            Y_valid,
        ) = self.load_flat_training_data(_stats=_stats)

        print(f"Training CatBoost with {X_train.shape[1]} features...")
        params = {
            "iterations": 5000,
            "learning_rate": 0.005,
            "depth": 10,  # Try 8-10, but not too high to avoid overfitting
            "l2_leaf_reg": 10,
            "early_stopping_rounds": 100,
            "verbose": 50,
            "task_type": "GPU",  # Use GPU for speed
            "loss_function": "Logloss",
            "eval_metric": "AUC",  # Or "Recall"
            "border_count": 128,
            "bagging_temperature": 1.0,
            "random_strength": 1.0,
            "grow_policy": "Depthwise",  # Try "SymmetricTree" too
            # No class_weights since classes are balanced
        }

        clf = CatBoostClassifier(**params)

        # Prepare Pool objects (CatBoost optimized data container)
        train_pool = Pool(X_train, label=Y_train)
        valid_pool = Pool(X_valid, label=Y_valid)

        # Train model with early stopping
        clf.fit(train_pool, eval_set=valid_pool, use_best_model=True)

        self.catboost_model = clf  # Save model

        # Evaluation and ROC
        for name, X, Y in [
            ("Train", X_train, Y_train),
            ("Validation", X_valid, Y_valid),
            ("Test", X_test, Y_test),
        ]:
            y_pred = clf.predict(X)
            y_prob = clf.predict_proba(X)[:, 1]

            print(f"\n{name} Accuracy: {accuracy_score(Y, y_pred):.4f}")
            print(f"{name} ROC AUC: {roc_auc_score(Y, y_prob):.4f}")
            print(classification_report(Y, y_pred))

        # Plot ROC on test set
        y_prob_test = clf.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(Y_test, y_prob_test)
        roc_auc = auc(fpr, tpr)

        plt.figure()
        plt.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.2f})")
        plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("CatBoost ROC Curve")
        plt.legend(loc="lower right")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        # --- Confusion Matrix (Test set, Percentages) ---
        y_pred_test = clf.predict(X_test)
        prediction_dict = {0: "sweep", 1: "neutral"}

        pred_data = np.vectorize(prediction_dict.get)(y_pred_test)
        y_model = np.vectorize(prediction_dict.get)(Y_test)
        # Generate confusion matrix, normalized by true labels (rows sum to 1)
        cm = confusion_matrix(
            y_model,
            pred_data,
            labels=["neutral", "sweep"],
            normalize="true",
        )

        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm, display_labels=["neutral", "sweep"]
        )
        fig, ax = plt.subplots(figsize=(4, 4))
        cm_plot = disp.plot(cmap="Blues", ax=ax, colorbar=False)
        plt.show()

        print(cm)
        return clf


def rank_probabilities(prediction, gene_coordinates, include_xy=False):
    """
    Ranks genes based on their associated maximum sweep probability.

    Parameters
    ----------
    data_dir : str
        Path to the directory containing prediction files (*.predictions.txt).
        Each file should contain genomic positions and their associated sweep probability.

    gene_coordinates : str
        Path to a BED file containing gene coordinates with the following columns:
        chromosome, start position, end position, gene ID, and strand.

    include_xy : bool, default=False
        If True, includes genes from X and Y chromosomes in the analysis.
        If False, only includes genes from autosomal chromosomes (1-22).

    Returns
    -------
    tuple
        A tuple containing:
        - DataFrame with ranked genes (columns: gene_id, rank, prob_sweep, start, end),
          sorted by rank in ascending order.
        - Integer count of genes that have the maximum probability score found.

    """
    # data_dir = "/labstorage/jmurgamoreno/bchak/mno/mno_260325/"
    # Always filtering
    # gene_coordinates = "//home/jmurgamoreno/train/ensembl_gene_coords_v109.bed"
    df_genes = (
        pl.read_csv(
            gene_coordinates,
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
    )

    df_genes_filtered = (
        df_genes.filter(pl.col("chr").is_in(np.arange(1, 23).astype(str)))
        .with_columns(("chr" + pl.col("chr")).alias("chr"))
        .sort("chr", "start")
    )

    df_pred = (
        pl.read_csv(prediction)
        .select("chr", "start", "end", "prob_sweep")
        .with_columns(
            # pl.col("chr").str.replace("chr", "").cast(pl.Int64),
            # pl.col("chr").str.replace("chr", ""),
            (((pl.col("start") + pl.col("end")) / 2)).alias("center_1").cast(pl.Int64),
            (((pl.col("start") + pl.col("end")) / 2) + 1)
            .alias("center_2")
            .cast(pl.Int64),
        )
        .select(pl.exclude(["start", "end"]))
        .rename({"center_1": "start", "center_2": "end"})
        .select("chr", "start", "end", "prob_sweep")
    ).sort("chr", "start")

    schema_a = [
        "chr_gene",
        "start_gene",
        "end_gene",
        "strand",
        "gene_id",
        "chr",
        "start",
        "end",
        "prob_sweep",
        "d",
    ]

    gene_bed = BedTool.from_dataframe(df_genes_filtered.to_pandas())
    pred_bed = BedTool.from_dataframe(
        df_pred.select(["chr", "start", "end", "prob_sweep"]).to_pandas()
    )

    # Get distances to 111 elements, sort by distance and higher probability
    w_closest_bed = pl.DataFrame(
        gene_bed.closest(pred_bed, d=True, k=111).to_dataframe(
            disable_auto_names=True, header=None
        ),
        schema=schema_a,
    ).sort(
        ["chr", "gene_id", "d", "prob_sweep"], descending=[False, False, False, True]
    )

    # Since it is already ordered, just get first gene_id row, where d is sorted min and prob_sweep is max, sorted d asc and prob_sweep desc at a time
    w_closest = (
        w_closest_bed.group_by(["chr", "gene_id"], maintain_order=True)
        .agg(
            [
                pl.first("start_gene").alias("start"),
                pl.first("end_gene").alias("end"),
                pl.first("prob_sweep"),
                pl.first("d").alias("d"),
            ]
        )
        .sort(["chr", "start"])
    )

    if rank_distance:
        min_d = w_closest.select("gene_id", "d").rename({"d": "d_min"})

        w_length = (
            w_closest_bed.join(min_d, on="gene_id")
            .filter(
                (pl.col("d") >= pl.col("d_min") - 5e5)
                & (pl.col("d") <= pl.col("d_min") + 5e5)
            )
            .group_by("gene_id")
            .agg(
                pl.col("prob_sweep").sum(),
                pl.col("d").sum(),
            )
            .sort(["prob_sweep", "d"], descending=[True, False])
        )

        w_rank = (
            w_length.with_columns(
                pl.col("prob_sweep")
                .rank(method="ordinal", descending=True)
                .alias("rank")
                .cast(pl.Int64)
            )
            .select("gene_id", "rank", "prob_sweep")
            .sort("rank")
        )

        return w_rank
    else:
        w_rank = (
            w_closest.with_columns(
                pl.col("prob_sweep")
                .rank(method="ordinal", descending=True)
                .alias("_rank")
                .cast(pl.Int64),
                pl.col("prob_sweep")
                .rank(method="min", descending=True)
                .alias("rank_min")
                .cast(pl.Int64),
            )
            .select("gene_id", "_rank", "rank_min", "prob_sweep", "chr", "start", "end")
            .sort("rank_min")
        )

        rank_min = (
            w_rank.filter(pl.col("rank_min").is_duplicated())["rank_min"]
            .unique()
            .to_numpy()
            .flatten()
        )
        all_ranks = w_rank["rank_min"].unique()

        gene_order = []
        for i in tqdm(all_ranks):
            genes_rank_order = w_rank.filter(pl.col("rank_min") == i).select("gene_id")

            if i not in rank_min:
                gene_order.append(genes_rank_order.select("gene_id"))
                continue

            w_closest_bed_filtered = w_closest_bed.filter(
                pl.col("gene_id").is_in(genes_rank_order.select("gene_id"))
            )
            min_d = w_closest_bed_filtered.group_by("chr", "gene_id").agg(
                pl.col("d").min().alias("d_min")
            )

            w_closest_bed_filtered = (
                w_closest_bed_filtered.join(min_d, on="gene_id")
                .filter(
                    (pl.col("d") >= pl.col("d_min") - 5e5)
                    & (pl.col("d") <= pl.col("d_min") + 5e5)
                )
                .select(w_closest_bed_filtered.columns)
            )

            genes_ranked = (
                w_closest_bed_filtered.group_by("gene_id")
                .agg(pl.col("prob_sweep").sum(), pl.col("d").sum())
                .sort(["prob_sweep", "d"], descending=[True, False])
                .select("gene_id")
            )

            gene_order.append(genes_ranked)

        df_gene_order = pl.concat(gene_order)
        df_gene_order = df_gene_order.with_columns(
            pl.arange(1, df_gene_order.height + 1).alias("rank")
        )
        w_closest_force_rank = w_closest_rank.join(df_gene_order, on="gene_id").sort(
            "rank"
        )

        n_rank_max = w_closest_force_rank.filter(
            pl.col("prob_sweep") == pl.col("prob_sweep").max()
        ).shape[0]

        return w_closest_force_rank.select("gene_id", "rank", "prob_sweep"), n_rank_max


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

    fig.savefig(f"{data_dir}/roc_curve_04.svg")

    return fig


def plot_confusion_matrix(
    cm, target_names, title="Confusion matrix", cmap="Blues", normalize=True
):
    """
    given a sklearn confusion matrix (cm), make a nice plot

    Arguments
    ---------
    cm:           confusion matrix from sklearn.metrics.confusion_matrix

    target_names: given classification classes such as [0, 1, 2]
                  the class names, for example: ['high', 'medium', 'low']

    title:        the text to display at the top of the matrix

    cmap:         the gradient of the values displayed from matplotlib.pyplot.cm
                  see http://matplotlib.org/examples/color/colormaps_reference.html
                  plt.get_cmap('jet') or plt.cm.Blues

    normalize:    If False, plot the raw numbers
                  If True, plot the proportions

    Usage
    -----
    plot_confusion_matrix(cm           = cm,                  # confusion matrix created by
                                                              # sklearn.metrics.confusion_matrix
                          normalize    = True,                # show proportions
                          target_names = y_labels_vals,       # list of names of the classes
                          title        = best_estimator_name) # title of graph

    Citiation
    ---------
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

    """
    import matplotlib.pyplot as plt
    import numpy as np
    import itertools

    accuracy = np.trace(cm) / np.sum(cm).astype("float")
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap("Blues")

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation="nearest", cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(
                j,
                i,
                "{:0.4f}".format(cm[i, j]),
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black",
            )
        else:
            plt.text(
                j,
                i,
                "{:,}".format(cm[i, j]),
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    plt.tight_layout()
    plt.ylabel("True label")
    plt.xlabel(
        "Predicted label\naccuracy={:0.4f}; misclass={:0.4f}".format(accuracy, misclass)
    )
    plt.show()


class PopGenAugmentation:
    """Custom augmentations designed for population genetics data"""

    @staticmethod
    def shuffle_axis(data, axis):
        idx = tf.range(tf.shape(data)[axis])
        shuffled_idx = tf.random.shuffle(idx)
        return tf.gather(data, shuffled_idx, axis=axis)

    @staticmethod
    def add_noise(data, stddev):
        noise = tf.random.normal(
            tf.shape(data), mean=0.0, stddev=stddev, dtype=data.dtype
        )
        return data + noise

    @staticmethod
    def dropout_axis(data, axis, rate):
        shape = [1] * len(data.shape)
        shape[axis] = tf.shape(data)[axis]
        mask = tf.cast(tf.random.uniform(shape) > rate, data.dtype)
        return data * mask

    @staticmethod
    def flip_horizontal(data):
        return tf.reverse(data, axis=[1])

    @staticmethod
    def rotate_90(data):
        return tf.transpose(data, perm=[0, 2, 1, 3])

    @staticmethod
    def swap_centers(data, swap_prob=0.3):
        batch_size = tf.shape(data)[0]
        num_centers = tf.shape(data)[1]

        # Create mask for whether to swap each center
        swap_mask = tf.random.uniform((batch_size, num_centers), 0, 1) < swap_prob

        # Generate shuffled indices for each batch
        shuffled = tf.argsort(tf.random.uniform((batch_size, num_centers)))

        # Gather shuffled data
        batch_indices = tf.reshape(tf.range(batch_size), (batch_size, 1))
        batch_indices = tf.tile(batch_indices, [1, num_centers])
        full_indices = tf.stack([batch_indices, shuffled], axis=-1)
        shuffled_data = tf.gather_nd(data, full_indices)

        # Expand mask and apply swap conditionally
        # swap_mask_exp = tf.cast(tf.expand_dims(tf.expand_dims(swap_mask, -1), -1), data.dtype)
        swap_mask_exp = tf.expand_dims(
            tf.expand_dims(swap_mask, -1), 1
        )  # keep bool dtype
        return data * (1 - swap_mask_exp) + shuffled_data * swap_mask_exp


class AugmentationPipeline:
    def __init__(
        self,
        rotation_prob=None,
        flip_prob=None,
        shuffle_centers_prob=None,
        shuffle_windows_prob=None,
        statistic_dropout_prob=None,
        noise_factor=None,
        dropout_centers_prob=None,
        dropout_windows_prob=None,
        swap_centers_prob=None,
    ):
        self.rotation_prob = rotation_prob
        self.flip_prob = flip_prob
        self.shuffle_centers_prob = shuffle_centers_prob
        self.shuffle_windows_prob = shuffle_windows_prob
        self.statistic_dropout_prob = statistic_dropout_prob
        self.noise_factor = noise_factor
        self.dropout_centers_prob = dropout_centers_prob
        self.dropout_windows_prob = dropout_windows_prob
        self.swap_centers_prob = swap_centers_prob

        self.aug = PopGenAugmentation()

    def augment_sample(self, data, label):
        if self.shuffle_centers_prob is not None:
            if tf.random.uniform(()) < self.shuffle_centers_prob:
                data = self.aug.shuffle_axis(data, axis=1)

        if self.shuffle_windows_prob is not None:
            if tf.random.uniform(()) < self.shuffle_windows_prob:
                data = self.aug.shuffle_axis(data, axis=2)

        if self.statistic_dropout_prob is not None:
            if tf.random.uniform(()) < 1.0:
                data = self.aug.dropout_axis(
                    data, axis=3, rate=self.statistic_dropout_prob
                )

        if self.noise_factor is not None:
            if tf.random.uniform(()) < 1.0:
                data = self.aug.add_noise(data, self.noise_factor)

        if self.dropout_centers_prob is not None:
            if tf.random.uniform(()) < 1.0:
                data = self.aug.dropout_axis(
                    data, axis=1, rate=self.dropout_centers_prob
                )

        if self.dropout_windows_prob is not None:
            if tf.random.uniform(()) < 1.0:
                data = self.aug.dropout_axis(
                    data, axis=2, rate=self.dropout_windows_prob
                )

        if self.swap_centers_prob is not None:
            if tf.random.uniform(()) < 1.0:
                data = self.aug.swap_centers(data, self.swap_centers_prob)

        if self.flip_prob is not None:
            if tf.random.uniform(()) < self.flip_prob:
                data = self.aug.flip_horizontal(data)

        if self.rotation_prob is not None:
            if tf.random.uniform(()) < self.rotation_prob:
                data = self.aug.rotate_90(data)

        return data, label

    def create_augmented_dataset(self, dataset, batch_size=32):
        return (
            dataset.map(self.augment_sample, num_parallel_calls=tf.data.AUTOTUNE)
            .batch(batch_size)
            .prefetch(tf.data.AUTOTUNE)
        )
