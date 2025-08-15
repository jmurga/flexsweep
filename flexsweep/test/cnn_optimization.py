import tensorflow as tf
from tensorflow.keras import layers, models
import keras_tuner as kt


def se_block(x, reduction=16):
    filters = x.shape[-1]
    se = layers.GlobalAveragePooling2D()(x)
    se = layers.Dense(filters // reduction, activation="relu")(se)
    se = layers.Dense(filters, activation="sigmoid")(se)
    se = layers.Reshape((1, 1, filters))(se)
    return layers.Multiply()([x, se])

def cbam_block(x, reduction=16):
    filters = x.shape[-1]

    # Channel attention
    avg_pool = layers.GlobalAveragePooling2D()(x)
    max_pool = layers.GlobalMaxPooling2D()(x)

    shared_dense_one = layers.Dense(filters // reduction, activation="relu")
    shared_dense_two = layers.Dense(filters)

    avg_fc = shared_dense_two(shared_dense_one(avg_pool))
    max_fc = shared_dense_two(shared_dense_one(max_pool))

    channel_attention = layers.Add()([avg_fc, max_fc])
    channel_attention = layers.Activation("sigmoid")(channel_attention)
    channel_attention = layers.Reshape((1, 1, filters))(channel_attention)
    x = layers.Multiply()([x, channel_attention])

    # Spatial attention
    avg_pool_spatial = layers.Lambda(
        lambda x: tf.reduce_mean(x, axis=-1, keepdims=True)
    )(x)
    max_pool_spatial = layers.Lambda(
        lambda x: tf.reduce_max(x, axis=-1, keepdims=True)
    )(x)

    concat = layers.Concatenate(axis=-1)([avg_pool_spatial, max_pool_spatial])
    spatial_attention = layers.Conv2D(
        1, kernel_size=7, padding="same", activation="sigmoid"
    )(concat)

    return layers.Multiply()([x, spatial_attention])


def build_model_flexible(hp, input_shape=(21, 5, 11)):
    """
    Highly flexible CNN for sweep vs. neutral classification.
    Expands the search space to explore:
      - Number of residual blocks (1–6)
      - Conv vs. SeparableConv, kernel sizes (1–7)
      - Whether to use dilation or not
      - Number of convs per block (1–3)
      - Activation choice (ReLU, LeakyReLU, Swish)
      - BatchNorm on/off
      - Squeeze-and-Excite on/off
      - Pooling type per block (None, Max, Avg)
      - Global pooling vs. flatten
      - Dense head width (32–512), dropout (0–0.7)
      - Learning‐rate schedule parameters
    """
    # 1) Input
    inputs = layers.Input(shape=input_shape)
    x = inputs

    # 2) Block‐level hyperparameters
    num_blocks = hp.Int("num_blocks", 1, 3)
    for i in range(num_blocks):
        shortcut = x

        # Choose block style: simple vs. residual
        block_type = hp.Choice(f"block_type_{i}", ["residual", "plain"])
        # Choose number of conv layers in this block
        conv_per_block = hp.Int(f"conv_count_{i}", 1, 5)

        # Loop over conv layers in this block
        for j in range(conv_per_block):
            # Conv type
            Conv = (
                layers.SeparableConv2D
                if hp.Choice(f"conv_type_{i}_{j}", ["standard", "separable"])
                == "separable"
                else layers.Conv2D
            )

            # Filters
            filters = hp.Int(f"filters_{i}_{j}", 32, 256, step=32)
            # Kernel size
            k = hp.Choice(f"kernel_size_{i}_{j}", [1, 3, 5, 7])
            # Dilation
            use_dilation = hp.Boolean(f"use_dilation_{i}_{j}")
            dilation = hp.Choice(f"dilation_{i}_{j}", [1, 2, 3]) if use_dilation else 1

            # Activation
            act = hp.Choice(f"activation_{i}_{j}", ["relu", "leaky_relu", "swish"])
            if act == "leaky_relu":
                activation_fn = lambda x: layers.LeakyReLU(alpha=0.1)(x)
            elif act == "swish":
                activation_fn = tf.keras.activations.swish
            else:
                activation_fn = tf.keras.activations.relu

            # First conv in this small sub-block
            x = Conv(
                filters, kernel_size=k, padding="same", dilation_rate=(1, dilation)
            )(x)
            # BatchNorm?
            if hp.Boolean(f"use_bn_{i}_{j}"):
                x = layers.BatchNormalization()(x)
            # Activation
            x = layers.Activation(activation_fn)(x)

        # Optionally add a pointwise conv to match channels for residual
        if block_type == "residual":
            if shortcut.shape[-1] != x.shape[-1]:
                shortcut = layers.Conv2D(x.shape[-1], 1, padding="same")(shortcut)
                if hp.Boolean(f"use_bn_proj_{i}"):
                    shortcut = layers.BatchNormalization()(shortcut)
            x = layers.Add()([shortcut, x])
            x = layers.Activation(activation_fn)(x)

        # Optional Squeeze-and-Excite
        if hp.Boolean(f"use_attention_{i}"):
            attn_type = hp.Choice(f"attn_type_{i}", ["se", "cbam"])
            if attn_type == "se":
                x = se_block(x)
            else:
                x = cbam_block(x)

        # Pooling choice for this block
        pool_type = hp.Choice(f"pool_type_{i}", ["none", "max", "avg"])
        if pool_type == "max":
            x = layers.MaxPooling2D(pool_size=(1, 2), padding="same")(x)
        elif pool_type == "avg":
            x = layers.AveragePooling2D(pool_size=(1, 2), padding="same")(x)
        # else: no pooling

        # Optional dropout after block
        if hp.Boolean(f"dropout_block_{i}"):
            rate = hp.Float(f"drop_rate_block_{i}", 0.0, 0.5, step=0.1)
            x = layers.Dropout(rate)(x)

    # 3) Final pooling choice
    final_pool = hp.Choice("final_pool", ["flatten", "global_avg", "global_max"])
    if final_pool == "flatten":
        x = layers.Flatten()(x)
    elif final_pool == "global_avg":
        x = layers.GlobalAveragePooling2D()(x)
    else:
        x = layers.GlobalMaxPooling2D()(x)

    # 4) Dense head hyperparameters
    dense_units = hp.Int("dense_units", 32, 512, step=32)
    x = layers.Dense(dense_units, activation="relu")(x)
    if hp.Boolean("dense_dropout"):
        rate = hp.Float("dense_drop_rate", 0.0, 0.7, step=0.1)
        x = layers.Dropout(rate)(x)

    # 5) Output
    outputs = layers.Dense(1, activation="sigmoid")(x)
    model = models.Model(inputs, outputs)

    # 6) Learning‐rate schedule hyperparameters
    initial_lr = hp.Float("initial_lr", 1e-5, 1e-2, sampling="log")
    first_decay = hp.Int("first_decay_steps", 50, 500, step=50)
    lr_schedule = tf.keras.optimizers.schedules.CosineDecayRestarts(
        initial_learning_rate=initial_lr,
        first_decay_steps=first_decay,
        t_mul=hp.Float("t_mul", 1.0, 2.0, step=0.5),
        m_mul=hp.Float("m_mul", 0.5, 1.0, step=0.25),
        alpha=hp.Float("alpha", 0.0, 0.5, step=0.1),
    )
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=lr_schedule,
        epsilon=hp.Float("epsilon", 1e-8, 1e-6, sampling="log"),
        amsgrad=hp.Boolean("amsgrad"),
    )

    smoothing = hp.Float("label_smoothing", 0.0, 0.1, step=0.01)
    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.BinaryCrossentropy(label_smoothing=smoothing),
        metrics=["accuracy"],
    )
    return model


# Example tuner using Hyperband
tuner = kt.Hyperband(
    build_model_flexible,
    objective="val_accuracy",
    max_epochs=20,
    factor=3,
    directory="flexsweep_tuner",
    project_name="sweep_classification",
)

# Then run:
earlystop = tf.keras.callbacks.EarlyStopping(
    monitor="val_accuracy",
    min_delta=0.001,
    patience=5,
    restore_best_weights=True,
)
tuner.search(
    X_train,
    Y_train,
    validation_data=(X_valid, Y_valid),
    epochs=20,
    batch_size=32,
    callbacks=[earlystop],
)

best_hps = tuner.get_best_hyperparameters(num_trials=5)

for idx, hp in enumerate(best_hps, 1):
    print(f"\n--- Top trial #{idx} ---")
    for key, val in hp.values.items():
        print(f"{key:25}: {val}")


top_trials = tuner.oracle.get_best_trials(num_trials=5)
top_hps = [t.hyperparameters.values for t in top_trials]


with open("hyperband_top_trials.json", "w") as f:
    json.dump(top_hps, f, indent=4)

with open("hyperband_top_trials.json", "r") as f:
    top_hps = json.load(f)


#######################

from tensorflow.keras import layers, models
import tensorflow as tf
import keras_tuner as kt

from tensorflow.keras import layers, models
import tensorflow as tf


class PositionalEncoding2D(layers.Layer):
    def call(self, x):
        # we can use the static shape for sin-encoding,
        # since input_shape=(21,5,11) is fixed at model-build
        h, w = x.shape[1], x.shape[2]
        # build a (1,h,w,1) pos grid
        pos = tf.range(h * w, dtype=x.dtype)
        pos = tf.reshape(pos, (h, w))
        pos = tf.reshape(pos, (1, h, w, 1))
        return x + tf.sin(pos)


class LearnablePositionalEncoding2D(layers.Layer):
    def build(self, input_shape):
        _, h, w, channels = input_shape
        self.pe = self.add_weight(
            shape=(1, h, w, channels),
            initializer='random_normal',
            trainable=True,
            name="learnable_pe"
        )
        super().build(input_shape)

    def call(self, x):
        return x + self.pe

def grouped_conv_by_stat(x):
    # channel dimension is known statically, so .shape[-1] works
    ch = x.shape[-1]
    hap, sfs = 5, 2
    div = ch - hap - sfs
    groups = [hap, sfs, max(div, 0)]
    slices = []
    idx = 0
    for count in groups:
        if count <= 0:
            idx += count
            continue
        # capture 'idx' and 'count' in default args
        group = layers.Lambda(
            lambda t, i=idx, c=count: t[..., i : i + c],
            output_shape=lambda in_shape, i=idx, c=count: (
                in_shape[0],
                in_shape[1],
                in_shape[2],
                c,
            ),
        )(x)
        conv = layers.Conv2D(64, 3, padding="same", activation="relu")(group)
        slices.append(conv)
        idx += count
    if not slices:
        return x
    return layers.Concatenate(axis=-1)(slices)


def se_block(input_tensor, ratio=8):
    channels = input_tensor.shape[-1]
    se = layers.GlobalAveragePooling2D()(input_tensor)
    se = layers.Dense(channels // ratio, activation="relu")(se)
    se = layers.Dense(channels, activation="sigmoid")(se)
    se = layers.Reshape((1, 1, channels))(se)
    return layers.Multiply()([input_tensor, se])


def cbam_block(x, reduction=16):
    filters = x.shape[-1]

    # Channel attention
    avg_pool = layers.GlobalAveragePooling2D()(x)
    max_pool = layers.GlobalMaxPooling2D()(x)

    shared_dense_one = layers.Dense(filters // reduction, activation="relu")
    shared_dense_two = layers.Dense(filters)

    avg_fc = shared_dense_two(shared_dense_one(avg_pool))
    max_fc = shared_dense_two(shared_dense_one(max_pool))

    channel_attention = layers.Add()([avg_fc, max_fc])
    channel_attention = layers.Activation("sigmoid")(channel_attention)
    channel_attention = layers.Reshape((1, 1, filters))(channel_attention)
    x = layers.Multiply()([x, channel_attention])

    # Spatial attention
    avg_pool_spatial = layers.Lambda(
        lambda x: tf.reduce_mean(x, axis=-1, keepdims=True)
    )(x)
    max_pool_spatial = layers.Lambda(
        lambda x: tf.reduce_max(x, axis=-1, keepdims=True)
    )(x)

    concat = layers.Concatenate(axis=-1)([avg_pool_spatial, max_pool_spatial])
    spatial_attention = layers.Conv2D(
        1, kernel_size=7, padding="same", activation="sigmoid"
    )(concat)

    return layers.Multiply()([x, spatial_attention])


class MHSA2D(layers.Layer):
    def __init__(self, num_heads=2, key_dim=8, **kwargs):
        super().__init__(**kwargs)
        self.mha = layers.MultiHeadAttention(num_heads=num_heads, key_dim=key_dim)

    def call(self, x):
        # x is [batch,h,w,ch] — turn into [batch,h*w,ch] for attention
        b, h, w, ch = tf.shape(x)[0], x.shape[1], x.shape[2], x.shape[3]
        flat = tf.reshape(x, (b, h * w, ch))
        att = self.mha(flat, flat)
        return tf.reshape(att, (b, h, w, ch))


def build_model_refactored(hp, input_shape=(21, 5, 11)):
    inp = layers.Input(shape=input_shape)
    x = inp

    if hp.Boolean("use_positional_encoding"):
        x = PositionalEncoding2D()(x)

    # if hp.Boolean("use_grouped_conv"):
    #     x = grouped_conv_by_stat(x)

    for i in range(hp.Int("num_blocks", 1, 3)):
        shortcut = x
        f = hp.Int(f"filters_{i}", 32, 128, step=32)
        k = hp.Choice(f"kernel_{i}", [1, 3, 5])
        c = hp.Int(f"conv_count_{i}", 1, 3)
        for j in range(c):
            x = layers.Conv2D(f, k, padding="same")(x)
            if hp.Boolean(f"bn_{i}_{j}"):
                x = layers.BatchNormalization()(x)
            x = layers.Activation("relu")(x)
        if hp.Boolean(f"use_se_{i}"):
            x = se_block(x)
        if hp.Boolean(f"use_pool_{i}"):
            p = hp.Choice(f"pool_type_{i}", ["max", "avg"])
            if p == "max":
                x = layers.MaxPooling2D((1, 2), padding="same")(x)
            else:
                x = layers.AveragePooling2D((1, 2), padding="same")(x)
        # only add if channels still match
        if shortcut.shape[-1] == x.shape[-1]:
            x = layers.Add()([shortcut, x])

    if hp.Boolean("use_mhsa"):
        x = MHSA2D(
            num_heads=hp.Int("attn_heads", 2, 4), key_dim=hp.Int("attn_keydim", 4, 16)
        )(x)

    # global pooling
    if hp.Choice("final_pool", ["avg", "max"]) == "avg":
        x = layers.GlobalAveragePooling2D()(x)
    else:
        x = layers.GlobalMaxPooling2D()(x)

    # dense head
    # x = layers.Dense(hp.Int("dense_units", 64, 256, step=64), activation="relu")(x)
    if hp.Boolean("dense_dropout"):
        x = layers.Dropout(hp.Float("dense_drop_rate", 0.1, 0.5, step=0.1))(x)

    out = layers.Dense(1, activation="sigmoid")(x)
    model = models.Model(inp, out)

    lr = tf.keras.optimizers.schedules.CosineDecayRestarts(
        initial_learning_rate=hp.Float("initial_lr", 1e-4, 1e-2, sampling="log"),
        first_decay_steps=hp.Int("decay_steps", 50, 500, step=50),
        t_mul=hp.Float("t_mul", 1.0, 2.0, step=0.5),
        m_mul=hp.Float("m_mul", 0.5, 1.0, step=0.25),
        alpha=hp.Float("alpha", 0.0, 0.5, step=0.1),
    )
    metrics_measures = [
        tf.keras.metrics.BinaryAccuracy(name="val_accuracy"),
        tf.keras.metrics.Precision(name="val_precision"),
        tf.keras.metrics.AUC(name="roc", curve="ROC"),
    ]
    opt = tf.keras.optimizers.Adam(learning_rate=lr)
    model.compile(
        optimizer=opt,
        loss=tf.keras.losses.BinaryCrossentropy(
            label_smoothing=hp.Float("label_smoothing", 0.0, 0.1, step=0.01)
        ),
        metrics=metrics_measures,
    )
    return model


def build_model_three_inputs(hp):
    # 1) define three inputs with fixed channel dims:
    hap_in = layers.Input(shape=(5, 21, 8), name="inp_haplotype")
    sfs_in = layers.Input(shape=(5, 21, 4), name="inp_sfs")
    div_in = layers.Input(shape=(5, 21, 6), name="inp_diversity")
    
    def one_branch(inp, name):
        x = inp
        # optional positional encoding
        if hp.Boolean(f"use_pe_{name}"):
            x = PositionalEncoding2D(name=f"pe_{name}")(x)
        
        # a few conv layers
        for bi in range(hp.Int(f"{name}_blocks", 1, 3)):
            f = hp.Int(f"{name}_filters_{bi}", 16, 64, step=16)
            k = hp.Choice(f"{name}_kernel_{bi}", [1, 3, 5])
            c = hp.Int(f"{name}_conv_count_{bi}", 1, 5)
            for ci in range(c):
                x = layers.Conv2D(f, k, padding="same", name=f"{name}_conv{bi}_{ci}")(x)
                if hp.Boolean(f"{name}_bn_{bi}_{ci}"):
                    x = layers.BatchNormalization(name=f"{name}_bn{bi}_{ci}")(x)
                x = layers.Activation("relu", name=f"{name}_act{bi}_{ci}")(x)
        
        # FIXED: Remove duplicate SE block logic
        if hp.Boolean(f"use_se_{name}"):
            attn_type = hp.Choice(f"attn_type_{name}", ["se", "cbam"])  # Make unique per branch
            if attn_type == "se":
                x = se_block(x)
            elif attn_type == "cbam":
                x = cbam_block(x)
        
        return x
    
    # 2) build each branch
    bh = one_branch(hap_in, "hap")
    bs = one_branch(sfs_in, "sfs")
    bd = one_branch(div_in, "div")
    
    # FIXED: Apply pooling consistently to ALL branches or NONE
    # Option 1: Global pooling flag (recommended)
    if hp.Boolean("use_global_pool"):
        pool_type = hp.Choice("global_pool_type", ["max", "avg"])
        if pool_type == "max":
            bh = layers.MaxPooling2D((1, 2), padding="same")(bh)
            bs = layers.MaxPooling2D((1, 2), padding="same")(bs)
            bd = layers.MaxPooling2D((1, 2), padding="same")(bd)
        else:
            bh = layers.AveragePooling2D((1, 2), padding="same")(bh)
            bs = layers.AveragePooling2D((1, 2), padding="same")(bs)
            bd = layers.AveragePooling2D((1, 2), padding="same")(bd)
    
    # 3) merge
    x = layers.Concatenate(axis=-1, name="merge_stats")([bh, bs, bd])
    
    # 4) optional MHSA
    if hp.Boolean("use_mhsa"):
        x = MHSA2D(
            num_heads=hp.Int("mhsa_heads", 2, 4),
            key_dim=hp.Int("mhsa_keydim", 4, 16),
            name="mhsa",
        )(x)
    
    # 5) global pooling
    if hp.Choice("final_global_pool", ["avg", "max"]) == "avg":
        x = layers.GlobalAveragePooling2D(name="pool_avg")(x)
    else:
        x = layers.GlobalMaxPooling2D(name="pool_max")(x)
    
    # 6) dense head
    units = hp.Int("dense_units", 64, 256, step=64)
    # x = layers.Dense(units, activation="relu", name="dense")(x)
    if hp.Boolean("use_dropout"):
        rate = hp.Float("dropout_rate", 0.1, 0.5, step=0.1)
        x = layers.Dropout(rate, name="dropout")(x)
    
    out = layers.Dense(1, activation="sigmoid", name="out")(x)
    model = keras.Model(inputs=[hap_in, sfs_in, div_in], outputs=out)
    
    # compile with your LR schedule
    lr = tf.keras.optimizers.schedules.CosineDecayRestarts(
        initial_learning_rate=hp.Float("lr", 1e-4, 1e-2, sampling="log"),
        first_decay_steps=hp.Int("decay_steps", 50, 500, step=50),
        t_mul=hp.Float("t_mul", 1.0, 2.0, step=0.5),
        m_mul=hp.Float("m_mul", 0.5, 1.0, step=0.25),
        alpha=hp.Float("alpha", 0.0, 0.5, step=0.1),
    )
    opt = tf.keras.optimizers.Adam(lr)
    model.compile(
        optimizer=opt,
        loss="binary_crossentropy",
        metrics=["accuracy", "precision", "auc"],
    )
    return model

# 1) Instantiate the tuner
tuner = kt.Hyperband(
    build_model_with_branch_weighting,
    objective="val_auc",
    max_epochs=20,
    factor=2,
    directory="tuner_group",
    project_name="flexsweep_group",
)

# 2) Early‐stopping callback
stop_early = tf.keras.callbacks.EarlyStopping(monitor="val_auc", patience=5)

# 3) Kick off the search
tuner.search(
    [X_train_ld,X_train_sfs,X_train_div],
    Y_train,
    epochs=20,
    validation_data=([X_valid_ld,X_valid_sfs,X_valid_div], Y_valid),
    batch_size=32,
    callbacks=[stop_early],
)


######
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

# Custom layers for branch weighting

# Custom layers for branch weighting
class LearnableWeightingConcat(layers.Layer):
    """Learnable weighting for branches with different channel dimensions - concatenates then weights"""
    def __init__(self, num_branches, **kwargs):
        super().__init__(**kwargs)
        self.num_branches = num_branches
        
    def build(self, input_shape):
        # Initialize weights to be equal (1/num_branches each)
        self.branch_weights = self.add_weight(
            name='branch_weights',
            shape=(self.num_branches,),
            initializer='ones',
            trainable=True
        )
        super().build(input_shape)
    
    def call(self, inputs):
        # Apply softmax to ensure weights sum to 1
        normalized_weights = tf.nn.softmax(self.branch_weights)
        
        # First concatenate all branches
        concat_features = layers.Concatenate(axis=-1)(inputs)
        
        # Calculate channel splits for each branch
        channel_starts = [0]
        for inp in inputs[:-1]:
            channel_starts.append(channel_starts[-1] + inp.shape[-1])
        
        # Apply weights to each branch's channels
        weighted_branches = []
        for i, (start_idx, inp) in enumerate(zip(channel_starts, inputs)):
            end_idx = start_idx + inp.shape[-1]
            branch_channels = concat_features[..., start_idx:end_idx]
            weighted_branches.append(branch_channels * normalized_weights[i])
        
        return layers.Concatenate(axis=-1)(weighted_branches)

class LearnableWeightingPooled(layers.Layer):
    """Learnable weighting after global pooling - works with different channel dimensions"""
    def __init__(self, num_branches, **kwargs):
        super().__init__(**kwargs)
        self.num_branches = num_branches
        
    def build(self, input_shape):
        # Initialize weights to be equal (1/num_branches each)
        self.branch_weights = self.add_weight(
            name='branch_weights',
            shape=(self.num_branches,),
            initializer='ones',
            trainable=True
        )
        super().build(input_shape)
    
    def call(self, inputs):
        # Apply global pooling to each branch first
        pooled_branches = []
        for inp in inputs:
            pooled = layers.GlobalAveragePooling2D()(inp)
            pooled_branches.append(pooled)
        
        # Apply softmax to ensure weights sum to 1
        normalized_weights = tf.nn.softmax(self.branch_weights)
        
        # Weight each pooled branch
        weighted_branches = []
        for i, pooled_branch in enumerate(pooled_branches):
            weighted_branches.append(pooled_branch * normalized_weights[i])
        
        return layers.Concatenate()(weighted_branches)

class AttentionWeightingPooled(layers.Layer):
    """Attention-based weighting after global pooling"""
    def __init__(self, num_branches, hidden_units=32, **kwargs):
        super().__init__(**kwargs)
        self.num_branches = num_branches
        self.hidden_units = hidden_units
        
    def build(self, input_shape):
        # Small network to compute attention weights
        self.attention_dense1 = layers.Dense(self.hidden_units, activation='relu')
        self.attention_dense2 = layers.Dense(self.num_branches, activation='softmax')
        super().build(input_shape)
    
    def call(self, inputs):
        # Apply global pooling to each branch first
        pooled_branches = []
        for inp in inputs:
            pooled = layers.GlobalAveragePooling2D()(inp)
            pooled_branches.append(pooled)
        
        # Use concatenated pooled features to compute attention
        concat_pooled = layers.Concatenate()(pooled_branches)
        
        # Compute attention weights
        attention_weights = self.attention_dense1(concat_pooled)
        attention_weights = self.attention_dense2(attention_weights)
        
        # Apply weights to each branch
        weighted_branches = []
        for i, pooled_branch in enumerate(pooled_branches):
            weight = tf.expand_dims(attention_weights[:, i], -1)
            weighted_branches.append(pooled_branch * weight)
        
        return layers.Concatenate()(weighted_branches)

def build_model_with_branch_weighting(hp):
    # 1) Define three inputs
    hap_in = layers.Input(shape=(5, 21, 8), name="inp_haplotype")
    sfs_in = layers.Input(shape=(5, 21, 4), name="inp_sfs")
    div_in = layers.Input(shape=(5, 21, 6), name="inp_diversity")
    
    def one_branch(inp, name):
        x = inp
        # Optional positional encoding
        if hp.Boolean(f"use_pe_{name}"):
            x = PositionalEncoding2D(name=f"pe_{name}")(x)
        
        # Conv layers
        for bi in range(hp.Int(f"{name}_blocks", 1, 3)):
            f = hp.Int(f"{name}_filters_{bi}", 16, 64, step=16)
            k = hp.Choice(f"{name}_kernel_{bi}", [1, 3, 5])
            c = hp.Int(f"{name}_conv_count_{bi}", 1, 5)
            for ci in range(c):
                x = layers.Conv2D(f, k, padding="same", name=f"{name}_conv{bi}_{ci}")(x)
                if hp.Boolean(f"{name}_bn_{bi}_{ci}"):
                    x = layers.BatchNormalization(name=f"{name}_bn{bi}_{ci}")(x)
                x = layers.Activation("relu", name=f"{name}_act{bi}_{ci}")(x)
        
        # Attention blocks
        if hp.Boolean(f"use_se_{name}"):
            attn_type = hp.Choice(f"attn_type_{name}", ["se", "cbam"])
            if attn_type == "se":
                x = se_block(x)
            elif attn_type == "cbam":
                x = cbam_block(x)
        
        return x
    
    # 2) Build each branch
    bh = one_branch(hap_in, "hap")
    bs = one_branch(sfs_in, "sfs")
    bd = one_branch(div_in, "div")
    
    # 3) Apply consistent pooling if needed
    if hp.Boolean("use_global_pool"):
        pool_type = hp.Choice("global_pool_type", ["max", "avg"])
        if pool_type == "max":
            bh = layers.MaxPooling2D((1, 2), padding="same")(bh)
            bs = layers.MaxPooling2D((1, 2), padding="same")(bs)
            bd = layers.MaxPooling2D((1, 2), padding="same")(bd)
        else:
            bh = layers.AveragePooling2D((1, 2), padding="same")(bh)
            bs = layers.AveragePooling2D((1, 2), padding="same")(bs)
            bd = layers.AveragePooling2D((1, 2), padding="same")(bd)
    
    # 4) BRANCH WEIGHTING - Choose the weighting strategy
    weighting_strategy = hp.Choice("branch_weighting", ["none", "learnable", "attention", "fixed"])
    
    if weighting_strategy == "none":
        # Original concatenation approach
        x = layers.Concatenate(axis=-1, name="merge_stats")([bh, bs, bd])
    
    elif weighting_strategy == "learnable":
        # Learnable weights for each branch
        x = LearnableWeighting(num_branches=3, name="learnable_weights")([bh, bs, bd])
    
    elif weighting_strategy == "attention":
        # Attention-based weighting
        x = AttentionWeighting(num_branches=3, name="attention_weights")([bh, bs, bd])
    
    elif weighting_strategy == "fixed":
        # Fixed weights that you can tune as hyperparameters
        hap_weight = hp.Float("hap_weight", 0.1, 1.0, step=0.1)
        sfs_weight = hp.Float("sfs_weight", 0.1, 1.0, step=0.1)
        div_weight = hp.Float("div_weight", 0.1, 1.0, step=0.1)
        
        # Normalize weights
        total_weight = hap_weight + sfs_weight + div_weight
        hap_weight_norm = hap_weight / total_weight
        sfs_weight_norm = sfs_weight / total_weight
        div_weight_norm = div_weight / total_weight
        
        # Apply weights
        bh_weighted = layers.Lambda(lambda x: x * hap_weight_norm, name="hap_weighted")(bh)
        bs_weighted = layers.Lambda(lambda x: x * sfs_weight_norm, name="sfs_weighted")(bs)
        bd_weighted = layers.Lambda(lambda x: x * div_weight_norm, name="div_weighted")(bd)
        
        x = layers.Add(name="weighted_sum")([bh_weighted, bs_weighted, bd_weighted])
    
    # 5) Optional MHSA after weighting
    if hp.Boolean("use_mhsa"):
        x = MHSA2D(
            num_heads=hp.Int("mhsa_heads", 2, 4),
            key_dim=hp.Int("mhsa_keydim", 4, 16),
            name="mhsa",
        )(x)
    
    # 6) Global pooling
    if hp.Choice("final_global_pool", ["avg", "max"]) == "avg":
        x = layers.GlobalAveragePooling2D(name="pool_avg")(x)
    else:
        x = layers.GlobalMaxPooling2D(name="pool_max")(x)
    
    # 7) Dense head
    units = hp.Int("dense_units", 64, 256, step=64)
    # x = layers.Dense(units, activation="relu", name="dense")(x)
    if hp.Boolean("use_dropout"):
        rate = hp.Float("dropout_rate", 0.1, 0.5, step=0.1)
        x = layers.Dropout(rate, name="dropout")(x)
    
    out = layers.Dense(1, activation="sigmoid", name="out")(x)
    model = keras.Model(inputs=[hap_in, sfs_in, div_in], outputs=out)
    
    # Compile
    lr = tf.keras.optimizers.schedules.CosineDecayRestarts(
        initial_learning_rate=hp.Float("lr", 1e-4, 1e-2, sampling="log"),
        first_decay_steps=hp.Int("decay_steps", 50, 500, step=50),
        t_mul=hp.Float("t_mul", 1.0, 2.0, step=0.5),
        m_mul=hp.Float("m_mul", 0.5, 1.0, step=0.25),
        alpha=hp.Float("alpha", 0.0, 0.5, step=0.1),
    )
    opt = tf.keras.optimizers.Adam(lr)
    model.compile(
        optimizer=opt,
        loss="binary_crossentropy",
        metrics=["accuracy", "precision", "auc"],
    )
    return model

################
Best val_auc So Far: 0.9297690391540527

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Stage 1: Core Architecture - Branch Structure and Basic Conv Layers
def build_model_stage1(hp):
    """Stage 1: Focus on core conv architecture and basic branch structure"""
    
    hap_in = layers.Input(shape=(5, 21, 8), name="inp_haplotype")
    sfs_in = layers.Input(shape=(5, 21, 4), name="inp_sfs")
    div_in = layers.Input(shape=(5, 21, 6), name="inp_diversity")
    
    def one_branch(inp, name):
        x = inp
        
        # Core conv parameters - most impactful
        for bi in range(hp.Int(f"{name}_blocks", 1, 3)):
            f = hp.Int(f"{name}_filters_{bi}", 16, 64, step=16)
            k = hp.Choice(f"{name}_kernel_{bi}", [1, 3, 5])
            c = hp.Int(f"{name}_conv_count_{bi}", 1, 5)
            
            for ci in range(c):
                x = layers.Conv2D(f, k, padding="same", name=f"{name}_conv{bi}_{ci}")(x)
                # Always use BatchNorm in Stage 1 for stability
                x = layers.BatchNormalization(name=f"{name}_bn{bi}_{ci}")(x)
                x = layers.Activation("relu", name=f"{name}_act{bi}_{ci}")(x)
        
        return x
    
    bh = one_branch(hap_in, "hap")
    bs = one_branch(sfs_in, "sfs")
    bd = one_branch(div_in, "div")
    
    # Simple concatenation for Stage 1
    x = layers.Concatenate(axis=-1, name="merge_stats")([bh, bs, bd])
    
    # Basic global pooling
    if hp.Choice("global_pool", ["avg", "max"]) == "avg":
        x = layers.GlobalAveragePooling2D(name="pool_avg")(x)
    else:
        x = layers.GlobalMaxPooling2D(name="pool_max")(x)
    
    # Basic dropout
    if hp.Boolean("use_dropout"):
        rate = hp.Float("dropout_rate", 0.1, 0.5, step=0.1)
        x = layers.Dropout(rate, name="dropout")(x)
    
    out = layers.Dense(1, activation="sigmoid", name="out")(x)
    model = keras.Model(inputs=[hap_in, sfs_in, div_in], outputs=out)
    
    # Basic optimizer
    lr = hp.Float("lr", 1e-4, 1e-2, sampling="log")
    opt = tf.keras.optimizers.Adam(lr)
    model.compile(
        optimizer=opt,
        loss="binary_crossentropy",
        metrics=["accuracy", "precision", "auc"],
    )
    return model


# 1) Instantiate the tuner
tuner = kt.Hyperband(
    build_model_stage2_fixed,
    objective="val_accuracy",
    max_epochs=20,
    factor=3,
    directory="tuner_group2",
    project_name="flexsweep_group2",
)

# 2) Early‐stopping callback
stop_early = tf.keras.callbacks.EarlyStopping(monitor="val_accuracy",mode='min', patience=5)

# 3) Kick off the search
tuner.search(
    [X_train_ld,X_train_sfs,X_train_div],
    Y_train,
    epochs=20,
    validation_data=([X_valid_ld,X_valid_sfs,X_valid_div], Y_valid),
    batch_size=32,
    callbacks=[stop_early],
)

# Stage 2: Add Attention Mechanisms and BatchNorm Options
def build_model_stage2_fixed(hp):
    """Stage 2: Add attention blocks and BatchNorm options using fixed best Stage 1 values"""

    hap_in = layers.Input(shape=(5, 21, 8), name="inp_haplotype")
    sfs_in = layers.Input(shape=(5, 21, 4), name="inp_sfs")
    div_in = layers.Input(shape=(5, 21, 6), name="inp_diversity")

    def one_branch(inp, name, num_blocks, filters_list, kernel_list, conv_count_list):
        x = inp

        if hp.Boolean(f"use_pe_{name}"):
            x = LearnablePositionalEncoding2D(name=f"pe_{name}")(x)

        for bi in range(num_blocks):
            f = filters_list[bi]
            k = kernel_list[bi]
            c = conv_count_list[bi]
            for ci in range(c):
                x = layers.Conv2D(f, k, padding="same", name=f"{name}_conv{bi}_{ci}")(x)
                # BatchNorm: tunable in Stage 2
                if hp.Boolean(f"{name}_bn_{bi}_{ci}"):
                    x = layers.BatchNormalization(name=f"{name}_bn{bi}_{ci}")(x)
                x = layers.Activation("relu", name=f"{name}_act{bi}_{ci}")(x)

        # Attention: tunable in Stage 2
        if hp.Boolean(f"use_attention_{name}"):
            attn_type = hp.Choice(f"attention_type_{name}", ["se", "cbam"])
            if attn_type == "se":
                x = se_block(x)
            elif attn_type == "cbam":
                x = cbam_block(x)
        return x

    # Use exactly best configuration from Stage 1
    bh = one_branch(
        hap_in, "hap", 3,
        [32, 64, 48],
        [5, 5, 1],
        [2, 1, 2]
    )
    bs = one_branch(
        sfs_in, "sfs", 1,
        [32, 16, 48],
        [5, 3, 3],
        [4, 5, 5]
    )
    bd = one_branch(
        div_in, "div", 3,
        [48, 32, 48],
        [5, 3, 1],
        [1, 4, 4]
    )

    # Merge
    x = layers.Concatenate(axis=-1, name="merge_stats")([bh, bs, bd])

    # Global pooling: fixed to best
    x = layers.GlobalAveragePooling2D(name="pool_avg")(x)

    out = layers.Dense(1, activation="sigmoid", name="out")(x)
    model = keras.Model(inputs=[hap_in, sfs_in, div_in], outputs=out)

    # Optimizer: fixed lr from Stage 1 best
    opt = tf.keras.optimizers.Adam(learning_rate=0.00010007891408507613)

    model.compile(
        optimizer=opt,
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.Precision(), tf.keras.metrics.AUC(name="auc")]
    )

    return model

# Stage 3: Add Branch Weighting
def build_model_stage3_fixed(hp):
    """Stage 3: Add branch weighting using fixed best Stage 2 config"""

    hap_in = layers.Input(shape=(5, 21, 8), name="inp_haplotype")
    sfs_in = layers.Input(shape=(5, 21, 4), name="inp_sfs")
    div_in = layers.Input(shape=(5, 21, 6), name="inp_diversity")

    def one_branch_fixed(inp, name, num_blocks, filters_list, kernel_list, conv_count_list, bn_dict, use_attention, attention_type):

        x = inp


        if hp.Boolean(f"use_pe_{name}"):
            x = LearnablePositionalEncoding2D(name=f"pe_{name}")(x)


        for bi in range(num_blocks):
            f = filters_list[bi]
            k = kernel_list[bi]
            c = conv_count_list[bi]
            for ci in range(c):
                x = layers.Conv2D(f, k, padding="same", name=f"{name}_conv{bi}_{ci}")(x)
                if bn_dict.get(f"{name}_bn_{bi}_{ci}", False):
                    x = layers.BatchNormalization(name=f"{name}_bn{bi}_{ci}")(x)
                x = layers.Activation("relu", name=f"{name}_act{bi}_{ci}")(x)

        if use_attention:
            if attention_type == "se":
                x = se_block(x)
            elif attention_type == "cbam":
                x = cbam_block(x)
        return x

    bh = one_branch_fixed(
        hap_in, "hap", 3,
        [32, 64, 48],
        [5, 5, 1],
        [2, 1, 2],
        {
            'use_pe_hap': False,
            'hap_bn_0_0': False, 
            'hap_bn_0_1': False,
            'hap_bn_1_0': True,  
            'hap_bn_2_0': False, 
            'hap_bn_2_1': False,
        },
        True,
        'se',
    )

    bs = one_branch_fixed(
        sfs_in, "sfs", 1,
        [32, 16, 48],
        [5, 3, 3],
        [4, 5, 5],
        {
            'use_pe_sfs': True,                  
            'sfs_bn_0_0': True,
            'sfs_bn_0_1': False,      
            'sfs_bn_0_2': False,               
            'sfs_bn_0_3': False,
        },
        False,
        "cbam"
    )

    bd = one_branch_fixed(
        div_in, "div", 3,
        [48, 32, 48],
        [5, 3, 1],
        [1, 4, 4],
        {
            'use_pe_div': False,
            'div_bn_0_0': True,
            'div_bn_1_0': True,
            'div_bn_1_1': False,
            'div_bn_1_2': True,
            'div_bn_1_3': False,
            'div_bn_2_0': True,
            'div_bn_2_1': False,
            'div_bn_2_2': False,
            'div_bn_2_3': False,
        },
        True,
        "cbam"
    )

    # Stage 3: test branch weighting strategy
    weighting_strategy = hp.Choice("branch_weighting", ["none", "learnable_concat", "learnable_pooled", "attention_pooled"])

    if weighting_strategy == "none":
        x = layers.Concatenate(axis=-1, name="merge_stats")([bh, bs, bd])
        skip_global_pool = False
    elif weighting_strategy == "learnable_concat":
        x = LearnableWeightingConcat(num_branches=3, name="learnable_weights_concat")([bh, bs, bd])
        skip_global_pool = False
    elif weighting_strategy == "learnable_pooled":
        x = LearnableWeightingPooled(num_branches=3, name="learnable_weights_pooled")([bh, bs, bd])
        skip_global_pool = True
    elif weighting_strategy == "attention_pooled":
        x = AttentionWeightingPooled(num_branches=3, name="attention_weights_pooled")([bh, bs, bd])
        skip_global_pool = True

    # Global pooling if needed
    if not skip_global_pool:
        x = layers.GlobalAveragePooling2D(name="pool_avg")(x)

    out = layers.Dense(1, activation="sigmoid", name="out")(x)
    model = keras.Model(inputs=[hap_in, sfs_in, div_in], outputs=out)

    # Optimizer: fixed best from Stage 2
    opt = tf.keras.optimizers.Adam(learning_rate=0.00010007891408507613)

    model.compile(
        optimizer=opt,
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.Precision(), tf.keras.metrics.AUC(name="auc")]
    )

    return model

# Stage 4: Add MHSA and Positional Encoding
def build_model_stage4_fixed(hp):
    """Stage 4: Add MHSA and positional encoding to best Stage 3 config"""

    hap_in = layers.Input(shape=(5, 21, 8), name="inp_haplotype")
    sfs_in = layers.Input(shape=(5, 21, 4), name="inp_sfs")
    div_in = layers.Input(shape=(5, 21, 6), name="inp_diversity")


    def one_branch_fixed(inp, name, num_blocks, filters_list, kernel_list, conv_count_list, bn_dict, use_attention, attention_type, use_pe):

        x = inp
        if use_pe:
            x = LearnablePositionalEncoding2D(name=f"pe_{name}")(x)


        for bi in range(num_blocks):
            f = filters_list[bi]
            k = kernel_list[bi]
            c = conv_count_list[bi]
            for ci in range(c):
                x = layers.Conv2D(f, k, padding="same", name=f"{name}_conv{bi}_{ci}")(x)
                if bn_dict.get(f"{name}_bn_{bi}_{ci}", False):
                    x = layers.BatchNormalization(name=f"{name}_bn{bi}_{ci}")(x)
                x = layers.Activation("relu", name=f"{name}_act{bi}_{ci}")(x)

        if use_attention:
            if attention_type == "se":
                x = se_block(x)
            elif attention_type == "cbam":
                x = cbam_block(x)
        return x

    bh = one_branch_fixed(
        hap_in, "hap", 3,
        [32, 64, 48],
        [5, 5, 1],
        [2, 1, 2],
        {
            'use_pe_hap': False,
            'hap_bn_0_0': False, 
            'hap_bn_0_1': False,
            'hap_bn_1_0': True,  
            'hap_bn_2_0': False, 
            'hap_bn_2_1': False,
        },
        True,
        'se',
        True
    )

    bs = one_branch_fixed(
        sfs_in, "sfs", 1,
        [32, 16, 48],
        [5, 3, 3],
        [4, 5, 5],
        {
            'use_pe_sfs': True,                  
            'sfs_bn_0_0': True,
            'sfs_bn_0_1': False,      
            'sfs_bn_0_2': False,               
            'sfs_bn_0_3': False,
        },
        False,
        "cbam",
        True
    )

    bd = one_branch_fixed(
        div_in, "div", 3,
        [48, 32, 48],
        [5, 3, 1],
        [1, 4, 4],
        {
            'use_pe_div': True,
            'div_bn_0_0': True,
            'div_bn_1_0': True,
            'div_bn_1_1': False,
            'div_bn_1_2': True,
            'div_bn_1_3': False,
            'div_bn_2_0': True,
            'div_bn_2_1': False,
            'div_bn_2_2': False,
            'div_bn_2_3': False,
        },
        True,
        "cbam",
        True
    )

    # Fixed Stage 3 branch weighting best config
    x = LearnableWeightingConcat(num_branches=3, name="learnable_weights_concat")([bh, bs, bd])

    # Stage 4: MHSA layer (tunable hyperparameters)
    if hp.Boolean("use_mhsa"):

        num_heads = hp.Int("mhsa_heads", 2, 4)
        key_dim = hp.Int("mhsa_keydim", 4, 16)
        
        x = MHSA2D(num_heads=num_heads, key_dim=key_dim, name="mhsa")(x)

        # This is correct — only one pooling
        mhsa_pool = hp.Choice("mhsa_pooling", ["avg", "max", "concat"])
        if mhsa_pool == "avg":
            x = layers.GlobalAveragePooling2D(name="pool_avg_mhsa")(x)
        elif mhsa_pool == "max":
            x = layers.GlobalMaxPooling2D(name="pool_max_mhsa")(x)
        else:
            # Make sure this is right after MHSA
            _avg = layers.GlobalAveragePooling2D()(x)
            _max = layers.GlobalMaxPooling2D()(x)
            x = layers.Concatenate(name="mhsa_concat_pool")([_avg, _max])

    else:
        non_mhsa_pool = hp.Choice("mhsa_pooling", ["avg", "max", "concat"])
       if non_mhsa_pool == "avg":
            x = layers.GlobalAveragePooling2D(name="pool_avg_mhsa")(x)
        elif non_mhsa_pool == "max":
            x = layers.GlobalMaxPooling2D(name="pool_max_mhsa")(x)
        else:
            _avg = layers.GlobalAveragePooling2D()(x)
            _max = layers.GlobalMaxPooling2D()(x)
            x = layers.Concatenate(name="mhsa_concat_pool")([_avg, _max])


    out = layers.Dense(1, activation="sigmoid", name="out")(x)

    model = keras.Model(inputs=[hap_in, sfs_in, div_in], outputs=out)

    opt = tf.keras.optimizers.Adam(learning_rate=0.00010007891408507613)

    model.compile(
        optimizer=opt,
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.Precision(), tf.keras.metrics.AUC(name="auc")]
    )

    return model

# Stage 5: Advanced Optimization - Learning Rate Schedules and Pooling Options
def build_model_stage5_fixed(hp):
    """Stage 4: Add MHSA and positional encoding to best Stage 3 config"""

    hap_in = layers.Input(shape=(5, 21, 8), name="inp_haplotype")
    sfs_in = layers.Input(shape=(5, 21, 4), name="inp_sfs")
    div_in = layers.Input(shape=(5, 21, 6), name="inp_diversity")


    def one_branch_fixed(inp, name, num_blocks, filters_list, kernel_list, conv_count_list, bn_dict, use_attention, attention_type, use_pe):

        x = inp
        if use_pe:
            x = LearnablePositionalEncoding2D(name=f"pe_{name}")(x)


        for bi in range(num_blocks):
            f = filters_list[bi]
            k = kernel_list[bi]
            c = conv_count_list[bi]
            for ci in range(c):
                x = layers.Conv2D(f, k, padding="same", name=f"{name}_conv{bi}_{ci}")(x)
                if bn_dict.get(f"{name}_bn_{bi}_{ci}", False):
                    x = layers.BatchNormalization(name=f"{name}_bn{bi}_{ci}")(x)
                x = layers.Activation("relu", name=f"{name}_act{bi}_{ci}")(x)

        if use_attention:
            if attention_type == "se":
                x = se_block(x)
            elif attention_type == "cbam":
                x = cbam_block(x)
        return x

    bh = one_branch_fixed(
        hap_in, "hap", 3,
        [32, 64, 48],
        [5, 5, 1],
        [2, 1, 2],
        {
            'use_pe_hap': False,
            'hap_bn_0_0': False,
            'hap_bn_0_1': False,
            'hap_bn_1_0': True,
            'hap_bn_2_0': False,
            'hap_bn_2_1': False,
        },
        True,
        'se',
        True
    )

    bs = one_branch_fixed(
        sfs_in, "sfs", 1,
        [32, 16, 48],
        [5, 3, 3],
        [4, 5, 5],
        {
            'use_pe_sfs': True,
            'sfs_bn_0_0': True,
            'sfs_bn_0_1': False,
            'sfs_bn_0_2': False,
            'sfs_bn_0_3': False,
        },
        False,
        "cbam",
        True
    )

    bd = one_branch_fixed(
        div_in, "div", 3,
        [48, 32, 48],
        [5, 3, 1],
        [1, 4, 4],
        {
            'use_pe_div': True,
            'div_bn_0_0': True,
            'div_bn_1_0': True,
            'div_bn_1_1': False,
            'div_bn_1_2': True,
            'div_bn_1_3': False,
            'div_bn_2_0': True,
            'div_bn_2_1': False,
            'div_bn_2_2': False,
            'div_bn_2_3': False,
        },
        True,
        "cbam",
        True
    )

    # Fixed Stage 3 branch weighting best config
    x = LearnableWeightingConcat(num_branches=3, name="learnable_weights_concat")([bh, bs, bd])

    # Stage 4: MHSA layer (tunable hyperparameters)

    x = MHSA2D(num_heads=4, key_dim=9, name="mhsa")(x)

    x = layers.GlobalAveragePooling2D(name="pool_avg_mhsa")(x)

    out = layers.Dense(1, activation="sigmoid", name="out")(x)

    model = keras.Model(inputs=[hap_in, sfs_in, div_in], outputs=out)

    # TEST advanced learning rate schedules
    lr_schedule_type = hp.Choice("lr_schedule", ["constant", "cosine_decay"])

    if lr_schedule_type == "constant":
        lr = hp.Float("lr", 1e-4, 1e-2, sampling="log")
        opt = tf.keras.optimizers.Adam(lr)
    else:
        lr = tf.keras.optimizers.schedules.CosineDecayRestarts(
            initial_learning_rate=hp.Float("lr", 1e-4, 1e-2, sampling="log"),
            first_decay_steps=hp.Int("decay_steps", 50, 500, step=50),
            t_mul=hp.Float("t_mul", 1.0, 2.0, step=0.5),
            m_mul=hp.Float("m_mul", 0.5, 1.0, step=0.25),
            alpha=hp.Float("alpha", 0.0, 0.5, step=0.1),
        )
        opt = tf.keras.optimizers.Adam(lr)

    model.compile(
        optimizer=opt,
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.Precision(), tf.keras.metrics.AUC(name="auc")]
    )

    return model


def cnn_stage_5(hap_in,sfs_in,div_in):

    hap_in = layers.Input(shape=input_ld, name="inp_haplotype")
    sfs_in = layers.Input(shape=input_sfs, name="inp_sfs")
    div_in = layers.Input(shape=input_div, name="inp_diversity")


    def one_branch_fixed(inp, name, num_blocks, filters_list, kernel_list, conv_count_list, bn_dict, use_attention, attention_type, use_pe):

        x = inp
        if use_pe:
            x = LearnablePositionalEncoding2D(name=f"pe_{name}")(x)


        for bi in range(num_blocks):
            f = filters_list[bi]
            k = kernel_list[bi]
            c = conv_count_list[bi]
            for ci in range(c):
                x = layers.Conv2D(f, k, padding="same", name=f"{name}_conv{bi}_{ci}")(x)
                if bn_dict.get(f"{name}_bn_{bi}_{ci}", False):
                    x = layers.BatchNormalization(name=f"{name}_bn{bi}_{ci}")(x)
                x = layers.Activation("relu", name=f"{name}_act{bi}_{ci}")(x)

        if use_attention:
            if attention_type == "se":
                x = se_block(x)
            elif attention_type == "cbam":
                x = cbam_block(x)
        return x

    bh = one_branch_fixed(
        hap_in, "hap", 3,
        [32, 64, 48],
        [5, 5, 1],
        [2, 1, 2],
        {
            'use_pe_hap': False,
            'hap_bn_0_0': False,
            'hap_bn_0_1': False,
            'hap_bn_1_0': True,
            'hap_bn_2_0': False,
            'hap_bn_2_1': False,
        },
        True,
        'se',
        True
    )

    bs = one_branch_fixed(
        sfs_in, "sfs", 1,
        [32, 16, 48],
        [5, 3, 3],
        [4, 5, 5],
        {
            'use_pe_sfs': True,
            'sfs_bn_0_0': True,
            'sfs_bn_0_1': False,
            'sfs_bn_0_2': False,
            'sfs_bn_0_3': False,
        },
        False,
        "cbam",
        True
    )

    bd = one_branch_fixed(
        div_in, "div", 3,
        [48, 32, 48],
        [5, 3, 1],
        [1, 4, 4],
        {
            'use_pe_div': True,
            'div_bn_0_0': True,
            'div_bn_1_0': True,
            'div_bn_1_1': False,
            'div_bn_1_2': True,
            'div_bn_1_3': False,
            'div_bn_2_0': True,
            'div_bn_2_1': False,
            'div_bn_2_2': False,
            'div_bn_2_3': False,
        },
        True,
        "cbam",
        True
    )

    # Fixed Stage 3 branch weighting best config
    x = LearnableWeightingConcat(num_branches=3, name="learnable_weights_concat")([bh, bs, bd])

    # Stage 4: MHSA layer (tunable hyperparameters)
    x = MHSA2D(num_heads=4, key_dim=9, name="mhsa")(x)

    x = layers.GlobalAveragePooling2D(name="pool_avg_mhsa")(x)

    out = layers.Dense(1, activation="sigmoid", name="out")(x)

    model = keras.Model(inputs=[hap_in, sfs_in, div_in], outputs=out)

    lr = tf.keras.optimizers.schedules.CosineDecayRestarts(
        initial_learning_rate=0.001,
        first_decay_steps=50,
        t_mul=2,
        m_mul=1,
        alpha=0,
    )
    opt = tf.keras.optimizers.Adam(lr)

    model.compile(
        optimizer=opt,
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.Precision(), tf.keras.metrics.AUC(name="auc")]
    )

    return model

def cnn_stage_5(hap_in,sfs_in,div_in):

    hap_in = layers.Input(shape=input_ld, name="inp_haplotype")
    sfs_in = layers.Input(shape=input_sfs, name="inp_sfs")
    div_in = layers.Input(shape=input_div, name="inp_diversity")


    def one_branch_fixed(inp, name, num_blocks, filters_list, kernel_list, conv_count_list, bn_dict, use_attention, attention_type, use_pe):

        x = inp
        if use_pe:
            x = LearnablePositionalEncoding2D(name=f"pe_{name}")(x)


        for bi in range(num_blocks):
            f = filters_list[bi]
            k = kernel_list[bi]
            c = conv_count_list[bi]
            for ci in range(c):
                x = layers.Conv2D(f, k, padding="same", name=f"{name}_conv{bi}_{ci}")(x)
                if bn_dict.get(f"{name}_bn_{bi}_{ci}", False):
                    x = layers.BatchNormalization(name=f"{name}_bn{bi}_{ci}")(x)
                x = layers.Activation("relu", name=f"{name}_act{bi}_{ci}")(x)

        if use_attention:
            if attention_type == "se":
                x = se_block(x)
            elif attention_type == "cbam":
                x = cbam_block(x)
        return x

    bh = one_branch_fixed(
        hap_in, "hap", 3,
        [32, 64, 48],
        [5, 5, 1],
        [2, 1, 2],
        {
            'use_pe_hap': False,
            'hap_bn_0_0': False,
            'hap_bn_0_1': False,
            'hap_bn_1_0': True,
            'hap_bn_2_0': False,
            'hap_bn_2_1': False,
        },
        True,
        'se',
        True
    )

    bs = one_branch_fixed(
        sfs_in, "sfs", 1,
        [32, 16, 48],
        [5, 3, 3],
        [4, 5, 5],
        {
            'use_pe_sfs': True,
            'sfs_bn_0_0': True,
            'sfs_bn_0_1': False,
            'sfs_bn_0_2': False,
            'sfs_bn_0_3': False,
        },
        False,
        "cbam",
        True
    )

    bd = one_branch_fixed(
        div_in, "div", 3,
        [48, 32, 48],
        [5, 3, 1],
        [1, 4, 4],
        {
            'use_pe_div': True,
            'div_bn_0_0': True,
            'div_bn_1_0': True,
            'div_bn_1_1': False,
            'div_bn_1_2': True,
            'div_bn_1_3': False,
            'div_bn_2_0': True,
            'div_bn_2_1': False,
            'div_bn_2_2': False,
            'div_bn_2_3': False,
        },
        True,
        "cbam",
        True
    )

    # Fixed Stage 3 branch weighting best config
    x = LearnableWeightingConcat(num_branches=3, name="learnable_weights_concat")([bh, bs, bd])

    # Stage 4: MHSA layer (tunable hyperparameters)
    x = MHSA2D(num_heads=4, key_dim=9, name="mhsa")(x)

    x = layers.GlobalAveragePooling2D(name="pool_avg_mhsa")(x)

    out = layers.Dense(1, activation="sigmoid", name="out")(x)

    model = keras.Model(inputs=[hap_in, sfs_in, div_in], outputs=out)

    lr = tf.keras.optimizers.schedules.CosineDecayRestarts(
        initial_learning_rate=0.001,
        first_decay_steps=50,
        t_mul=2,
        m_mul=1,
        alpha=0,
    )
    opt = tf.keras.optimizers.Adam(lr)

    model.compile(
        optimizer=opt,
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.Precision(), tf.keras.metrics.AUC(name="auc")]
    )

    return model


# 1) Instantiate the tuner
tuner5 = kt.Hyperband(
    build_model_stage5_fixed,
    objective="val_accuracy",
    max_epochs=10,
    factor=3,
    directory="tuner_group5",
    project_name="flexsweep_group5",
)

# 2) Early‐stopping callback
stop_early = tf.keras.callbacks.EarlyStopping(monitor="val_accuracy",mode='min', patience=5)

# 3) Kick off the search
tuner5.search(
    [X_train_ld,X_train_sfs,X_train_div],
    Y_train,
    epochs=10,
    validation_data=([X_valid_ld,X_valid_sfs,X_valid_div], Y_valid),
    batch_size=32,
    callbacks=[stop_early],
)


###############
import tensorflow as tf
from tensorflow import keras
import keras_tuner as kt
import numpy as np

class FlexSweepCNNTuner(kt.HyperModel):
    def __init__(self, input_shape):
        """
        Keras Tuner version of Flex-sweep CNN for genomic data classification.

        Args:
            input_shape (tuple): Shape of input data (height, width, channels)
        """
        self.input_shape = input_shape

    def build(self, hp):
        """
        Build hyperparameter-tuned model.

        Args:
            hp: HyperParameters object from keras_tuner
        """
        # Input layer
        inputs = keras.Input(shape=self.input_shape, name='genomic_input')

        # Hyperparameters for global architecture
        use_batch_norm = hp.Boolean('use_batch_norm', default=True)
        use_attention = hp.Boolean('use_attention', default=False)
        # use_positional_encoding = hp.Boolean('use_positional_encoding', default=True)
        activation_type = hp.Choice('activation', ['relu', 'elu', 'swish', 'gelu'])

        # Apply positional encoding if enabled
        # if use_positional_encoding:
        #     inputs_with_pos = self._add_positional_encoding(inputs, hp)
        # else:
        inputs_with_pos = inputs

        # TUNABLE NUMBER OF BRANCHES (1-4)
        num_branches = hp.Int('num_branches', min_value=1, max_value=4, default=3)

        branches = []


        # Branch configurations - each branch has different characteristics
        branch_configs = [
            {'name': 'branch1', 'kernel_size': 3, 'dilation_rate': 1, 'desc': 'Traditional 3x3'},
            {'name': 'branch2', 'kernel_size': 2, 'dilation_rate': [1, 3], 'desc': 'Horizontal dilation (genomic windows)'},
            {'name': 'branch3', 'kernel_size': 2, 'dilation_rate': [5, 1], 'desc': 'Vertical dilation (statistics)'},
            {'name': 'branch4', 'kernel_size': 1, 'dilation_rate': 1, 'desc': 'Pointwise feature mixing'}
        ]

        # Build selected number of branches
        for i in range(num_branches):
            config = branch_configs[i]

            # Special handling for branch 4 (pointwise)
            max_layers = 2 if config['name'] == 'branch4' else None

            branch = self._build_conv_branch(
                inputs_with_pos, hp,
                branch_name=config['name'],
                kernel_size=config['kernel_size'],
                dilation_rate=config['dilation_rate'],
                use_batch_norm=use_batch_norm,
                activation_type=activation_type,
                max_layers=max_layers
            )
            branches.append(branch)

            # Log branch info during tuning
            if hp.Boolean('verbose_branches', default=False):
                print(f"Branch {i+1}: {config['desc']}")

        # Concatenate branches
        if len(branches) > 1:
            concat = keras.layers.concatenate(branches, name='branch_concat')
        else:
            concat = branches[0]

        # Optional attention mechanism
        if use_attention:
            concat = self._add_attention_layer(concat, hp)

        # Dense layers with tunable architecture
        x = self._build_dense_layers(concat, hp, activation_type)

        # Output layer
        outputs = keras.layers.Dense(
            1,
            activation='sigmoid',
            name='binary_output',
            kernel_initializer=hp.Choice('output_initializer',
                                       ['glorot_uniform', 'he_normal', 'lecun_normal'])
        )(x)

        # Create model
        model = keras.Model(inputs=inputs, outputs=outputs, name='flexsweep_cnn')

        # Compile with tunable parameters
        optimizer = self._get_optimizer(hp)

        model.compile(
            optimizer=optimizer,
            loss='binary_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )

        return model

    def _build_conv_branch(self, inputs, hp, branch_name, kernel_size, dilation_rate,
                          use_batch_norm, activation_type, max_layers=None):
        """Build a convolutional branch with tunable parameters."""

        if max_layers is None:
            max_layers = hp.Int(f'{branch_name}_num_layers', min_value=2, max_value=4, default=3)

        # Initial filter size
        filters = hp.Int(f'{branch_name}_initial_filters', min_value=32, max_value=128, step=32, default=64)

        # Kernel initializer
        initializer = hp.Choice(f'{branch_name}_initializer',
                              ['he_normal', 'glorot_uniform', 'lecun_normal'], default='he_normal')

        x = inputs

        conv_type = hp.Choice(f'{branch_name}_conv_type', ['conv2d', 'separable_conv2d'])


        # Convolution type choice
        conv_type = hp.Choice(f'{branch_name}_conv_type', ['conv2d', 'separable_conv2d'], default='conv2d')

        for i in range(max_layers):
            # Progressive filter increase
            current_filters = filters * (2 ** i) if i < 2 else filters * 4
            current_filters = min(current_filters, 512)  # Cap at 512


            if conv_type == 'conv2d':
                x = keras.layers.Conv2D(
                    current_filters,
                    kernel_size,
                    padding='same',
                    dilation_rate=dilation_rate,
                    kernel_initializer=initializer,
                    name=f'{branch_name}_conv_{i+1}'
                )(x)
            else:
                # Use SeparableConv2D
                x = keras.layers.SeparableConv2D(
                    current_filters,
                    kernel_size,
                    padding='same',
                    dilation_rate=dilation_rate,
                    depthwise_initializer=initializer,
                    pointwise_initializer=initializer,
                    name=f'{branch_name}_sep_conv_{i+1}'
                )(x)

            # Batch normalization (optional)
            if use_batch_norm:
                x = keras.layers.BatchNormalization(name=f'{branch_name}_bn_{i+1}')(x)

            # Activation
            x = self._get_activation(activation_type, f'{branch_name}_act_{i+1}')(x)

        # Pooling strategy
        pool_type = hp.Choice(f'{branch_name}_pool_type', ['max', 'avg', 'adaptive'])
        pool_size = hp.Int(f'{branch_name}_pool_size', min_value=2, max_value=3, default=2)

        if pool_type == 'max':
            x = keras.layers.MaxPooling2D(pool_size=pool_size, name=f'{branch_name}_pool')(x)
        elif pool_type == 'avg':
            x = keras.layers.AveragePooling2D(pool_size=pool_size, name=f'{branch_name}_pool')(x)
        else:  # adaptive
            x = keras.layers.GlobalAveragePooling2D(name=f'{branch_name}_global_pool')(x)
            return x  # Skip flatten for global pooling

        # Dropout
        dropout_rate = hp.Float(f'{branch_name}_dropout', min_value=0.1, max_value=0.5, default=0.2)
        x = keras.layers.Dropout(dropout_rate, name=f'{branch_name}_dropout')(x)

        # Flatten
        x = keras.layers.Flatten(name=f'{branch_name}_flatten')(x)

        return x

    def _add_positional_encoding(self, inputs, hp):
            """
            Add positional encoding for genomic data structure.

            For genomic data with shape (batch, windows, statistics, window_sizes):
            - Windows: 21 genomic positions (sequential)
            - Statistics: 11 pop-gen statistics (categorical)
            - Window_sizes: 5 different sizes (hierarchical)
            """
            pos_encoding_type = hp.Choice('pos_encoding_type',
                                        ['sinusoidal', 'learned', 'hybrid', 'none'])

            if pos_encoding_type == 'none':
                return inputs

            # Get input dimensions
            if len(inputs.shape) == 4:  # (batch, height, width, channels)
                batch_size, height, width, channels = inputs.shape
            else:
                return inputs  # Skip if not 4D

            x = inputs

            if pos_encoding_type == 'sinusoidal':
                # Sinusoidal positional encoding (like Transformer)
                # For genomic windows (height dimension)
                pos_encoding_dim = hp.Int('pos_encoding_dim', min_value=16, max_value=64, default=32)

                # Create positional encoding for windows (rows)
                window_positions = tf.range(height, dtype=tf.float32)
                window_encoding = self._create_sinusoidal_encoding(window_positions, pos_encoding_dim)

                # Create positional encoding for statistics (columns)
                stat_positions = tf.range(width, dtype=tf.float32)
                stat_encoding = self._create_sinusoidal_encoding(stat_positions, pos_encoding_dim)

                # Combine encodings
                window_encoding = tf.expand_dims(tf.expand_dims(window_encoding, 1), 0)  # (1, height, 1, pos_dim)
                stat_encoding = tf.expand_dims(tf.expand_dims(stat_encoding, 0), 0)      # (1, 1, width, pos_dim)

                # Broadcast to full shape
                window_encoding = tf.tile(window_encoding, [tf.shape(x)[0], 1, width, 1])
                stat_encoding = tf.tile(stat_encoding, [tf.shape(x)[0], height, 1, 1])

                # Project to match input channels
                pos_projection = keras.layers.Dense(channels, name='pos_projection')
                window_encoding = pos_projection(window_encoding)
                stat_encoding = pos_projection(stat_encoding)

                # Add to input
                if hp.Boolean('additive_pos_encoding', default=True):
                    x = keras.layers.Add(name='add_window_pos')([x, window_encoding])
                    x = keras.layers.Add(name='add_stat_pos')([x, stat_encoding])
                else:
                    x = keras.layers.Concatenate(axis=-1, name='concat_pos')([x, window_encoding, stat_encoding])

            elif pos_encoding_type == 'learned':
                # Learned positional embeddings
                pos_embedding_dim = hp.Int('learned_pos_dim', min_value=8, max_value=32, default=16)

                # Window position embeddings
                window_pos_layer = keras.layers.Embedding(
                    input_dim=height,
                    output_dim=pos_embedding_dim,
                    name='window_pos_embedding'
                )

                # Statistics position embeddings
                stat_pos_layer = keras.layers.Embedding(
                    input_dim=width,
                    output_dim=pos_embedding_dim,
                    name='stat_pos_embedding'
                )

                # Create position indices
                window_indices = tf.range(height)
                stat_indices = tf.range(width)

                # Get embeddings
                window_pos_emb = window_pos_layer(window_indices)  # (height, pos_dim)
                stat_pos_emb = stat_pos_layer(stat_indices)        # (width, pos_dim)

                # Expand and tile to match input shape
                window_pos_emb = tf.expand_dims(tf.expand_dims(window_pos_emb, 1), 0)
                stat_pos_emb = tf.expand_dims(tf.expand_dims(stat_pos_emb, 0), 0)

                window_pos_emb = tf.tile(window_pos_emb, [tf.shape(x)[0], 1, width, 1])
                stat_pos_emb = tf.tile(stat_pos_emb, [tf.shape(x)[0], height, 1, 1])

                # Project to input channels
                window_proj = keras.layers.Dense(channels, name='window_pos_proj')(window_pos_emb)
                stat_proj = keras.layers.Dense(channels, name='stat_pos_proj')(stat_pos_emb)

                # Add to input
                x = keras.layers.Add(name='add_learned_pos')([x, window_proj, stat_proj])

            elif pos_encoding_type == 'hybrid':
                # Hybrid: learned for statistics, sinusoidal for windows
                pos_dim = hp.Int('hybrid_pos_dim', min_value=16, max_value=48, default=24)

                # Sinusoidal for genomic windows (sequential nature)
                window_positions = tf.range(height, dtype=tf.float32)
                window_encoding = self._create_sinusoidal_encoding(window_positions, pos_dim)

                # Learned for statistics (categorical nature)
                stat_embedding = keras.layers.Embedding(
                    input_dim=width,
                    output_dim=pos_dim,
                    name='hybrid_stat_embedding'
                )(tf.range(width))

                # Process and add similar to above methods
                window_encoding = tf.expand_dims(tf.expand_dims(window_encoding, 1), 0)
                stat_encoding = tf.expand_dims(tf.expand_dims(stat_embedding, 0), 0)

                window_encoding = tf.tile(window_encoding, [tf.shape(x)[0], 1, width, 1])
                stat_encoding = tf.tile(stat_encoding, [tf.shape(x)[0], height, 1, 1])

                # Project and add
                combined_encoding = keras.layers.Add()([window_encoding, stat_encoding])
                pos_projection = keras.layers.Dense(channels, name='hybrid_pos_proj')(combined_encoding)

                x = keras.layers.Add(name='add_hybrid_pos')([x, pos_projection])

            return x


    def _create_sinusoidal_encoding(self, positions, d_model):
        """
        Create sinusoidal encoding matrix of shape (positions, d_model).
        """
        positions = tf.cast(positions, tf.float32)  # <- Fix here
        angle_rates = 1 / tf.pow(10000.0, (2 * (tf.range(d_model) // 2)) / tf.cast(d_model, tf.float32))
        angle_rads = tf.expand_dims(positions, -1) * angle_rates

        # Apply sin to even indices, cos to odd
        sines = tf.sin(angle_rads[:, 0::2])
        cosines = tf.cos(angle_rads[:, 1::2])

        pos_encoding = tf.concat([sines, cosines], axis=-1)
        return pos_encoding

    def _build_dense_layers(self, inputs, hp, activation_type):
        """Build dense layers with tunable architecture."""

        num_dense_layers = hp.Int('num_dense_layers', min_value=1, max_value=3, default=2)

        x = inputs

        for i in range(num_dense_layers):
            # Dense layer size
            if i == 0:
                units = hp.Int(f'dense_{i+1}_units', min_value=128, max_value=1024, step=128, default=512)
            else:
                units = hp.Int(f'dense_{i+1}_units', min_value=64, max_value=512, step=64, default=128)

            x = keras.layers.Dense(
                units,
                name=f'dense_{i+1}',
                kernel_initializer=hp.Choice('dense_initializer',
                                           ['he_normal', 'glorot_uniform', 'lecun_normal'])
            )(x)

            # Batch normalization for dense layers
            if hp.Boolean('dense_batch_norm', default=False):
                x = keras.layers.BatchNormalization(name=f'dense_bn_{i+1}')(x)

            # Activation
            x = self._get_activation(activation_type, f'dense_act_{i+1}')(x)

            # Dropout
            dropout_rate = hp.Float(f'dense_dropout_{i+1}', min_value=0.1, max_value=0.6, default=0.3)
            x = keras.layers.Dropout(dropout_rate, name=f'dense_dropout_{i+1}')(x)

        return x

    def _add_attention_layer(self, inputs, hp):
        """Add attention mechanism - improved for genomic data."""
        attention_type = hp.Choice('attention_type', ['channel_attention', 'spatial_attention', 'self_attention'])

        if attention_type == 'channel_attention':
            # Channel attention (Squeeze-and-Excitation style) - BEST for genomic features
            x = inputs
            if len(x.shape) == 4:  # If still 4D from conv layers
                # Global average pooling and max pooling
                avg_pool = keras.layers.GlobalAveragePooling2D()(x)
                max_pool = keras.layers.GlobalMaxPooling2D()(x)

                # Combine both pooling methods
                if hp.Boolean('use_dual_pooling_attention', default=True):
                    pooled = keras.layers.Add()([avg_pool, max_pool])
                else:
                    pooled = avg_pool
            else:
                pooled = x

            # Attention mechanism
            attention_ratio = hp.Int('channel_attention_ratio', min_value=4, max_value=16, default=8)
            reduced_dim = max(pooled.shape[-1] // attention_ratio, 1)

            # Two-layer MLP
            attention = keras.layers.Dense(reduced_dim, activation='relu', name='channel_att_reduce')(pooled)
            attention = keras.layers.Dropout(0.1)(attention)
            attention = keras.layers.Dense(pooled.shape[-1], activation='sigmoid', name='channel_att_expand')(attention)

            # Apply attention
            x = keras.layers.Multiply(name='channel_attention_apply')([pooled, attention])

        elif attention_type == 'spatial_attention':
            # Spatial attention for genomic windows
            x = inputs
            if len(x.shape) == 4:  # 4D tensor
                # Average and max pooling across channel dimension
                avg_pool = keras.layers.Lambda(lambda x: tf.reduce_mean(x, axis=-1, keepdims=True))(x)
                max_pool = keras.layers.Lambda(lambda x: tf.reduce_max(x, axis=-1, keepdims=True))(x)

                # Concatenate
                spatial_input = keras.layers.Concatenate(axis=-1)([avg_pool, max_pool])

                # Convolution to generate attention map
                attention_map = keras.layers.Conv2D(1, kernel_size=7, padding='same', activation='sigmoid')(spatial_input)

                # Apply attention
                x = keras.layers.Multiply()([x, attention_map])
                x = keras.layers.Flatten()(x)
            else:
                # For flattened input, reshape and apply
                spatial_dim = int(np.sqrt(x.shape[-1]))
                if spatial_dim * spatial_dim == x.shape[-1]:
                    reshaped = keras.layers.Reshape((spatial_dim, spatial_dim, 1))(x)
                    # Apply similar logic as above
                    attention_map = keras.layers.Conv2D(1, kernel_size=3, padding='same', activation='sigmoid')(reshaped)
                    attended = keras.layers.Multiply()([reshaped, attention_map])
                    x = keras.layers.Flatten()(attended)
                else:
                    x = inputs  # Skip if can't reshape nicely

        else:  # self_attention
            # Multi-head self-attention for sequence modeling
            x = inputs
            if len(x.shape) == 2:  # Flattened features
                # Determine sequence structure for genomic data
                # Assuming: 21 windows × 11 statistics × 5 window_sizes = features
                seq_length = hp.Choice('attention_seq_length', [21, 55, 105])  # Different ways to structure

                if x.shape[-1] % seq_length == 0:
                    feature_dim = x.shape[-1] // seq_length
                    x = keras.layers.Reshape((seq_length, feature_dim))(x)

                    # Multi-head attention
                    num_heads = hp.Int('attention_heads', min_value=2, max_value=8, default=4)
                    key_dim = feature_dim // num_heads

                    # Self-attention layer
                    attention_output = keras.layers.MultiHeadAttention(
                        num_heads=num_heads,
                        key_dim=key_dim,
                        dropout=hp.Float('attention_dropout', min_value=0.0, max_value=0.3, default=0.1),
                        name='genomic_self_attention'
                    )(x, x)

                    # Add residual connection if dimensions match
                    if hp.Boolean('attention_residual', default=True):
                        attention_output = keras.layers.Add()([x, attention_output])

                    # Layer normalization
                    attention_output = keras.layers.LayerNormalization()(attention_output)

                    # Flatten back
                    x = keras.layers.Flatten()(attention_output)
                else:
                    # If can't reshape nicely, use dense-based attention
                    attention_dim = hp.Int('dense_attention_dim', min_value=64, max_value=256, default=128)

                    # Query, Key, Value projections
                    q = keras.layers.Dense(attention_dim, name='attention_query')(x)
                    k = keras.layers.Dense(attention_dim, name='attention_key')(x)
                    v = keras.layers.Dense(attention_dim, name='attention_value')(x)

                    # Scaled dot-product attention
                    attention_scores = keras.layers.Lambda(
                        lambda inputs: tf.nn.softmax(tf.matmul(inputs[0], inputs[1], transpose_b=True) / tf.sqrt(tf.cast(attention_dim, tf.float32)))
                    )([q, k])

                    attention_output = keras.layers.Lambda(lambda inputs: tf.matmul(inputs[0], inputs[1]))([attention_scores, v])

                    # Combine with original
                    if hp.Boolean('dense_attention_residual', default=True):
                        x = keras.layers.Add()([x, attention_output])
                    else:
                        x = attention_output
            else:
                x = inputs  # Skip if already processed

        return x

    def _get_activation(self, activation_type, name):
        """Get activation layer based on type."""
        if activation_type == 'relu':
            return keras.layers.ReLU(name=name)
        elif activation_type == 'elu':
            return keras.layers.ELU(name=name)
        elif activation_type == 'swish':
            return keras.layers.Activation('swish', name=name)
        elif activation_type == 'gelu':
            return keras.layers.Activation('gelu', name=name)
        else:
            return keras.layers.ReLU(name=name)

    def _build_adaptive_branch_architecture(self, inputs, hp):
        """
        Build adaptive branch architecture based on genomic data characteristics.

        This method automatically determines optimal branch configurations
        based on the input data structure and hyperparameter search.
        """

        # Analyze input characteristics
        height, width, channels = inputs.shape[1:]

        # Branch design strategies
        branch_strategy = hp.Choice('branch_strategy',
                                  ['multi_scale', 'multi_dilation', 'mixed', 'minimal'])

        branches = []

        if branch_strategy == 'multi_scale':
            # Focus on different kernel sizes
            kernel_sizes = [1, 2, 3, 5]
            num_branches = hp.Int('multiscale_branches', min_value=2, max_value=4, default=3)

            for i in range(num_branches):
                kernel_size = kernel_sizes[i]
                branch = self._build_conv_branch(
                    inputs, hp, branch_name=f'scale_branch_{i+1}',
                    kernel_size=kernel_size, dilation_rate=1,
                    use_batch_norm=hp.Boolean('use_batch_norm', default=True),
                    activation_type=hp.Choice('activation', ['relu', 'elu', 'swish'])
                )
                branches.append(branch)

        elif branch_strategy == 'multi_dilation':
            # Focus on different dilation rates
            dilations = [1, [1, 2], [1, 3], [2, 1], [3, 1], [2, 2]]
            num_branches = hp.Int('multidilation_branches', min_value=2, max_value=4, default=3)

            for i in range(num_branches):
                dilation = dilations[i]
                branch = self._build_conv_branch(
                    inputs, hp, branch_name=f'dilation_branch_{i+1}',
                    kernel_size=2, dilation_rate=dilation,
                    use_batch_norm=hp.Boolean('use_batch_norm', default=True),
                    activation_type=hp.Choice('activation', ['relu', 'elu', 'swish'])
                )
                branches.append(branch)

        elif branch_strategy == 'mixed':
            # Mix of different approaches
            configs = [
                {'kernel_size': 3, 'dilation_rate': 1, 'name': 'standard'},
                {'kernel_size': 2, 'dilation_rate': [1, 3], 'name': 'horizontal'},
                {'kernel_size': 2, 'dilation_rate': [3, 1], 'name': 'vertical'},
                {'kernel_size': 1, 'dilation_rate': 1, 'name': 'pointwise'}
            ]

            num_branches = hp.Int('mixed_branches', min_value=2, max_value=4, default=3)

            for i in range(num_branches):
                config = configs[i]
                branch = self._build_conv_branch(
                    inputs, hp, branch_name=f'mixed_branch_{i+1}',
                    kernel_size=config['kernel_size'],
                    dilation_rate=config['dilation_rate'],
                    use_batch_norm=hp.Boolean('use_batch_norm', default=True),
                    activation_type=hp.Choice('activation', ['relu', 'elu', 'swish'])
                )
                branches.append(branch)

        else:  # minimal
            # Single branch with optimal configuration
            branch = self._build_conv_branch(
                inputs, hp, branch_name='single_branch',
                kernel_size=hp.Int('minimal_kernel_size', min_value=2, max_value=5, default=3),
                dilation_rate=hp.Choice('minimal_dilation', [1, [1, 2], [2, 1]]),
                use_batch_norm=hp.Boolean('use_batch_norm', default=True),
                activation_type=hp.Choice('activation', ['relu', 'elu', 'swish'])
            )
            branches.append(branch)

        return branches

    def _get_optimizer(self, hp):
        """Get optimizer with tunable parameters."""
        optimizer_type = hp.Choice('optimizer', ['adam', 'adamw', 'rmsprop', 'sgd'])

        learning_rate = hp.Float('learning_rate', min_value=1e-5, max_value=1e-2,
                                sampling='LOG', default=1e-3)

        if optimizer_type == 'adam':
            return keras.optimizers.Adam(
                learning_rate=learning_rate,
                beta_1=hp.Float('adam_beta1', min_value=0.8, max_value=0.95, default=0.9),
                beta_2=hp.Float('adam_beta2', min_value=0.95, max_value=0.999, default=0.999)
            )
        elif optimizer_type == 'adamw':
            return keras.optimizers.AdamW(
                learning_rate=learning_rate,
                weight_decay=hp.Float('weight_decay', min_value=1e-6, max_value=1e-3,
                                    sampling='LOG', default=1e-4)
            )
        elif optimizer_type == 'rmsprop':
            return keras.optimizers.RMSprop(
                learning_rate=learning_rate,
                decay=hp.Float('rmsprop_decay', min_value=0.9, max_value=0.99, default=0.9)
            )
        else:  # sgd
            return keras.optimizers.SGD(
                learning_rate=learning_rate,
                momentum=hp.Float('sgd_momentum', min_value=0.8, max_value=0.99, default=0.9)
            )

def create_lr_scheduler(hp):
    """Create learning rate scheduler."""
    scheduler_type = hp.Choice('lr_scheduler', ['cosine', 'exponential', 'reduce_on_plateau', 'none'])

    if scheduler_type == 'cosine':
        return keras.callbacks.CosineRestartScheduler(
            first_restart_step=hp.Int('cosine_restart_step', min_value=10, max_value=50, default=20),
            t_mul=hp.Float('cosine_t_mul', min_value=1.0, max_value=2.0, default=1.5)
        )
    elif scheduler_type == 'exponential':
        return keras.callbacks.ExponentialDecay(
            decay_steps=hp.Int('exp_decay_steps', min_value=100, max_value=1000, default=500),
            decay_rate=hp.Float('exp_decay_rate', min_value=0.8, max_value=0.98, default=0.9)
        )
    elif scheduler_type == 'reduce_on_plateau':
        return keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=hp.Float('reduce_factor', min_value=0.1, max_value=0.5, default=0.2),
            patience=hp.Int('reduce_patience', min_value=3, max_value=10, default=5),
            min_lr=1e-7
        )
    else:
        return None

# Example usage - HYPERBAND FIRST!
def run_hyperband_tuner(X_train, Y_train, X_valid, Y_valid, input_shape):
    """
    Run Hyperband tuning for the Flex-sweep CNN (RECOMMENDED FIRST APPROACH).

    Args:
        X_train, y_train: Training data
        X_val, y_val: Validation data
        input_shape: Shape of input data (height, width, channels)
    """

    # Initialize Hyperband tuner
    tuner = kt.Hyperband(
        FlexSweepCNNTuner(input_shape),
        objective='val_accuracy',
        max_epochs=20,
        factor=3,  # Reduction factor for successive halving
        hyperband_iterations=2,  # Number of times to repeat the hyperband process
        directory='flexsweep_hyperband',
        project_name='genomic_cnn_hyperband',
        overwrite=True
    )

    # Define callbacks for Hyperband
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,  # Shorter patience for Hyperband
            restore_best_weights=True
        )
    ]

    # Search for best hyperparameters
    tuner.search(
        X_train, Y_train,
        epochs=100,  # Hyperband will decide actual epochs per trial
        validation_data=(X_valid, Y_valid),
        callbacks=callbacks,
        verbose=1
    )

    # Get best model
    best_model = tuner.get_best_models(num_models=1)[0]

    # Print best hyperparameters
    print("Best hyperparameters from Hyperband:")
    print(tuner.get_best_hyperparameters()[0].values)

    return best_model, tuner



def run_tuner_search(X_train, y_train, X_val, y_val, input_shape):
    """
    Run random search tuning (use after Hyperband for fine-tuning).
    """

    # Initialize tuner
    tuner = kt.RandomSearch(
        FlexSweepCNNTuner(input_shape),
        objective='val_accuracy',
        max_trials=50,
        executions_per_trial=2,
        directory='flexsweep_tuning',
        project_name='genomic_cnn_optimization'
    )

# Alternative: Bayesian Optimization tuner (often more efficient)
def run_bayesian_tuner(X_train, y_train, X_val, y_val, input_shape):
    """Run Bayesian optimization tuner."""

    tuner = kt.BayesianOptimization(
        FlexSweepCNNTuner(input_shape),
        objective='val_accuracy',
        max_trials=30,
        directory='flexsweep_bayesian',
        project_name='genomic_cnn_bayesian'
    )

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=8,
            restore_best_weights=True
        )
    ]

    tuner.search(
        X_train, y_train,
        epochs=30,
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        verbose=1
    )

    return tuner.get_best_models(num_models=1)[0], tuner
