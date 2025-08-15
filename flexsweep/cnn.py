import time, os

from . import pl, np

import importlib
import glob
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, roc_auc_score
from itertools import product
from collections import defaultdict
from tqdm import tqdm
from pybedtools import BedTool
from copy import deepcopy

# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


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

    def __init__(self, train_data=None, test_data=None, output_folder=None, model=None):
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

    def cnn_flexsweep(self, model_input, num_classes=2):
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
        layer1 = tf.keras.layers.Conv2D(
            64,
            3,
            padding="same",
            name="convlayer1_1",
            kernel_initializer="glorot_uniform",
        )(model_input)
        layer1 = tf.keras.layers.ReLU(negative_slope=0)(layer1)
        layer1 = tf.keras.layers.Conv2D(
            128,
            3,
            padding="same",
            name="convlayer1_2",
            kernel_initializer="glorot_uniform",
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
            kernel_initializer="glorot_uniform",
        )(model_input)
        layer2 = tf.keras.layers.ReLU(negative_slope=0)(layer2)
        layer2 = tf.keras.layers.Conv2D(
            128,
            2,
            dilation_rate=[1, 3],
            padding="same",
            name="convlayer2_2",
            kernel_initializer="glorot_uniform",
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
            dilation_rate=[1, 5],
            padding="same",
            name="convlayer4_1",
            kernel_initializer="glorot_uniform",
        )(model_input)
        layer3 = tf.keras.layers.ReLU(negative_slope=0)(layer3)
        layer3 = tf.keras.layers.Conv2D(
            128,
            2,
            dilation_rate=[1, 5],
            padding="same",
            name="convlayer4_2",
            kernel_initializer="glorot_uniform",
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
            num_classes,
            name="out_dense",
            activation="sigmoid",
            kernel_initializer="glorot_uniform",
        )(concat)

        return output

    def load_training_data(self):
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

        if self.num_stats < 17:
            tmp = tmp.select([col for col in tmp.columns if "flip" not in col])

        tmp = tmp.with_columns(
            pl.when((pl.col("model") != "neutral"))
            .then(pl.lit("sweep"))
            .otherwise(pl.lit("neutral"))
            .alias("model")
        )
        tmp = tmp.filter(
            ((pl.col("f_t") >= 0.4) & (pl.col("model") == "sweep"))
            | (pl.col("model") == "neutral")
        )

        sweep_parameters = tmp.filter("model" != "neutral").select(tmp.columns[:5])

        stats = ["iter", "model"]
        _stats = [
            "dind",
            "haf",
            "hapdaf_o",
            "isafe",
            "high_freq",
            "hapdaf_s",
            "nsl",
            "s_ratio",
            "low_freq",
            "ihs",
            "h12",
        ]
        if self.seed is not None:
            np.random.seed(self.seed)
            _stats = np.random.choice(_stats, self.num_stats)
        stats = stats + _stats
        train_stats = []
        for i in stats:
            # train_stats.append(df_train.iloc[:, df_train.columns.str.contains(i)])
            if i == "ihs":
                train_stats.append(
                    tmp.select(pl.col("^.*" + i + ".*$")).select(
                        pl.exclude("^.*hapbin.*$")
                    )
                )
            # elif i == 'haf' or i == 'h12':
            #     train_stats.append(df_train.select(pl.col("^.*" + i + ".*600000$")))
            else:
                train_stats.append(tmp.select(pl.col("^.*" + i + ".*$")))
        train_stats = pl.concat(train_stats, how="horizontal").select(
            pl.exclude("^.*ihh.*$")
        )
        train_stats = pl.concat(
            [tmp.select("s", "f_i", "f_t", "t"), train_stats], how="horizontal"
        )

        y = train_stats.select(
            (pl.col("model").str.contains("neutral").cast(pl.Int8)).alias(
                "neutral_flag"
            )
        )["neutral_flag"].to_numpy()

        test_split = round(1 - self.train_split, 2)

        X_train, X_test, y_train, y_test = train_test_split(
            train_stats, y, test_size=test_split, shuffle=True
        )

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
        Y_train = tf.keras.utils.to_categorical(y_train, 2)
        Y_test = tf.keras.utils.to_categorical(y_test, 2)

        X_valid, X_test, Y_valid, Y_test = train_test_split(
            X_test, Y_test, test_size=0.5
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

        self.test_data = [X_test, X_test_params, Y_test]

        return X_train, X_test, Y_train, Y_test, X_valid, Y_valid

    def train(self, cnn=None):
        """
        Trains the CNN model on the training data.

        This method preprocesses the data, sets up data augmentation, defines the model architecture,
        compiles the model, and fits it to the training data while saving the best model and training history.
        """
        tf = self.check_tf()

        if cnn is None:
            cnn = self.cnn_flexsweep

        (X_train, X_test, Y_train, Y_test, X_valid, Y_valid) = self.load_training_data()

        datagen = tf.keras.preprocessing.image.ImageDataGenerator(
            featurewise_center=True,
            featurewise_std_normalization=True,
            horizontal_flip=True,
        )

        validation_gen = tf.keras.preprocessing.image.ImageDataGenerator(
            featurewise_center=True,
            featurewise_std_normalization=True,
            horizontal_flip=False,
        )

        test_gen = tf.keras.preprocessing.image.ImageDataGenerator(
            featurewise_center=True,
            featurewise_std_normalization=True,
            horizontal_flip=False,
        )

        datagen.fit(X_train)
        validation_gen.fit(X_valid)
        test_gen.fit(X_test)

        # put model together
        input_to_model = tf.keras.Input(X_train.shape[1:])
        model = tf.keras.models.Model(
            inputs=[input_to_model], outputs=[cnn(input_to_model)]
        )
        model_path = self.output_folder + "/model.keras"
        weights_path = self.output_folder + "/model_weights.hdf5"

        metrics_measures = [
            tf.keras.metrics.TruePositives(name="tp"),
            tf.keras.metrics.FalsePositives(name="fp"),
            tf.keras.metrics.TrueNegatives(name="tn"),
            tf.keras.metrics.FalseNegatives(name="fn"),
            tf.keras.metrics.BinaryAccuracy(name="accuracy"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.AUC(name="auc"),
            tf.keras.metrics.AUC(name="prc", curve="PR"),  # precision-recall curve
        ]

        lr_decayed_fn = tf.keras.optimizers.schedules.CosineDecayRestarts(
            initial_learning_rate=0.001, first_decay_steps=300
        )

        opt_adam = tf.keras.optimizers.Adam(
            learning_rate=lr_decayed_fn, epsilon=0.0000001, amsgrad=True
        )
        model.compile(
            loss="binary_crossentropy", optimizer=opt_adam, metrics=metrics_measures
        )

        earlystop = tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            min_delta=0.001,
            patience=5,
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

        history = model.fit(
            datagen.flow(X_train, Y_train, batch_size=32),
            epochs=100,
            callbacks=callbacks_list,
            validation_data=validation_gen.flow(X_valid, Y_valid, batch_size=32),
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
            model.save(self.output_folder + "/model.keras")

    def predict(self, _iter=1):
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
        if isinstance(self.test_data, str):
            assert (
                isinstance(self.test_data, pl.DataFrame)
                or "txt" in self.test_data
                or "csv" in self.test_data
                or self.test_data.endswith(".parquet")
            ), "Please input a pl.DataFrame or save it as CSV or parquet"
            try:
                df_test = pl.read_parquet(self.test_data)
            except:
                df_test = pl.read_csv(self.test_data, separator=",")
            if self.num_stats < 17:
                df_test = df_test.select(
                    [col for col in df_test.columns if "flip" not in col]
                )

            regions = df_test["iter"].to_numpy()
            # Same folder custom fvs name based on input VCF.
            self.output_prediction = (
                os.path.basename(self.test_data)
                .replace("fvs_", "")
                .replace(".parquet", "_predictions.txt")
            )
        else:
            test_X, test_X_params, test_Y = self.test_data

        # d_prediction = {}
        # for m, df_m in df_test.group_by("model"):
        #     test_X = []
        #     for i in [
        #         "dind",
        #         "haf",
        #         "hapdaf_o",
        #         "isafe",
        #         "high_freq",
        #         "hapdaf_s",
        #         "nsl",
        #         "s_ratio",
        #         "low_freq",
        #         "ihs",
        #         "h12",
        #     ]:
        #         if i == "ihs":
        #             test_X.append(
        #                 df_m.select(pl.col("^.*" + i + ".*$").exclude("^.*hapbin.*$"))
        #             )
        #         else:
        #             test_X.append(
        #                 df_m.select(pl.col("^.*" + i + ".*$").exclude("^.*flip.*$"))
        #             )

        #     test_X = (
        #         pl.concat(test_X, how="horizontal")
        #         .select(pl.exclude("^.*ihh.*$"))
        #         .to_numpy()
        #     )

        #     test_X = test_X.reshape(
        #         test_X.shape[0],
        #         self.num_stats,
        #         self.windows.size * self.center.size,
        #         1,
        #     )

        # batch size, image width, image height,number of channels
        if isinstance(self.model, str):
            model = tf.keras.models.load_model(self.model)
        else:
            model = self.model

        metrics_measures = [
            tf.keras.metrics.TruePositives(name="tp"),
            tf.keras.metrics.FalsePositives(name="fp"),
            tf.keras.metrics.TrueNegatives(name="tn"),
            tf.keras.metrics.FalseNegatives(name="fn"),
            tf.keras.metrics.BinaryAccuracy(name="accuracy"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.AUC(name="auc"),
            tf.keras.metrics.AUC(name="prc", curve="PR"),
        ]

        lr_decayed_fn = tf.keras.optimizers.schedules.CosineDecayRestarts(
            initial_learning_rate=0.001, first_decay_steps=300
        )
        opt_adam = tf.keras.optimizers.Adam(
            learning_rate=lr_decayed_fn, epsilon=0.0000001, amsgrad=True
        )
        model.compile(
            loss="binary_crossentropy", optimizer=opt_adam, metrics=metrics_measures
        )

        validation_gen = tf.keras.preprocessing.image.ImageDataGenerator(
            featurewise_center=True,
            featurewise_std_normalization=True,
            horizontal_flip=False,
        )

        validation_gen.fit(test_X)

        # make predictions
        preds = model.predict(validation_gen.standardize(test_X))
        predictions = np.argmax(preds, axis=1)
        prediction_dict = {0: "sweep", 1: "neutral"}
        predictions_class = np.vectorize(prediction_dict.get)(predictions)

        df_prediction = pl.concat(
            [
                test_X_params.select(pl.nth([5, 0, 1, 2, 3])),
                # df_m.select(pl.nth([0])),
                pl.DataFrame(
                    np.column_stack([predictions_class, preds]),
                    schema=["predicted_model", "prob_sweep", "prob_neutral"],
                ),
            ],
            how="horizontal",
        )
        # d_prediction[m[0]] = df_prediction

        # df_prediction = pl.concat(d_prediction.values(), how="vertical")
        # df_prediction.iloc[:, -2:] = df_prediction.iloc[:, -2:].astype(float)

        if isinstance(self.test_data, str):
            df_prediction = df_prediction.with_columns(pl.Series("region", regions))
            chr_start_end = np.array(
                [item.replace(":", "-").split("-") for item in regions]
            )

            df_prediction = df_prediction.with_columns(
                pl.Series("chr", chr_start_end[:, 0]),
                pl.Series("start", chr_start_end[:, 1], dtype=pl.Int64),
                pl.Series("end", chr_start_end[:, 2], dtype=pl.Int64),
            )
            df_prediction = (
                df_prediction.select(pl.exclude("region"))
                .sort("start")
                .select(df_prediction.columns[-3:] + df_prediction.columns[:-4])
            )

        self.prediction = df_prediction

        if self.output_folder is not None:
            df_prediction.write_csv(
                # f"{self.output_folder}/predictions.txt"
                f"{self.output_folder}/predictions.txt"
            )

        return df_prediction

    def roc_curve(self):
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

        if self.output_folder is not None:
            plot_roc.savefig(self.output_folder + "/roc_curve.svg")
            plot_history.savefig(self.output_folder + "/train_history.svg")
        return plot_roc, plot_history


# # import tensorflow as tf
# from tensorflow.keras.utils import Sequence


# # --- Siepel-style GRL and masked-BCE ---
# @tf.custom_gradient
# def grad_reverse(x):
#     y = tf.identity(x)

#     def custom_grad(dy):
#         return -dy

#     return y, custom_grad


# class GradReverse(tf.keras.layers.Layer):
#     def call(self, x):
#         return grad_reverse(x)


# def masked_bce():
#     bce = tf.keras.losses.BinaryCrossentropy(reduction=tf.keras.losses.Reduction.NONE)

#     def _loss(y_true, y_pred):
#         mask = tf.not_equal(y_true, -1.0)
#         y_true_m = tf.boolean_mask(y_true, mask)
#         y_pred_m = tf.boolean_mask(y_pred, mask)
#         return tf.cond(
#             tf.size(y_true_m) > 0,
#             lambda: tf.reduce_mean(bce(y_true_m, y_pred_m)),
#             lambda: tf.constant(0.0, tf.float32),
#         )

#     return _loss


# @tf.custom_gradient
# def _grl_with_lambda(x, lambd):
#     y = tf.identity(x)

#     def grad(dy):
#         return -lambd * dy, tf.zeros_like(lambd)  # no grad wrt lambda

#     return y, grad


# class GradReverse(tf.keras.layers.Layer):
#     def __init__(self, lambd=0.0, **kw):
#         super().__init__(**kw)
#         # trainable=False so you control it from a callback
#         self.lambd = tf.Variable(
#             float(lambd), trainable=False, dtype=tf.float32, name="grl_lambda"
#         )

#     def call(self, x):
#         return _grl_with_lambda(x, self.lambd)


# class GRLRamp(tf.keras.callbacks.Callback):
#     def __init__(self, grl_layer, max_lambda=0.5, epochs=50):
#         """
#         epochs = number of ramp epochs (not total training epochs).
#         After this many epochs, λ will be held at max_lambda.
#         """
#         super().__init__()
#         self.grl_layer = grl_layer
#         self.max_lambda = float(max_lambda)
#         self.ramp_epochs = int(max(1, epochs))

#     def on_epoch_begin(self, epoch, logs=None):
#         # linear warmup 0 → max_lambda over `ramp_epochs`, then hold
#         if epoch < self.ramp_epochs:
#             t = epoch / max(1, self.ramp_epochs - 1)
#             lam = self.max_lambda * t
#         else:
#             lam = self.max_lambda
#         self.grl_layer.lambd.assign(lam)


# def se_block_2d(x, reduction=8):
#     ch = int(x.shape[-1])
#     s = tf.keras.layers.GlobalAveragePooling2D()(x)
#     s = tf.keras.layers.Dense(max(4, ch // reduction), activation="relu")(s)
#     s = tf.keras.layers.Dense(ch, activation="sigmoid")(s)
#     s = tf.keras.layers.Reshape((1, 1, ch))(s)
#     return tf.keras.layers.Multiply()([x, s])


# class CNN:
#     """
#     A class for building and training a Convolutional Neural Network (CNN) for classification tasks using flex-sweep input statistics.
#     Attributes:
#         train_data (str or pl.DataFrame): Path to the training data file or a pandas DataFrame containing the training data.
#         test_data (pl.DataFrame): DataFrame containing test data after being loaded from a file.
#         output_folder (str): Directory where the trained model and history will be saved.
#         num_stats (int): Number of statistics/features in the training data. Default is 11.
#         center (np.ndarray): Array defining the center positions for processing. Default ranges from 500000 to 700000.
#         windows (np.ndarray): Array defining the window sizes for the CNN. Default values are [50000, 100000, 200000, 500000, 1000000].
#         train_split (float): Fraction of the training data used for training. Default is 0.8.
#         model (tf.keras.Model): Keras model instance for the CNN.
#         gpu (bool): Indicates whether to use GPU for training. Default is True.
#     """

#     def __init__(
#         self,
#         train_data=None,
#         source_data=None,
#         target_data=None,
#         test_data=None,
#         valid_data=None,
#         output_folder=None,
#         normalize=False,
#         model=None,
#     ):
#         """
#         Initializes the CNN class with training data and output folder.

#         Args:
#             train_data (str or pl.DataFrame): Path to the training data file or a pandas DataFrame containing the training data.
#             output_folder (str): Directory to save the trained model and history.
#         """
#         # self.sweep_data = sweep_data
#         self.normalize = normalize
#         self.train_data = train_data
#         self.test_data = test_data
#         self.output_folder = output_folder
#         self.output_prediction = "predictions.txt"
#         self.num_stats = 11
#         self.center = np.arange(5e5, 7e5 + 1e4, 1e4).astype(int)
#         self.windows = np.array([50000, 100000, 200000, 500000, 1000000])
#         self.train_split = 0.8
#         self.prediction = None
#         self.history = None
#         self.model = model
#         self.gpu = True
#         self.tf = None
#         self.seed = None
#         self.misspecification = False
#         self.channel = 11
#         self.split = True
#         self.test_data_splitted = None
#         self.lstm = False
#         self.source_data = source_data
#         self.target_data = target_data

#     def check_tf(self):
#         """
#         Checks and imports the TensorFlow library.

#         Returns:
#             tf.Module: The TensorFlow module.
#         """
#         if self.gpu is False:
#             os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
#         tf = importlib.import_module("tensorflow")
#         return tf

#     def cnn_flexsweep(self, model_input, num_classes=1):
#         """
#         Flex-sweep CNN architecture with multiple convolutional and pooling layers.

#         Args:            input_shape (tuple): Shape of the input data, e.g., (224, 224, 3). Default Flex-sweep input statistics, windows and centers
#             num_classes (int): Number of output classes in the classification problem. Default: Flex-sweep binary classification

#         Returns:
#             Model: A Keras model instance representing the Flex-sweep CNN architecture.
#         """
#         tf = self.check_tf()
#         # 3x3 layer
#         initializer = tf.keras.initializers.HeNormal()
#         layer1 = tf.keras.layers.Conv2D(
#             64,
#             3,
#             padding="same",
#             name="convlayer1_1",
#             kernel_initializer=initializer,
#         )(model_input)
#         layer1 = tf.keras.layers.ReLU(negative_slope=0)(layer1)
#         layer1 = tf.keras.layers.Conv2D(
#             128,
#             3,
#             padding="same",
#             name="convlayer1_2",
#             kernel_initializer=initializer,
#         )(layer1)
#         layer1 = tf.keras.layers.ReLU(negative_slope=0)(layer1)
#         layer1 = tf.keras.layers.Conv2D(256, 3, padding="same", name="convlayer1_3")(
#             layer1
#         )
#         layer1 = tf.keras.layers.ReLU(negative_slope=0)(layer1)
#         layer1 = tf.keras.layers.MaxPooling2D(
#             pool_size=3, name="poollayer1", padding="same"
#         )(layer1)
#         layer1 = tf.keras.layers.Dropout(0.15, name="droplayer1")(layer1)
#         layer1 = tf.keras.layers.Flatten(name="flatlayer1")(layer1)

#         # 2x2 layer with 1x3 dilation
#         layer2 = tf.keras.layers.Conv2D(
#             64,
#             2,
#             dilation_rate=[1, 3],
#             padding="same",
#             name="convlayer2_1",
#             kernel_initializer=initializer,
#         )(model_input)
#         layer2 = tf.keras.layers.ReLU(negative_slope=0)(layer2)
#         layer2 = tf.keras.layers.Conv2D(
#             128,
#             2,
#             dilation_rate=[1, 3],
#             padding="same",
#             name="convlayer2_2",
#             kernel_initializer=initializer,
#         )(layer2)
#         layer2 = tf.keras.layers.ReLU(negative_slope=0)(layer2)
#         layer2 = tf.keras.layers.Conv2D(
#             256, 2, dilation_rate=[1, 3], padding="same", name="convlayer2_3"
#         )(layer2)
#         layer2 = tf.keras.layers.ReLU(negative_slope=0)(layer2)
#         layer2 = tf.keras.layers.MaxPooling2D(pool_size=2, name="poollayer2")(layer2)
#         layer2 = tf.keras.layers.Dropout(0.15, name="droplayer2")(layer2)
#         layer2 = tf.keras.layers.Flatten(name="flatlayer2")(layer2)

#         # 2x2 with 1x5 dilation
#         layer3 = tf.keras.layers.Conv2D(
#             64,
#             2,
#             # dilation_rate=[1, 5],
#             dilation_rate=[5, 1],
#             padding="same",
#             name="convlayer4_1",
#             kernel_initializer=initializer,
#         )(model_input)
#         layer3 = tf.keras.layers.ReLU(negative_slope=0)(layer3)
#         layer3 = tf.keras.layers.Conv2D(
#             128,
#             2,
#             dilation_rate=[1, 5],
#             padding="same",
#             name="convlayer4_2",
#             kernel_initializer=initializer,
#         )(layer3)
#         layer3 = tf.keras.layers.ReLU(negative_slope=0)(layer3)
#         layer3 = tf.keras.layers.Conv2D(
#             256, 2, dilation_rate=[1, 5], padding="same", name="convlayer4_3"
#         )(layer3)
#         layer3 = tf.keras.layers.ReLU(negative_slope=0)(layer3)
#         layer3 = tf.keras.layers.MaxPooling2D(pool_size=2, name="poollayer3")(layer3)
#         layer3 = tf.keras.layers.Dropout(0.15, name="droplayer3")(layer3)
#         layer3 = tf.keras.layers.Flatten(name="flatlayer3")(layer3)

#         # concatenate convolution layers
#         concat = tf.keras.layers.concatenate([layer1, layer2, layer3])
#         concat = tf.keras.layers.Dense(512, name="512dense", activation="relu")(concat)
#         concat = tf.keras.layers.Dropout(0.2, name="dropconcat1")(concat)
#         concat = tf.keras.layers.Dense(128, name="last_dense", activation="relu")(
#             concat
#         )
#         concat = tf.keras.layers.Dropout(0.2 / 2, name="dropconcat2")(concat)
#         output = tf.keras.layers.Dense(
#             1,
#             name="out_dense",
#             activation="sigmoid",
#             kernel_initializer=initializer,
#         )(concat)

#         return output

#     def cnn_flexsweep_2d(self, model_input, num_classes=1):
#         """
#         Build a CNN as described in the provided JSON, following the explicit, layered cnn_flexsweep style.
#         All parameters and layer names are from your config.
#         """
#         tf = self.check_tf()

#         # --- Main branch ---
#         x = tf.keras.layers.Dropout(0.1, name="dropout")(model_input)
#         x = tf.keras.layers.Flatten(name="flatten")(x)
#         x = tf.keras.layers.Dense(64, activation="relu", name="dense_2")(x)
#         x = tf.keras.layers.Dropout(0.2, name="dropout_1")(x)
#         x = tf.keras.layers.Dense(128, activation="relu", name="dense_3")(x)
#         x = tf.keras.layers.Dropout(0.2, name="dropout_2")(x)
#         out = tf.keras.layers.Dense(num_classes, activation="sigmoid", name="dense_4")(
#             x
#         )

#         return out

#     def cnn_flexsweep_small_v2(self, model_input, num_classes=1):
#         """
#         Fine-scale Flex-sweep CNN with:
#           • Per-stat depthwise spatial filters
#           • Cross-stat mixing via 1×1 conv
#           • Branches for local, mid, and broad patterns
#         Designed for (windows=5, centers=21, stats=11) when channel==11.
#         """
#         tf = self.check_tf()
#         He = tf.keras.initializers.HeNormal()

#         def se_block(x, reduction=8):
#             c = x.shape[-1]
#             s = tf.keras.layers.GlobalAveragePooling2D()(x)
#             s = tf.keras.layers.Dense(
#                 max(c // reduction, 4), activation="relu", kernel_initializer=He
#             )(s)
#             s = tf.keras.layers.Dense(c, activation="sigmoid", kernel_initializer=He)(s)
#             s = tf.keras.layers.Reshape((1, 1, c))(s)
#             return tf.keras.layers.Multiply()([x, s])

#         def per_stat_then_mix(
#             x, k, d=(1, 1), filters_out=24, padding="same", name=None
#         ):
#             """Depthwise conv per stat -> pointwise conv to mix stats."""
#             x = tf.keras.layers.DepthwiseConv2D(
#                 kernel_size=k,
#                 padding=padding,
#                 dilation_rate=d,
#                 depthwise_initializer=He,
#                 name=None if name is None else f"{name}_dw",
#             )(x)
#             x = tf.keras.layers.LayerNormalization(epsilon=1e-5)(x)
#             x = tf.keras.layers.ReLU()(x)

#             x = tf.keras.layers.Conv2D(
#                 filters_out,
#                 (1, 1),
#                 padding="same",
#                 kernel_initializer=He,
#                 name=None if name is None else f"{name}_pw",
#             )(x)
#             x = tf.keras.layers.LayerNormalization(epsilon=1e-5)(x)
#             x = tf.keras.layers.ReLU()(x)
#             return x

#         # --- Stage 0: Early 1×1 stat mixing (light)
#         x0 = tf.keras.layers.Conv2D(
#             16, (1, 1), padding="same", kernel_initializer=He, name="mix_stats"
#         )(model_input)
#         x0 = tf.keras.layers.ReLU()(x0)
#         x0 = se_block(x0)  # channel attention

#         # === Branch A: ultra-local (fine detail)
#         a = per_stat_then_mix(
#             x0, k=(1, 3), filters_out=16, name="A_c1"
#         )  # across centers
#         a = per_stat_then_mix(
#             a, k=(3, 1), filters_out=24, name="A_c2"
#         )  # across windows
#         a = tf.keras.layers.MaxPooling2D(
#             pool_size=(1, 2), padding="same", name="A_pool"
#         )(a)
#         a = tf.keras.layers.SpatialDropout2D(0.10)(a)
#         a = tf.keras.layers.GlobalAveragePooling2D()(a)

#         # === Branch B: mid-context
#         b = per_stat_then_mix(
#             x0, k=(1, 5), filters_out=16, name="B_c1"
#         )  # 5-center span
#         b = per_stat_then_mix(
#             b, k=(3, 1), filters_out=24, name="B_c2"
#         )  # small window context
#         b = tf.keras.layers.MaxPooling2D(
#             pool_size=(1, 2), padding="same", name="B_pool"
#         )(b)
#         b = tf.keras.layers.SpatialDropout2D(0.10)(b)
#         b = tf.keras.layers.GlobalAveragePooling2D()(b)

#         # === Branch C: broad context (late dilation)
#         c = per_stat_then_mix(
#             x0, k=(1, 3), d=(1, 2), filters_out=16, name="C_c1"
#         )  # ~5 centers effective
#         c = per_stat_then_mix(
#             c, k=(3, 1), d=(2, 1), filters_out=24, name="C_c2"
#         )  # ~5 windows effective
#         c = tf.keras.layers.MaxPooling2D(
#             pool_size=(1, 2), padding="same", name="C_pool"
#         )(c)
#         c = tf.keras.layers.SpatialDropout2D(0.10)(c)
#         c = tf.keras.layers.GlobalAveragePooling2D()(c)

#         # === Merge branches
#         x = tf.keras.layers.Concatenate()([a, b, c])

#         # === Head
#         x = tf.keras.layers.Dense(64, activation="relu", kernel_initializer=He)(x)
#         x = tf.keras.layers.Dropout(0.25)(x)
#         act = "sigmoid" if num_classes == 1 else "softmax"
#         out = tf.keras.layers.Dense(num_classes, activation=act, kernel_initializer=He)(
#             x
#         )

#         return out

#     def cnn_flexsweep_small(self, model_input, num_classes=1):
#         """
#         Fine-scale Flex-sweep CNN with:
#           • Per-stat depthwise spatial filters
#           • Cross-stat mixing via 1×1 conv
#           • Branches for local, mid, and broad patterns
#         Designed for (windows=5, centers=21, stats=11) when channel==11.
#         """
#         tf = self.check_tf()
#         He = tf.keras.initializers.HeNormal()

#         def se_block(x, reduction=8):
#             c = x.shape[-1]
#             s = tf.keras.layers.GlobalAveragePooling2D()(x)
#             s = tf.keras.layers.Dense(
#                 max(c // reduction, 4), activation="relu", kernel_initializer=He
#             )(s)
#             s = tf.keras.layers.Dense(c, activation="sigmoid", kernel_initializer=He)(s)
#             s = tf.keras.layers.Reshape((1, 1, c))(s)
#             return tf.keras.layers.Multiply()([x, s])

#         def per_stat_then_mix(
#             x, k, d=(1, 1), filters_out=24, padding="same", name=None
#         ):
#             """Depthwise conv per stat -> pointwise conv to mix stats."""
#             x = tf.keras.layers.DepthwiseConv2D(
#                 kernel_size=k,
#                 padding=padding,
#                 dilation_rate=d,
#                 depthwise_initializer=He,
#                 name=None if name is None else f"{name}_dw",
#             )(x)
#             x = tf.keras.layers.LayerNormalization(epsilon=1e-5)(x)
#             x = tf.keras.layers.ReLU()(x)

#             x = tf.keras.layers.Conv2D(
#                 filters_out,
#                 (1, 1),
#                 padding="same",
#                 kernel_initializer=He,
#                 name=None if name is None else f"{name}_pw",
#             )(x)
#             x = tf.keras.layers.LayerNormalization(epsilon=1e-5)(x)
#             x = tf.keras.layers.ReLU()(x)
#             return x

#         # --- Stage 0: Early 1×1 stat mixing (light)
#         x0 = tf.keras.layers.Conv2D(
#             16, (1, 1), padding="same", kernel_initializer=He, name="mix_stats"
#         )(model_input)
#         x0 = tf.keras.layers.ReLU()(x0)
#         x0 = se_block(x0)  # channel attention

#         # === Branch A: ultra-local (fine detail)
#         a = per_stat_then_mix(
#             x0, k=(1, 3), filters_out=16, name="A_c1"
#         )  # across centers
#         a = per_stat_then_mix(
#             a, k=(3, 1), filters_out=24, name="A_c2"
#         )  # across windows
#         a = tf.keras.layers.MaxPooling2D(
#             pool_size=(1, 2), padding="same", name="A_pool"
#         )(a)
#         a = tf.keras.layers.SpatialDropout2D(0.10)(a)
#         a = tf.keras.layers.GlobalAveragePooling2D()(a)

#         # === Branch B: mid-context
#         b = per_stat_then_mix(
#             x0, k=(1, 5), filters_out=16, name="B_c1"
#         )  # 5-center span
#         b = per_stat_then_mix(
#             b, k=(3, 1), filters_out=24, name="B_c2"
#         )  # small window context
#         b = tf.keras.layers.MaxPooling2D(
#             pool_size=(1, 2), padding="same", name="B_pool"
#         )(b)
#         b = tf.keras.layers.SpatialDropout2D(0.10)(b)
#         b = tf.keras.layers.GlobalAveragePooling2D()(b)

#         # === Branch C: broad context (late dilation)
#         c = per_stat_then_mix(
#             x0, k=(1, 3), d=(1, 2), filters_out=16, name="C_c1"
#         )  # ~5 centers effective
#         c = per_stat_then_mix(
#             c, k=(3, 1), d=(2, 1), filters_out=24, name="C_c2"
#         )  # ~5 windows effective
#         c = tf.keras.layers.MaxPooling2D(
#             pool_size=(1, 2), padding="same", name="C_pool"
#         )(c)
#         c = tf.keras.layers.SpatialDropout2D(0.10)(c)
#         c = tf.keras.layers.GlobalAveragePooling2D()(c)

#         # === Merge branches
#         x = tf.keras.layers.Concatenate()([a, b, c])

#         # === Head
#         x = tf.keras.layers.Dense(64, activation="relu", kernel_initializer=He)(x)
#         x = tf.keras.layers.Dropout(0.25)(x)
#         act = "sigmoid" if num_classes == 1 else "softmax"
#         out = tf.keras.layers.Dense(num_classes, activation=act, kernel_initializer=He)(
#             x
#         )

#         return out

#     def cnn_simple_opt(self, model_input, num_classes=1):
#         """
#         Improved CNN for popgen summary statistics with tiny spatial dimensions.
#         Includes BN, global pooling, and tuned dropout to help generalization and recall.
#         """
#         tf = self.check_tf()

#         initializer = tf.keras.initializers.HeNormal()

#         x = tf.keras.layers.Conv2D(
#             64, (2, 1), padding="same", kernel_initializer=initializer, name="conv_2x1"
#         )(model_input)
#         x = tf.keras.layers.BatchNormalization()(x)
#         x = tf.keras.layers.ReLU()(x)

#         x = tf.keras.layers.Conv2D(
#             64, (1, 2), padding="same", kernel_initializer=initializer, name="conv_1x2"
#         )(x)
#         x = tf.keras.layers.BatchNormalization()(x)
#         x = tf.keras.layers.ReLU()(x)

#         x = tf.keras.layers.Conv2D(
#             128, (2, 2), padding="same", kernel_initializer=initializer, name="conv_2x2"
#         )(x)
#         x = tf.keras.layers.BatchNormalization()(x)
#         x = tf.keras.layers.ReLU()(x)

#         # Optional deeper conv layer
#         x = tf.keras.layers.Conv2D(
#             128, (1, 1), padding="same", kernel_initializer=initializer, name="conv_1x1"
#         )(x)
#         x = tf.keras.layers.BatchNormalization()(x)
#         x = tf.keras.layers.ReLU()(x)

#         x = tf.keras.layers.MaxPooling2D(pool_size=(2, 1), name="gentle_pool")(x)
#         x = tf.keras.layers.Dropout(0.15, name="dropout_1")(x)

#         # Swap Flatten → GlobalAveragePooling2D
#         x = tf.keras.layers.GlobalAveragePooling2D(name="global_avg_pool")(x)
#         # x = self.attention_pool_2d(x, name="attn_pool")

#         x = tf.keras.layers.Dense(128, activation="relu", name="dense_1")(x)
#         x = tf.keras.layers.Dropout(0.2, name="dropout_2")(x)

#         x = tf.keras.layers.Dense(32, activation="relu", name="dense_2")(x)
#         x = tf.keras.layers.Dropout(0.1, name="dropout_3")(x)

#         output = tf.keras.layers.Dense(
#             num_classes, activation="sigmoid", name="output"
#         )(x)

#         return output

#     def load_training_data(self, _stats=None, w=None, n=None):
#         """
#         Loads training data from specified files and preprocesses it for training.

#         Returns:
#             tuple: Contains the training and validation datasets:
#                 - X_train (np.ndarray): Input features for training.
#                 - X_test (np.ndarray): Input features for testing.
#                 - Y_train (np.ndarray): One-hot encoded labels for training.
#                 - Y_test (np.ndarray): One-hot encoded labels for testing.
#                 - X_valid (np.ndarray): Input features for validation.
#                 - Y_valid (np.ndarray): One-hot encoded labels for validation.
#         """
#         tf = self.check_tf()

#         assert self.train_data is not None, "Please input training data"
#         assert (
#             "txt" in self.train_data
#             or "csv" in self.train_data
#             or self.train_data.endswith(".parquet")
#         ), "Please save your dataframe as CSV or parquet"

#         if isinstance(self.train_data, pl.DataFrame):
#             pass
#         elif self.train_data.endswith(".gz"):
#             tmp = pl.read_csv(self.train_data, separator=",")
#         elif self.train_data.endswith(".parquet"):
#             tmp = pl.read_parquet(self.train_data)

#             if n is not None:
#                 tmp = tmp.sample(n)

#         tmp = tmp.with_columns(
#             pl.when((pl.col("model") != "neutral"))
#             .then(pl.lit("sweep"))
#             .otherwise(pl.lit("neutral"))
#             .alias("model")
#         )

#         if w is not None:
#             try:
#                 self.center = np.array([int(w)])
#                 tmp = tmp.select(
#                     "iter", "s", "t", "f_i", "f_t", "model", f"^*._{int(w)}$"
#                 )
#             except:
#                 self.center = np.sort(np.array(w).astype(int))
#                 _tmp = []
#                 _h = tmp.select("iter", "s", "t", "f_i", "f_t", "model")
#                 for window in self.center:
#                     _tmp.append(tmp.select(f"^*._{int(window)}$"))
#                 tmp = pl.concat(_tmp, how="horizontal")
#                 tmp = pl.concat([_h, tmp], how="horizontal")

#         sweep_parameters = tmp.filter("model" != "neutral").select(tmp.columns[:5])

#         stats = [
#             # "dind",
#             # "hapdaf_o",
#             # "isafe",
#             # "high_freq",
#             # "hapdaf_s",
#             # "nsl",
#             # "s_ratio",
#             # "low_freq",
#             # "ihs",
#             # "h12",
#             # "haf"
#         ]
#         # ld_stats = ["ihs", "nsl", "isafe", "haf", "h12","omega_max","zns",'mu_ld']
#         # sfs_stats = ["hapdaf_o", "hapdaf_s",'fay_wu_h','mu_sfs']
#         # div_stats = ["dind", "s_ratio", "low_freq", "high_freq","pi",'mu_var']
#         # stats = ['model','iter'] + ld_stats + sfs_stats + div_stats
#         # stats = ["model","iter","ihs", "nsl", "isafe", "haf", "h12","hapdaf_o", "hapdaf_s","dind", "s_ratio", "low_freq", "high_freq"]

#         if _stats is not None:
#             stats = stats + _stats

#         train_stats = []
#         for i in stats:
#             # train_stats.append(tmp.select(pl.col("^.*" + i + ".*$")))
#             train_stats.append(tmp.select(pl.col(f"^{i}_[0-9]+_[0-9]+$")))

#         # train_stats = pl.concat(train_stats, how="horizontal").select(
#         #     pl.exclude("^.*flip.*$"),
#         # )
#         train_stats = pl.concat(train_stats, how="horizontal")
#         train_stats = pl.concat(
#             [tmp.select("model", "iter", "s", "f_i", "f_t", "t"), train_stats],
#             how="horizontal",
#         )

#         y = train_stats.select(
#             ((~pl.col("model").str.contains("neutral")).cast(pl.Int8)).alias(
#                 "neutral_flag"
#             )
#         )["neutral_flag"].to_numpy()

#         test_split = round(1 - self.train_split, 2)

#         if self.misspecification:
#             stratify_key = np.char.add(
#                 tmp["demo"].to_numpy().astype(str), np.char.add("_", y.astype(str))
#             )
#             X_train, X_test, y_train, y_test = train_test_split(
#                 train_stats,
#                 y,
#                 test_size=test_split,
#                 shuffle=True,
#                 stratify=stratify_key,
#             )
#         else:
#             (
#                 X_train,
#                 X_test,
#                 Y_train,
#                 y_test,
#             ) = train_test_split(train_stats, y, test_size=test_split, shuffle=True)

#         X_train = (
#             X_train.select(train_stats.columns[6:])
#             .to_numpy()
#             .reshape(
#                 X_train.shape[0],
#                 self.num_stats,
#                 self.windows.size * self.center.size,
#                 1,
#             )
#         )

#         X_valid, X_test, Y_valid, Y_test = train_test_split(
#             X_test, y_test, test_size=0.5
#         )

#         X_test_params = X_test.select(X_test.columns[:6])
#         X_test = (
#             X_test.select(train_stats.columns[6:])
#             .to_numpy()
#             .reshape(
#                 X_test.shape[0],
#                 self.num_stats,
#                 self.windows.size * self.center.size,
#                 1,
#             )
#         )
#         X_valid = (
#             X_valid.select(train_stats.columns[6:])
#             .to_numpy()
#             .reshape(
#                 X_valid.shape[0],
#                 self.num_stats,
#                 self.windows.size * self.center.size,
#                 1,
#             )
#         )

#         # === NEW: adapt Normalization on training data ===
#         if self.normalize:
#             self.stat_norm = tf.keras.layers.Normalization(axis=-1, name="stat_norm")
#             self.stat_norm.adapt(X_train)  # learns mean/std from training set only

#         # Y_train = np.argmax(Y_train, axis=1)
#         # Y_test = np.argmax(Y_test, axis=1)
#         # Y_valid = np.argmax(Y_valid, axis=1)

#         self.test_data = [X_test, X_test_params, Y_test]

#         return (
#             X_train,
#             X_test,
#             Y_train,
#             Y_test,
#             X_valid,
#             Y_valid,
#         )

#     def train(self, _iter=1, _stats=None, w=None, cnn=None, input_data=None):
#         """
#         Trains the CNN model on the training data.

#         This method preprocesses the data, sets up data augmentation, defines the model architecture,
#         compiles the model, and fits it to the training data while saving the best model and training history.
#         """
#         tf = self.check_tf()

#         if cnn is None:
#             cnn = self.cnn_flexsweep

#         if self.split is True:
#             (
#                 X_train,
#                 X_test,
#                 Y_train,
#                 Y_test,
#                 X_valid,
#                 Y_valid,
#             ) = self.load_training_data(w=w, _stats=_stats)
#             # (X_train_ld,X_test_ld,X_valid_ld,X_train_sfs,X_test_sfs,X_valid_sfs,X_train_div,X_test_div,X_valid_div) = self.load_training_data_stat()
#             self.test_data_splitted = (
#                 X_train,
#                 X_test,
#                 Y_train,
#                 Y_test,
#                 X_valid,
#                 Y_valid,
#                 # X_train_hap,
#                 # X_test_hap,
#                 # X_valid_hap,
#             )

#         else:
#             (
#                 X_train,
#                 X_test,
#                 Y_train,
#                 Y_test,
#                 X_valid,
#                 Y_valid,
#                 # X_train_hap,
#                 # X_test_hap,
#                 # X_valid_hap,
#             ) = self.test_data_splitted

#         if self.channel == 5:
#             X_train = X_train.reshape(
#                 X_train.shape[0], self.num_stats, self.center.size, self.windows.size
#             )
#             X_test = X_test.reshape(
#                 X_test.shape[0], self.num_stats, self.center.size, self.windows.size
#             )
#             X_valid = X_valid.reshape(
#                 X_valid.shape[0], self.num_stats, self.center.size, self.windows.size
#             )
#             input_shape = (self.num_stats, self.center.size, self.windows.size)
#         elif self.channel == 11:
#             X_train = X_train.reshape(
#                 X_train.shape[0], self.windows.size, self.center.size, self.num_stats
#             )
#             X_test = X_test.reshape(
#                 X_test.shape[0], self.windows.size, self.center.size, self.num_stats
#             )
#             X_valid = X_valid.reshape(
#                 X_valid.shape[0], self.windows.size, self.center.size, self.num_stats
#             )
#             input_shape = (self.center.size, self.windows.size, self.num_stats)
#         elif self.channel == 21:
#             X_train = X_train.reshape(
#                 X_train.shape[0], self.windows.size, self.num_stats, self.center.size
#             )
#             X_test = X_test.reshape(
#                 X_test.shape[0], self.windows.size, self.num_stats, self.center.size
#             )
#             X_valid = X_valid.reshape(
#                 X_valid.shape[0], self.windows.size, self.num_stats, self.center.size
#             )
#             input_shape = X_train.shape[1:]

#         # put model together
#         input_to_model = tf.keras.Input(X_train.shape[1:])
#         # input_ld = X_train_ld.shape[1:]
#         # input_sfs = X_train_sfs.shape[1:]
#         # input_div = X_train_div.shape[1:]
#         batch_size = 32

#         # Instantiate your augmentation pipeline
#         # aug_pipeline = AugmentationPipeline(
#         #     rotation_prob=0.1,
#         #     flip_prob=0.3,
#         #     # shuffle_centers_prob=0.5,
#         #     # shuffle_windows_prob=0.3,
#         #     # noise_factor=0.02,
#         #     # dropout_centers_prob=0.15,
#         #     # dropout_windows_prob=0.15,
#         #     # statistic_dropout_prob=0.1,
#         #     # swap_centers_prob=0.3
#         # )

#         train_dataset = (
#             tf.data.Dataset.from_tensor_slices((X_train, Y_train))
#             # tf.data.Dataset.from_tensor_slices(((X_train_ld,X_train_sfs,X_train_div), Y_train))
#             .shuffle(10000)
#             .batch(batch_size)
#             .prefetch(tf.data.AUTOTUNE)
#         )

#         # train_dataset = aug_pipeline.create_augmented_dataset(raw_train_dataset, batch_size=batch_size)
#         valid_dataset = (
#             tf.data.Dataset.from_tensor_slices((X_valid, Y_valid))
#             # tf.data.Dataset.from_tensor_slices(((X_valid_ld,X_valid_sfs,X_valid_div), Y_valid))
#             .batch(batch_size).prefetch(tf.data.AUTOTUNE)
#         )
#         test_dataset = (
#             tf.data.Dataset.from_tensor_slices((X_test, Y_test))
#             # tf.data.Dataset.from_tensor_slices(((X_test_ld,X_test_sfs,X_test_div), Y_test))
#             .batch(batch_size).prefetch(tf.data.AUTOTUNE)
#         )

#         model = tf.keras.models.Model(
#             inputs=[input_to_model], outputs=[cnn(input_to_model)]
#         )

#         # model = cnn(input_ld,input_sfs,input_div)

#         model_path = f"{self.output_folder}/model5.keras"

#         metrics_measures = [
#             tf.keras.metrics.BinaryAccuracy(name="val_accuracy"),
#             tf.keras.metrics.Precision(name="val_precision"),
#             tf.keras.metrics.AUC(name="roc", curve="ROC"),
#         ]

#         lr_decayed_fn = tf.keras.optimizers.schedules.CosineDecayRestarts(
#             initial_learning_rate=1e-4, first_decay_steps=300
#         )
#         opt_adam = tf.keras.optimizers.Adam(
#             learning_rate=lr_decayed_fn, epsilon=0.0000001, amsgrad=True
#         )

#         custom_loss = tf.keras.losses.BinaryCrossentropy(label_smoothing=0.05)
#         # Keep only one compilation

#         # model.compile(loss=focal_loss(gamma=2.0, alpha=0.5))
#         model.compile(
#             optimizer=opt_adam,
#             loss=custom_loss,
#             # loss = focal_loss(gamma=2.0, alpha=0.5),
#             metrics=[
#                 tf.keras.metrics.BinaryAccuracy(name="accuracy"),
#                 tf.keras.metrics.AUC(name="auc", curve="ROC"),
#                 tf.keras.metrics.Precision(name="precision"),
#             ],
#         )
#         earlystop = tf.keras.callbacks.EarlyStopping(
#             monitor="val_auc",
#             min_delta=0.0001,
#             patience=10,
#             verbose=2,
#             mode="max",
#             restore_best_weights=True,
#         )

#         # monitor="val_auc","val_prc","sweep_recall",
#         checkpoint = tf.keras.callbacks.ModelCheckpoint(
#             model_path,
#             monitor="val_auc",
#             verbose=2,
#             save_best_only=True,
#             mode="max",
#         )

#         callbacks_list = [checkpoint, earlystop]

#         start = time.time()

#         # class_weight = {0: 1.0, 1: 1.2}
#         # model.fit(..., )

#         history = model.fit(
#             train_dataset,
#             # class_weight=class_weight,
#             epochs=1000,
#             validation_data=valid_dataset,
#             callbacks=callbacks_list,
#         )

#         val_score = model.evaluate(
#             valid_dataset,
#             batch_size=32,
#             steps=len(Y_valid) // 32,
#         )
#         test_score = model.evaluate(
#             test_dataset,
#             batch_size=32,
#             steps=len(Y_test) // 32,
#         )

#         train_score = model.evaluate(
#             train_dataset,
#             batch_size=32,
#             steps=len(Y_train) // 32,
#         )
#         self.model = model

#         df_history = pl.DataFrame(history.history)
#         self.history = df_history

#         print(
#             "Training and testing model took {} seconds".format(
#                 round(time.time() - start, 3)
#             )
#         )

#         if self.output_folder is not None:
#             model.save(model_path)

#         # return self.model.evaluate(X_valid, Y_valid, verbose=0)[3]

#     def predict(self, _stats=None, w=None, input_data=None, _iter=1):
#         """
#         Makes predictions on the test data using the trained CNN model.

#         This method loads test data, processes it, and applies the trained model to generate predictions.

#         Raises:
#             AssertionError: If the model has not been trained or loaded.
#         """
#         tf = self.check_tf()

#         assert self.model is not None, "Please input the CNN trained model"

#         assert self.test_data is not None, "Please input training data"

#         # import data to predict
#         _output_prediction = self.output_folder + "/" + self.output_prediction
#         if isinstance(self.test_data, str):
#             assert (
#                 isinstance(self.test_data, pl.DataFrame)
#                 or "txt" in self.test_data
#                 or "csv" in self.test_data
#                 or self.test_data.endswith(".parquet")
#             ), "Please input a pl.DataFrame or save it as CSV or parquet"
#             try:
#                 df_test = pl.read_parquet(self.test_data)
#                 if "test" in self.test_data:
#                     df_test = df_test.sample(
#                         with_replacement=False, fraction=1.0, shuffle=True
#                     )
#             except:
#                 df_test = pl.read_csv(self.test_data, separator=",")

#             df_test = df_test.with_columns(
#                 pl.when((pl.col("model") != "neutral"))
#                 .then(pl.lit("sweep"))
#                 .otherwise(pl.lit("neutral"))
#                 .alias("model")
#             )
#             regions = df_test["iter"].to_numpy()

#             stats = []
#             if _stats is not None:
#                 stats = stats + _stats
#             test_stats = []

#             for i in stats:
#                 test_stats.append(df_test.select(pl.col(f"^{i}_[0-9]+_[0-9]+$")))

#             X_test = pl.concat(test_stats, how="horizontal")

#             if w is not None:
#                 try:
#                     self.center = np.array([int(w)])
#                     X_test = X_test.select(f"^*._{int(w)}$")
#                 except:
#                     self.center = np.sort(np.array(w).astype(int))
#                     _X_test = []
#                     for window in self.center:
#                         _X_test.append(X_test.select(f"^*._{int(window)}$"))
#                     X_test = pl.concat(_X_test, how="horizontal")

#             test_X_params = df_test.select("model", "iter", "s", "f_i", "f_t", "t")

#             test_X = (
#                 X_test.select(X_test)
#                 .to_numpy()
#                 .reshape(
#                     X_test.shape[0],
#                     self.num_stats,
#                     self.windows.size * self.center.size,
#                     1,
#                 )
#             )

#             test_Y = df_test.select(
#                 ((~pl.col("model").str.contains("neutral")).cast(pl.Int8)).alias(
#                     "neutral_flag"
#                 )
#             )["neutral_flag"].to_numpy()

#             # Same folder custom fvs name based on input VCF.
#             self.output_prediction = (
#                 os.path.basename(self.test_data)
#                 .replace("fvs_", "")
#                 .replace(".parquet", "_predictions.txt")
#             )

#         else:
#             try:
#                 # test_X, test_X_hap, test_X_params, test_Y = deepcopy(self.test_data)
#                 (X_test_ld, X_test_sfs, X_test_div, Y_test) = deepcopy(self.test_data)
#             except:
#                 test_X, test_X_params, test_Y = deepcopy(self.test_data)

#         if self.channel == 5:
#             test_X = test_X.reshape(
#                 test_X.shape[0], self.num_stats, self.center.size, self.windows.size
#             )
#         elif self.channel == 11:
#             test_X = test_X.reshape(
#                 test_X.shape[0], self.windows.size, self.center.size, self.num_stats
#             )
#         elif self.channel == 21:
#             test_X = test_X.reshape(
#                 test_X.shape[0], self.windows.size, self.num_stats, self.center.size
#             )

#         # batch size, image width, image height,number of channels
#         if isinstance(self.model, str):
#             model = tf.keras.models.load_model(self.model)
#         else:
#             model = self.model

#         try:
#             preds = model.predict((X_test_ld, X_test_sfs, X_test_div))
#         except:
#             preds = model.predict(test_X)

#         preds = np.column_stack([1 - preds, preds])
#         predictions = np.argmax(preds, axis=1)

#         y_true = (test_X_params["model"] != "neutral").to_numpy().astype(int)
#         # y_true = Y_test
#         fpr, tpr, thresh = roc_curve(y_true, preds[:, 1])

#         # 2) Pick threshold: FPR <= max_fpr, maximize TPR
#         max_fpr = 0.9
#         valid = np.where(fpr <= max_fpr)[0]
#         if len(valid) > 0:
#             idx = valid[np.argmax(tpr[valid])]
#         else:
#             idx = np.argmin(fpr)
#         best_thresh = thresh[idx]

#         prediction_dict = {
#             0: "neutral",
#             1: "sweep",
#         }
#         predictions_class = np.vectorize(prediction_dict.get)(predictions)
#         # predictions_class = np.where(preds[:, 1] >= best_thresh, "sweep", "neutral")

#         df_prediction = pl.concat(
#             [
#                 # pl.DataFrame(np.column_stack([np.vectorize(prediction_dict.get)(y_true),np.zeros((preds.shape[0],4))]),schema = ['model','f_i','f_t','s','t']),
#                 test_X_params.select("model", "f_i", "f_t", "s", "t"),
#                 pl.DataFrame(
#                     {
#                         "predicted_model": predictions_class,
#                         "prob_sweep": preds[:, 1],
#                         "prob_neutral": preds[:, 0],
#                     }
#                 ),
#             ],
#             how="horizontal",
#         )

#         # df_prediction = df_prediction.with_columns(pl.lit(np.vectorize(prediction_dict.get)(y_true)).alias('model'))
#         if isinstance(self.test_data, str) and "test" not in self.test_data:
#             df_prediction = df_prediction.with_columns(pl.Series("region", regions))
#             chr_start_end = np.array(
#                 [item.replace(":", "-").split("-") for item in regions]
#             )

#             df_prediction = df_prediction.with_columns(
#                 pl.Series("chr", chr_start_end[:, 0]),
#                 pl.Series("start", chr_start_end[:, 1], dtype=pl.Int64),
#                 pl.Series("end", chr_start_end[:, 2], dtype=pl.Int64),
#                 pl.Series(
#                     "nchr",
#                     pl.Series(chr_start_end[:, 0]).str.replace("chr", "").cast(int),
#                 ),
#             )
#             df_prediction = df_prediction.sort("nchr", "start").select(
#                 pl.exclude("region", "iter", "model", "nchr")
#             )

#         self.prediction = df_prediction
#         self.roc_curve()

#         if self.output_folder is not None:
#             df_prediction.write_csv(_output_prediction)

#         return df_prediction, test_X

#     def roc_curve(self, _iter=1):
#         """
#         Generates and plots ROC curves along with a history plot of model metrics.

#         Returns:
#             tuple: A tuple containing:
#                 - plot_roc (Figure): The ROC curve plot.
#                 - plot_history (Figure): The history plot of model metrics (loss, validation loss, accuracy, validation accuracy).

#         Example:
#             roc_plot, history_plot = model.roc_curves()
#             plt.show(roc_plot)
#             plt.show(history_plot)
#         """
#         import matplotlib.pyplot as plt

#         if isinstance(self.prediction, str):
#             pred_data = pl.read_csv(self.prediction)
#         else:
#             pred_data = self.prediction

#         # Create confusion dataframe
#         confusion_data = pred_data.group_by(["model", "predicted_model"]).agg(
#             pl.len().alias("n")
#         )

#         expected_combinations = pl.DataFrame(
#             {
#                 "model": ["sweep", "sweep", "neutral", "neutral"],
#                 "predicted_model": ["sweep", "neutral", "neutral", "sweep"],
#             }
#         )

#         confusion_data = expected_combinations.join(
#             confusion_data, on=["model", "predicted_model"], how="left"
#         ).fill_null(
#             0
#         )  # Fill missing values with 0
#         # Adding the "true_false" column
#         confusion_data = confusion_data.with_columns(
#             pl.when(
#                 (pl.col("model") == pl.col("predicted_model"))
#                 & (pl.col("model") == "neutral")
#             )
#             .then(pl.lit("true_negative"))  # Explicit literal for Polars
#             .when(
#                 (pl.col("model") == pl.col("predicted_model"))
#                 & (pl.col("model") == "sweep")
#             )
#             .then(pl.lit("true_positive"))  # Explicit literal for Polars
#             .when(
#                 (pl.col("model") != pl.col("predicted_model"))
#                 & (pl.col("model") == "neutral")
#             )
#             .then(pl.lit("false_positive"))  # Explicit literal for Polars
#             .otherwise(pl.lit("false_negative"))  # Explicit literal for Polars
#             .alias("true_false")
#         )

#         confusion_pivot = confusion_data.pivot(
#             values="n", index=None, on="true_false", aggregate_function="sum"
#         ).fill_null(0)

#         # Copying the pivoted data (optional as Polars is immutable)
#         rate_data = confusion_pivot.select(
#             ["false_negative", "false_positive", "true_negative", "true_positive"]
#         ).sum()

#         # Compute the required row sums for normalization
#         required_cols = [
#             "false_negative",
#             "false_positive",
#             "true_negative",
#             "true_positive",
#         ]
#         for col in required_cols:
#             if col not in rate_data.columns:
#                 rate_data[col] = 0

#         # Calculate row sums for normalization
#         rate_data = rate_data.with_columns(
#             (pl.col("false_negative") + pl.col("true_positive")).alias("sum_fn_tp")
#         )
#         rate_data = rate_data.with_columns(
#             (pl.col("false_positive") + pl.col("true_negative")).alias("sum_fp_tn")
#         )

#         # Compute normalized rates
#         rate_data = rate_data.with_columns(
#             (pl.col("false_negative") / pl.col("sum_fn_tp")).alias("false_negative"),
#             (pl.col("false_positive") / pl.col("sum_fp_tn")).alias("false_positive"),
#             (pl.col("true_negative") / pl.col("sum_fp_tn")).alias("true_negative"),
#             (pl.col("true_positive") / pl.col("sum_fn_tp")).alias("true_positive"),
#         )

#         # Replace NaN values with 0
#         rate_data = rate_data.with_columns(
#             [
#                 pl.col("false_negative").fill_null(0).alias("false_negative"),
#                 pl.col("false_positive").fill_null(0).alias("false_positive"),
#                 pl.col("true_negative").fill_null(0).alias("true_negative"),
#                 pl.col("true_positive").fill_null(0).alias("true_positive"),
#             ]
#         )

#         # Calculate accuracy and precision
#         rate_data = rate_data.with_columns(
#             [
#                 (
#                     (pl.col("true_positive") + pl.col("true_negative"))
#                     / (
#                         pl.col("true_positive")
#                         + pl.col("true_negative")
#                         + pl.col("false_positive")
#                         + pl.col("false_negative")
#                     )
#                 ).alias("accuracy"),
#                 (
#                     pl.col("true_positive")
#                     / (pl.col("true_positive") + pl.col("false_positive"))
#                 )
#                 .fill_null(0)
#                 .alias("precision"),
#             ]
#         )

#         # Compute ROC AUC and prepare roc_data. Set 'sweep' as the positive class
#         pred_rate_auc_data = pred_data.clone().with_columns(
#             pl.col("model").cast(pl.Categorical).alias("model")
#         )

#         # Calculate ROC AUC
#         roc_auc_value = roc_auc_score(
#             (pred_rate_auc_data["model"] == "sweep").cast(int),
#             pred_rate_auc_data["prob_sweep"].cast(float),
#         )

#         # Create roc_data DataFrame
#         roc_data = pl.DataFrame({"AUC": [roc_auc_value]})

#         rate_roc_data = pl.concat([pl.DataFrame(rate_data), roc_data], how="horizontal")

#         first_row_values = rate_roc_data.row(0)

#         pred_rate_auc_data = pred_rate_auc_data.with_columns(
#             [
#                 pl.lit(value).alias(col)
#                 for col, value in zip(rate_roc_data.columns, first_row_values)
#             ]
#         )

#         # Compute ROC curve using sklearn
#         fpr, tpr, thresholds = roc_curve(
#             (pred_rate_auc_data["model"] == "sweep").cast(int),
#             pred_rate_auc_data["prob_sweep"].cast(float),
#         )

#         roc_df = pl.DataFrame({"false_positive_rate": fpr, "sensitivity": tpr})

#         fig, ax = plt.subplots(figsize=(10, 6))
#         ax.plot(
#             roc_df["false_positive_rate"],
#             roc_df["sensitivity"],
#             color="orange",
#             linewidth=2,
#             label="ROC Curve",
#         )
#         ax.plot(
#             [0, 1], [0, 1], color="grey", linestyle="--"
#         )  # Diagonal line for reference
#         ax.set_xlabel("false positive rate")
#         ax.set_ylabel("power")
#         ax.set_title("ROC Curve")
#         ax.axis("equal")  # Equivalent to coord_equal in ggplot
#         ax.grid(True, which="both", linestyle="--", linewidth=0.5)
#         ax.legend()
#         fig.tight_layout()
#         plot_roc = fig

#         ## History
#         # Load and preprocess the data
#         history_data = self.history
#         h = history_data.select(
#             ["loss", "val_loss", "accuracy", "val_accuracy"]
#         ).clone()

#         h = h.with_columns((pl.arange(0, h.height) + 1).alias("epoch"))

#         h_melted = h.unpivot(
#             index=["epoch"],
#             on=["loss", "val_loss", "accuracy", "val_accuracy"],
#             variable_name="metric_name",
#             value_name="metric_val",
#         )

#         line_styles = {
#             "loss": "-",
#             "val_loss": "--",
#             "accuracy": "-",
#             "val_accuracy": "--",
#         }
#         colors = {
#             "loss": "orange",
#             "val_loss": "orange",
#             "accuracy": "blue",
#             "val_accuracy": "blue",
#         }

#         fig, ax = plt.subplots(figsize=(10, 6))

#         for group_name, group_df in h_melted.group_by("metric_name"):
#             ax.plot(
#                 group_df["epoch"].to_numpy(),
#                 group_df["metric_val"].to_numpy(),
#                 label=group_name[0],
#                 linestyle=line_styles[group_name[0]],
#                 color=colors[group_name[0]],
#                 linewidth=2,
#             )
#         ax.set_title("History")
#         ax.set_xlabel("Epoch")
#         ax.set_ylabel("Value")
#         ax.tick_params(axis="both", labelsize=10)
#         ax.grid(True)
#         ax.legend(title="", loc="upper right")

#         plot_history = fig

#         ##############
#         cm = confusion_matrix(
#             pred_data["model"],
#             pred_data["predicted_model"],
#             labels=["neutral", "sweep"],
#             normalize="true",
#         )
#         disp = ConfusionMatrixDisplay(
#             confusion_matrix=cm, display_labels=["neutral", "sweep"]
#         )
#         cm_plot = disp.plot(cmap="Blues")

#         print(cm)

#         if self.output_folder is not None:
#             plt.savefig(self.output_folder + "/confusion_matrix.svg")
#             plt.close()
#             plot_roc.savefig(self.output_folder + f"/roc_curve.svg")
#             plot_history.savefig(
#                 # self.output_folder + f"/train_history_{ihs}_{_iter}.svg"
#                 self.output_folder
#                 + f"/train_history.svg"
#             )
#         return plot_roc, plot_history

#     def _select_stats_matrix_like_old(self, df: pl.DataFrame, stats: list[str], w=None):
#         # Standardize model: anything not 'neutral' -> 'sweep'
#         df = df.with_columns(
#             pl.when(pl.col("model") != "neutral")
#             .then(pl.lit("sweep"))
#             .otherwise(pl.lit("neutral"))
#             .alias("model")
#         )

#         # (optional) keep only requested center(s) like your old code
#         if w is not None:
#             try:
#                 self.center = np.array([int(w)])
#                 df = df.select(
#                     "iter", "s", "t", "f_i", "f_t", "model", f"^*._{int(w)}$"
#                 )
#             except:
#                 self.center = np.sort(np.array(w).astype(int))
#                 _tmp = []
#                 _h = df.select("iter", "s", "t", "f_i", "f_t", "model")
#                 for ww in self.center:
#                     _tmp.append(df.select(f"^*._{int(ww)}$"))
#                 df = pl.concat(_tmp, how="horizontal")
#                 df = pl.concat([_h, df], how="horizontal")

#         # Your original regex selection per stat
#         blocks = []
#         for s in stats:
#             blk = df.select(pl.col(f"^{s}_[0-9]+_[0-9]+$"))

#             # sort columns deterministically by (window, center)
#             cols = blk.columns

#             def key(col):
#                 base, a, b = col.rsplit("_", 2)  # safe for stat names with underscores
#                 a = int(a)
#                 b = int(b)
#                 # try to infer which token is window vs center
#                 if (a in set(self.windows.tolist())) and (
#                     b in set(self.center.tolist())
#                 ):
#                     wv, cv = a, b
#                 elif (a in set(self.center.tolist())) and (
#                     b in set(self.windows.tolist())
#                 ):
#                     wv, cv = b, a
#                 else:
#                     # fallback: assume {stat}_{center}_{window}
#                     cv, wv = a, b
#                 return (wv, cv)

#             cols_sorted = sorted(cols, key=key)
#             blocks.append(blk.select(cols_sorted))

#         X = pl.concat(blocks, how="horizontal")
#         y = (df["model"] != "neutral").cast(pl.Int8).to_numpy().astype(np.float32)
#         params = df.select("iter", "s", "t", "f_i", "f_t", "model")

#         # reshape to (N, W, C, S)
#         N = df.height
#         X = (
#             X.to_numpy()
#             .reshape(N, self.windows.size, self.center.size, len(stats))
#             .astype(np.float32)
#         )

#         return X, y, params

#     def load_da_data(
#         self,
#         _stats=None,
#         w=None,
#         n_src=None,
#         n_tgt=None,
#         test_size=0.20,
#         val_size=0.10,
#         seed=42,
#     ):
#         tf = self.check_tf()
#         assert (
#             self.source_data is not None and self.target_data is not None
#         ), "Set source_data and target_data"
#         assert self.source_data.endswith(".parquet") and self.target_data.endswith(
#             ".parquet"
#         )

#         src_df = pl.read_parquet(self.source_data)
#         tgt_df = pl.read_parquet(self.target_data)

#         if n_src is not None:
#             src_df = src_df.sample(n_src)
#         if n_tgt is not None:
#             tgt_df = tgt_df.sample(n_tgt)

#         # stats comes from caller (like your old loader)
#         stats = []
#         if _stats is not None:
#             stats = stats + _stats

#         # Build tensors exactly like your old code
#         X_src, y_src, _ = self._select_stats_matrix_like_old(src_df, stats, w=w)
#         X_tgt, y_tgt, tgt_params = self._select_stats_matrix_like_old(
#             tgt_df, stats, w=w
#         )

#         # Split target into train/val/test
#         from sklearn.model_selection import train_test_split

#         X_t_tr, X_t_te, y_t_tr, y_t_te, p_tr, p_te = train_test_split(
#             X_tgt,
#             y_tgt,
#             tgt_params,
#             test_size=test_size,
#             stratify=y_tgt,
#             random_state=seed,
#         )
#         val_frac = val_size / (1.0 - test_size)
#         X_t_tr, X_t_va, y_t_tr, y_t_va, p_tr, p_va = train_test_split(
#             X_t_tr, y_t_tr, p_tr, test_size=val_frac, stratify=y_t_tr, random_state=seed
#         )

#         # keep for training/prediction
#         self.da_data = {
#             "stats": stats,
#             "X_src": X_src,
#             "y_src": y_src,
#             "X_tgt_train": X_t_tr,
#             "y_tgt_train": y_t_tr,
#             "X_tgt_val": X_t_va,
#             "y_tgt_val": y_t_va,
#             "tgt_val_params": p_va,
#             "X_tgt_test": X_t_te,
#             "y_tgt_test": y_t_te,
#             "tgt_test_params": p_te,
#         }
#         # set test_data so predict_da mimics your predict()
#         self.test_data = [X_t_te, p_te, y_t_te]
#         return self.da_data

#     def train_da(self, _stats=None, w=None, batch_size=32, epochs=100, seed=42):
#         tf = self.check_tf()
#         if not hasattr(self, "da_data") or self.da_data is None:
#             self.load_da_data(_stats=_stats, w=w, seed=seed)

#         da = self.da_data

#         # if self.normalize:
#         #     # # 2) ---- Fit a single Normalization layer on (source-train ∪ target-train) ----
#         #     X_adapt = np.concatenate([da["X_src"], da["X_tgt_train"]], axis=0).astype(np.float32)

#         #     self.stat_norm_da = tf.keras.layers.Normalization(axis=-1, name="stat_norm_da")
#         #     self.stat_norm_da.adapt(X_adapt)   # <-- fit ON TRAIN ONLY (no val/test leakage)

#         # generator: builds 2B batches with masking exactly like Siepel
#         gen = DAParquetSequence(
#             da["X_src"],
#             da["y_src"],
#             da["X_tgt_train"],
#             da["y_tgt_train"],
#             batch_size=batch_size,
#             shuffle=True,
#             rng_seed=seed,
#         )

#         input_shape = (self.windows.size, self.center.size, len(da["stats"]))
#         model = self.build_grl_model_from_cnn_simple_opt(input_shape)

#         # small mixed-domain validation (so discriminator metrics aren't 0/NaN)
#         k = min(2000, len(da["X_src"]))
#         idx_src = np.random.RandomState(seed).choice(
#             len(da["X_src"]), size=k, replace=False
#         )
#         Xv = np.concatenate([da["X_src"][idx_src], da["X_tgt_val"]], axis=0)
#         yv_cls = np.concatenate(
#             [-np.ones((k, 1), np.float32), da["y_tgt_val"][:, None].astype(np.float32)],
#             axis=0,
#         )
#         yv_dom = np.concatenate(
#             [
#                 np.zeros((k, 1), np.float32),
#                 np.ones((da["X_tgt_val"].shape[0], 1), np.float32),
#             ],
#             axis=0,
#         )

#         lr_sched = tf.keras.optimizers.schedules.CosineDecayRestarts(
#             initial_learning_rate=5e-5, first_decay_steps=300
#         )
#         opt = tf.keras.optimizers.Adam(
#             learning_rate=lr_sched, epsilon=1e-7, amsgrad=True
#         )

#         # compile with masked BCE (Siepel semantics)
#         model.compile(
#             optimizer=opt,
#             loss={"classifier": masked_bce(), "discriminator": masked_bce()},
#             # loss_weights={"classifier": 1.0, "discriminator": 1.0},
#             loss_weights={"classifier": 1.0, "discriminator": 0.5},
#             metrics={
#                 "classifier": [
#                     tf.keras.metrics.AUC(name="auc"),
#                     tf.keras.metrics.BinaryAccuracy(name="accuracy"),
#                 ],
#                 "discriminator": [tf.keras.metrics.BinaryAccuracy(name="accuracy")],
#             },
#         )

#         ckpt_path = (
#             f"{self.output_folder}/model_da.keras" if self.output_folder else None
#         )
#         callbacks = []
#         callbacks.append(
#             tf.keras.callbacks.EarlyStopping(
#                 monitor="val_classifier_auc",
#                 mode="max",
#                 patience=20,
#                 min_delta=1e-4,
#                 restore_best_weights=True,
#             )
#         )

#         ramp_frac = 0.7
#         max_lambda = 0.5
#         ramp_epochs = int(ramp_frac * epochs)

#         callbacks.append(GRLRamp(self.grl, max_lambda=max_lambda, epochs=ramp_epochs))

#         if ckpt_path:
#             callbacks.append(
#                 tf.keras.callbacks.ModelCheckpoint(
#                     ckpt_path,
#                     monitor="val_classifier_auc",
#                     mode="max",
#                     save_best_only=True,
#                     verbose=2,
#                 )
#             )

#         # === fit ===
#         history = model.fit(
#             gen,
#             epochs=epochs,
#             steps_per_epoch=len(gen),
#             validation_data=(Xv, {"classifier": yv_cls, "discriminator": yv_dom}),
#             callbacks=callbacks,
#             verbose=2,
#         )

#         # store history with the names your roc_curve() expects
#         hh = history.history
#         self.history = pl.DataFrame(
#             {
#                 "loss": hh.get("loss", []),
#                 "val_loss": hh.get("val_loss", []),
#                 "accuracy": hh.get("classifier_accuracy", []),
#                 "val_accuracy": hh.get("val_classifier_accuracy", []),
#                 "auc": hh.get("classifier_auc", []),
#                 "val_auc": hh.get("val_classifier_auc", []),
#             }
#         )
#         self.model = model
#         if self.output_folder:
#             model.save(f"{self.output_folder}/model_da.keras")
#         return model

#     def predict_da(self, _stats=None):
#         tf = self.check_tf()
#         assert self.model is not None, "Call train_da() first"
#         assert isinstance(self.test_data, (list, tuple)) and len(self.test_data) == 3

#         X_test, test_params, Y_test = self.test_data
#         out = self.model.predict(X_test, verbose=0, batch_size=32)
#         cls = out[0] if isinstance(out, (list, tuple)) else out
#         p = cls.ravel().astype(np.float32)

#         df_pred = pl.concat(
#             [
#                 test_params,
#                 pl.DataFrame(
#                     {
#                         "predicted_model": np.where(p >= 0.5, "sweep", "neutral"),
#                         "prob_sweep": p,
#                         "prob_neutral": 1.0 - p,
#                     }
#                 ),
#             ],
#             how="horizontal",
#         )

#         # overwrite 'model' with clean neutral/sweep from numeric Y for ROC/CM
#         df_pred = df_pred.drop("model").with_columns(
#             pl.Series("model", np.where(Y_test == 1, "sweep", "neutral"))
#         )

#         self.prediction = df_pred
#         self.roc_curve()
#         if self.output_folder:
#             df_pred.write_csv(self.output_folder + "/predictions_da.txt")
#         return df_pred, X_test

#     def feature_extractor_from_cnn_simple_opt(self, model_input):
#         # tf = self.check_tf()
#         # initializer = tf.keras.initializers.HeNormal()

#         # x = tf.keras.layers.Conv2D(64, (2, 1), padding="same", kernel_initializer=initializer, name="conv_2x1")(model_input)
#         # x = tf.keras.layers.LayerNormalization(axis=[-1], epsilon=1e-5)(x); x = tf.keras.layers.ReLU()(x)

#         # x = tf.keras.layers.Conv2D(64, (1, 2), padding="same", kernel_initializer=initializer, name="conv_1x2")(x)
#         # x = tf.keras.layers.LayerNormalization(axis=[-1], epsilon=1e-5)(x); x = tf.keras.layers.ReLU()(x)

#         # x = tf.keras.layers.Conv2D(128, (2, 2), padding="same", kernel_initializer=initializer, name="conv_2x2")(x)
#         # x = tf.keras.layers.LayerNormalization(axis=[-1], epsilon=1e-5)(x); x = tf.keras.layers.ReLU()(x)

#         # x = tf.keras.layers.Conv2D(128, (1, 1), padding="same", kernel_initializer=initializer, name="conv_1x1")(x)
#         # x = tf.keras.layers.LayerNormalization(axis=[-1], epsilon=1e-5)(x); x = tf.keras.layers.ReLU()(x)

#         # # <-- SE attention across the stat channels
#         # x = se_block_2d(x, reduction=4)

#         # x = tf.keras.layers.MaxPooling2D(pool_size=(2, 1), name="gentle_pool")(x)
#         # x = tf.keras.layers.Dropout(0.15, name="dropout_after_2d")(x)

#         # # feat = tf.keras.layers.Flatten(name="flatten_pool")(x)
#         # feat = tf.keras.layers.GlobalAveragePooling2D(name="gap")(x)
#         # tf = self.check_tf()
#         He = tf.keras.initializers.HeNormal()

#         # --- Branch 1: 3x3 convs ---
#         b1 = tf.keras.layers.Conv2D(
#             64, 3, padding="same", kernel_initializer=He, name="fx_b1_c1"
#         )(model_input)
#         b1 = tf.keras.layers.ReLU()(b1)
#         b1 = tf.keras.layers.Conv2D(
#             128, 3, padding="same", kernel_initializer=He, name="fx_b1_c2"
#         )(b1)
#         b1 = tf.keras.layers.ReLU()(b1)
#         b1 = tf.keras.layers.Conv2D(
#             256, 3, padding="same", kernel_initializer=He, name="fx_b1_c3"
#         )(b1)
#         b1 = tf.keras.layers.ReLU()(b1)
#         b1 = tf.keras.layers.MaxPooling2D(
#             pool_size=3, padding="same", name="fx_b1_pool"
#         )(b1)
#         b1 = tf.keras.layers.Dropout(0.15, name="fx_b1_drop")(b1)
#         b1 = tf.keras.layers.Flatten(name="fx_b1_flat")(b1)

#         # --- Branch 2: 2x2 convs with dilation (1,3) ---
#         b2 = tf.keras.layers.Conv2D(
#             64,
#             2,
#             dilation_rate=[1, 3],
#             padding="same",
#             kernel_initializer=He,
#             name="fx_b2_c1",
#         )(model_input)
#         b2 = tf.keras.layers.ReLU()(b2)
#         b2 = tf.keras.layers.Conv2D(
#             128,
#             2,
#             dilation_rate=[1, 3],
#             padding="same",
#             kernel_initializer=He,
#             name="fx_b2_c2",
#         )(b2)
#         b2 = tf.keras.layers.ReLU()(b2)
#         b2 = tf.keras.layers.Conv2D(
#             256,
#             2,
#             dilation_rate=[1, 3],
#             padding="same",
#             kernel_initializer=He,
#             name="fx_b2_c3",
#         )(b2)
#         b2 = tf.keras.layers.ReLU()(b2)
#         b2 = tf.keras.layers.MaxPooling2D(pool_size=2, name="fx_b2_pool")(b2)
#         b2 = tf.keras.layers.Dropout(0.15, name="fx_b2_drop")(b2)
#         b2 = tf.keras.layers.Flatten(name="fx_b2_flat")(b2)

#         # --- Branch 3: 2x2 convs with dilation (5,1) then (1,5) ---
#         b3 = tf.keras.layers.Conv2D(
#             64,
#             2,
#             dilation_rate=[5, 1],
#             padding="same",
#             kernel_initializer=He,
#             name="fx_b3_c1",
#         )(model_input)
#         b3 = tf.keras.layers.ReLU()(b3)
#         b3 = tf.keras.layers.Conv2D(
#             128,
#             2,
#             dilation_rate=[1, 5],
#             padding="same",
#             kernel_initializer=He,
#             name="fx_b3_c2",
#         )(b3)
#         b3 = tf.keras.layers.ReLU()(b3)
#         b3 = tf.keras.layers.Conv2D(
#             256,
#             2,
#             dilation_rate=[1, 5],
#             padding="same",
#             kernel_initializer=He,
#             name="fx_b3_c3",
#         )(b3)
#         b3 = tf.keras.layers.ReLU()(b3)
#         b3 = tf.keras.layers.MaxPooling2D(pool_size=2, name="fx_b3_pool")(b3)
#         b3 = tf.keras.layers.Dropout(0.15, name="fx_b3_drop")(b3)
#         b3 = tf.keras.layers.Flatten(name="fx_b3_flat")(b3)

#         feat = tf.keras.layers.Concatenate(name="fx_concat")(
#             [b1, b2, b3]
#         )  # shared representation

#         return feat

#     def build_grl_model_from_cnn_simple_opt(self, input_shape):
#         tf = self.check_tf()
#         inp = tf.keras.Input(shape=input_shape)  # (W, C, S), channels-last

#         feat = self.feature_extractor_from_cnn_simple_opt(inp)

#         # classifier head
#         h = tf.keras.layers.Dense(128, activation="relu")(feat)
#         h = tf.keras.layers.Dropout(0.20)(h)
#         h = tf.keras.layers.Dense(32, activation="relu")(h)
#         h = tf.keras.layers.Dropout(0.10)(h)
#         out_cls = tf.keras.layers.Dense(1, activation="sigmoid", name="classifier")(h)

#         # domain head via GRL (store the layer for ramping)
#         self.grl = GradReverse(lambd=0.0)
#         g = self.grl(feat)
#         g = tf.keras.layers.Dense(128, activation="relu")(g)
#         g = tf.keras.layers.Dropout(0.20)(g)
#         g = tf.keras.layers.Dense(32, activation="relu")(g)
#         out_dom = tf.keras.layers.Dense(1, activation="sigmoid", name="discriminator")(
#             g
#         )

#         model = tf.keras.Model(inputs=inp, outputs=[out_cls, out_dom])

#         return model


# class DAParquetSequence(Sequence):
#     def __init__(
#         self, X_src, y_src, X_tgt, y_tgt, batch_size=32, shuffle=True, rng_seed=42
#     ):
#         assert X_src.shape[1:] == X_tgt.shape[1:], "Source/Target shapes must match"
#         self.Xs, self.ys = X_src, y_src.astype(np.int32)
#         self.Xt, self.yt = X_tgt, y_tgt.astype(np.int32)
#         self.B = batch_size
#         self.shuffle = shuffle
#         self.rng = np.random.RandomState(rng_seed)
#         self._reset_epoch()
#         # limit batches by pools as Siepel does
#         self.n_batches = int(
#             np.floor(min(len(self.src_pool_cls), len(self.tgt_pool_dis)) / self.B)
#         )

#     def _reset_epoch(self):
#         self.src_pool_cls = self.rng.permutation(
#             len(self.ys)
#         )  # for classifier (source)
#         self.src_pool_dis = self.rng.permutation(
#             len(self.ys)
#         )  # for discriminator (source)
#         self.tgt_pool_dis = self.rng.permutation(
#             len(self.yt)
#         )  # for discriminator (target))

#     def __len__(self):
#         return self.n_batches

#     def __getitem__(self, idx):
#         # A) B source for classifier (0/1), domain masked
#         idxA = self.src_pool_cls[idx * self.B : (idx + 1) * self.B]
#         XA = self.Xs[idxA]
#         yA_cls = self.ys[idxA].astype(np.float32).reshape(-1, 1)
#         yA_dom = -np.ones((XA.shape[0], 1), np.float32)

#         # B) B/2 source for discriminator (domain=0), classifier masked
#         half = self.B // 2
#         idxB = self.src_pool_dis[idx * half : (idx + 1) * half]
#         XB = self.Xs[idxB]
#         yB_cls = -np.ones((XB.shape[0], 1), np.float32)
#         yB_dom = np.zeros((XB.shape[0], 1), np.float32)

#         # C) B/2 target for discriminator (domain=1), classifier masked
#         idxC = self.tgt_pool_dis[idx * half : (idx + 1) * half]
#         XC = self.Xt[idxC]
#         yC_cls = -np.ones((XC.shape[0], 1), np.float32)
#         yC_dom = np.ones((XC.shape[0], 1), np.float32)

#         X = np.concatenate([XA, XB, XC], axis=0)
#         y_cls = np.concatenate([yA_cls, yB_cls, yC_cls], axis=0)
#         y_dom = np.concatenate([yA_dom, yB_dom, yC_dom], axis=0)

#         assert X.shape[0] == 2 * self.B
#         return X, {"classifier": y_cls, "discriminator": y_dom}

#     def on_epoch_end(self):
#         if self.shuffle:
#             self._reset_epoch()


def rank_probabilities(data_dir, gene_coordinates, pop, include_xy=False):
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
    # gene_coordinates = "/labstorage/jmurgamoreno/bchak/ensembl_gene_coords_v109.bed"
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
