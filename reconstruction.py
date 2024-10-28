# In this script, we read in the teacher model to create reconstruction plots
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import argparse
import numpy as np
import numpy.typing as npt
# import pandas as pd
import tensorflow as tf
import yaml

from drawing import Draw
from generator import RegionETGenerator
from pathlib import Path
from tensorflow import data
from tensorflow import keras
from tensorflow.keras import Model
# from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger
# from tensorflow.keras.optimizers import Adam
# from typing import List
from utils import IsValidFile

from qkeras import *


def loss(y_true: npt.NDArray, y_pred: npt.NDArray) -> npt.NDArray:
    return np.mean((y_true - y_pred) ** 2, axis=(1, 2, 3))


def quantize(arr: npt.NDArray, precision: tuple = (16, 8)) -> npt.NDArray:
    word, int_ = precision
    decimal = word - int_
    step = 1 / 2**decimal
    max_ = 2**int_ - step
    arrq = step * np.round(arr / step)
    arrc = np.clip(arrq, 0, max_)
    return arrc


def get_student_targets(
    teacher: Model, gen: RegionETGenerator, X: npt.NDArray
) -> data.Dataset:
    X_hat = teacher.predict(X, batch_size=512, verbose=0)
    y = loss(X, X_hat)
    y = np.sqrt(y)
    # y = quantize(np.log(y) * 512)
    return gen.get_generator(X.reshape((-1, 2880, 1)), y, 1024, True)


def train_model(
    model: Model,
    gen_train: tf.data.Dataset,
    gen_val: tf.data.Dataset,
    epoch: int = 1,
    steps: int = 1,
    callbacks=None,
    verbose: bool = False,
) -> None:
    model.fit(
        gen_train,
        steps_per_epoch=len(gen_train),
        initial_epoch=epoch,
        epochs=epoch + steps,
        validation_data=gen_val,
        callbacks=callbacks,
        verbose=verbose,
    )


def reconstruction(
    config: dict, verbose: bool = False
) -> None:
    draw = Draw()

    datasets = [i["path"] for i in config["background"] if i["use"]]
    datasets = [path for paths in datasets for path in paths]

    gen = RegionETGenerator()
    X_train, X_val, X_test = gen.get_data_split(datasets)
    print(X_test.shape)
    X_signal, _ = gen.get_benchmark(config["signal"], filter_acceptance=False)


    teacher = keras.models.load_model("models/teacher")

    # Evaluation
    y_pred_background_teacher = teacher.predict(X_test, batch_size=512, verbose=verbose)
    y_loss_background_teacher = loss(X_test, y_pred_background_teacher)

    results_teacher = dict()
    results_teacher["2023 Zero Bias (Test)"] = y_loss_background_teacher

    # sort the loss
    sort_index_background_teacher = np.argsort(y_loss_background_teacher)

    # plot reconstruction according to loss
    thresh_small = 4
    thresh_large = 400
    # large loss 1
    event_index = sort_index_background_teacher[-1]
    X_example = X_test[event_index: event_index + 1]
    y_example = teacher.predict(X_example, verbose=verbose)
    draw.plot_reconstruction_results(
        X_example,
        y_example,
        loss=loss(X_example, y_example)[0],
        name="reconstruction-background-large-loss-1",
        max_deposit=thresh_large,
    )
    # large loss 2
    event_index = sort_index_background_teacher[-2]
    X_example = X_test[event_index:event_index + 1]
    y_example = teacher.predict(X_example, verbose=verbose)
    draw.plot_reconstruction_results(
        X_example,
        y_example,
        loss=loss(X_example, y_example)[0],
        name="reconstruction-background-large-loss-2",
        max_deposit=thresh_large,
    )
    # small loss 1
    event_index = sort_index_background_teacher[0]
    X_example = X_test[event_index: event_index + 1]
    y_example = teacher.predict(X_example, verbose=verbose)
    draw.plot_reconstruction_results(
        X_example,
        y_example,
        loss=loss(X_example, y_example)[0],
        name="reconstruction-background-small-loss-1",
        max_deposit=thresh_small,
    )
    # small loss 2
    event_index = sort_index_background_teacher[1]
    X_example = X_test[event_index: event_index + 1]
    y_example = teacher.predict(X_example, verbose=verbose)
    draw.plot_reconstruction_results(
        X_example,
        y_example,
        loss=loss(X_example, y_example)[0],
        name="reconstruction-background-small-loss-2",
        max_deposit=thresh_small,
    )

    y_true, y_pred_teacher = [], []
    inputs = []
    for name, data in X_signal.items():
        inputs.append(np.concatenate((data, X_test)))

        y_loss_teacher = loss(
            data, teacher.predict(data, batch_size=512, verbose=verbose)
        )

        results_teacher[name] = y_loss_teacher

        # sort the loss
        sort_index_teacher = np.argsort(y_loss_teacher)

        # reconstruction plots
        # large loss 1
        event_index = sort_index_teacher[-1]
        X_example = X_signal[name][event_index:event_index+1]
        y_example = teacher.predict(X_example, verbose=verbose)
        draw.plot_reconstruction_results(
            X_example,
            y_example,
            loss=loss(X_example, y_example)[0],
            name=f"reconstruction-{name}-large-loss-1",
            max_deposit=thresh_large,
        )

        # large loss 2
        event_index = sort_index_teacher[-2]
        X_example = X_signal[name][event_index:event_index + 1]
        y_example = teacher.predict(X_example, verbose=verbose)
        draw.plot_reconstruction_results(
            X_example,
            y_example,
            loss=loss(X_example, y_example)[0],
            name=f"reconstruction-{name}-large-loss-2",
            max_deposit=thresh_large,
        )

        # small loss 1
        event_index = sort_index_teacher[0]
        X_example = X_signal[name][event_index:event_index + 1]
        y_example = teacher.predict(X_example, verbose=verbose)
        draw.plot_reconstruction_results(
            X_example,
            y_example,
            loss=loss(X_example, y_example)[0],
            name=f"reconstruction-{name}-small-loss-1",
            max_deposit=thresh_small,
        )

        # small loss 2
        event_index = sort_index_teacher[1]
        X_example = X_signal[name][event_index:event_index + 1]
        y_example = teacher.predict(X_example, verbose=verbose)
        draw.plot_reconstruction_results(
            X_example,
            y_example,
            loss=loss(X_example, y_example)[0],
            name=f"reconstruction-{name}-small-loss-2",
            max_deposit=thresh_small,
        )





def parse_arguments():
    parser = argparse.ArgumentParser(description="""CICADA training scripts""")
    parser.add_argument(
        "--config",
        "-c",
        action=IsValidFile,
        type=Path,
        default="misc/config.yml",
        help="Path to config file",
    )
    parser.add_argument(
        "--evaluate-only",
        action="store_true",
        help="Skip training",
        default=False,
    )
    parser.add_argument(
        "-e",
        "--epochs",
        type=int,
        help="Number of training epochs",
        default=100,
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Output verbosity",
        default=False,
    )
    args = parser.parse_args()
    config = yaml.safe_load(open(args.config))
    return args, config


def main(args_in=None) -> None:
    args, config = parse_arguments()
    reconstruction(config, verbose=args.verbose)


if __name__ == "__main__":
    main()
