# In this script, we only train the teacher model
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import argparse
import numpy as np
import numpy.typing as npt
import pandas as pd
import qkeras
import tensorflow as tf
import yaml

from drawing import Draw
from generator import RegionETGenerator
from models_v1 import TeacherAutoencoder
from pathlib import Path
from tensorflow import data
from tensorflow import keras
from tensorflow.keras import Model
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger
from tensorflow.keras.optimizers import Adam
from typing import List
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


def run_training(
    config: dict, eval_only: bool, epochs: int = 100, verbose: bool = False
) -> None:
    draw = Draw()

    datasets = [i["path"] for i in config["background"] if i["use"]]
    datasets = [path for paths in datasets for path in paths]

    gen = RegionETGenerator()
    X_train, X_val, X_test = gen.get_data_split(datasets)
    print(X_test.shape)
    X_signal, _ = gen.get_benchmark(config["signal"], filter_acceptance=False)
    gen_train = gen.get_generator(X_train, X_train, 512, True)
    gen_val = gen.get_generator(X_val, X_val, 512)
    outlier_train = gen.get_data(config["exposure"]["training"])
    outlier_val = gen.get_data(config["exposure"]["validation"])

    X_train_student = np.concatenate([X_train, outlier_train])
    X_val_student = np.concatenate([X_val, outlier_val])

    if not eval_only:
        teacher = TeacherAutoencoder((72, 40, 1)).get_model()
        teacher.compile(optimizer=Adam(learning_rate=0.001), loss="mse")
        t_mc = ModelCheckpoint(f"models/{teacher.name}", save_best_only=True)
        t_log = CSVLogger(f"models/{teacher.name}/training.log", append=True)

        for epoch in range(epochs):
            print(f"epoch {epoch}")
            train_model(
                teacher,
                gen_train,
                gen_val,
                epoch=epoch,
                callbacks=[t_mc, t_log],
                verbose=verbose,
            )
            tmp_teacher = keras.models.load_model("models/teacher")

        for model in [teacher]:
            log = pd.read_csv(f"models/{model.name}/training.log")
            draw.plot_loss_history(
                log["loss"], log["val_loss"], f"{model.name}-training-history"
            )

    teacher = keras.models.load_model("models/teacher")

    # Comparison between original and reconstructed inputs
    # background dataset
    # example 1: test data in the background
    X_example = X_test[:1]
    y_example = teacher.predict(X_example, verbose=verbose)
    draw.plot_reconstruction_results(
        X_example,
        y_example,
        loss=loss(X_example, y_example)[0],
        name="comparison-background_1",
    )

    # example 2: test data in the background
    # X_example = X_test[-1]
    # y_example = teacher.predict(X_example, verbose=verbose)
    # draw.plot_reconstruction_results(
    #     X_example,
    #     y_example,
    #     loss=loss(X_example, y_example)[0],
    #     name="comparison-background_2",
    # )

    # example 3: test data in the background
    # X_example = X_test[len(X_test)//2]
    # y_example = teacher.predict(X_example, verbose=verbose)
    # draw.plot_reconstruction_results(
    #     X_example,
    #     y_example,
    #     loss=loss(X_example, y_example)[0],
    #     name="comparison-background_3",
    # )

    # example 4: test data in the background
    # X_example = X_test[len(X_test)//4]
    # y_example = teacher.predict(X_example, verbose=verbose)
    # draw.plot_reconstruction_results(
    #     X_example,
    #     y_example,
    #     loss=loss(X_example, y_example)[0],
    #     name="comparison-background_4",
    # )

    # HTo2LongLivedTo4b
    # example 1: signal data HTo2LongLivedTo4b
    X_example = X_signal["HTo2LongLivedTo4b"][:1]
    y_example = teacher.predict(X_example, verbose=verbose)
    draw.plot_reconstruction_results(
        X_example,
        y_example,
        loss=loss(X_example, y_example)[0],
        name="comparison-HTo2LongLivedTo4b_1",
    )

    # example 2: signal data HTo2LongLivedTo4b
    # X_example = X_signal["HTo2LongLivedTo4b"][-1]
    # y_example = teacher.predict(X_example, verbose=verbose)
    # draw.plot_reconstruction_results(
    #     X_example,
    #     y_example,
    #     loss=loss(X_example, y_example)[0],
    #     name="comparison-HTo2LongLivedTo4b_2",
    # )

    # RelValTTbar_SemiLeptonic
    # example 1: signal data RelValTTbar_SemiLeptonic
    X_example = X_signal["RelValTTbar_SemiLeptonic"][:1]
    y_example = teacher.predict(X_example, verbose=verbose)
    draw.plot_reconstruction_results(
        X_example,
        y_example,
        loss=loss(X_example, y_example)[0],
        name="comparison-RelValTTbar_SemiLeptonic_1",
    )

    # example 2: signal data RelValTTbar_SemiLeptonic
    # X_example = X_signal["RelValTTbar_SemiLeptonic"][-1]
    # y_example = teacher.predict(X_example, verbose=verbose)
    # draw.plot_reconstruction_results(
    #     X_example,
    #     y_example,
    #     loss=loss(X_example, y_example)[0],
    #     name="comparison-RelValTTbar_SemiLeptonic_2",
    # )


    # VBFHToTauTau
    # example 1: signal data VBFHToTauTau
    X_example = X_signal["VBFHToTauTau"][:1]
    y_example = teacher.predict(X_example, verbose=verbose)
    draw.plot_reconstruction_results(
        X_example,
        y_example,
        loss=loss(X_example, y_example)[0],
        name="comparison-VBFHToTauTau_1",
    )

    # example 2: signal data VBFHToTauTau
    # X_example = X_signal["VBFHToTauTau"][-1]
    # y_example = teacher.predict(X_example, verbose=verbose)
    # draw.plot_reconstruction_results(
    #     X_example,
    #     y_example,
    #     loss=loss(X_example, y_example)[0],
    #     name="comparison-VBFHToTauTau_2",
    # )

    # Evaluation
    y_pred_background_teacher = teacher.predict(X_test, batch_size=512, verbose=verbose)
    y_loss_background_teacher = loss(X_test, y_pred_background_teacher)

    results_teacher = dict()
    results_teacher["2023 Zero Bias (Test)"] = y_loss_background_teacher

    y_true, y_pred_teacher = [], []
    inputs = []
    for name, data in X_signal.items():
        inputs.append(np.concatenate((data, X_test)))

        y_loss_teacher = loss(
            data, teacher.predict(data, batch_size=512, verbose=verbose)
        )
        results_teacher[name] = y_loss_teacher

        y_true.append(
            np.concatenate((np.ones(data.shape[0]), np.zeros(X_test.shape[0])))
        )
        y_pred_teacher.append(
            np.concatenate((y_loss_teacher, y_loss_background_teacher))
        )

    draw.plot_anomaly_score_distribution(
        list(results_teacher.values()),
        [*results_teacher],
        "anomaly-score-teacher",
    )

    # ROC Curves with Cross-Validation
    draw.plot_roc_curve(y_true, y_pred_teacher, [*X_signal], inputs, "roc-teacher")


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
    run_training(config, args.evaluate_only, epochs=args.epochs, verbose=args.verbose)


if __name__ == "__main__":
    main()
