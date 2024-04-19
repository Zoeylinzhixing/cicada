from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import (
    Activation,
    AveragePooling2D,
    Conv2D,
    Dense,
    Dropout,
    Flatten,
    Input,
    Reshape,
    UpSampling2D,
    BatchNormalization
)
from qkeras import QActivation, QConv2D, QDense, QDenseBatchnorm, quantized_bits




class TeacherAutoencoder:
    def __init__(self, input_shape: tuple):
        self.input_shape = input_shape
    def get_model(self):
        inputs = Input(shape=self.input_shape, name="inputs")
        x = Reshape((72, 40, 1), name="reshape")(inputs)
        x = Conv2D(16, (3, 3), strides=1, padding="same", name="conv2d_1")(x)
        x = BatchNormalization(name="bn_1")(x)
        x = Activation("relu", name="relu_1")(x)
        x = AveragePooling2D((2, 2), name="pool_1")(x)
        x = Conv2D(32, (3, 3), strides=1, padding="same", name="conv2d_2")(x)
        x = BatchNormalization(name="bn_2")(x)
        x = Activation("relu", name="relu_2")(x)
        x = Dropout(0.25, name="dropout_1")(x)
        x = Flatten(name="flatten")(x)
        x = Dense(128, activation="relu", name="dense_1")(x)
        x = Dropout(0.5, name="dropout_2")(x)
        x = Dense(36 * 20 * 30, activation="relu", name="dense_2")(x)  
        x = Reshape((36, 20, 30), name="reshape2")(x)
        x = Activation("relu", name="teacher_relu_3")(x)
        x = Conv2D(30, (3, 3), strides=1, padding="same", name="teacher_conv2d_3")(x)
        x = Activation("relu", name="teacher_relu_4")(x)
        x = UpSampling2D((2, 2), name="teacher_upsampling")(x)
        x = Conv2D(20, (3, 3), strides=1, padding="same", name="teacher_conv2d_4")(x)
        x = Activation("relu", name="teacher_relu_5")(x)
        outputs = Conv2D(
            1,
            (3, 3),
            activation="relu",
            strides=1,
            padding="same",
            name="teacher_outputs",
        )(x)
        return Model(inputs, outputs, name="teacher")


class CicadaV1:
    def __init__(self, input_shape: tuple):
        self.input_shape = input_shape

    def get_model(self):
        inputs = Input(shape=self.input_shape, name="inputs_")
        x = QDenseBatchnorm(
            16,
            kernel_quantizer=quantized_bits(8, 1, 1, alpha=1.0),
            bias_quantizer=quantized_bits(8, 3, 1, alpha=1.0),
            name="dense1",
        )(inputs)
        x = QActivation("quantized_relu(10, 6)", name="relu1")(x)
        x = Dropout(1 / 8)(x)
        x = QDense(
            1,
            kernel_quantizer=quantized_bits(12, 3, 1, alpha=1.0),
            use_bias=False,
            name="dense2",
        )(x)
        outputs = QActivation("quantized_relu(16, 8)", name="outputs")(x)
        return Model(inputs, outputs, name="cicada-v1")


class CicadaV2:
    def __init__(self, input_shape: tuple):
        self.input_shape = input_shape

    def get_model(self):
        inputs = Input(shape=self.input_shape, name="inputs_")
        x = Reshape((72, 40, 1), name="reshape")(inputs)
        x = QConv2D(
            32,
            (5, 5),
            strides=(1, 1),
            padding="same",
            use_bias=False,
            kernel_quantizer=quantized_bits(8, 2, 1, alpha=1.0),
            name="conv1",
        )(x)
        x = QActivation("quantized_relu(8, 4)", name="relu0")(x)
        x = Flatten(name="flatten")(x)
        x = QDense(
            64,
            use_bias=False,
            kernel_quantizer=quantized_bits(8, 2, 1, alpha=1.0),
            bias_quantizer=quantized_bits(8, 2, 1, alpha=1.0),
            name="dense1"
        )(x)
        # x = Dropout(1 / 9)(x)
        # x = QDenseBatchnorm(
        #     16,
        #     kernel_quantizer=quantized_bits(8, 1, 1, alpha=1.0),
        #     bias_quantizer=quantized_bits(8, 3, 1, alpha=1.0),
        #     name="dense1",
        # )(x)
        x = QActivation("quantized_relu(8, 4)", name="relu1")(x)
        # x = Dropout(1 / 8)(x)
        x = QDense(
            1,
            kernel_quantizer=quantized_bits(8, 2, 1, alpha=1.0),
            use_bias=False,
            name="dense2",
        )(x)
        outputs = QActivation("quantized_relu(8, 4)", name="outputs")(x)
        return Model(inputs, outputs, name="cicada-v2")

