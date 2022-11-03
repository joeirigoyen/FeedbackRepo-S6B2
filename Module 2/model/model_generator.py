import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import numpy as np
import keras.utils
import keras.models
from PIL import Image
from pathlib import Path
from numpy import linspace
from scipy import interpolate
from keras.models import load_model
from matplotlib import pyplot as plt
from tensorflow import nn, expand_dims
from keras import Sequential, layers, callbacks
from keras.losses import SparseCategoricalCrossentropy

tf.get_logger().setLevel('ERROR')
tf.autograph.set_verbosity(3)

SAVED_MODEL_PATH = Path.cwd().joinpath('saved')
SAVED_PLOTS_PATH = Path.cwd().joinpath('plots')


def convert_to_path(path_to_convert: str | Path):
    return Path(path_to_convert) if isinstance(path_to_convert, str) else path_to_convert


def get_saved_model(filename: str):
    return load_model(SAVED_MODEL_PATH.joinpath(filename))


def get_history_plot(history: callbacks.History, filename: str, epochs: int, title: str):
    acc, val_acc = history.history['accuracy'], history.history['val_accuracy']
    loss, val_loss = history.history['loss'], history.history['val_loss']
    rng = range(epochs)

    acc_spline = interpolate.make_interp_spline(rng, acc)
    val_acc_spline = interpolate.make_interp_spline(rng, val_acc)
    loss_spline = interpolate.make_interp_spline(rng, loss)
    val_loss_spline = interpolate.make_interp_spline(rng, val_loss)

    new_rng = linspace(1, epochs - 1, 50)
    new_acc = acc_spline(new_rng)
    new_val_acc = val_acc_spline(new_rng)
    new_loss = loss_spline(new_rng)
    new_val_loss = val_loss_spline(new_rng)

    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(new_rng, new_acc, label='Training Accuracy')
    plt.plot(new_rng, new_val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title(title)

    plt.subplot(1, 2, 2)
    plt.plot(new_rng, new_loss, label='Training Loss')
    plt.plot(new_rng, new_val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Model Loss')
    plt.savefig(SAVED_PLOTS_PATH.joinpath(filename))


def convert_images_in(class_dir: str | Path, src_suffix: str = 'webp', out_suffix: str = 'jpeg'):
    class_dir = convert_to_path(class_dir)
    for child in class_dir.rglob('*'):
        print(child.suffix)
        if child.suffix == '.' + src_suffix:
            img = Image.open(child).convert('RGB')
            new_name = child.with_suffix('.' + out_suffix)
            img.save(new_name, out_suffix)


def get_classes_in(class_dir: str | Path):
    class_dir = convert_to_path(class_dir)
    return [child.name for child in class_dir.rglob('*') if child.is_dir()]


def get_datasets(data_path: str | Path, validation_split: float = 0.2, image_size: tuple = (96, 96), batch_size: int = 10) -> tuple:
    training_data = keras.utils.image_dataset_from_directory(
        data_path,
        validation_split=validation_split,
        subset='training',
        seed=666,
        image_size=image_size
    )
    validation_data = keras.utils.image_dataset_from_directory(
        data_path,
        validation_split=validation_split,
        subset='validation',
        seed=666,
        image_size=image_size
    )
    return training_data, validation_data


def get_augmentation_layer(image_size: tuple):
    return Sequential([
        layers.RandomFlip("horizontal", input_shape=(image_size[0], image_size[1], 3)),
        layers.RandomRotation(0.1),
        layers.RandomBrightness(0.125),
        layers.RandomContrast(0.1)
    ])


def get_model(num_classes: int, image_height: int, image_width: int, batch_size: int = 10):
    model = Sequential([
        get_augmentation_layer((image_height, image_width)),
        layers.Rescaling(1. / 255, input_shape=(image_height, image_width, 3)),
        layers.Conv2D(8, 3, padding='same', activation='relu'),
        layers.Conv2D(8, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(strides=2, padding='same'),
        layers.Conv2D(16, 3, padding='same', activation='relu'),
        layers.Conv2D(16, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(strides=2, padding='same'),
        layers.Conv2D(32, 3, padding='same', activation='relu'),
        layers.Conv2D(32, 3, padding='same', activation='relu'),
        layers.Conv2D(32, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(strides=2, padding='same'),
        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(strides=2, padding='same'),
        layers.Dropout(0.2),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(32, activation='relu'),
        layers.Dense(num_classes)
    ])
    model.compile(optimizer='adam',
                  loss=SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    return model


def fit_model(model: keras.models.Model, training_data: keras.utils.image_dataset, validation_data: keras.utils.image_dataset, epochs: int = 15):
    history = model.fit(training_data,
                        validation_data=validation_data,
                        epochs=epochs)
    model.save(SAVED_MODEL_PATH.joinpath('my_model.h5'))
    return history


def get_prediction(model: keras.models.Model, image_path: Path, image_size: tuple, classes: list):
    image = keras.utils.load_img(image_path, target_size=image_size)
    img_arr = keras.utils.img_to_array(image)
    img_arr = expand_dims(img_arr, 0)
    predicts = model.predict(img_arr)
    score = nn.softmax(predicts[0])
    return classes[np.argmax(score)], 100 * np.max(score)
