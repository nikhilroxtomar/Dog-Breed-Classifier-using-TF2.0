
import os
import numpy as np
import pandas as pd
import cv2
from glob import glob
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import SGD, Adam
from keras import backend as K
from sklearn.model_selection import train_test_split

def read_image(path, size):
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    image = cv2.resize(image, (size, size))
    image = image / 255.0
    image = image.astype(np.float32)
    return image

def build_model(size, num_classes=120, trainable=False):
    inputs = Input((size, size, 3))
    backbone = MobileNetV2(input_tensor=inputs, include_top=False, weights="imagenet")
    backbone.trainable = trainable
    x = backbone.output

    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.2)(x)
    x = Dense(1024, activation="relu")(x)
    x = Dense(num_classes, activation="softmax")(x)

    model = tf.keras.Model(inputs, x)
    return model

def tf_parse(x, y):
    def _prase(x, y):
        x = x.decode()

        num_classes = 120
        size = 224
        image = read_image(x, size)
        label = [0] * num_classes
        label[y] = 1
        label = np.array(label, dtype=np.int32)

        return image, label

    return tf.numpy_function(_prase, [x, y], [tf.float32, tf.int32])

def tf_dataset(x, y, batch=8):
    dataset = tf.data.Dataset.from_tensor_slices((x, y))
    dataset = dataset.map(tf_parse)
    dataset = dataset.batch(batch)
    return dataset

if __name__ == "__main__":
    path = "Dog Breed Identification/"
    train_path = os.path.join(path, "train/*")
    labels_path = os.path.join(path, "labels.csv")

    labels_df = pd.read_csv(labels_path)
    breed = labels_df["breed"].unique()
    print("Number of Breeds: ", len(breed))
    breed2id = {name:i for i, name in enumerate(breed)}

    images = glob(train_path)
    labels = []
    for image_path in images:
        image_id = image_path.split("/")[-1].split(".")[0]
        breed_name = list(labels_df[labels_df.id == image_id]["breed"])[0]
        breed_idx = breed2id[breed_name]
        labels.append(breed_idx)

    ##
    images = images[:2000]
    labels = labels[:2000]

    ## Spliting b/w training and testing data
    train_x, valid_x = train_test_split(images, test_size=0.2, random_state=42)
    train_y, valid_y = train_test_split(labels, test_size=0.2, random_state=42)

    ## Hyperparameters
    size = 224
    num_classes = len(breed)
    lr = 1e-4
    batch = 16
    epochs = 5

    ## Model
    model = build_model(size, num_classes=num_classes, trainable=False)
    model.compile(loss="categorical_crossentropy", optimizer=Adam(lr), metrics=["acc"])

    ## Dataset
    train_dataset = tf_dataset(train_x, train_y)
    valid_dataset = tf_dataset(valid_x, valid_y)

    ## Training
    callbacks = [
        ModelCheckpoint("model.h5", verbose=1),
        ReduceLROnPlateau(factor=0.1, patience=5, min_lr=1e-6)
    ]
    model.fit(train_dataset, validation_data=valid_dataset, epochs=epochs, callbacks=callbacks)
