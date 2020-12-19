import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf

from tensorflow.keras.preprocessing import image_dataset_from_directory

class Classify:

    def __init__(self, base_dir):
        # Path to cropped images from detector
        self.path = base_dir
        self.input_path = os.path.join(self.path, "cascade-image")

    def train(self, img_size=(160, 160), batch_size=32):
        train_dataset = image_dataset_from_directory(self.input_path,
                                                     shuffle=True,
                                                     batch_size=BATCH_SIZE,
                                                     image_size=IMG_SIZE,
                                                     validation_split=0.2,
                                                     subset="training",
                                                     seed=420)
        validation_dataset = image_dataset_from_directory(self.input_path,
                                                     shuffle=True,
                                                     batch_size=BATCH_SIZE,
                                                     image_size=IMG_SIZE,
                                                     validation_split=0.2,
                                                     subset="validation",
                                                     seed=420)

        # Create test set by move 20% of the validation
        val_batches = tf.data.experimental.cardinality(validation_dataset)
        test_dataset = validation_dataset.take(val_batches // 5)
        validation_dataset = validation_dataset.skip(val_batches // 5)

        # Use buffered prefetching to load images without having I/O blocking
        AUTOTUNE = tf.data.experimental.AUTOTUNE

        train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
        validation_dataset = validation_dataset.prefetch(buffer_size=AUTOTUNE)
        test_dataset = test_dataset.prefetch(buffer_size=AUTOTUNE)

        # Data augmentation
        # Apply random transformations to augment data set
        # These layers are only active during training when calling model.fit
        # Inactive during inference
        data_augmentation = tf.keras.Sequential([
            tf.keras.layers.experimental.preprocessing.RandomFlip('horizontal'),
            tf.keras.layers.experimental.preprocessing.RandomRotation(0.2),
        ])

        # Rescale pixel values so that they can be accepted by the base model
        # For now use MobileNetV3 (expects pixel values [-1,1])
        # Method comes with model
        preprocess_input = tf.keras.applications.mobilenet_v3.preprocess_input
        # Alternatively could add rescaling layer to do this
        # rescale = tf.keras.layers.experimental.preprocessing.Rescaling(1./127.5, offset=-1)



    def classify(self):
        print('lol not complete yet ho')