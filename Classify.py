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
                                                     batch_size=batch_size,
                                                     image_size=img_size,
                                                     validation_split=0.2,
                                                     subset="training",
                                                     seed=420)
        validation_dataset = image_dataset_from_directory(self.input_path,
                                                     shuffle=True,
                                                     batch_size=batch_size,
                                                     image_size=img_size,
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
        # For now use MobileNetV2 (expects pixel values [-1,1])
        # Method comes with model
        preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input
        # Alternatively could add rescaling layer to do this
        # rescale = tf.keras.layers.experimental.preprocessing.Rescaling(1./127.5, offset=-1)

        # Create base model
        # Will follow common practice to depend on last layer before the 'flatten' operation
        # ie the bottleneck layer
        # 1st instantiate a v2 model pre-loaded with weights
        # include_top=False excludes top layers
        IMG_SHAPE = img_size + (3,)
        base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
                                                       include_top=False,
                                                       weights='imagenet')
        # Covert image into a 5x5x1280 block of features
        image_batch, label_batch = next(iter(train_dataset))
        feature_batch = base_model(image_batch)

        # Freeze the base
        base_model.trainable = False

        # Add a classification head
        # 1st average over the 5x5 spatial locations to covert features to a single 1280 vector/img
        global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
        feature_batch_average = global_average_layer(feature_batch)





    def classify(self):
        print('lol not complete yet ho')