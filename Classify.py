# Modified code adapted from TensorFlow website
# Copyright 2019 The TensorFlow Authors.
# @title Licensed under the Apache License, Version 2.0 (the "License");

import matplotlib.pyplot as plt
import os
import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory
import glob
import cv2
import numpy as np


class Classify:

    def __init__(self, base_dir, model_path=None, batch=32, img_size=(160, 160),
                 base_learning_rate=0.0001, initial_epochs=100,
                 fine_tune_epochs=70):
        print(tf.__version__)

        # Import model.h5. If None, then use pre-trained MobileNet V2 weights
        self.model_path = model_path
        # Path to cropped faces (from CascadeDetect.py and/or Detect.py) and annotated faces
        self.path = base_dir
        self.train_path = os.path.join(self.path, "annotated")
        self.input_path = os.path.join(self.path, "cropped_faces")

        # Training parameters
        self.batch = batch
        self.img_size = img_size
        self.base_learning_rate = base_learning_rate
        self.initial_epochs = initial_epochs
        self.fine_tune_epochs = fine_tune_epochs

        # Initialize instance variables for tensorflow data objects/layers I'll be passing around methods
        self.train_dataset = None   # preprocess
        self.validation_dataset = None  # preprocess
        self.test_dataset = None    # preprocess
        self.data_augmentation = None   # preprocess
        self.preprocess_input = None    # preprocess
        self.model = None   # base_model and import_model

        # If GPU exists, don't use too much memory
        # https://www.tensorflow.org/guide/gpu#limiting_gpu_memory_growth
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                # Currently, memory growth needs to be the same across GPUs
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
                    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
            except RuntimeError as e:
                # Memory growth must be set before GPUs have been initialized
                print(e)

    # Data preprocessing
    def preprocess(self):
        # We will use a dataset containing annotated images of anime characters
        # Please see the README for more information
        train_dir = self.train_path
        validation_dir = self.train_path

        if os.path.exists(self.train_path) is False:
            print("%s does not exist" % self.train_path)
            quit()

        BATCH_SIZE = self.batch
        IMG_SIZE = self.img_size

        # Import cropped images to tf.dataset objects
        train_dataset = image_dataset_from_directory(train_dir,
                                                     shuffle=True,
                                                     batch_size=BATCH_SIZE,
                                                     image_size=IMG_SIZE,
                                                     validation_split=0.2,
                                                     subset="training",
                                                     seed=420)

        validation_dataset = image_dataset_from_directory(validation_dir,
                                                          shuffle=True,
                                                          batch_size=BATCH_SIZE,
                                                          image_size=IMG_SIZE,
                                                          validation_split=0.2,
                                                          subset="validation",
                                                          seed=420)
        # Saving class info to csv
        class_names = train_dataset.class_names
        with open(os.path.join(self.path, 'class_names.csv'), 'w') as f:
            for n, c in enumerate(class_names):
                f.write("%s,%s\n" % (n, c))

        print('Saving model to: ', os.path.join(self.path, 'transfer_model.h5'))
        # Create test set. To do so, determine how many batches of data are available in the validation set
        # using ```tf.data.experimental.cardinality```, then move 20% of them to a test set.
        val_batches = tf.data.experimental.cardinality(validation_dataset)
        test_dataset = validation_dataset.take(val_batches // 5)
        validation_dataset = validation_dataset.skip(val_batches // 5)

        print('Number of validation batches: %d' % tf.data.experimental.cardinality(validation_dataset))
        print('Number of test batches: %d' % tf.data.experimental.cardinality(test_dataset))

        # Configure the dataset for performance
        # Use buffered prefetching to load images from disk without having I/O become blocking
        AUTOTUNE = tf.data.experimental.AUTOTUNE
        self.train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
        self.validation_dataset = validation_dataset.prefetch(buffer_size=AUTOTUNE)
        self.test_dataset = test_dataset.prefetch(buffer_size=AUTOTUNE)

        # Use data augmentation
        # When you don't have a large image dataset:
        # it's a good practice to artificially introduce sample diversity
        # Apply random, yet realistic, transformations to the training images, such as rotation and horizontal flipping.
        self.data_augmentation = tf.keras.Sequential([
            tf.keras.layers.experimental.preprocessing.RandomFlip('horizontal'),
            tf.keras.layers.experimental.preprocessing.RandomRotation(0.2),
        ])

        # Note: These layers are active only during training, when you call `model.fit`.
        # They are inactive when the model is used in inference mode in `model.evaulate` or `model.fit`.

        # Rescale pixel values
        # In a moment, you will download `tf.keras.applications.MobileNetV2` for use as your base model.
        # MobileNetV2 expects pixel values in `[-1,1]`, but the pixel values in our images are in `[0-255]`.

        self.preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input

    def importModel(self):
        # The 'preprocess' method must be run before before 'importModel'
        # Import a model from an h5 file
        self.model = tf.keras.models.load_model(self.model_path)

    def baseModel(self):
        # The 'preprocess' method must be run before before 'baseModel'
        # Base model: **MobileNet V2** model developed at Google.
        # This is pre-trained on the ImageNet dataset, a large dataset consisting of 1.4M images and 1000 classes.
        # This base of knowledge will help us classify anime characters from our specific dataset.
        # First, you need to pick which layer of MobileNet V2 you will use for feature extraction.
        # Follow the common practice to depend on the very last layer before the flatten operation.
        # First, instantiate a MobileNet V2 model pre-loaded with weights trained on ImageNet.

        # Create the base model from the pre-trained model MobileNet V2
        # By specifying the **include_top=False** argument,
        # we load a network that doesn't include the classification layers at the top,
        # which is ideal for feature extraction.
        IMG_SHAPE = self.img_size + (3,)
        base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
                                                       include_top=False,
                                                       weights='imagenet')

        # This feature extractor converts each `160x160x3` image into a `5x5x1280` block of features.
        image_batch, label_batch = next(iter(self.train_dataset))
        feature_batch = base_model(image_batch)
        # print(feature_batch.shape)
        # print(image_batch)

        # Feature extraction
        # Freeze the convolutional base created from the previous step and to use as a feature extractor.
        # Additionally, you add a classifier on top of it and train the top-level classifier.
        base_model.trainable = False
        # base_model.summary()

        # Add a classification head
        # To generate predictions from the block of features
        # average over the spatial `5x5` spatial locations
        # using a `tf.keras.layers.GlobalAveragePooling2D` layer to convert the features to
        # a single 1280-element vector per image.
        global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
        feature_batch_average = global_average_layer(feature_batch)
        # print(feature_batch_average.shape)

        # Apply a `tf.keras.layers.Dense` layer to convert these features into a single prediction per image.
        # Need to know how many categories there are
        class_names = self.train_dataset.class_names
        prediction_layer = tf.keras.layers.Dense(len(class_names), activation='relu')
        # prediction_batch = prediction_layer(feature_batch_average)
        # print(prediction_batch.shape)

        # Build a model by chaining together the data augmentation, rescaling,
        # base_model and feature extractor layers using keras.
        # As previously mentioned, use training=False as our model contains a BatchNormalization layer.
        inputs = tf.keras.Input(shape=IMG_SHAPE)
        x = self.data_augmentation(inputs)
        x = self.preprocess_input(x)
        x = base_model(x, training=False)
        x = global_average_layer(x)
        x = tf.keras.layers.Dropout(0.2)(x)
        outputs = prediction_layer(x)
        self.model = tf.keras.Model(inputs, outputs)

    def train(self):
        # The methods 'preprocess' and 'importModel' or 'baseModel' must be run before before 'train'
        # Train self.model
        # First just make sure that the base moblenetv2 model is frozen
        # Then compile the model before training it
        # Use sparse categorical cross-entropy loss with `from_logits=True` since the model provides a linear output.
        model = self.model
        base_model = model.get_layer(index=4)   # moblenetv2 should be the 4th index
        base_model.trainable = False
        model.compile(optimizer=tf.keras.optimizers.Adam(lr=self.base_learning_rate),
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                      metrics=['accuracy'])

        # model.summary()
        # Train the model
        loss0, accuracy0 = model.evaluate(self.validation_dataset)
        print("initial loss: {:.2f}".format(loss0))
        print("initial accuracy: {:.2f}".format(accuracy0))
        history = model.fit(self.train_dataset,
                            epochs=self.initial_epochs,
                            validation_data=self.validation_dataset)

        # Initial variables to plot Learning curves
        acc = history.history['accuracy']
        val_acc = history.history['val_accuracy']

        loss = history.history['loss']
        val_loss = history.history['val_loss']

        # Fine tune model
        # 1st unfreeze moblenet2 base layers
        base_model.trainable = True

        # Let's take a look to see how many layers are in the base model
        print("Number of layers in the base model: ", len(base_model.layers))

        # Fine-tune from this layer onwards
        fine_tune_at = 50

        # Freeze all the layers before the `fine_tune_at` layer
        for layer in base_model.layers[:fine_tune_at]:
            layer.trainable = False

        # Recompile the model
        model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                      optimizer=tf.keras.optimizers.RMSprop(lr=self.base_learning_rate/10),
                      metrics=['accuracy'])

        total_epochs = self.initial_epochs + self.fine_tune_epochs
        history_fine = model.fit(self.train_dataset,
                                 epochs=total_epochs,
                                 initial_epoch=history.epoch[-1],
                                 validation_data=self.validation_dataset)

        # Save learning curves
        acc += history_fine.history['accuracy']
        val_acc += history_fine.history['val_accuracy']

        loss += history_fine.history['loss']
        val_loss += history_fine.history['val_loss']

        plt.figure(figsize=(8, 8))
        plt.subplot(2, 1, 1)
        plt.plot(acc, label='Training Accuracy')
        plt.plot(val_acc, label='Validation Accuracy')
        plt.ylim([0, 1])
        plt.plot([self.initial_epochs-1, self.initial_epochs-1],
                 plt.ylim(), label='Start Fine Tuning')
        plt.legend(loc='lower right')
        plt.title('Training and Validation Accuracy')

        plt.subplot(2, 1, 2)
        plt.plot(loss, label='Training Loss')
        plt.plot(val_loss, label='Validation Loss')
        plt.ylim([0, 1.0])
        plt.plot([self.initial_epochs-1, self.initial_epochs-1],
                 plt.ylim(), label='Start Fine Tuning')
        plt.legend(loc='upper right')
        plt.title('Training and Validation Loss')
        plt.xlabel('epoch')
        plt.savefig(os.path.join(self.path, 'learning_curves.png'))
        plt.close()

        # Evaluate the performance of the model on new data using test set
        loss, accuracy = model.evaluate(self.test_dataset)
        print('Test accuracy :', accuracy)

        # Save model
        model.save(os.path.join(self.path, 'transfer_model.h5'))
        print('Saving model to: ', os.path.join(self.path, 'transfer_model.h5'))

    def predict(self):
        # Predict on a list of images and return a dictionary with how many frames for each category
        if os.path.exists(self.input_path) is False:
            print("%s does not exist" % self.input_path)
            quit()
        img_list = []
        img_paths = sorted(glob.glob(os.path.join(self.input_path, '*.png')))

        for i in img_paths:
            im = cv2.imread(i)
            img_list.append(im)
        img_list = np.array(img_list)

        model = self.model
        predictions = model.predict(img_list)
        labels = np.zeros(np.shape(predictions), dtype=np.float32)
        predictions_label = tf.math.argmax(predictions, axis=1)

        # I will categorize things that the classifier isnt sure about as 'Unkown'
        # Look at the cross entropy from the predictied dist vs the max of the dist
        for row in range(0, len(labels)):
            labels[row][predictions_label[row]] = np.float32(1)
        entropy = tf.nn.softmax_cross_entropy_with_logits(logits=predictions, labels=labels)

        # Get class names
        class_names = []
        with open(os.path.join(self.path, 'class_names.csv'), 'r') as f:
            for line in f:
                class_names.append(line.strip().split(',')[1])

        # Only count toward category if the cross entropy is < 0.01
        for i in range(0, len(img_paths)):
            if float(entropy[i]) < 0.01:
                print(img_paths[i], class_names[predictions_label[i]], float(entropy[i]), '\t', predictions[i])
            else:
                print(img_paths[i], 'Unknown', float(entropy[i]), '\t', predictions[i])
