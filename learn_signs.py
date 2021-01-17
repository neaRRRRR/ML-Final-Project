import os
from skimage import io, transform

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, activations
from tensorflow.keras import applications, callbacks

from sklearn.model_selection import train_test_split


# all data IO functions
class DataIO(object):

  @staticmethod
  def load_images_labels(path: str) -> np.ndarray:
    """Loads images from specified directory.

    Arguments:
      - path  : path of the images directory"""

    folders = os.listdir(path)
    images = list()
    labels = list()

    # read image from directory
    for folder in folders:
      folder_path = os.path.join(path, folder)
      img_list = os.listdir(folder_path)
      label = int(folder)
      for img_name in img_list:
        image_path = os.path.join(folder_path, img_name)
        image = io.imread(image_path)
        resized = transform.resize(image, (32, 32))

        # add images and labels to the lists
        images.append(resized)
        labels.append(label)

    return np.array(images), np.array(labels)

# loading images and labels
IMAGES, LABELS = DataIO.load_images_labels("gtsrb")

# split into test and train set from all images
x_train, x_test, y_train, y_test = train_test_split(IMAGES, LABELS, test_size = 0.1)

# split x_train and y_train sets into the validation and train sets
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size = 0.2)

# convert label shapes to the categorical
y_train = tf.keras.utils.to_categorical(y_train)
y_test = tf.keras.utils.to_categorical(y_test)
y_val = tf.keras.utils.to_categorical(y_val)


class TrainingCallbacks(object):

  @classmethod
  def early_stopping_callback(cls, patience = 2) -> callbacks.EarlyStopping:
    """Returns tf.keras.callbacks.EarlyStopping object"""
    
    # constructed EarlyStopping callback
    callback = callbacks.EarlyStopping(patience = patience)
    
    return callback
  
  @classmethod
  def model_checkpoint_callback(cls, 
      filepath = 'model.checkpoint.h5') -> callbacks.ModelCheckpoint:
      """Returns tf.keras.callbacks.ModelCheckpoint object"""

      # constructed ModelCheckpoint callback
      callback = callbacks.ModelCheckpoint(filepath = filepath)

      return callback

  @classmethod
  def tensorboard_callback(cls, log_dir = './models/logs') -> callbacks.TensorBoard:
    """Returns tf.keras.callbacks.TensorBoard object"""
    
    # constructed TensorBoard callback
    callback = callbacks.TensorBoard(log_dir= log_dir)
    
    return callback
    
  @classmethod
  def get_default_callbacks(cls) -> list:
    """Returns all callbacks with default values"""

    early_stopping = cls.early_stopping_callback()
    checkpoint     = cls.model_checkpoint_callback()
    tensorboard    = cls.tensorboard_callback()

    return [early_stopping, checkpoint, tensorboard]

class Models(object):

  @staticmethod
  def MobileNetModel(input_shape: tuple, classes: int) -> tf.keras.models.Model:
    """Classical convolutional model with Pretrained MobileNet model
    in it's structure.

    Arguments:
      - input_shape : input shape of the input images
      - classes     : number of classes for classification process

    Returns:
      - Compiled tf.keras.models.Model object
    """

    # input layer of the model
    input_layer  = layers.Input(shape = input_shape)
    output_shape = tuple(input_layer.shape.as_list())[1:]

    # load MobileNet model from tf.keras.applications
    mobilenet = applications.MobileNet(input_shape = output_shape,
                                       include_top = False,
                                       classes = None)(input_layer)
    
    # convolution layers
    conv1 = layers.Conv2D(128, (1, 1), (1, 1))(mobilenet)
    conv2 = layers.Conv2D(64, (1, 1), (1, 1))(conv1)

    # add dense for classification to end of the model
    flatten   = layers.Flatten()(conv2)
    batchnorm = layers.BatchNormalization()(flatten)


    dense1 = layers.Dense(128, activation = activations.relu)(batchnorm)
    dense2 = layers.Dense(64, activation = activations.relu)(dense1)
    output = layers.Dense(classes, activation = activations.softmax)(dense2)

    # construct the model
    model = tf.keras.models.Model(inputs = input_layer, outputs = output,
                                  name = "MobileNetConvModel")

    # convert the MobileNet part of the structure: non-trainable
    model.layers[1].trainable = False
    
    # compile the model
    model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate = 1e-4),
                  loss      = tf.keras.losses.categorical_crossentropy,
                  metrics   = ["accuracy"])
    
    return model

  @staticmethod
  def ConvModel(input_shape: tuple, classes: int) -> tf.keras.models.Model:
    """Classical convolutional model with 4 convolutions and 2 dense layers.

    Arguments:
      - input_shape : input shape of the input images
      - classes     : number of classes for classification process

    Returns:
      - Compiled tf.keras.models.Model object
    """

    # input layer
    input_layer = layers.Input(shape = input_shape)

    # convolution layers
    conv1 = layers.Conv2D(128, (3, 3), (1, 1))(input_layer)
    pool1 = layers.MaxPooling2D(pool_size = (2, 2))(conv1)
    drop1 = layers.Dropout(0.2)(pool1)
    
    conv2 = layers.Conv2D(64, (3, 3), (1, 1))(drop1)
    pool2 = layers.MaxPooling2D(pool_size = (2, 2))(conv2)
    drop2 = layers.Dropout(0.2)(pool2)
    
    conv3 = layers.Conv2D(64, (2, 2), (1, 1))(drop2)
    drop3 = layers.Dropout(0.2)(conv3)
    
    conv4 = layers.Conv2D(32, (1, 1), (1, 1))(drop3)
    drop4 = layers.Dropout(0.2)(conv4)

    # flatten layers to the 1D array and normalize the batch
    flatten   = layers.Flatten()(drop4)
    batchnorm = layers.BatchNormalization()(flatten)

    # dense layers for final classification process
    dense1 = layers.Dense(128, activation = activations.relu)(batchnorm)
    output = layers.Dense(classes, activation = activations.softmax)(dense1)

    # construct the model
    model = tf.keras.models.Model(inputs = input_layer, outputs = output,
                                  name = "ConvModel")
    
    # compile the model
    model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate = 1e-4),
                  loss      = tf.keras.losses.categorical_crossentropy,
                  metrics   = ["accuracy"])
    
    return model


if __name__ == "__main__":
	# Train the models
	CLASSES = 43
	IMG_SIZE = (32, 32, 3)
	training_callbacks = TrainingCallbacks.get_default_callbacks()


	""" ----- MobileNet model training ----- """
	# construct the model
	mobilenet_model = Models.MobileNetModel(input_shape = IMG_SIZE, classes = CLASSES)

	# train the model
	mobilenet_history = mobilenet_model.fit(x = x_train, y = y_train, epochs = 10,
	                                        validation_data = (x_val, y_val),
	                                        callbacks = training_callbacks)

	# MobileNetModel Accuracy per epoch
	plt.title("MobileNetModel's Accuracy") 
	plt.plot(mobilenet_history.epoch, mobilenet_history.history["accuracy"])
	plt.plot(mobilenet_history.epoch, mobilenet_history.history["val_accuracy"])
	plt.xlabel("Epochs")
	plt.legend(["Accuracy", "Value Accuracy"])
	plt.show()

	# MobileNetModel Loss per epoch
	plt.title("MobileNetModel's Loss") 
	plt.plot(mobilenet_history.epoch, mobilenet_history.history["loss"])
	plt.plot(mobilenet_history.epoch, mobilenet_history.history["val_loss"])
	plt.xlabel("Epochs")
	plt.legend(["Loss", "Value Loss"])
	plt.show()


	""" ----- ConvModel model training ----- """
	# construct the model
	conv_model = Models.ConvModel(input_shape = IMG_SIZE, classes = CLASSES)

	# train the model
	conv_model_history = conv_model.fit(x = x_train, y = y_train, epochs = 10,
	                                        validation_data = (x_val, y_val),
	                                        callbacks = training_callbacks)

	# Conv Model Accuracy per epoch
	plt.title("ConvModel model's Accuracy") 
	plt.plot(conv_model_history.epoch, conv_model_history.history["accuracy"])
	plt.plot(conv_model_history.epoch, conv_model_history.history["val_accuracy"])
	plt.xlabel("Epochs")
	plt.legend(["Accuracy", "Value Accuracy"])
	plt.show()

	# Conv Model Loss per epoch
	plt.title("ConvModel model's Loss") 
	plt.plot(conv_model_history.epoch, conv_model_history.history["loss"])
	plt.plot(conv_model_history.epoch, conv_model_history.history["val_loss"])
	plt.xlabel("Epochs")
	plt.legend(["Loss", "Value Loss"])
	plt.show()

  # save the models
  convnet_save_path = os.path.join("trained_models", "convmodel.h5")
  mobilenet_save_path = os.path.join("trained_models", "mobilenet_model.h5")
  tf.keras.models.save_model(mobilenet_model, convnet_save_path)
  tf.keras.models.save_model(conv_model, mobilenet_save_path)
