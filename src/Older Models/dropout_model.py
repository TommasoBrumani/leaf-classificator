import tensorflow as tf
import numpy as np
import os
import random
from datetime import datetime
import pandas as pd
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.metrics import confusion_matrix
from tensorflow.keras.preprocessing.image import ImageDataGenerator


tfk = tf.keras
tfkl = tf.keras.layers

seed = 1

random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
tf.random.set_seed(seed)
tf.compat.v1.set_random_seed(seed)

dataset_dir = 'training'

# Model hyperparameters
model_name = "Model_MK_XVI_Batch24"
epochs = 500
patience = 50
metrics = ["categorical_accuracy"]
monitor = "val_categorical_accuracy"
regularization = 1e-5

# Data hyperparameters
image_size = (128, 128)
color_mode = "rgb"
input_shape = (128, 128, 3)
height_shift_range = 0.3
width_shift_range = 0.3
rotation_range = 30
zoom_range = 0.3
validation_split = 0.2
batch_size = 24


labels = ['Apple', 'Blueberry', 'Cherry', 'Corn', 'Grape', 'Orange', 'Peach', 'Pepper', 'Potato', 'Raspberry',
          'Soybean', 'Squash', 'Strawberry', 'Tomato']

# Images are divided into folders, one for each class.
# If the images are organized in such a way, we can exploit the
# ImageDataGenerator to read them from disk.

# Create an instance of ImageDataGenerator for training, validation, and test sets
train_data_gen = ImageDataGenerator(rotation_range=rotation_range,
                                    height_shift_range=height_shift_range,
                                    width_shift_range=width_shift_range,
                                    zoom_range=zoom_range,
                                    shear_range=30,
                                    horizontal_flip=True,
                                    vertical_flip=True,
                                    fill_mode='constant',
                                    validation_split=validation_split,
                                    rescale=1 / 255.)
valid_data_gen = ImageDataGenerator(rotation_range=rotation_range,
                                    height_shift_range=height_shift_range,
                                    width_shift_range=width_shift_range,
                                    zoom_range=zoom_range,
                                    shear_range=30,
                                    horizontal_flip=True,
                                    vertical_flip=True,
                                    fill_mode='constant',
                                    validation_split=validation_split,
                                    rescale=1 / 255.)

# Obtain a data generator with the 'ImageDataGenerator.flow_from_directory' method
train_gen = train_data_gen.flow_from_directory(directory=dataset_dir,
                                               target_size=image_size,
                                               color_mode=color_mode,
                                               classes=None, # can be set to labels
                                               class_mode='categorical',
                                               batch_size=batch_size,
                                               shuffle=True,
                                               subset="training",
                                               seed=seed)
valid_gen = valid_data_gen.flow_from_directory(directory=dataset_dir,
                                               target_size=image_size,
                                               color_mode=color_mode,
                                               classes=None, # can be set to labels
                                               class_mode='categorical',
                                               batch_size=batch_size,
                                               shuffle=False,
                                               subset="validation",
                                               seed=seed)

# Utility function to create folders and callbacks for training


def create_folders_and_callbacks(model_name):
    exps_dir = os.path.join('models_saved')
    if not os.path.exists(exps_dir):
        os.makedirs(exps_dir)

    now = datetime.now().strftime('%b%d_%H-%M-%S')

    exp_dir = os.path.join(exps_dir, model_name + '_' + str(now))
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)

    callbacks = []

    # Model checkpoint
    # ----------------
    ckpt_dir = os.path.join(exp_dir, 'ckpts')
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)

    ckpt_callback = tf.keras.callbacks.ModelCheckpoint(filepath=os.path.join(ckpt_dir, 'cp.ckpt'),
                                                       save_weights_only=False,  # True to save only weights
                                                       save_best_only=False)  # True to save only the best epoch
    callbacks.append(ckpt_callback)

    # Visualize Learning on Tensorboard
    # ---------------------------------
    tb_dir = os.path.join(exp_dir, 'tb_logs')
    if not os.path.exists(tb_dir):
        os.makedirs(tb_dir)

    # By default shows losses and metrics for both training and validation
    tb_callback = tf.keras.callbacks.TensorBoard(log_dir=tb_dir,
                                                 profile_batch=0,
                                                 histogram_freq=1)  # if > 0 (epochs) shows weights histograms
    callbacks.append(tb_callback)

    # Early Stopping
    es_callback = tf.keras.callbacks.EarlyStopping(monitor=monitor, mode='max', patience=patience,
                                                   restore_best_weights=True, verbose=1)
    callbacks.append(es_callback)

    return callbacks


def build_model(input_shape):
    # Build the neural network layer by layer
    model = tfk.Sequential()
    model.add(tfkl.Input(shape=input_shape, name='Input'))
    model.add(tfkl.Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu',
                          kernel_initializer=tfk.initializers.GlorotUniform(seed)))
    model.add(tfkl.BatchNormalization())
    model.add(tfkl.MaxPooling2D(pool_size=(2, 2)))
    model.add(tfkl.Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu',
                          kernel_initializer=tfk.initializers.GlorotUniform(seed)))
    model.add(tfkl.BatchNormalization())
    model.add(tfkl.MaxPooling2D(pool_size=(2, 2)))
    model.add(tfkl.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu',
                          kernel_initializer=tfk.initializers.GlorotUniform(seed)))
    model.add(tfkl.BatchNormalization())
    model.add(tfkl.MaxPooling2D(pool_size=(2, 2)))
    model.add(tfkl.Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu',
                          kernel_initializer=tfk.initializers.GlorotUniform(seed)))
    model.add(tfkl.BatchNormalization())
    model.add(tfkl.MaxPooling2D(pool_size=(2, 2)))
    model.add(tfkl.Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu',
                          kernel_initializer=tfk.initializers.GlorotUniform(seed)))
    model.add(tfkl.BatchNormalization())
    model.add(tfkl.MaxPooling2D(pool_size=(2, 2)))
    model.add(tfkl.Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu',
                          kernel_initializer=tfk.initializers.GlorotUniform(seed)))
    model.add(tfkl.BatchNormalization())
    model.add(tfkl.MaxPooling2D(pool_size=(2, 2)))

    model.add(tfkl.Flatten(name='Flatten'))
    model.add(tfkl.Dropout(0.3, seed=seed))
    model.add(tfkl.Dense(units=512, name='Classifier', kernel_initializer=tfk.initializers.GlorotUniform(seed),
                         activation='relu', kernel_regularizer=tf.keras.regularizers.l2(regularization)))
    model.add(tfkl.Dropout(0.3, seed=seed))
    model.add(tfkl.Dense(units=14, activation='softmax', name='Output'))

    # Compile the model
    model.compile(loss=tfk.losses.CategoricalCrossentropy(), optimizer=tfk.optimizers.Adam(), metrics=metrics)

    # Return the model
    return model


# Build model
model = build_model(input_shape)
#model = tfk.models.load_model(model_name)
model.summary()

# Create folders and callbacks and fit
callbacks = create_folders_and_callbacks(model_name=model_name)

# Train the model
history = model.fit(
    x=train_gen,
    epochs=epochs,
    validation_data=valid_gen,
    callbacks=callbacks,
).history

# Save best epoch model
model.save(model_name)

model = tfk.models.load_model(model_name)

# Plot the training

plt.figure(figsize=(15, 5))
plt.plot(history['loss'], label='Training', alpha=.8, color='#ff7f0e')
plt.plot(history['val_loss'], label='Validation', alpha=.8, color='#4D61E2')
plt.legend(loc='upper left')
plt.title('Binary Crossentropy')
plt.grid(alpha=.3)

plt.figure(figsize=(15, 5))
plt.plot(history['categorical_accuracy'], label='Training', alpha=.8, color='#ff7f0e')
plt.plot(history['val_categorical_accuracy'], label='Validation', alpha=.8, color='#4D61E2')
plt.legend(loc='upper left')
plt.title('Accuracy')
plt.grid(alpha=.3)

plt.show()
