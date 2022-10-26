from datetime import datetime
import tensorflow as tf
import numpy as np
import os
import random
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import coolvisuals

tfk = tf.keras
tfkl = tf.keras.layers

# set seed
seed = 42

random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
tf.random.set_seed(seed)
tf.compat.v1.set_random_seed(seed)

# load dataset
dataset_dir = 'data_divided'
training_dir = os.path.join(dataset_dir, 'train')
validation_dir = os.path.join(dataset_dir, 'val')
test_dir = os.path.join(dataset_dir, 'test')
model_name = "Model_inspired_VGG19_batch"
image_size = (64, 64)
color_mode = "rgb"
input_shape = (64, 64, 3)
epochs = 200

# Image labels
labels = ['Apple', 'Blueberry', 'Cherry', 'Corn', 'Grape', 'Orange', 'Peach', 'Pepper', 'Potato', 'Raspberry',
          'Soybean', 'Squash', 'Strawberry', 'Tomato']

# Create generator for each split of the dataset
train_data_gen = ImageDataGenerator(rotation_range=30,
                                    horizontal_flip=True,
                                    vertical_flip=True,
                                    fill_mode='constant',
                                    height_shift_range=0.2,
                                    width_shift_range=0.2,
                                    zoom_range=0.3,
                                    cval=0,
                                    rescale=1 / 255.)
valid_data_gen = ImageDataGenerator(rotation_range=30,
                                    horizontal_flip=True,
                                    vertical_flip=True,
                                    fill_mode='constant',
                                    height_shift_range=0.2,
                                    width_shift_range=0.2,
                                    zoom_range=0.3,
                                    cval=0,
                                    rescale=1 / 255.)
test_data_gen = ImageDataGenerator(rescale=1 / 255.)

# Obtain a data generator with the 'ImageDataGenerator.flow_from_directory' method
train_gen = train_data_gen.flow_from_directory(directory=training_dir,
                                               target_size=image_size,
                                               color_mode=color_mode,
                                               classes=None,  # can be set to labels
                                               class_mode='categorical',
                                               batch_size=16,
                                               shuffle=True,
                                               seed=seed)
valid_gen = valid_data_gen.flow_from_directory(directory=validation_dir,
                                               target_size=image_size,
                                               color_mode=color_mode,
                                               classes=None,  # can be set to labels
                                               class_mode='categorical',
                                               batch_size=8,
                                               shuffle=False,
                                               seed=seed)
test_gen = test_data_gen.flow_from_directory(directory=test_dir,
                                             target_size=image_size,
                                             color_mode=color_mode,
                                             classes=None,  # can be set to labels
                                             class_mode='categorical',
                                             batch_size=8,
                                             shuffle=False,
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

    ckpt_callback = tf.keras.callbacks.ModelCheckpoint(filepath=os.path.join(ckpt_dir,
                                                                             'cp.ckpt'),
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
    es_callback = tf.keras.callbacks.EarlyStopping(monitor='val_categorical_accuracy', mode='max', patience=20,
                                                   restore_best_weights=True)
    callbacks.append(es_callback)

    return callbacks

# Build the network model
def build_model(input_shape):
    model = tfk.Sequential()
    # Build the neural network layer by layer
    model.add(tfkl.Input(shape=input_shape, name='input_layer'))

    model.add(tfkl.Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu',
                          kernel_initializer=tfk.initializers.GlorotUniform(seed)))
    model.add(tfkl.Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu',
                          kernel_initializer=tfk.initializers.GlorotUniform(seed)))

    model.add(tfkl.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))

    model.add(tfkl.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu',
                          kernel_initializer=tfk.initializers.GlorotUniform(seed)))
    model.add(tfkl.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu',
                          kernel_initializer=tfk.initializers.GlorotUniform(seed)))

    model.add(tfkl.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))

    model.add(tfkl.Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu',
                          kernel_initializer=tfk.initializers.GlorotUniform(seed)))

    model.add(tfkl.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))

    model.add(tfkl.Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu',
                          kernel_initializer=tfk.initializers.GlorotUniform(seed)))

    model.add(tfkl.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))

    model.add(tfkl.Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu',
                          kernel_initializer=tfk.initializers.GlorotUniform(seed)))

    model.add(tfkl.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))

    model.add(tfkl.Flatten(name='Flatten'))

    model.add(tfkl.Dense(units=512, activation='relu', kernel_initializer=tfk.initializers.GlorotUniform(seed=seed),
                         kernel_regularizer=tf.keras.regularizers.l2(1e-5)))

    model.add(tfkl.Dense(units=64, activation='relu', kernel_initializer=tfk.initializers.GlorotUniform(seed=seed),
                         kernel_regularizer=tf.keras.regularizers.l2(1e-5)))

    model.add(tfkl.Dense(units=14, activation='softmax', name='Output'))

    # Compile the model
    model.compile(loss=tfk.losses.CategoricalCrossentropy(), optimizer=tfk.optimizers.Adam(),
                  metrics=['categorical_accuracy'])

    # Return the model
    return model


"""Construction of the model, summary e training"""

model = build_model(input_shape)
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

model.save(model_name)

# Plot some useful metrics

model = tfk.models.load_model(model_name)

coolvisuals.plot_early_stopping(history)
coolvisuals.get_confusion_matrix(test_gen, model, labels)
coolvisuals.get_prediction_statistics(model, labels, next(test_gen))

model_test_metrics = model.evaluate(test_gen, return_dict=True)
print(model_test_metrics)
