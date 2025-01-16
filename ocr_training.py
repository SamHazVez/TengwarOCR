import tensorflow as tf 
from keras.models import Sequential
from keras.layers import Conv2D, Flatten, MaxPooling2D, Dense
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle

def build_data(type):
  images = []
  labels = []
  path = 'data/' + type + '/training_data'
  dir_list = os.listdir(path)
  for i in dir_list:
    dir = os.path.join(path, i)
    file_list = os.listdir(dir)
    for j in file_list:
      files = os.path.join(dir, j)
      image = cv2.imread(files)
      image = cv2.resize(image, (64,64))
      image = np.array(image, dtype=np.float32)
      image = image/255
      images.append(image)
      labels.append(i)

  X = np.array(images)
  y = np.array(labels)

  le = LabelEncoder()
  y = le.fit_transform(y)
  X_sh, y_sh = shuffle(X, y, random_state=42)
  return X_sh, y_sh, le

def build_ocr_model():
    model = Sequential()
    model.add(Conv2D(filters=16, kernel_size=(3,3), activation='relu', input_shape=(64,64,3)))
    model.add(MaxPooling2D())
    model.add(Conv2D(filters=32, kernel_size=(3,3),  activation='relu'))
    model.add(MaxPooling2D())
    model.add(Conv2D(filters=64, kernel_size=(3,3),  activation='relu'))
    model.add(MaxPooling2D())
    model.add(Conv2D(filters=128, kernel_size=(3,3), activation='relu'))
    model.add(Flatten())
    model.add(Dense(units=128, activation='relu'))
    model.add(Dense(units=64, activation='relu'))
    model.add(Dense(units=36, activation='softmax'))
    return model

def train_model(X_sh, y_sh, le):
  model = build_ocr_model()
  model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics = ['accuracy'])
  history = model.fit(X_sh, y_sh ,validation_split=0.2, batch_size=4, epochs=50)
  return model, le

def latin():
  X_sh, y_sh, le = build_data('latin')
  return train_model(X_sh, y_sh, le)

def beleriand():
  X_sh, y_sh, le = build_data('beleriand')
  return train_model(X_sh, y_sh, le)


# def build_ocr_model(input_shape, num_classes):
#     inputs = layers.Input(shape=input_shape)
#     x = layers.Conv2D(32, (3, 3), activation="relu", padding="same")(inputs)
#     x = layers.MaxPooling2D(pool_size=(2, 2))(x)
#     x = layers.Conv2D(64, (3, 3), activation="relu", padding="same")(x)
#     x = layers.MaxPooling2D(pool_size=(2, 2))(x)
#     x = layers.Reshape((-1, x.shape[-1]))(x)  # Flatten spatial dimensions
#     x = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(x)
#     outputs = layers.Dense(num_classes + 1, activation="softmax")(x)  # +1 for CTC blank token

#     return tf.keras.Model(inputs, outputs)

# Example: Build model for 26 characters + 10 digits + CTC blank token
# input_shape = (128, 32, 1)  # Height, Width, Channels
# num_classes = 36
# model = build_ocr_model(input_shape, num_classes)
# model.summary()

# def ctc_loss(y_true, y_pred):
#     y_pred = tf.nn.log_softmax(y_pred)
#     input_length = tf.fill([tf.shape(y_pred)[0]], tf.shape(y_pred)[1])
#     label_length = tf.fill([tf.shape(y_true)[0]], tf.shape(y_true)[1])
#     return tf.reduce_mean(tf.keras.backend.ctc_batch_cost(y_true, y_pred, input_length, label_length))

# model.compile(optimizer="adam", loss=ctc_loss, metrics=['accuracy'])

# # Example training loop
# model.fit(train_dataset, validation_data=val_dataset, epochs=10)
