import tensorflow as tf 
from keras.models import Sequential
from keras.layers import Conv2D, Flatten, MaxPooling2D, Dense
import os
import cv2
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle

def __build_data(data_type):
  images = []
  labels = []
  path = f'data/{data_type}/training'
  dir_list = os.listdir(path)
  for i in dir_list:
    dir = os.path.join(path, i)
    file_list = os.listdir(dir)
    for j in file_list:
      files = os.path.join(dir, j)
      image = cv2.imread(files)
      image = __preprocess_image(image)
      images.append(image)
      labels.append(i)

  X = np.array(images)
  y = np.array(labels)

  le = LabelEncoder()
  y = le.fit_transform(y)
  X_sh, y_sh = shuffle(X, y, random_state=42)
  return X_sh, y_sh, le

def __preprocess_image(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, (64, 64))
    image = np.array(image, dtype=np.float32) / 255.0
    image = np.expand_dims(image, axis=-1)
    image = tf.image.random_brightness(image, max_delta=0.1)
    image = tf.image.random_contrast(image, lower=0.9, upper=1.1)
    image = image.numpy()
    return image

def __build_ocr_model():
    model = Sequential([
        Conv2D(filters=16, kernel_size=(3, 3), activation='relu', input_shape=(64, 64, 1)),
        MaxPooling2D(),
        Conv2D(filters=32, kernel_size=(3, 3), activation='relu'),
        MaxPooling2D(),
        Conv2D(filters=64, kernel_size=(3, 3), activation='relu'),
        MaxPooling2D(),
        Conv2D(filters=128, kernel_size=(3, 3), activation='relu'),
        Flatten(),
        Dense(units=128, activation='relu'),
        Dense(units=64, activation='relu'),
        Dense(units=38, activation='softmax')
    ])
    return model

def __train_model(X_sh, y_sh, le):
  model = __build_ocr_model()
  model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics = ['accuracy'])
  model.fit(X_sh, y_sh ,validation_split=0.2, batch_size=4, epochs=50)
  return model, le

def train_beleriand():
  X_sh, y_sh, le = __build_data('tengwar')
  return __train_model(X_sh, y_sh, le)