import tensorflow as tf 
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from ocr_training import beleriand

images = []
labels = []

def predict_test() :
  path = 'data/tengwar/testing'
  file_list = os.listdir(path)
  for i in file_list:
    files = os.path.join(path, i)
    image = cv2.imread(files)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, (64, 64))
    image = np.array(image, dtype=np.float32) / 255.0
    image = np.expand_dims(image, axis=-1)
    images.append(image)
    labels.append(i)
  X = np.array(images)
  y = np.array(labels)
  model, le = beleriand()
  __predict(X, model, le)
 
def predict_by_rois(rois, model, le):
  for roi in rois:
    image = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, (64, 64))
    image = np.array(image, dtype=np.float32) / 255.0
    image = np.expand_dims(image, axis=-1)
    images.append(image)
  X = np.array(images)
  __predict(X, model, le)

def __predict(X, model, le):
  preds = model.predict(X)
  predicted_labels = le.inverse_transform(np.argmax(preds, axis=1))

  for k, test in enumerate(X):
    plt.imshow(X[k])
    plt.title('Predicted: ' + predicted_labels[k])
    plt.show()