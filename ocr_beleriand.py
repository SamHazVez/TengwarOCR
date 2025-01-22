import tensorflow as tf 
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from ocr_training import train_beleriand

def predict_test() :
  images = []
  labels = []
  path = 'data/tengwar/testing'
  model, le = train_beleriand()
  dir_list = os.listdir(path)
  for i in dir_list:
    dir = os.path.join(path, i)
    file_list = os.listdir(dir)
    for j in file_list:
      files = os.path.join(dir, j)
      image = cv2.imread(files)
      image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
      image = cv2.resize(image, (64, 64))
      image = np.array(image, dtype=np.float32) / 255.0
      image = np.expand_dims(image, axis=-1)
      images.append(image)
      labels.append(i)
  X = np.array(images)
  y = np.array(labels)
  __predict(X, model, le)
 
def __predict_by_rois(rois, model, le):
  images = []
  for roi in rois:
    image = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, (64, 64))
    image = np.array(image, dtype=np.float32) / 255.0
    image = np.expand_dims(image, axis=-1)
    images.append(image)
  X = np.array(images)
  return __predict(X, model, le)

def __predict(X, model, le):
  preds = model.predict(X)
  predicted_labels = le.inverse_transform(np.argmax(preds, axis=1))

  for k, test in enumerate(X):
    plt.imshow(X[k])
    plt.title('Predicted: ' + predicted_labels[k])
    plt.show()
  return predicted_labels

def __extract(image):
  rois = []
  image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  image_uint8 = image_gray.astype(np.uint8)
  _, threshold = cv2.threshold(image_uint8, 0, 255, cv2.THRESH_OTSU)    
  contours, _ = cv2.findContours(threshold.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
  sorted_contours = sorted(contours, key=lambda ctr: cv2.boundingRect(ctr)[0])
  
  image_boxed = image.copy()
  for i, ctr in enumerate(sorted_contours):
    x, y, w, h = cv2.boundingRect(ctr)
    area = w*h

    if 700 < area < 8000:
      rect = cv2.rectangle(image_boxed, (x, y), (x + w, y + h), (0, 255, 0), 2)
      roi = image[y:y + h, x:x + w]
      rois.append(roi)
      cv2.imshow('rect', rect)
  cv2.waitKey(0)
  return rois

def predict_beleriand(image, model, le):
  rois = __extract(image)
  return __predict_by_rois(rois, model, le) 