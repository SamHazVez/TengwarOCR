import tensorflow as tf 
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from ocr_training import beleriand

test_images = []
test_labels = []

path = 'data/beleriand/training_data'

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
    test_images.append(image)
    test_labels.append(i)
    
X_test = np.array(test_images)
y_test = np.array(test_labels)

model, le = beleriand()

preds = model.predict(X_test)
predicted_labels = le.inverse_transform(np.argmax(preds, axis=1))

for k, test in enumerate(X_test):
  plt.imshow(X_test[k])
  plt.title('Label: ' + predicted_labels[k])
  plt.show()

y_test = le.fit_transform(y_test)
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print('Test Accuracy:' + str(test_accuracy))