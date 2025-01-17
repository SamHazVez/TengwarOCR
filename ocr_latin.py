import tensorflow as tf 
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from ocr_training import latin

test_images = []
test_labels = []

path = 'data/latin/testing'

dir_list = os.listdir(path)
for i in dir_list:
  dir = os.path.join(path, i)
  file_list = os.listdir(dir)
  for j in file_list:
    files = os.path.join(dir, j)
    img = cv2.imread(files)
    img = cv2.resize(img, (64,64))
    img = np.array(img, dtype=np.float32)
    img = img/255
    test_images.append(img)
    test_labels.append(i)
    
X_test = np.array(test_images)
y_test = np.array(test_labels)

model, le = latin()

preds = model.predict(X_test)
predicted_labels = le.inverse_transform(np.argmax(preds, axis=1))

plt.imshow(X_test[147])
plt.title('Label: ' + predicted_labels[147])
plt.show()

y_test = le.fit_transform(y_test)
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print('Test Accuracy:' + str(test_accuracy))