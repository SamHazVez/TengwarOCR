import os
import cv2

def extract_test():
  rois = []
  path = 'data/tengwar/sentences'
  file_list = os.listdir(path)
  for i in file_list:
    files = os.path.join(path, i)
    image = cv2.imread(files)
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, threshold = cv2.threshold(image_gray, 0, 255, cv2.THRESH_OTSU)    
    contours, _ = cv2.findContours(threshold.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    sorted_contours = sorted(contours, key=lambda ctr: cv2.boundingRect(ctr)[0])
    
    image_boxed = image.copy()
    for j, ctr in enumerate(sorted_contours):
      x, y, w, h = cv2.boundingRect(ctr)
      area = w*h

      if 700 < area < 8000:
        rect = cv2.rectangle(image_boxed, (x, y), (x + w, y + h), (0, 255, 0), 2)
        roi = image[y:y + h, x:x + w]
        rois.append(roi)
        cv2.imshow('rect', rect)
    cv2.waitKey(0)
  return rois