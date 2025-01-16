import os
import cv2
import numpy as np

def extract_contour(input_file, output_dir, size):
  image = cv2.imread('import/' + input_file, cv2.IMREAD_GRAYSCALE)
  assert image is not None, "file could not be read"
  
  _, binary_image = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY_INV)
  contours = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  
  cv2.drawContours(image, contours, -1, (0, 255, 0), 2)
  cv2.imshow("Contours", image)
  cv2.waitKey(0)
  cv2.destroyAllWindows()
  
  
def extract_grid(input_file, output_dir, size):
  img = cv2.imread('import/' + input_file)