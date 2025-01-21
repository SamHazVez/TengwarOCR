from ocr_training import beleriand
from ocr_beleriand import predict_by_rois
from extract import extract_char

path = 'data/tengwar/testing'

model, le = beleriand()
predict_by_rois(extract_char(), model, le)