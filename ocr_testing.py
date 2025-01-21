from ocr_training import beleriand
from ocr_beleriand import predict_by_rois
from extract import extract_by_path

path = 'data/tengwar/sentences'

model, le = beleriand()
predict_by_rois(extract_by_path(path), model, le)