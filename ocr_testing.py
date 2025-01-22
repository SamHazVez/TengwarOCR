from ocr_training import beleriand
from ocr_beleriand import predict_by_rois
from extract import extract_test

model, le = beleriand()
predict_by_rois(extract_test(), model, le)