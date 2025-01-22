from ocr_training import train_beleriand
from ocr_beleriand import predict_test

model, le = train_beleriand()
predict_test()