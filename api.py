from flask import Flask, request, jsonify
import matplotlib.pyplot as plt
from ocr_training import train_beleriand
from ocr_beleriand import predict_beleriand, predict_test

app = Flask('TengwarOCR')

model = None
le = None

@app.route('/train', methods=['GET'])
def train():
    global model, le
    model, le = train_beleriand()
    return "Training complete and model saved.", 200

@app.route('/test', methods=['GET'])
def test():
    if model is None or le is None:
        return "Model not trained. Train the model using the /train endpoint.", 400
    
    try:
        predicted_labels = predict_test(model, le)
        predicted_label = '-'.join(predicted_labels)
        return jsonify({"prediction": str(predicted_label)}), 200
    except Exception as e:
        return str(e), 500

@app.route('/predict', methods=['POST'])
def predict():
    if model is None or le is None:
        return "Model not trained. Train the model using the /train endpoint.", 400
    
    if 'image' not in request.files:
        return "No image file provided.", 400
    
    try:
        image = plt.imread(request.files['image'])
        predicted_labels = predict_beleriand(image, model, le)
        predicted_label = '-'.join(predicted_labels)
        return jsonify({"prediction": predicted_label}), 200
    except Exception as e:
        return str(e), 500

if __name__ == '__main__':
    train()
    app.run(host='0.0.0.0', port=5555)
