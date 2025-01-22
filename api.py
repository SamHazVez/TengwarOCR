from flask import Flask, request
import matplotlib.pyplot as plt
from ocr_training import latin, beleriand

app = Flask('TengwarOCR')

model = None
le = None
    
@app.route('/train', methods = ['GET'])
def train_beleriand():
    model, le = beleriand()
    return "OK"
    
@app.route('/find/beleriand', methods=['POST'])
def find_beleriand():
    file = request.files['image']
    plt.imshow(file)
    plt.show()

if __name__ == '__main__': 
	app.run(debug=True, host='0.0.0.0', port = 5555)