import sys
import os
import flask
from flask import request, jsonify
sys.path.append('../')
from model.model_functions import predict

# app = Flask(__name__,
#             static_url_path='', 
#             static_folder='web/static',
#             template_folder='web/templates')
app = flask.Flask(__name__)
app.config["DEBUG"] = True

@app.route('/', methods=['GET'])
def home():
    return flask.render_template('index.html')

@app.route('/api/pneumonia/predict', methods=['POST'])
def api_predict():
    response, confidence = predict(request.data)

    # print(str(request.data).split(',')[1])
    if response == -1:
        result = {'image': -1, 'confidence': 0}
    else:
        result = {'image': str(response), 'confidence': confidence} 
        
    return jsonify(result)

app.run()
