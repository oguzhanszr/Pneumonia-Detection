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

books = [
    {'id': 0,
     'title': 'A Fire Upon the Deep',
     'author': 'Vernor Vinge',
     'first_sentence': 'The coldsleep itself was dreamless.',
     'year_published': '1992'},
    {'id': 1,
     'title': 'The Ones Who Walk Away From Omelas',
     'author': 'Ursula K. Le Guin',
     'first_sentence': 'With a clamor of bells that set the swallows soaring, the Festival of Summer came to the city Omelas, bright-towered by the sea.',
     'published': '1973'},
    {'id': 2,
     'title': 'Dhalgren',
     'author': 'Samuel R. Delany',
     'first_sentence': 'to wound the autumnal city.',
     'published': '1975'}
]

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