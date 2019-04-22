import numpy as np
import PIL.Image as Image
import tensorflow as tf
import requests, json
from keras.preprocessing.image import img_to_array, image
from io import BytesIO
from keras.models import load_model
from flask import Flask, jsonify, request
from flask_cors import CORS, cross_origin
from flask.json import JSONEncoder

# Custom Encoder
class CustomJSONEncoder(JSONEncoder):
  def default(self, obj):
    if isinstance(obj, np.integer):
      return int(obj)
    elif isinstance(obj, np.floating):
      return float(obj)
    elif isinstance(obj, np.ndarray):
      return obj.tolist()
    else:
      return super(CustomJSONEncoder, self).default(obj)

# LOAD PRE-TRAINED MODEL
model = load_model('Keras-Model-GC')
graph = tf.get_default_graph()

# FLASK REST API
app = Flask(__name__)
app.json_encoder = CustomJSONEncoder #use custom encoder for numphy

CORS(app)
@app.route('/', methods=['GET'])
def home():
  return 'GarbageAPI'
@app.route("/GarbageAPI", methods=['POST'])
def predict():
  global graph
  with graph.as_default():
    response = request.get_json()
    trashType = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]

    # LOAD IMAGE
    imageLoader = requests.get(response['imageURL'])
    img = Image.open(BytesIO(imageLoader.content))
    img = img.resize((300,300))
    img_tensor = image.img_to_array(img)                    
    img_tensor = np.expand_dims(img_tensor, axis=0)         
    img_tensor /= 255.                                      

    # PREDICT IMAGE
    prediction = model.predict(img_tensor)
    result = np.argmax(prediction)
    return jsonify({
      'index': result, 
      'type': trashType[result],
      'prediction': prediction
    }), 200

if __name__ == "__main__":
    app.run(debug=False, host='0.0.0.0', port=5000)