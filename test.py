import numpy as np
import PIL.Image as Image
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array, array_to_img, image
from keras.layers import Conv2D, Flatten, MaxPooling2D, Dense
from keras.models import Sequential
import tensorflow as tf

import glob, os, random

from keras.models import load_model
from flask import Flask, jsonify, request
from flask_cors import CORS, cross_origin
import pickle
import pandas as pd
import requests, json

base_path = './Garbage classification'

img_list = glob.glob(os.path.join(base_path, '*/*.jpg'))

print(len(img_list))
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.1,
    zoom_range=0.1,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    vertical_flip=True,
    validation_split=0.1
)

test_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.1
)

train_generator = train_datagen.flow_from_directory(
    base_path,
    target_size=(300, 300),
    batch_size=16,
    class_mode='categorical',
    subset='training',
    seed=0
)

validation_generator = test_datagen.flow_from_directory(
    base_path,
    target_size=(300, 300),
    batch_size=16,
    class_mode='categorical',
    subset='validation',
    seed=0
)

labels = (train_generator.class_indices)
labels = dict((v,k) for k,v in labels.items())

print(labels)

for i, img_path in enumerate(random.sample(img_list, 6)):
    img = load_img(img_path)
    img = img_to_array(img, dtype=np.uint8)
    plt.subplot(2, 3, i+1)
    plt.imshow(img.squeeze())

model = Sequential([
    Conv2D(filters=32, kernel_size=3, padding='same', activation='relu', input_shape=(300, 300, 3)),
    MaxPooling2D(pool_size=2),

    Conv2D(filters=64, kernel_size=3, padding='same', activation='relu'),
    MaxPooling2D(pool_size=2),
    
    Conv2D(filters=32, kernel_size=3, padding='same', activation='relu'),
    MaxPooling2D(pool_size=2),
    
    Conv2D(filters=32, kernel_size=3, padding='same', activation='relu'),
    MaxPooling2D(pool_size=2),

    Flatten(),

    Dense(64, activation='relu'),

    Dense(6, activation='softmax')
])

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

model.summary()

model = load_model('Keras-Model-GC')

test_x, test_y = validation_generator.__getitem__(1)

preds = model.predict(test_x)

plt.figure(figsize=(16, 16))
for i in range(16):
    plt.subplot(4, 4, i+1)
    plt.title('pred:%s / truth:%s' % (labels[np.argmax(preds[i])], labels[np.argmax(test_y[i])]))
    # plt.imshow(test_x[i])

app = Flask(__name__)
CORS(app)
@app.route('/', methods=['GET'])
def home():
    return 'Hallo'
@app.route("/GarbageAPI", methods=['POST'])
def predict():
    response = request.get_json()
    print(response['imageURL'])
    grace_hopper = tf.keras.utils.get_file(response['name'], response['imageURL'])
    grace_hopper = Image.open(grace_hopper)
    print(grace_hopper)
    # grace_hopper_shape = tf.constant(grace_hopper)
    # test_image = tf.expand_dims(grace_hopper_shape, 0)
    # im_arr = np.fromstring(grace_hopper.tobytes(), dtype=np.uint8)
    # im_arr = im_arr.reshape((grace_hopper.size[1], grace_hopper.size[0], 3))    
    grace_hopper.save(response['name'], "JPEG", quality=80, optimize=True, progressive=True)

    img = image.load_img(response['name'], target_size=(300, 300))
    img_tensor = image.img_to_array(img)                    # (height, width, channels)
    img_tensor = np.expand_dims(img_tensor, axis=0)         # (1, height, width, channels), add a dimension because the model expects this shape: (batch_size, height, width, channels)
    img_tensor /= 255.                                      # imshow expects values in the range [0, 1]

    prediction = model.predict(img_tensor)
    result = np.argmax(prediction)
    
    return jsonify({'index': result}), 201

if __name__ == "__main__":
    app.run(debug=False, host='0.0.0.0', port=5000)