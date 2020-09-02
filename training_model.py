import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # This will hide those Keras messages
current_dir = os.path.dirname(__file__) #<-- absolute dir the script is in

from PIL import Image
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.applications.imagenet_utils import decode_predictions
from keras.models import load_model
import numpy as np
import tensorflow


# model = InceptionV3()  # InceptionV3 Input (None, 299,299,3 / print(model.input_shape)
model = load_model('model.h5')  # Input (None, 299,299,3 / print(model.input_shape)

# Open file
image = Image.open('data/test/hare.jpg')

# Resize it
image = image.resize((299, 299))

# Convert to Array and ensure its dimension matches the Input of the model
image = np.array(image)  # Image -> Array
image = np.expand_dims(image, axis=0)  # (299,299,3) -> (1, 299,299,3)
image = preprocess_input(image)

# Prediction
prediction = decode_predictions(model.predict(image, verbose=1))
prediction = prediction[0]

result = []
for i in range(len(prediction)):
    class_predicted = {"class": prediction[i][1], "accuracy": float(prediction[i][2])}
    result.append(class_predicted)

print(result)

#model.save('inceptionV3_model.h5')

