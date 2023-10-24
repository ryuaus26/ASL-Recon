import os
import tensorflow as tf
from keras.preprocessing import image
from keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt

# Load your trained model
model = load_model("model_weights.h5")

# Function to translate model predictions to ASL letters using if statements
def translate_prediction(prediction):
    max_index = tf.argmax(prediction[0])
    if max_index == 0:
        return 'A'
    elif max_index == 1:
        return 'B'
    elif max_index == 2:
        return 'C'
    elif max_index == 3:
        return 'D'
    elif max_index == 4:
        return 'E'
    elif max_index == 5:
        return 'F'
    elif max_index == 6:
        return 'G'
    elif max_index == 7:
        return 'H'
    elif max_index == 8:
        return 'I'
    elif max_index == 9:
        return 'J'
    elif max_index == 10:
        return 'K'
    elif max_index == 11:
        return 'L'
    elif max_index == 12:
        return 'M'
    elif max_index == 13:
        return 'N'
    elif max_index == 14:
        return 'O'
    elif max_index == 15:
        return 'P'
    elif max_index == 16:
        return 'Q'
    elif max_index == 17:
        return 'R'
    elif max_index == 18:
        return 'S'
    elif max_index == 19:
        return 'T'
    elif max_index == 20:
        return 'U'
    elif max_index == 21:
        return 'V'
    elif max_index == 22:
        return 'W'
    elif max_index == 23:
        return 'X'
    elif max_index == 24:
        return 'Y'
    elif max_index == 25:
        return 'Z'

# Specify the image you want to predict
image_path = '/Users/ryuaus26/Desktop/Python/ASL-Recon/Test/E.jpg'  # Replace with the path to your image

# Load and preprocess the image
img = image.load_img(image_path, target_size=(128, 128))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array = img_array / 255.0  # Normalize the image data

# Make the prediction
prediction = model.predict(img_array)
predicted_letter = translate_prediction(prediction)

# Display the image and prediction
plt.imshow(img)
plt.title(f"Predicted letter: {predicted_letter}")
plt.show()
