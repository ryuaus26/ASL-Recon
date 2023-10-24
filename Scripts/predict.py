import os
import tensorflow as tf
from keras.preprocessing import image
from keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt
import random

train_path = os.path.join("Data","asl_alphabet_train/asl_alphabet_train")


class_names = sorted(os.listdir(train_path))

# Create a grid to display the images
rows, cols = 5, 6  # Adjust the number of rows and columns as needed
fig, axes = plt.subplots(rows, cols, figsize=(12, 10))

# for i, class_name in enumerate(class_names):
#     # Find an image from each class folder
#     class_path = os.path.join(train_path, class_name)
#     image_files = os.listdir(class_path)
#     image_file = os.path.join(class_path, image_files[0])

#     # Load and display the image
#     img = image.load_img(image_file, target_size=(200, 200))
#     ax = axes[i // cols, i % cols]
#     ax.imshow(img)
#     ax.set_title(class_name)
#     ax.axis('off')

# plt.tight_layout()
# plt.show()

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

# Directory containing ASL images
image_path = '/Users/ryuaus26/Desktop/Python/ASL-Recon/Data/asl_alphabet_train/asl_alphabet_train'

sentence = []

#Word to translate
word = "Austin"
word = word.upper()
#Accept input of multiple images
for i,filename in enumerate(sorted(os.listdir(image_path))):
    
        #Stop code once reached length of the word
        if i == len(word):
            break
        
        image_file = os.path.join(image_path, filename)
        #Keep track of letters
        letter = word[i]
        letter_file = os.path.join(image_path,letter)
        
        #Choose random image
        random_file = random.choice(os.listdir(letter_file))
        print(word[i])
        if (random_file.startswith(word[i])):
            
            # Load and preprocess the image
            img = image.load_img(os.path.join(image_path + '/' +  letter,random_file), target_size=(128, 128))
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = img_array / 255.0  # Normalize the image data
        
            # Make the prediction
            prediction = model.predict(img_array)
            predicted_letter = translate_prediction(prediction)
        
            # Display the image and prediction
            sentence.append(predicted_letter)
            plt.imshow(img)
            plt.title(f"Predicted letter: {predicted_letter}")
            plt.show()
            i += 1
        

#print out sentence
sentence_str = ''.join(sentence)
print(f"The sentence is {sentence_str}")