import keras
import os
import tensorflow as tf


train_path = os.path.join("Data","asl_alphabet_train/asl_alphabet_train")


#define tuning model


#Define hyperparameters
EPOCHS = 100
IMG_SIZE = (200,200)
LR = 0.01
BATCH_SIZE = 32

VGG16 = keras.applications.VGG16(include_top=False,input_shape=(IMG_SIZE + (3,)),classes=29)

#Define translation corresponding to each ASL letter


train_generator  = keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

train_data_gen = train_generator.flow_from_directory(
    directory=train_path,
    target_size=IMG_SIZE,
    class_mode="categorical",
    batch_size=BATCH_SIZE,
    
)



#Fine Tune the model
preprocess_input = tf.keras.applications.VGG16.preprocess_input



