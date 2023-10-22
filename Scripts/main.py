import keras
import os
import tensorflow as tf
from keras.layers import Conv2D,Dropout,MaxPooling2D,Dense,Flatten
from keras import Sequential

train_path = os.path.join("Data","asl_alphabet_train/asl_alphabet_train")


#define tuning model


#Define hyperparameters
EPOCHS = 100
IMG_SIZE = (200,200)
LR = 0.01
BATCH_SIZE = 32


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


model = Sequential()
# input layer
# Block 1
model.add(Conv2D(32,3,activation='relu',padding='same',input_shape = IMG_SIZE + (3,)))
model.add(Conv2D(32,3,activation='relu',padding='same'))
#model.add(BatchNormalization())
model.add(MaxPooling2D(padding='same'))
model.add(Dropout(0.2))

# Block 2
model.add(Conv2D(64,3,activation='relu',padding='same'))
model.add(Conv2D(64,3,activation='relu',padding='same'))
#model.add(BatchNormalization())
model.add(MaxPooling2D(padding='same'))
model.add(Dropout(0.3))

#Block 3
model.add(Conv2D(128,3,activation='relu',padding='same'))
model.add(Conv2D(128,3,activation='relu',padding='same'))
#model.add(BatchNormalization())
model.add(MaxPooling2D(padding='same'))
model.add(Dropout(0.4))

# fully connected layer
model.add(Flatten())

model.add(Dense(512,activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(128,activation='relu'))
model.add(Dropout(0.3))

# output layer
model.add(Dense(29, activation='softmax'))



model.summary()



model.compile(optimizer='adam', 
              loss='categorical_crossentropy', 
              metrics=['accuracy'])

history = model.fit(train_data_gen,
                    epochs=EPOCHS,batch_size=BATCH_SIZE)

model.save("asl_model.h5")
