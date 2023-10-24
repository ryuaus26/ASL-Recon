import keras
import os
import tensorflow as tf
from keras.layers import Conv2D,Dropout,MaxPooling2D,Dense,Flatten
from keras import Sequential
import matplotlib.pyplot as plt
from keras.preprocessing import image

train_path = os.path.join("Data","asl_alphabet_train/asl_alphabet_train")


class_names = sorted(os.listdir(train_path))

# Create a grid to display the images
rows, cols = 5, 6  # Adjust the number of rows and columns as needed
fig, axes = plt.subplots(rows, cols, figsize=(12, 10))

for i, class_name in enumerate(class_names):
    # Find an image from each class folder
    class_path = os.path.join(train_path, class_name)
    image_files = os.listdir(class_path)
    image_file = os.path.join(class_path, image_files[0])

    # Load and display the image
    img = image.load_img(image_file, target_size=(200, 200))
    ax = axes[i // cols, i % cols]
    ax.imshow(img)
    ax.set_title(class_name)
    ax.axis('off')

plt.tight_layout()
plt.show()



#Define hyperparameters
EPOCHS = 100
IMG_SIZE = (128,128)
LR = 0.01
BATCH_SIZE = 32


#Define translation corresponding to each ASL letter




train = keras.utils.image_dataset_from_directory(
    train_path,
    batch_size=BATCH_SIZE,
    label_mode='int',
    image_size=IMG_SIZE,
    validation_split=0.2,
    subset="training",
    seed= 1
)

valid = keras.utils.image_dataset_from_directory(
    train_path,
    label_mode='int',
    batch_size=BATCH_SIZE,
    image_size=IMG_SIZE,
    validation_split=0.2,
    subset="validation",
    seed = 1
)

def process(image,label):
    image = tf.cast(image/255. ,tf.float32)
    return image,label

train = train.map(process)

valid = valid.map(process)


model = Sequential()
# input layer
# Block 1
model.add(Conv2D(32,3,activation='relu',padding='same',input_shape = IMG_SIZE + (3,)))
model.add(Conv2D(32,3,activation='relu',padding='valid'))
#model.add(BatchNormalization())
model.add(MaxPooling2D((8,8),padding='valid'))
model.add(Dropout(0.2))

# fully connected layer
model.add(Flatten())

model.add(Dense(128,activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(64,activation='relu'))
model.add(Dropout(0.3))

# output layer
model.add(Dense(29, activation='softmax'))



model.summary()


checkpoint_callback = keras.callbacks.ModelCheckpoint(
    "model_weights.h5",
    monitor="val_loss",
    save_best_only=True,
    save_weights_only=False,
    mode="min",
    verbose=1
)

early_stopping_callback =keras.callbacks.EarlyStopping(
    monitor='val_loss',  # Monitor validation loss
    patience=10,  # Stop if there's no improvement for 10 epochs
    verbose=1,  # Print messages about early stopping
    restore_best_weights=True  # Restore model weights to the best epoch
)

model.compile(optimizer='adam', 
            loss = tf.keras.losses.SparseCategoricalCrossentropy(), 
              metrics=['accuracy'])

history = model.fit(train,
                    epochs=EPOCHS,batch_size=BATCH_SIZE,
                    validation_data=valid,
                    callbacks=[checkpoint_callback,early_stopping_callback])

model.save("asl_model.h5")
