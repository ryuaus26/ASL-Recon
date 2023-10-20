import keras
import os



train_path = os.path.join("Data","asl_alphabet_test")
test_path = os.path.join("Data","asl_alphabet_test")

#define tuning model


#Define hyperparameters
EPOCHS = 100
IMG_SIZE = (200,200)
LR = 0.01

print((200,200) + (3,))

VGG16 = keras.applications.VGG16(include_top=False,input_shape=(IMG_SIZE + (3,)),classes=29)

#Define translation corresponding to each ASL letter


train_generator  = keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255
)

train_data_gen = train_generator.flow_from_directory(
    directory=train_path,
    target_size=IMG_SIZE
)


