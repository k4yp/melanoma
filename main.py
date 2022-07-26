import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tensorflow.python.keras.preprocessing.image import load_img, img_to_array

from keras import models, regularizers, layers, optimizers, losses, metrics
from keras.models import Sequential
from keras.layers import Dense
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import ResNet50
from keras.preprocessing import image
from keras.models import load_model

import os


PATH = "../input/dermmel/DermMel/"

PATH_TRAIN = PATH + '/train_sep/'
PATH_VALID = PATH + '/valid/'
PATH_TEST = PATH + '/test/'


#checks the dataset for image counts
def dataset_check(path, type):
    labels = os.listdir(path)
    total = 0
    for label in labels:
        print(label,len(os.listdir(path + label)))
        total += len(os.listdir(path + label))

    print(type,"photos",total)
    print ("\n")

dataset_check(PATH_TRAIN,"training")
dataset_check(PATH_VALID,"validation")
dataset_check(PATH_TEST,"testing")

#displays sample image
def dataset_display(path, sample, type):
    img_path = path + '/' + type + '/'
    img_name = os.listdir(img_path)[sample]
    img_path_full = img_path + img_name
    img = load_img(img_path_full, target_size=(252, 252))
    imgplot = plt.imshow(img)
    plt.title(type)
    plt.show()
    return img_path_full

sample_num = 77

print(dataset_display(PATH_TRAIN, sample_num, 'Melanoma'),'sample:',sample_num,'\n')
print(dataset_display(PATH_TRAIN, sample_num, 'NotMelanoma'),'sample:',sample_num,'\n')
print(dataset_display(PATH_TEST, sample_num, 'Melanoma'),'sample:',sample_num,'\n')
print(dataset_display(PATH_TEST, sample_num, 'NotMelanoma'),'sample:',sample_num,'\n')

conv_base = ResNet50(weights='imagenet',
                     include_top=False,
                     input_shape=(224, 224, 3))

model = models.Sequential()
model.add(conv_base)
model.add(layers.Flatten())
model.add(layers.Dropout(0.5))
model.add(layers.Dense(64, activation='relu',kernel_regularizer=regularizers.l2(0.001)))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(2, activation='sigmoid'))

print(model.summary())

for layer in conv_base.layers[:]:
    layer.trainable = False

model.compile(optimizer=optimizers.Adam(),
              loss='binary_crossentropy',
              metrics=['accuracy'])

print(model.summary())

batch_size = 20
target_size = (224, 224)

train_datagen = ImageDataGenerator(rescale=1./255,
                                   rotation_range=40,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True,
                                   vertical_flip=True,
                                   fill_mode='nearest')

test_datagen = ImageDataGenerator(rescale=1./255)

train_gen = train_datagen.flow_from_directory(PATH_TRAIN,
                                              target_size=target_size,
                                              batch_size=batch_size)

valid_gen = test_datagen.flow_from_directory(PATH_VALID,
                                             target_size=target_size,
                                             batch_size=batch_size)

test_gen = test_datagen.flow_from_directory(PATH_TEST,
                                            target_size=target_size,
                                            batch_size=batch_size)


history = model.fit_generator(train_gen,
                              epochs=2,
                              steps_per_epoch = 4096 // batch_size,
                              validation_data = valid_gen,
                              validation_steps = 2048 // batch_size)

for layer in conv_base.layers[:165]:
    layer.trainable = False
for layer in conv_base.layers[165:]:
    layer.trainable = True

model.compile(optimizer=optimizers.Adam(lr=1e-5),
              loss='binary_crossentropy',
              metrics=['accuracy'])

history = model.fit_generator(train_gen,
                              epochs=32,
                              steps_per_epoch = 4096 // batch_size,
                              validation_data = valid_gen,
                              validation_steps = 2048 // batch_size)


acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'r', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()

def display_results(img_num, check_type):
    def load_image(img_path_full, show = False):
        img = image.load_img(img_path_full, target_size = (224, 224))
        img_tensor = image.img_to_array(img)
        img_tensor = np.expand_dims(img_tensor, axis = 0)
        img_tensor /= 255
        
        return img_tensor
    
    PROOF_PATH = dataset_display(PATH_TEST, img_num, check_type)
    pred = model.predict(load_image(PROOF_PATH))
    pred = round(float(pred[0][0]),4)*100
    actual = 100 if check_type == 'Melanoma' else 0
    diff = abs(round(pred-actual,4))
    
    y = ['Predicted','Actual','Accuracy']
    x = [pred+1,actual+1,(100-diff)]
    
    f = plt.figure()
    f.set_figwidth(3.4)
    f.set_figheight(1)
    plt.title('chance of melanoma')
    plt.barh(y,x,color=['white', 'lightgrey','b' if 100-diff > 76 else 'r'],edgecolor='black')
    plt.xlim([0,100])
    plt.show()
    
    print (f'Predicted chance of melanoma: {pred}%')
    print (f"Actual: {actual}%")
    print (f'Difference: {diff}%')
    print('\n')

for i in range(4,8):
    display_results(i,'NotMelanoma')
for i in range(4,8):
    display_results(i,'Melanoma')

test_loss, test_acc = model.evaluate_generator(test_gen, steps = 2048 // batch_size, verbose=1)
print('test accuracy:', test_acc)
print('test loss:', test_loss)

model.save('melanoma_resnet50.h5')