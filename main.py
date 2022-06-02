import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from keras.utils import load_img
from keras import models, regularizers, layers, optimizers, losses, metrics
from keras.models import Sequential
from keras.layers import Dense
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import ResNet50


PATH = "C:\data\melanoma"

PATH_TRAIN = PATH + '/train/'
PATH_VALID = PATH + '/valid/'
PATH_TEST = PATH + '/test/'

def dataset_count(path, type):
    labels = os.listdir(path)
    total = 0
    for label in labels:
        print(label,len(os.listdir(path + label)))
        total += len(os.listdir(path + label))

    print(type, "training photos", total)
    print ("\n")

dataset_count(PATH_TRAIN, "train")
dataset_count(PATH_VALID, "train")
dataset_count(PATH_TEST, "train")


def dataset_display(path, sample, cat):
    image_dir = path + '/' + cat + '/'
    img_name = os.listdir(image_dir)[sample]
    img_path = image_dir + img_name
    img = load_img(img_path, target_size=(224, 224))
    imgplot = plt.imshow(img)
    print(path, "photo number", sample)
    plt.show()

dataset_display(PATH_TRAIN, 77, 'melanoma')
dataset_display(PATH_VALID, 77, 'melanoma')
dataset_display(PATH_TEST, 77, 'melanoma')


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