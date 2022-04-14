# -*- coding: utf-8 -*-
"""
Created on Thu Apr 14 10:49:59 2022

@author: USER
"""

import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf

file_path = r"D:\Internship\Mida\Dataset"
SEED = 12345
IMAGE_SIZE = (160, 160)
BATCH_SIZE = 4

#Seperate to train and validation data
train_dataset = tf.keras.utils.image_dataset_from_directory(file_path, validation_split = 0.2, subset = 'training', seed = SEED, shuffle = True, image_size = IMAGE_SIZE, batch_size = BATCH_SIZE)
val_dataset = tf.keras.utils.image_dataset_from_directory(file_path, validation_split = 0.2, subset = 'validation', seed = SEED, shuffle = True, image_size = IMAGE_SIZE, batch_size = BATCH_SIZE)

#%%
class_names = train_dataset.class_names

##Show 4 images randomly in the train dataset
plt.figure(figsize = (10, 10))
for images, labels in train_dataset.take(1):
  for i in range(4):
    ax = plt.subplot(2, 2, i + 1)
    plt.imshow(images[i].numpy().astype("uint8"))
    plt.title(class_names[labels[i]])
    plt.axis("off")
    
#%%
#Split further from val_dataset to obtain test_dataset
val_batches = tf.data.experimental.cardinality(val_dataset)
test_dataset = val_dataset.take(val_batches // 5)
validation_dataset = val_dataset.skip(val_batches // 5)

#%%
#To create prefetch dataset for better performance
AUTOTUNE = tf.data.AUTOTUNE

train_dataset_pf = train_dataset.prefetch(buffer_size=AUTOTUNE)
validation_dataset_pf = validation_dataset.prefetch(buffer_size=AUTOTUNE)
test_dataset_pf = test_dataset.prefetch(buffer_size=AUTOTUNE)

#%%
#To increase the training size
data_augmentation = tf.keras.Sequential()
data_augmentation.add(tf.keras.layers.RandomFlip('horizontal'))
data_augmentation.add(tf.keras.layers.RandomRotation(0.2))

#%%
#Show example of applied image augmentation
for images, labels in train_dataset_pf.take(1):
    first_image = images[0]
    plt.figure(figsize = (10, 10))
    for i in range(9):
        plt.subplot(3, 3, i+1)
        augmented_image = data_augmentation(tf.expand_dims(first_image, axis = 0))
        plt.imshow(augmented_image[0] / 255.0)
        plt.axis('off')
        
#%%
#Mobilenet process
preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input

IMG_SHAPE = IMAGE_SIZE + (3,)
base_model = tf.keras.applications.MobileNetV2(input_shape = IMG_SHAPE, include_top = False, weights = 'imagenet')

#%%
#Freeze the base model
base_model.trainable = False
base_model.summary()

#%%
#Add classification layer using global average cooling
global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
#Add output layer
prediction_layer = tf.keras.layers.Dense(1)

#%%
#USe functional API to create the entire model
inputs = tf.keras.Input(shape = IMG_SHAPE)
x = data_augmentation(inputs)
x = preprocess_input(x)
x = base_model(x, training = False)
x = global_average_layer(x)
x = tf.keras.layers.Dropout(0.2)(x)
outputs = prediction_layer(x)

model = tf.keras.Model(inputs, outputs)
model.summary()

#%%
#Compile model
adam = tf.keras.optimizers.Adam(learning_rate = 0.0001)
loss = tf.keras.losses.BinaryCrossentropy(from_logits = True)

model.compile(optimizer = adam, loss = loss, metrics = ['accuracy'])
#%%
loss0, accuracy0 = model.evaluate(validation_dataset_pf)

print('--------------------Before Training-------------------')
print(f'Loss = {loss0}')
print(f'Accuracy = {accuracy0}')

#%%
EPOCHS = 10
history = model.fit(train_dataset_pf, validation_data = validation_dataset_pf, epochs = EPOCHS)

#%%
#Further improve after Feature Extraction
#Fine Tuning (train the higher level convolution layer)

base_model.trainable = True
fine_tune_at = 100

for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False
    
rmsprop = tf.keras.optimizers.RMSprop(learning_rate = 0.00001)
model.compile(optimizer = rmsprop, loss = loss, metrics = ['accuracy'])
model.summary()
#%%
#Continue to train for another 10 epochs
fine_tune_epoch = 10
total_epoch = EPOCHS + fine_tune_epoch

history_fine = model.fit(train_dataset_pf, validation_data = validation_dataset_pf, epochs = total_epoch, initial_epoch = history.epoch[-1])

#%% 
#Evaluate with test dataset
test_loss, test_accuracy = model.evaluate(test_dataset_pf)

print('-----------------Test Result-----------------')
print(f'Loss = {test_loss}')
print(f'Accuracy = {test_accuracy}')

#%%
#Deploy model to make prediction
image_batch, label_batch = test_dataset_pf.as_numpy_iterator().next()
predictions = model.predict_on_batch(image_batch).flatten()

#Apply sigmoid to outpot, since output is in a form of logits
predictions = tf.nn.sigmoid(predictions)
predictions = tf.where(predictions < 0.5, 0, 1)

#Compare predictions and labels
print(f'Prediction: {predictions.numpy()}')
print(f'Labels: {label_batch}')

#%%
#Show some prediction results
plt.figure(figsize = (10, 10))

for i in range(4):
    axs = plt.subplot(2, 2, i+1)
    plt.imshow(image_batch[i].astype('uint8'))
    plt.title(class_names[predictions[i]])
    plt.axis("off")
    