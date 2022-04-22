# -*- coding: utf-8 -*-
"""
Created on Fri Apr 15 11:52:32 2022

@author: ANEH
"""

import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow_examples.models.pix2pix import pix2pix
import numpy as np
import matplotlib.pyplot as plt
import cv2
import glob
import os
from IPython.display import clear_output
from scipy import io

#1. Load the data and prepare list for image and mask
images=[]
masks=[]
file_path = r"C:\Users\ANEH\Documents\Deep Learning Class\TensorFlow Deep Learning\Datasets\data-science-bowl-2018\data-science-bowl-2018-2\train"

#%%
#1.1 Load images
image_dir = os.path.join(file_path, 'inputs')
for image_file in os.listdir(image_dir):
    img = cv2.imread(os.path.join(image_dir, image_file))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img,(128,128))
    images.append(img)
    
    
#1.2 Load masks
mask_dir = os.path.join(file_path, 'masks_png')
for mask_file in os.listdir(mask_dir):
    mask = cv2.imread(os.path.join(mask_dir, mask_file), cv2.IMREAD_GRAYSCALE)
    mask = cv2.resize(mask,(128,128))
    masks.append(mask)
    
#%%
#1.3 Convert images and masks into numpy array
images_np = np.array(images)
masks_np = np.array(masks)

#%%
#1.4 check examples
plt.figure(figsize=(10,4))
for i in range(1,4):
    plt.subplot(1,3,i)
    img_plot = images[i]
    plt.imshow(img_plot)
    plt.axis('off')
plt.show()   

plt.figure(figsize=(10,4))
for i in range(1,4):
    plt.subplot(1,3,i)
    mask_plot = masks[i]
    plt.imshow(mask_plot, cmap='gray')
    plt.axis('off')
plt.show()  

#%%
#2. Data preprocessing
#2.1. Expand the mask dimension
masks_np_exp = np.expand_dims(masks_np,axis=-1)
#Check the mask output
print(masks[0].min(),masks[0].max())

#%%
#2.2. Change the mask value (1. normalize the value, 2. encode into numerical encoding)
converted_masks = np.round(masks_np_exp/255)
converted_masks = 1 - converted_masks

#%%
#2.3. Normalize the images
converted_images = images_np / 255.0

#%%
#2.4. Do train-test split
from sklearn.model_selection import train_test_split

SEED=12345
x_train,x_test,y_train,y_test = train_test_split(converted_images,converted_masks,test_size=0.2,random_state=SEED)

#%%
#2.5. Convert the numpy array data into tensor slice
train_x = tf.data.Dataset.from_tensor_slices(x_train)
test_x = tf.data.Dataset.from_tensor_slices(x_test)
train_y = tf.data.Dataset.from_tensor_slices(y_train)
test_y = tf.data.Dataset.from_tensor_slices(y_test)

#%%
#2.6. Zip tensor slice into dataset
train = tf.data.Dataset.zip((train_x,train_y))
test = tf.data.Dataset.zip((test_x,test_y))

#%%
#2.7. Convert into prefetch dataset
BATCH_SIZE = 16
AUTOTUNE = tf.data.AUTOTUNE
BUFFER_SIZE = 1000
STEPS_PER_EPOCH = 800//BATCH_SIZE
VALIDATION_STEPS = 200//BATCH_SIZE
train = train.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()
train = train.prefetch(buffer_size=AUTOTUNE)
test = test.batch(BATCH_SIZE).prefetch(buffer_size=AUTOTUNE)
