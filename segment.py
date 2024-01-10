import json
import numpy as np
import cv2
import os
import tensorflow as tf
from keras import backend as K
from sklearn.model_selection import train_test_split
from keras.models import Model
from keras.regularizers import l2
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate ,Conv2DTranspose,BatchNormalization,Dropout
from keras.callbacks import ModelCheckpoint, EarlyStopping,ReduceLROnPlateau
from keras.losses import binary_crossentropy
from keras.optimizers import Adam
from keras.metrics import MeanIoU
from keras.backend import epsilon
from keras.preprocessing.image import ImageDataGenerator
from keras.models import save_model

def dice_loss(y_true, y_pred, epsilon=1e-6):
    intersection = tf.reduce_sum(y_true * y_pred)
    union = tf.reduce_sum(y_true + y_pred)
    dice_loss = 1 - (2 * intersection + epsilon) / (union + epsilon)
    return dice_loss

def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2.0 * intersection + K.epsilon()) / (K.sum(y_true_f) + K.sum(y_pred_f) + K.epsilon())

def unet(input_shape, dropout_rate=0.2, l2_regularization=1e-4):
    # Encoder
    inputs = Input(input_shape)
    conv1 = Conv2D(32, 3, activation='relu', padding='same', kernel_regularizer=l2(l2_regularization))(inputs)
    conv1 = BatchNormalization()(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(64, 3, activation='relu', padding='same', kernel_regularizer=l2(l2_regularization))(pool1)
    conv2 = BatchNormalization()(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(128, 3, activation='relu', padding='same', kernel_regularizer=l2(l2_regularization))(pool2)
    conv3 = BatchNormalization()(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(256, 3, activation='relu', padding='same', kernel_regularizer=l2(l2_regularization))(pool3)
    conv4 = BatchNormalization()(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    # Decoder
    up5 = Conv2DTranspose(128, 2, strides=(2, 2), padding='same')(conv4)
    merge5 = concatenate([conv3, up5], axis=3)
    conv5 = Conv2D(128, 3, activation='relu', padding='same', kernel_regularizer=l2(l2_regularization))(merge5)
    conv5 = BatchNormalization()(conv5)
    conv5 = Dropout(dropout_rate)(conv5)

    up6 = Conv2DTranspose(64, 2, strides=(2, 2), padding='same')(conv5)
    merge6 = concatenate([conv2, up6], axis=3)
    conv6 = Conv2D(64, 3, activation='relu', padding='same', kernel_regularizer=l2(l2_regularization))(merge6)
    conv6 = BatchNormalization()(conv6)
    conv6 = Dropout(dropout_rate)(conv6)

    up7 = Conv2DTranspose(32, 2, strides=(2, 2), padding='same')(conv6)
    merge7 = concatenate([conv1, up7], axis=3)
    conv7 = Conv2D(32, 3, activation='relu', padding='same', kernel_regularizer=l2(l2_regularization))(merge7)
    conv7 = BatchNormalization()(conv7)
    conv7 = Dropout(dropout_rate)(conv7)

    outputs = Conv2D(1, 1, activation='sigmoid')(conv7)

    model = Model(inputs=inputs, outputs=outputs)
    return model


batch_size = 2
# Define directories for images and masks
image_dir = 'Unet Data/images'
mask_dir = 'Unet Data/masks'


# print(os.path.exists(image_dir))
# print(os.listdir(image_dir))
#
# print(os.path.exists(mask_dir))
# print(os.listdir(mask_dir))
#
# train_imggen = ImageDataGenerator(
#     rescale=1./255,
#     validation_split=0.2
#     # Specify the validation split
# )
# train_maskgen = ImageDataGenerator(
#     rescale=1./255,
#     validation_split=0.2
#
#       # Specify the validation split
# )
# # Create a data generator for the training set
# train_generator = train_imggen.flow_from_directory(
#     directory=image_dir,
#     target_size=(224, 224),
#     batch_size=batch_size,
#     class_mode='input',
#     subset='training',
#     color_mode='grayscale',
#     seed=1# Specify the subset as training set
# )
#
# # Create a data generator for the validation set
# val_generator = train_imggen.flow_from_directory(
#     directory= image_dir,
#     target_size=(224, 224),
#     batch_size=batch_size,
#     class_mode='input',
#     subset='validation',
#     color_mode='rgb',
#     seed=1# Specify the subset as validation set
# )
#
# train_maskgenerator = train_maskgen.flow_from_directory(
#     directory=mask_dir,
#     target_size=(224, 224),
#     batch_size=batch_size,
#     class_mode=None,
#     subset='training',
#     color_mode='rgb',
#     seed=1# Specify the subset as training set
# )
#
# # Create a data generator for the validation set
# val_maskgenerator = train_maskgen.flow_from_directory(
#     directory=mask_dir,
#     target_size=(224, 224),
#     batch_size=batch_size,
#     class_mode=None,
#     subset='validation',
#     seed=1 # Specify the subset as validation set
# )
#
# train_dataset = zip(train_generator, train_maskgenerator)
# val_dataset = zip(val_generator, val_maskgenerator)

# Get list of file names in image and mask directories
image_files = os.listdir( image_dir )
mask_files = os.listdir( mask_dir )

# Create arrays to hold images and masks
images = []
masks = []

# Iterate over files and load images and masks
for image_file, mask_file in zip(image_files, mask_files):
    # Load image
    image = cv2.imread(os.path.join(image_dir, image_file), cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (224, 224))
    image = np.expand_dims(image, axis=-1)
    images.append(image)

    # Load mask
    mask = cv2.imread(os.path.join(mask_dir, mask_file), cv2.IMREAD_GRAYSCALE)
    mask = cv2.resize(mask, (224, 224))
    mask = np.expand_dims(mask, axis=-1)
    masks.append(mask)

# Convert data to numpy arrays
images = np.array( images )
masks = np.array( masks )

# Normalize images to [0, 1]
images = images / 255.0
masks = masks/255.0

# Split data into training and validation sets
train_images, val_images, train_masks, val_masks = train_test_split(images, masks, test_size=0.2)


callbacks = [
    EarlyStopping(monitor='val_loss', min_delta=0.001, patience=5, mode='min', restore_best_weights=True),
    ModelCheckpoint(filepath='unet_weights.h5', monitor='val_loss', save_best_only=True, mode='min'),
    ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.001)
]
optimizer = Adam(learning_rate=0.001,beta_1 = 0.9,beta_2 = 0.999,epsilon = 1e-7)
model = unet(input_shape=(224, 224, 1))
model.compile(optimizer=optimizer, loss=dice_loss, metrics=[dice_coef,'accuracy'])



history = model.fit(train_images,train_masks, validation_data=(val_images,val_masks), epochs=75, batch_size=batch_size,callbacks=callbacks)

model.save('segment.h5')
