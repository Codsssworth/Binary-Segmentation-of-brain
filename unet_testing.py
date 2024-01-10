from PIL import Image
import numpy as np
from tensorflow import keras
import os
import tensorflow as tf
import cv2
from keras.preprocessing import image
from keras import backend as K
import matplotlib.pyplot as plt

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


model = keras.models.load_model('segment.h5', custom_objects={'dice_loss': dice_loss, 'dice_coef': dice_coef})
# Load the test image
dir_path = 'unet testing\BRAIN NORMAL IMAGES'
image_files = os.listdir(dir_path)
dir="Fetal MRI classifications/training/Normal"



for  i,test_img in enumerate(image_files) :
    # Load test image
    img_path = os.path.join( dir_path, test_img )
    img = cv2.imread( img_path, cv2.IMREAD_GRAYSCALE )
    img = cv2.resize( img, (224, 224) )
    output_path = os.path.join(dir, f"normal_{i}.png")
    # Output path with custom file name
    cv2.imwrite(output_path, img)
    # img = np.array( img ) / 255.0
    #
    # # Predict the segmentation mask
    # pred_mask = model.predict( img[np.newaxis, ...] )[0]
    #
    # # Convert the mask to binary (0 or 1) values
    # binary_mask = (pred_mask > 0.0).astype(np.uint8) * 255
    # binary_mask=np.squeeze(binary_mask)
    # # Save the binary mask
    # # Convert the grayscale mask to a 3-channel mask with the same dimensions as the original image
    # # color_mask = cv2.cvtColor(binary_mask,  cv2.IMREAD_GRAYSCALE )
    # # color_mask = cv2.resize( color_mask, (224, 224) )
    # # color_mask = np.array( color_mask ) / 255.0
    #
    # # Multiply the mask with the original image
    #
    # masked_image = cv2.multiply(img*255, binary_mask / 255.0)
    #
    # # Save the region of interest
    # roi = cv2.cvtColor(masked_image.astype(np.uint8), cv2.COLOR_BGR2RGB)
    # # plt.imshow( roi, cmap='gray' )
    # # plt.show()
    # # cv2.imwrite(f"roi_{i}.jpg", roi)
    # roi = roi * 255
    # roi_img = Image.fromarray( roi )
    # roi_img.save( f"abxyz_{i}.png" )