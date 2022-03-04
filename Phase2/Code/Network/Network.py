"""
CMSC733 Spring 2019: Classical and Deep Learning Approaches for
Geometric Computer Vision
Project 1: Autopano

Author(s):
Abhishek Nalawade (abhi1793@umd.edu)
Graduate Student in M.Eng Robotics,
University of Maryland, College Park

Aditya Jadhav (amjadhav@umd.edu)
Graduate Student in M.Eng Robotics,
University of Maryland, College Park
"""

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import sys
import numpy as np
from Misc.MiscUtils import *
from Misc.TFSpatialTransformer import transformer

# Don't generate pyc codes
sys.dont_write_bytecode = True


def superHomographyModel(image):
    """
    Inputs:
    Img is a MiniBatch of the current image
    ImageSize - Size of the Image
    Outputs:
    prLogits - logits output of the network
    prSoftMax - softmax output of the network
    """

    #############################
    # Fill your network here!
    #############################

    #Convolutional Layers
    x = tf.keras.layers.Conv2D(name = 'conv_1', padding = 'same',filters = 64, kernel_size = (3,3), activation = None)(image)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.nn.relu(x, name = 'relu_1')

    x = tf.keras.layers.Conv2D(name = 'conv_2', padding = 'same',filters = 64, kernel_size = (3,3), activation = None)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.nn.relu(x, name = 'relu_2')

    x = tf.keras.layers.MaxPooling2D(pool_size = (2,2), strides = 2)(x)

    x = tf.keras.layers.Conv2D(name = 'conv_3', padding = 'same',filters = 64, kernel_size = (3,3), activation = None)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.nn.relu(x, name = 'relu_3')

    x = tf.keras.layers.Conv2D(name = 'conv_4', padding = 'same',filters = 64, kernel_size = (3,3), activation = None)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.nn.relu(x, name = 'relu_4')

    x = tf.keras.layers.MaxPooling2D(pool_size = (2,2), strides = 2)(x)

    x = tf.keras.layers.Conv2D(name = 'conv_5', padding = 'same',filters = 128, kernel_size = (3,3), activation = None)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.nn.relu(x, name = 'relu_5')

    x = tf.keras.layers.Conv2D(name = 'conv_6', padding = 'same',filters = 128, kernel_size = (3,3), activation = None)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.nn.relu(x, name = 'relu_6')

    x = tf.keras.layers.MaxPooling2D(pool_size = (2,2), strides = 2)(x)

    x = tf.keras.layers.Conv2D(name = 'conv_7', padding = 'same',filters = 128, kernel_size = (3,3), activation = None)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.nn.relu(x, name = 'relu_7')

    x = tf.keras.layers.Conv2D(name = 'conv_8', padding = 'same',filters = 128, kernel_size = (3,3), activation = None)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.nn.relu(x, name = 'relu_8')

    x = tf.keras.layers.Flatten()(x)

    #Fully-connected layers
    x = tf.keras.layers.Dense(units = 1024, activation = tf.nn.relu)(x)
    x = tf.keras.layers.Dropout(rate = 0.5)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    h_4 = tf.keras.layers.Dense(units = 8, activation = None)(x)

    return h_4


''' Reference For TensorDLT:
    https://github.com/tynguyen/unsupervisedDeepHomographyRAL2018 '''

def tensorDLT(h_4_pts, c_4_A , min_batch):

    c_4_tile = tf.expand_dims(c_4_A, [2])

    # Solve for H using DLT
    predict_h_4_tile = tf.expand_dims(h_4_pts, [2])

    predict_c_4_tile = tf.add(predict_h_4_tile, c_4_tile)

    M1 = tf.tile(tf.expand_dims(tf.constant(Aux_M1, tf.float32) ,[0]), [min_batch, 1, 1])

    M2 = tf.tile(tf.expand_dims(tf.constant(Aux_M2,tf.float32), [0]) ,[min_batch, 1, 1])

    M3 = tf.tile(tf.expand_dims(tf.constant(Aux_M3, tf.float32), [0]), [min_batch, 1, 1])

    M4 = tf.tile(tf.expand_dims(tf.constant(Aux_M4, tf.float32), [0]), [min_batch, 1, 1])

    M5 = tf.tile(tf.expand_dims(tf.constant(Aux_M5, tf.float32), [0]), [min_batch, 1, 1])

    M6 = tf.tile(tf.expand_dims(tf.constant(Aux_M6, tf.float32), [0]), [min_batch, 1, 1])

    M71 = tf.tile(tf.expand_dims(tf.constant(Aux_M71, tf.float32), [0]), [min_batch, 1, 1])

    M72 = tf.tile(tf.expand_dims(tf.constant(Aux_M72, tf.float32), [0]), [min_batch, 1, 1])

    M8 = tf.tile(tf.expand_dims(tf.constant(Aux_M8, tf.float32), [0]), [min_batch, 1, 1])

    Mb = tf.tile(tf.expand_dims(tf.constant(Aux_Mb, tf.float32), [0]), [min_batch, 1, 1])

    A1 = tf.matmul(M1, c_4_tile)
    A2 = tf.matmul(M2, c_4_tile)

    A3 = M3
    A4 = tf.matmul(M4, c_4_tile)
    A5 = tf.matmul(M5, c_4_tile)

    A6 = M6
    A7 = tf.matmul(M71, predict_c_4_tile) * tf.matmul(M72, c_4_tile)
    A8 = tf.matmul(M71, predict_c_4_tile) * tf.matmul(M8, c_4_tile)

    A = tf.stack([tf.reshape(A1,[-1,8]),tf.reshape(A2,[-1,8]), tf.reshape(A3,[-1,8]),tf.reshape(A4,[-1,8]),tf.reshape(A5,[-1,8]),tf.reshape(A6,[-1,8]), tf.reshape(A7,[-1,8]),tf.reshape(A8,[-1,8])],axis=1)
    A = tf.transpose(A, perm=[0,2,1])

    B = tf.matmul(Mb, predict_c_4_tile)

    h_8 = tf.matrix_solve(A , B)

    h_ones = tf.ones([min_batch, 1, 1])
    h_9 = tf.concat([h_8, h_ones], 1)

    h_flat = tf.reshape(h_9, [-1, 9])
    h_matrix = tf.reshape(h_flat,[-1, 3, 3])

    return h_matrix


def unsuperHomographyModel(b_patches, a_image, c_4_A, i_patches):

    min_batch , h, w, c = a_image.get_shape().as_list()
    # h = image_a[1]
    # w = image_a[0]
    h_4_pts = superHomographyModel(b_patches)
    corners_A = tf.reshape(c_4_A, [min_batch, 8])

    h_mat_batches = tensorDLT(h_4_pts, corners_A, min_batch)
    M = np.array([[w/2.0, 0., w/2.0],
                  [0., h/2.0, h/2.0],
                  [0., 0., 1.]]).astype(np.float32)

    tensor_M = tf.constant(M, tf.float32)
    tile_M = tf.tile(tf.expand_dims(tensor_M, [0]), [min_batch, 1, 1])
    inv_M = np.linalg.inv(M)

    tensor_inv_M = tf.constant(inv_M, tf.float32)
    M_inv_tile_batches = tf.tile(tf.expand_dims(tensor_inv_M, [0]), [min_batch, 1, 1])

    h_matrix = tf.matmul(tf.matmul(M_inv_tile_batches, h_mat_batches), tile_M)

    warped_image_IA, _ = transformer(a_image, h_matrix, (h, w))
    warped_image_IA = tf.reshape(warped_image_IA, [min_batch, h, w])
    warped_patch_pa = tf.gather_nd(warped_image_IA, i_patches, name = None, batch_dims = 1)
    warped_patch_pa = tf.transpose(warped_patch_pa, perm = [0,2,1])
    warped_patch_pa = tf.reshape(warped_patch_pa, [min_batch, 128, 128, 1])

    return warped_patch_pa, h_mat_batches, h_4_pts
