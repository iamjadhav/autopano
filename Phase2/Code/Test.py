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


# Dependencies:
# opencv, do (pip install opencv-python)
# skimage, do (apt install python-skimage)

import cv2
import os
import sys
import glob
import random
from skimage import data, exposure, img_as_float
import matplotlib.pyplot as plt
from Network.Network import *
from Misc.MiscUtils import *
from Misc.DataUtils import *
from Wrapper import *
import numpy as np
import time
import argparse
import shutil
import string
import math as m
from tqdm import tqdm
from Misc.TFSpatialTransformer import *
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

# Don't generate pyc codes
sys.dont_write_bytecode = True


'''-------------GPU Verification-------------'''
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)


print("\n-->> TotalGPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))


if (len(tf.config.experimental.list_physical_devices('GPU')) > 0) :
       print("\n<<<<<<<<<<--------------------Preparing To Run on GPU-------------------->>>>>>>>>>")
else:
       print("\n<<<<<<<<<<--------------------NO GPU FOUND !!!!-------------------->>>>>>>>>>")


def GenerateBatch(BasePath, DirNamesTrain, TrainLabels, ImageSize, MiniBatchSize, NumTestSamples):
    """
    Inputs:
    BasePath - Path to COCO folder without "/" at the end
    DirNamesTrain - Variable with Subfolder paths to train files
    NOTE that Train can be replaced by Val/Test for generating batch corresponding to validation (held-out testing in this case)/testing
    TrainLabels - Labels corresponding to Train
    NOTE that TrainLabels can be replaced by Val/TestLabels for generating batch corresponding to validation (held-out testing in this case)/testing
    ImageSize - Size of the Image
    MiniBatchSize is the size of the MiniBatch
    Outputs:
    I1Batch - Batch of images
    LabelBatch - Batch of one-hot encoded labels
    """

    image_IA_batches = []
    corner_1_batches = []
    patch_2_batches = []
    patch_batches = []

    for n in range(NumTestSamples):
        index = n
        p_1_dir = BasePath + os.sep + "Patch_A/" + DirNamesTrain[index, 0]
        p_2_dir = BasePath + os.sep + "Patch_B/" + DirNamesTrain[index, 0]
        image_IA_dir = BasePath + os.sep + "Image_IA/" + DirNamesTrain[index, 0]
        p_1 = cv2.imread(p_1_dir, cv2.IMREAD_GRAYSCALE)
        p_2 = cv2.imread(p_2_dir, cv2.IMREAD_GRAYSCALE)
        image_IA = cv2.imread(image_IA_dir, cv2.IMREAD_GRAYSCALE)

        if(p_1 is None) or (p_2 is None):
            print("\nPatch empty moving on ..")
            continue

        p_1 = np.float32(p_1)
        p_2 = np.float32(p_2)
        image_IA = np.float32(image_IA)
        p_pair = np.dstack((p_1, p_2))
        o_corner = TrainLabels[index, :, :, 0]
        patch_batches.append(p_pair)
        corner_1_batches.append(o_corner)
        patch_2_batches.append(p_2.reshape(128, 128, 1))
        image_IA_batches.append(image_IA.reshape(image_IA.shape[0], image_IA.shape[1], 1))

    patch_batches = np.array(patch_batches)
    corner_1_batches = np.array(corner_1_batches)
    patch_2_batches = np.array(patch_2_batches)
    image_IA_batches = np.array(image_IA_batches)
    p_indices_batch = []

    for i in range(corner_1_batches.shape[0]):

        x_min, y_min = corner_1_batches[i, 0, 0], corner_1_batches[i, 0, 1]
        x_max, y_max = corner_1_batches[i, 3, 0], corner_1_batches[i, 3, 1]

        X_, Y_ = np.mgrid[x_min : x_max, y_min : y_max]
        p_indices_batch.append(np.dstack((Y_, X_)))

    return patch_batches, corner_1_batches, patch_2_batches, image_IA_batches, p_indices_batch


def testUnsupervised(ImgPH, LabelPH, DirNamesTrain, ImageSize, TrainLabels, CornerPH, Patch2PH, patchIndicesPH, SavePath, ModelPath, BasePath, NumTestSamples):
    if(not (os.path.isdir(SavePath))):
        print(SavePath, "\nCreating Results Dir ...")
        os.makedirs(SavePath)

    _, h_4_batch, _ = unsuperHomographyModel(LabelPH, ImgPH, CornerPH, patchIndicesPH)

    Saver = tf.train.Saver()
    with tf.Session() as sess:
        Saver.restore(sess, ModelPath)
        print('Number of parameters in this model are %d ' % np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()]))

        patch_batches, corner_1_batches, _, image_IA_batches, p_indices_batch = GenerateBatch(BasePath, DirNamesTrain, TrainLabels, ImageSize, _, NumTestSamples)
        FeedDict = {LabelPH: patch_batches, CornerPH: corner_1_batches, ImgPH: image_IA_batches, patchIndicesPH: p_indices_batch}

        pred_h = sess.run(h_4_batch, FeedDict)
        np.save(SavePath+"predicted_H.npy", pred_h)

def testSupervised(LabelPH, ModelPath, SavePath):
    """
    Inputs:
    ImgPH is the Input Image placeholder
    ImageSize is the size of the image
    ModelPath - Path to load trained model from
    DataPath - Paths of all images where testing will be run on
    LabelsPathPred - Path to save predictions
    Outputs:
    Predictions written to ./TxtFiles/PredOut.txt
    """
    image_IA_dir= "../Data/Val/"
    image_name = os.listdir(image_IA_dir)
    ran_index = random.randint(0, len(image_name) - 1)
    a = image_name[ran_index]
    ad = cv2.imread(image_IA_dir + a, 0)
    random_Image = cv2.resize(ad, (320, 240), interpolation = cv2.INTER_AREA)
    pA, pB, _, _, coors = patchCreation(random_Image)
    patch_batch = np.dstack((pA, pB))

    h_4_points = superHomographyModel(LabelPH)
    Saver = tf.train.Saver()

    with tf.Session() as sess:
        Saver.restore(sess, ModelPath)
        print('Number of parameters in this model are %d ' % np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()]))

        patch_batch = np.array(patch_batch).reshape(1, 128, 128, 2)
        # patch_batch = tf.reshape(patch_batch, shape=(1, 128, 128, 2))
        FeedDict = {LabelPH: patch_batch}
        Predicted = sess.run(h_4_points, FeedDict)

    pA_new = coors[0] + Predicted.reshape(4, 2)
    h_4_points_new = coors[1] - pA_new

    cv2.polylines(ad, np.int32([coors[0]]), True,(0, 255, 0), 3)
    cv2.polylines(ad, np.int32([coors[1]]), True,(255, 0, 0), 5)
    cv2.polylines(ad, np.int32([pA_new]), True,(0, 0, 255), 5)

    plt.figure()
    plt.imshow(ad)
    plt.show()
    cv2.imwrite(SavePath + "Stacked" + ".png", ad)

def main():
    """
    Inputs:
    None
    Outputs:
    Prints out the confusion matrix with accuracy
    """
    # Parse Command Line arguments
    Parser = argparse.ArgumentParser()
    Parser.add_argument('--ModelPath', dest='ModelPath', default='../Checkpoints/unsupervised/9model.ckpt', help='Path to load latest model from, Default:ModelPath')
    Parser.add_argument('--CheckPointPath', dest='CheckPointPath', default= '../Checkpoints/unsupervised/', help='Path to load latest model from, Default:CheckPointPath')
    Parser.add_argument('--BasePath', dest='BasePath', default='../Data/Validated_', help='Path to load images from, Default:BasePath')
    Parser.add_argument('--SavePath', dest='SavePath', default='./Results/', help='Path of labels file, Default: ./Results/')
    Parser.add_argument('--ModelType', default='Unsup', help='Model type, Supervised or Unsupervised? Choose from Sup and Unsup, Default:Unsup')

    Args = Parser.parse_args()
    ModelPath = Args.ModelPath
    BasePath = Args.BasePath
    CheckPointPath = Args.CheckPointPath
    SavePath = Args.SavePath
    ModelType = Args.ModelType

    # Plot Confusion Matrix
    # LabelsTrue, LabelsPred = ReadLabels(LabelsPath, LabelsPathPred)
    # ConfusionMatrix(LabelsTrue, LabelsPred)

    if ModelType == 'Unsup':

        DirNamesTrain, SaveCheckPoint, ImageSize, _, TrainLabels, NumClasses = SetupAll(BasePath, CheckPointPath)
        NumTestSamples = 100
        TrainLabels = np.load(BasePath + "/points_list.npy")

        CornerPH = tf.placeholder(tf.float32, shape = (NumTestSamples, 4, 2))
        LabelPH = tf.placeholder(tf.float32, shape = (NumTestSamples, 128, 128 ,2))
        Patch2PH = tf.placeholder(tf.float32, shape = (NumTestSamples, 128, 128, 1))
        ImgPH = tf.placeholder(tf.float32, shape = (NumTestSamples, 240, 320, 1))
        patchIndicesPH = tf.placeholder(tf.int32, shape = (NumTestSamples, 128, 128 , 2))

        testUnsupervised(ImgPH, LabelPH, DirNamesTrain, ImageSize, TrainLabels, CornerPH, Patch2PH, patchIndicesPH, SavePath + "unsupervised/", ModelPath, BasePath, NumTestSamples)

        ran_index = np.random.randint(0, NumTestSamples - 1, size = 5)

        for eachIndex in ran_index:
            TrainLabels = np.load(BasePath + "/points_list.npy")
            pred_y = np.load(SavePath + "unsupervised/predicted_H.npy")
            ground_t_h_4  = pd.read_csv(BasePath + "/h_4_list.csv", index_col = False)
            ground_t_h_4 = ground_t_h_4.to_numpy()
            patch_list = pd.read_csv(BasePath + "/images_list.csv")
            patch_list = patch_list.to_numpy()
            pA_corners = TrainLabels[eachIndex, :, :, 0]

            image_IA_dir = BasePath + "/Image_IA/" + patch_list[eachIndex,0]
            image_IA = cv2.imread(image_IA_dir)
            rm = ground_t_h_4[eachIndex].reshape(2, 4).T
            first_corners = np.array(pA_corners)
            second_corners = first_corners + rm
            H_a_b = cv2.getPerspectiveTransform(np.float32(first_corners), np.float32(second_corners))
            image = image_IA.copy()
            h, w, _ = image.shape

            corners = np.array(pA_corners)
            corners = corners.reshape((-1,1,2))

            image_transformed = cv2.warpPerspective(image, H_a_b, (w, h))
            corner_transformed = cv2.perspectiveTransform(np.float32(corners), H_a_b)
            corner_transformed = corner_transformed.astype(int)

            image_IA_corners = visualizationCorners(image_IA, pA_corners, (0, 0, 255))
            image_B_corners = visualizationCorners(image_transformed, corner_transformed, (0, 0, 255))

            pA_corners = pA_corners.reshape((-1, 1, 2))
            pred_corners_B = cv2.perspectiveTransform(np.float32(pA_corners), pred_y[eachIndex])
            pred_corners_B = pred_corners_B.astype(int)

            pred_corners_image_B = visualizationCorners(image_B_corners, pred_corners_B, (255,255,0))
            mean_corner_error = np.mean(np.abs(pred_corners_B -  corner_transformed))
            print("MC Error for image --> ", eachIndex , " --> ", mean_corner_error)

            pred_corners_image_B = cv2.putText(pred_corners_image_B, "Error: " + str(round(mean_corner_error, 3)), (150, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 200, 255), 2, cv2.LINE_AA)
            cv2.imwrite(SavePath + "unsupervised/stacked" + str(eachIndex) + ".png", np.hstack((image_IA_corners, pred_corners_image_B)))

        print("\n---->>>> Results are Stored in the Results/unsupervised directory  ...")

    else:

        DirNamesTrain, SaveCheckPoint, ImageSize, _, TrainLabels, NumClasses = SetupAll(BasePath, CheckPointPath)
        LabelPH = tf.placeholder(tf.float32, shape = (1, 128, 128, 2))
        testSupervised(LabelPH, ModelPath, SavePath + "supervised/")


if __name__ == '__main__':
    main()
