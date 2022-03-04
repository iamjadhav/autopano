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
# termcolor, do (pip install termcolor)

import cv2
import sys
import os
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
from termcolor import colored, cprint
import math as m
from tqdm import tqdm
from Misc.TFSpatialTransformer import *

# Don't generate pyc codes
sys.dont_write_bytecode = True


'''-------------GPU Verification-------------'''
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)


print("\n-->> TotalGPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))


if (len(tf.config.experimental.list_physical_devices('GPU')) > 0) :
       print("\n<<<<<<<<<<--------------------Preparing To Run on GPU-------------------->>>>>>>>>>")
else:
       print("\n<<<<<<<<<<--------------------NO GPU FOUND !!!!-------------------->>>>>>>>>>")


def GenerateBatch(BasePath, DirNamesTrain, TrainLabels, ImageSize, MiniBatchSize):
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

    if(len(DirNamesTrain) < MiniBatchSize):
        print("\n-->> Not enough images in the data ... ")
        return 0

    image_IA_batches = []
    corner_1_batches = []
    patch_2_batches = []
    patch_batches = []

    for n in range(MiniBatchSize):

        i = random.randint(0, len(DirNamesTrain) - 1)

        p_1_dir = BasePath + os.sep + "Patch_A/" + DirNamesTrain[i, 0]
        p_2_dir = BasePath + os.sep + "Patch_B/" + DirNamesTrain[i, 0]
        image_IA_dir = BasePath + os.sep + "Image_IA/" + DirNamesTrain[i, 0]
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
        corner_1 = TrainLabels[i, :, :, 0]

        patch_batches.append(p_pair)
        corner_1_batches.append(corner_1)
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



def PrettyPrint(NumEpochs, DivTrain, MiniBatchSize, NumTrainSamples, LatestFile):
    """
    Prints all stats with all arguments
    """
    print('Number of Epochs Training will run for ' + str(NumEpochs))
    print('Factor of reduction in training data is ' + str(DivTrain))
    print('Mini Batch Size ' + str(MiniBatchSize))
    print('Number of Training Images ' + str(NumTrainSamples))
    if LatestFile is not None:
        print('Loading latest checkpoint with the name ' + LatestFile)



def TrainOperation(ImgPH, LabelPH, DirNamesTrain, TrainLabels, CornerPH, H_4PH, Patch2PH, patchIndicesPH, NumTrainSamples, ImageSize,
                   NumEpochs, MiniBatchSize, SaveCheckPoint, CheckPointPath, LatestFile, BasePath, LogsPath, ModelType):

    if ModelType.lower() == "sup":
        print("---------->>>>>>>>>> Training Supervised Model <<<<<<<<<<----------")
        h_4_pt = superHomographyModel(LabelPH)

        with tf.name_scope("Loss"):
            loss = tf.sqrt(tf.reduce_sum((tf.squared_difference(h_4_pt, H_4PH))))

        with tf.name_scope("Adam"):
            Optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=1e-4).minimize(loss)
    else:
        print("---------->>>>>>>>>> Training Un-Supervised Model <<<<<<<<<<----------")
        warped_IA, _, _ = unsuperHomographyModel(LabelPH, ImgPH, CornerPH, patchIndicesPH)

        with tf.name_scope("Loss"):
            loss = tf.reduce_mean(tf.abs(warped_IA - Patch2PH))

        with tf.name_scope("Adam"):
            Optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=1e-5).minimize(loss)

    # Tensorboard
    # Create a summary to monitor loss tensor
    EpochLossPH = tf.compat.v1.placeholder(tf.float32, shape=None)
    loss_summary = tf.compat.v1.summary.scalar('LossEveryIter', loss)
    epoch_loss_summary = tf.compat.v1.summary.scalar('LossPerEpoch', EpochLossPH)
    # tf.summary.image('Anything you want', AnyImg)

    # Merge all summaries into a single operation
    MergedSummaryOP1 = tf.compat.v1.summary.merge([loss_summary])
    MergedSummaryOP2 = tf.compat.v1.summary.merge([epoch_loss_summary])
    # MergedSummaryOP = tf.summary.merge_all()

    # Setup Saver
    Saver = tf.compat.v1.train.Saver()
    AccOverEpochs = np.array([0, 0])
    with tf.compat.v1.Session() as sess:
        if LatestFile is not None:
            Saver.restore(sess, CheckPointPath + LatestFile + '.ckpt')
            StartEpoch = int(''.join(c for c in LatestFile.split('a')[0] if c.isdigit()))
            print('Loaded latest checkpoint with the name ' + LatestFile + '....')
        else:
            sess.run(tf.compat.v1.global_variables_initializer())
            StartEpoch = 0
            print('New model initialized....')

        # Tensorboard
        Writer = tf.compat.v1.summary.FileWriter(LogsPath, graph=tf.compat.v1.get_default_graph())
        e_loss = []
        for Epochs in tqdm(range(StartEpoch, NumEpochs)):
            NumIterationsPerEpoch = int(NumTrainSamples/MiniBatchSize)
            Loss = []
            epoch_loss = 0
            for PerEpochCounter in tqdm(range(NumIterationsPerEpoch)):
                if ModelType.lower() == "sup":
                    image_IA_dir= "../Data/Trained_/Image_IA/"
                    image_name = os.listdir(image_IA_dir)
                    ran_index = random.randint(0, len(image_name) - 1)
                    a = image_name[ran_index]
                    ad = cv2.imread(image_IA_dir + a)
                    random_Image = cv2.resize(ad, (320,240), interpolation = cv2.INTER_AREA)
                    _, _, h_4_labels, _, _ = patchCreation(random_Image)
                    h_4_labels = h_4_labels.reshape(1, 8)
                    patch_batches, _, _, _, _ = GenerateBatch(BasePath, DirNamesTrain, TrainLabels, ImageSize, MiniBatchSize)
                    FeedDict = {LabelPH: patch_batches, H_4PH: h_4_labels}
                else:
                    patch_batches, corner_1_batches, patch_2_batches, image_IA_batches, p_indices_batch = GenerateBatch(BasePath, DirNamesTrain, TrainLabels, ImageSize, MiniBatchSize)
                    FeedDict = {LabelPH: patch_batches, CornerPH: corner_1_batches, Patch2PH: patch_2_batches, ImgPH: image_IA_batches, patchIndicesPH: p_indices_batch}

                _, LossThisBatch, Summary = sess.run([Optimizer, loss, MergedSummaryOP1], feed_dict=FeedDict)
                Loss.append(LossThisBatch)
                epoch_loss = epoch_loss + LossThisBatch

          # Tensorboard
            Writer.add_summary(Summary, Epochs*NumIterationsPerEpoch + PerEpochCounter)
            epoch_loss = epoch_loss/NumIterationsPerEpoch

            print("Printing Epoch:  ",  np.mean(Loss), "\n")
            e_loss.append(np.mean(Loss))

          # Save model every epoch
            SaveName = CheckPointPath + str(Epochs) + 'model.ckpt'
            Saver.save(sess, save_path=SaveName)
            print('\n' + SaveName + ' Model Saved...')
            Summary_epoch = sess.run(MergedSummaryOP2,feed_dict={EpochLossPH: epoch_loss})
            Writer.add_summary(Summary_epoch,Epochs)
            Writer.flush()

    np.savetxt(LogsPath + "Unsupervised_Loss.txt", np.array(e_loss), delimiter = ",")


def main():
    """
    Inputs:
    None
    Outputs:
    Runs the Training and testing code based on the Flag
    """
    # Parse Command Line arguments
    Parser = argparse.ArgumentParser()
    Parser.add_argument('--BasePath', default="../Data/Trained", help='Base path of images, Default: ../Data/Trained')
    Parser.add_argument('--CheckPointPath', default='../Checkpoints/', help='Path to save Checkpoints, Default: ../Checkpoints/')
    Parser.add_argument('--ModelType', default='Unsup', help='Model type, Supervised or Unsupervised? Choose from Sup and Unsup, Default:Unsup')

    Parser.add_argument('--NumEpochs', type=int, default=50, help='Number of Epochs to Train for, Default:50')
    Parser.add_argument('--DivTrain', type=int, default=1, help='Factor to reduce Train data by per epoch, Default:1')
    Parser.add_argument('--MiniBatchSize', type=int, default=1, help='Size of the MiniBatch to use, Default:1')
    Parser.add_argument('--LoadCheckPoint', type=int, default=0, help='Load Model from latest Checkpoint from CheckPointsPath?, Default:0')
    Parser.add_argument('--LogsPath', default='../Logs/', help='Path to save Logs for Tensorboard, Default=Logs/')


    Args = Parser.parse_args()
    NumEpochs = Args.NumEpochs
    BasePath = Args.BasePath
    DivTrain = float(Args.DivTrain)
    MiniBatchSize = Args.MiniBatchSize
    LoadCheckPoint = Args.LoadCheckPoint
    CheckPointPath = Args.CheckPointPath
    LogsPath = Args.LogsPath
    ModelType = Args.ModelType

    if(not (os.path.isdir(LogsPath)) and not (os.path.isdir(CheckPointPath))):
        print("Creating Log and Checkpoint folder...")
        os.makedirs(LogsPath)
        os.makedirs(CheckPointPath)

    # Setup all needed parameters including file reading
    DirNamesTrain, SaveCheckPoint, ImageSize, NumTrainSamples, TrainLabels, NumClasses = SetupAll(BasePath, CheckPointPath)

    TrainLabels = np.load(BasePath + "/points_list.npy")

    H_4PH = tf.compat.v1.placeholder(tf.float32, shape = (1, 8))
    CornerPH = tf.compat.v1.placeholder(tf.float32, shape = (MiniBatchSize, 4, 2))
    LabelPH = tf.compat.v1.placeholder(tf.float32, shape = (MiniBatchSize, 128, 128 ,2))
    Patch2PH = tf.compat.v1.placeholder(tf.float32, shape = (MiniBatchSize, 128, 128, 1))
    ImgPH = tf.compat.v1.placeholder(tf.float32, shape = (MiniBatchSize, 240, 320, 1))
    patchIndicesPH = tf.compat.v1.placeholder(tf.int32, shape = (MiniBatchSize, 128, 128 , 2))
    LatestFile = None

    # Pretty print stats
    # PrettyPrint(NumEpochs, DivTrain, MiniBatchSize, NumTrainSamples, LatestFile)


    TrainOperation(ImgPH, LabelPH, DirNamesTrain, TrainLabels, CornerPH, H_4PH, Patch2PH, patchIndicesPH, NumTrainSamples, ImageSize,
                   NumEpochs, MiniBatchSize, SaveCheckPoint, CheckPointPath, LatestFile, BasePath, LogsPath, ModelType)


if __name__ == '__main__':
    main()
