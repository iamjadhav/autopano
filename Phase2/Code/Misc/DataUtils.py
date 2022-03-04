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

import os
import cv2
import numpy as np
import random
import skimage
import PIL
import sys
import pandas as pd
# Don't generate pyc codes
sys.dont_write_bytecode = True

def SetupAll(BasePath, CheckPointPath):
    """
    Inputs:
    BasePath is the base path where Images are saved without "/" at the end
    CheckPointPath - Path to save checkpoints/model
    Outputs:
    DirNamesTrain - Variable with Subfolder paths to train files
    SaveCheckPoint - Save checkpoint every SaveCheckPoint iteration in every epoch, checkpoint saved automatically after every epoch
    ImageSize - Size of the image
    NumTrainSamples - length(Train)
    NumTestRunsPerEpoch - Number of passes of Val data with MiniBatchSize
    Trainabels - Labels corresponding to Train
    NumClasses - Number of classes
    """
    # Setup DirNames
    DirNamesTrain =  SetupDirNames(BasePath + '/images_list.csv')

    # Read and Setup Labels
    LabelsPathTrain = BasePath + '/h_4_list.csv'
    TrainLabels = ReadLabels(LabelsPathTrain)

    if(not (os.path.isdir(CheckPointPath))):
       os.makedirs(CheckPointPath)
    SaveCheckPoint = 100
    NumTestRunsPerEpoch = 5

    ImageSize = [128, 128, 1]

    NumTrainSamples = len(DirNamesTrain)

    NumClasses = 10
    return DirNamesTrain, SaveCheckPoint, ImageSize, NumTrainSamples, TrainLabels, NumClasses

def ReadLabels(LabelsPathTrain):
    if(not (os.path.isfile(LabelsPathTrain))):
        print('ERROR: Train Labels do not exist in '+LabelsPathTrain)
        sys.exit()
    else:
        labels = pd.read_csv(LabelsPathTrain, index_col = False)
    return labels.to_numpy()


def SetupDirNames(BasePath):
    """
    Inputs:
    BasePath is the base path where Images are saved without "/" at the end
    Outputs:
    Writes a file ./TxtFiles/DirNames.txt with full path to all image files without extension
    """
    DirNamesTrain = ReadDirNames(BasePath)
    return DirNamesTrain

def ReadDirNames(ReadPath):
    """
    Inputs:
    ReadPath is the path of the file you want to read
    Outputs:
    DirNames is the data loaded from ./TxtFiles/DirNames.txt which has full path to all image files without extension
    """
    # Read text files
    load_data = pd.read_csv(ReadPath, index_col = False)
    return load_data.to_numpy()
