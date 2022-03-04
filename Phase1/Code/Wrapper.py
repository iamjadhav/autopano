#!/usr/bin/evn python

"""
CMSC733 Spring 2019: Classical and Deep Learning Approaches for
Geometric Computer Vision
Project1: MyAutoPano: Phase 1 Starter Code

Author(s):
Abhishek Nalawade (abhi1793@umd.edu)
Graduate Student in M.Eng Robotics,
University of Maryland, College Park

Aditya Jadhav (amjadhav@umd.edu)
Graduate Student in M.Eng Robotics,
University of Maryland, College Park
"""

# Code starts here:

import numpy as np
import cv2
# Add any python libraries here
import copy
import argparse
from include.common import ANMS
from include.common import FeatureDescriptors
from include.common import RANSAC
from include.common import Blend


def main():
    # Add any Command Line arguments here
    Parser = argparse.ArgumentParser()
    Parser.add_argument('--path', type=str, default='Train/Set1',
                        help='Path to the Images Example Train/CustomSet1 or Train/Set1 or Test/TestSet1')
    Parser.add_argument('--save', type=bool, default=True,
                        help='Save images or not')

    Args = Parser.parse_args()
    path = Args.path
    var = path.split('/')
    var = var[1]
    save_on = Args.save
    """
    Read a set of images for Panorama stitching
    """
    image_set = 2
    num_of_images = 0
    z = 1
    tmp_images = list()
    while True:
        # tmp_images.append(cv2.imread("../Data/Train/Set%s/%s.jpg"%(image_set,z)))
        tp = cv2.imread("../Data/%s/%s.jpg"%(path,z))
        z += 1
        try:
            a = tp.shape
            tmp_images.append(tp)
            num_of_images += 1
        except:
            break


    if num_of_images > 4:
        div_factor = num_of_images - 4
    else:
        div_factor = num_of_images
    # print("one ",one.shape)
    # cv2.imshow("wind", one)
    # cv2.waitKey(0)
    aspect_ratio = tmp_images[0].shape[0]/tmp_images[0].shape[1]
    width_scale = 1280/(div_factor*tmp_images[0].shape[1])
    width = int(tmp_images[0].shape[1]*width_scale)
    height = int(aspect_ratio*width)
    # cv2.imshow("tmp",tmp_images[0])
    # cv2.waitKey(0)
    images = [cv2.resize(i, (width, height)) for i in tmp_images]

    """
    Corner Detection
    Save Corner detection output as corners.png
    """
    def makePanorama(two_images):
        grays = [cv2.cvtColor(i, cv2.COLOR_BGR2GRAY) for i in two_images]
        all_corners = [cv2.goodFeaturesToTrack(i, 10000, 0.001, 6) for i in grays]
        all_corners = [np.int0(i) for i in all_corners]

        all_imgs_corners = copy.deepcopy(two_images)
        for i in range(len(all_imgs_corners)):
            img = all_imgs_corners[i]
            corner = all_corners[i]
            for j in corner:
                x, y = j.ravel()
                cv2.circle(img, (x,y), 3, 255, -1)
        if save_on:
            for i in range(len(all_imgs_corners)):
                cv2.imwrite("results/Corners/corners%s%s.png"%(var,i+1), all_imgs_corners[i])

        """
        Perform ANMS: Adaptive Non-Maximal Suppression
        Save ANMS output as anms.png
        """
        images_anms = copy.deepcopy(two_images)
        suppress = ANMS(1100)

        Nbest = list()
        for i in range(len(images_anms)):
            Nbest.append(suppress.runANMS(all_corners[i]))

        for i in range(len(Nbest)):
            for j in Nbest[i]:
                cv2.circle(images_anms[i], (j[0],j[1]), 3, 255, -1)


        cv2.imshow("wind",images_anms[0])
        cv2.imshow("window",images_anms[1])
        cv2.waitKey(1)
        if save_on:
            for i in range(len(images_anms)):
                cv2.imwrite("results/ANMS/anms%s%s.png"%(var,i+1), images_anms[i])

        """
        Feature Descriptors
        Save Feature Descriptor output as FD.png
        """
        descriptors = list()
        features = FeatureDescriptors(40)
        for i in range(len(grays)):
            descriptors.append(features.getFeatureDescriptors(grays[i], Nbest[i], var, save_on))

        """
        Feature Matching
        Save Feature Matching output as matching.png
        """
        pairs = list()
        for i in range(len(descriptors)-1):
            pairs.append(features.matchFeatures(descriptors[i], descriptors[i+1]))

        for i in range(len(pairs)):
            features.drawMatch(Nbest[i], two_images[i], Nbest[i+1], two_images[i+1], pairs[i], var, save_on)

        """
        Refine: RANSAC, Estimate Homography
        """
        ransac = RANSAC()
        best1, best2, temp_H, inliers = ransac.runRANSAC(Nbest[0], Nbest[1], pairs[0])
        if inliers >= 6:
            ransac.drawMatch(best1, two_images[0], best2, two_images[1], var, save_on)
            # print("homography from RANSAC ",temp_H)
            H = ransac.EstimateHomography(best1, best2)
            # print("final Homography ",H)
            """
            Image Warping + Blending
            Save Panorama output as mypano.png
            """
            blender = Blend(1)
            pano = blender.warpAndStitch(two_images[0], best1, two_images[1], best2, H)
        else:
            print("\n\n\n\n\n\n\n\n This image has less common features \n\n\n\n\n\n\n\n\n")
            pano = two_images[0]
        return pano

     # logic for starting from center and stitching either side of images
    if num_of_images%2 == 0:
        base_idx = int(len(images)/2)-1
    else:
        base_idx = int((len(images)/2))
    tmp = [images[base_idx], images[base_idx-1]]

    count = 1
    for i in range(1,len(images)):
        pano = makePanorama(tmp)
        if save_on:
            cv2.imwrite("results/Panoramas/mypano%s.png"%(var), pano)
        # break
        if i%2 == 0:
            tmp = [pano, images[base_idx - count]]
        else:
            tmp = [pano, images[base_idx + count]]
            count += 1
    pano = cv2.resize(pano, (800,800))
    cv2.imshow("ultimate panorama",pano)
    if save_on:
        cv2.imwrite("results/Panoramas/mypano%s.png"%(var), pano)
    cv2.waitKey(0)

    # # logic for stitching images in pairs from middle
    #
    # tmp_images = copy.deepcopy(images)
    # # for j in range(base_idx-1):
    # counter = 0
    # while counter < num_of_images-1:
    #     count = 0
    #     right_side = list()
    #     left_side = list()
    #     print("\n\n\n\n\n\n\n\n\n%s"%(len(tmp_images)))
    #     if len(tmp_images)%2 == 0:
    #         base_idx = int(len(tmp_images)/2) - 1
    #     else:
    #         base_idx = int(len(tmp_images)/2)
    #
    #     # tmp_left = [tmp_images[base_idx], tmp_images[base_idx-1]]
    #     if base_idx == 1:
    #         itr = base_idx
    #     else:
    #         itr = base_idx - 1
    #     for j in range(itr):
    #         tmp_left = [tmp_images[base_idx-count], tmp_images[base_idx-count-1]]
    #         left_side.insert(0, makePanorama(tmp_left))
    #
    #         tmp_right = [tmp_images[base_idx+count+1], tmp_images[base_idx+count+2]]
    #         right_side.append(makePanorama(tmp_right))
    #         count += 2
    #         print("j and count are: ",j, " ", count)
    #     tmp_images = copy.deepcopy(left_side) + copy.deepcopy(right_side)
    #     print("newwwwwwwwwwwwwwwwww: ",len(tmp_images))
    #     if len(tmp_images) == 2:
    #         print("condition satisfied")
    #         pano = makePanorama(tmp_images)
    #         pano = cv2.resize(pano, (800,800))
    #         cv2.imshow("ultimate panorama",pano)
    #         cv2.waitKey(0)
    #         break
    #     counter += 1
    #     # print("\n\n\n\n\n\n\n\n%s\n\n\n\n\n\n\n"%counter)

if __name__ == '__main__':
    main()
