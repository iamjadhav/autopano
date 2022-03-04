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

import numpy as np
import os 
import cv2
import csv
import tensorflow as tf
import pandas as pd
# Add any python libraries here


def patchCreation(input_image):
    
    height, width = input_image.shape[:2]
    margin = 42
    radius = 32
    max_patch = 128
   
    image_min = max_patch + 2 * margin +  1 
    
    if ((width > image_min) and (height > image_min)):
        right_bottom = max_patch + margin 
    
        x_coor = np.random.randint(margin, width - right_bottom) 
        y_coor = np.random.randint(margin, height - right_bottom)
        pa_coors = np.array([[x_coor, y_coor], [x_coor, max_patch + y_coor] , [max_patch + x_coor, y_coor], \
                         [max_patch + x_coor, max_patch + y_coor]])
        pb_coors = np.zeros_like(pa_coors)
        
        for i, points in enumerate(pa_coors):
            pb_coors[i][0] = points[0] + np.random.randint(- radius, radius)
            pb_coors[i][1] = points[1] + np.random.randint(- radius, radius)

        inv_homography = np.linalg.inv(cv2.getPerspectiveTransform(np.float32(pa_coors), np.float32(pb_coors))) 

        b_image = cv2.warpPerspective(input_image, inv_homography, (width,height))

        patch_PA = input_image[y_coor : y_coor + max_patch, x_coor : x_coor + max_patch]
        patch_PB = b_image[y_coor : y_coor + max_patch, x_coor : x_coor + max_patch]
        H_4_points = (pb_coors - pa_coors) 

        return patch_PA, patch_PB, H_4_points.astype(np.float32), np.dstack((pa_coors, pb_coors)), [pa_coors, pb_coors]
    else:
        return patch_PA == None, patch_PB == None, H_4_points == None, None



def main():
        
    iterations = ['one', 'two', 'three'] # Repeat 3 times
    h_4_points_list = []
    images_list = [] 
    points_list = []
    data = ['train','validate']

    for dataType in data:

        if dataType == 'train':
            print("Training data --> ")
            get_dir = '../Data/Train/'
            save_dir = '../Data/Trained_/'
            total = os.listdir(get_dir)
            no_images = len(total) + 1
            
        elif dataType == 'validate':
            print("\nValidation data --> ")
            get_dir = '../Data/Val/'
            save_dir = '../Data/Validated_/'
            total = os.listdir(get_dir)
            no_images = len(total) + 1
            
        if(not (os.path.isdir(save_dir))):
            print(save_dir, "\nCreating Directory to save Trained images ...")
            os.makedirs(save_dir)

        for eachIter in iterations:
            print(" Iteration :  ", eachIter)
            for image in range(1, no_images):

                a_image = cv2.imread(get_dir + str(image) + '.jpg')
                a_image = cv2.resize(a_image, (320,240), interpolation = cv2.INTER_AREA)

                pA, pB, h_4_pts, points, _ = patchCreation(a_image) 
                
                if ((pA is None)&(pB is None)&(h_4_pts is None)):
                    print("\nCurrent Image Errored Out ..")
                else:
                    if(not (os.path.isdir(save_dir +'Patch_A/'))):
                        os.makedirs(save_dir + 'Patch_A/')
                        os.makedirs(save_dir + 'Patch_B/')
                        os.makedirs(save_dir + 'Image_IA/')
 
                    patch_A_dir = save_dir + 'Patch_A/' + str(image) + '_' + eachIter + '.jpg'
                    patch_B_dir = save_dir + 'Patch_B/' + str(image) + '_' + eachIter + '.jpg'
                    image_IA_dir = save_dir + 'Image_IA/' + str(image) + '_' + eachIter + '.jpg'

                    cv2.imwrite(patch_A_dir, pA)
                    cv2.imwrite(patch_B_dir, pB)
                    cv2.imwrite(image_IA_dir, a_image)

                    h_4_points_list.append(np.hstack((h_4_pts[:,0] , h_4_pts[:,1])))
                    points_list.append(points)
                    images_list.append(str(image) + '_' + eachIter + '.jpg')
        
        np.save(save_dir + "points_list", np.array(points_list))
        print("\nPoints data saved @:  ", save_dir)
    
        if(not (os.path.isdir(save_dir + 'h_4_list.csv'))):
            print("\nCreating H4pt List ... ")
            with open(save_dir + 'h_4_list.csv', 'w', newline ='') as h4:
                w = csv.writer(h4)
                w.writerows(h_4_points_list)
        else:
            with open(save_dir + 'h_4_list.csv', 'a', newline ='') as h4:
                w = csv.writer(h4)
                w.writerows(h_4_points_list)
        
        df = pd.DataFrame(images_list)
        print("\nSaving Image Names List ... ")
        df.to_csv(save_dir + "images_list.csv", index = False)
            
            
            
if __name__ == '__main__':
    main()
 
