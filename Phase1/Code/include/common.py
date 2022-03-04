'''
Author(s):
Abhishek Nalawade (abhi1793@umd.edu)
Graduate Student in M.Eng Robotics,
University of Maryland, College Park

Aditya Jadhav (amjadhav@umd.edu)
Graduate Student in M.Eng Robotics,
University of Maryland, College Park
'''

import numpy as np
import cv2

class ANMS:
    def __init__(self, points):
        self.num_points = points

    def runANMS(self, corners):
        print("Number of Feature points %s\n"%(len(corners)))
        best = list()
        score = list()
        for i in range(len(corners)):
            if i == 0:
                best.append([corners[i][0][0], corners[i][0][1]])
                continue
            ED = corners[:i] - corners[i][0]     # since all above it have strong corner response value
            ED = ED**2
            ED = np.sum(ED, axis=2)
            minimumED = min(ED)
            ind = np.where(minimumED==ED)
            if [corners[ind[0][0]][0][0], corners[ind[0][0]][0][1]] in best:
                continue
            best.append([int(corners[ind[0][0]][0][0]), int(corners[ind[0][0]][0][1])])
            score.append(minimumED[0])
        score = np.array(score)
        ind = np.argsort(score)
        best = np.array(best)
        best = best[ind]
        return best[:self.num_points]

class FeatureDescriptors:
    def __init__(self, wsize):
        self.window_size = wsize

    def getFeatureDescriptors(self, image, points, var, save_on):
        print("Number of ANMS points %s\n"%(len(points)))
        features = list()
        count = 1
        for i in points:
            diff_lowery = 0
            diff_lowerx = 0
            lowery = i[1]-int(self.window_size/2)
            if lowery < 0:
                lowery = i[1] + abs(lowery) - int(self.window_size/2)
                diff_lowery = abs(i[1]-int(self.window_size/2))   # if lower y is shifted then shit uppery as well

            uppery = i[1]+int(self.window_size/2) + diff_lowery
            if uppery > image.shape[0]:
                diff = uppery - image.shape[0]
                uppery = i[1] - diff + int(self.window_size/2)
                lowery = lowery - diff

            lowerx = i[0]-int(self.window_size/2)
            if lowerx < 0:
                lowerx = i[0] + abs(lowerx) - int(self.window_size/2)
                diff_lowerx = abs(i[0]-int(self.window_size/2))   # if lower x is shifted then shit upperx as well

            upperx = i[0]+int(self.window_size/2)  + diff_lowerx
            if upperx > image.shape[1]:
                diff = upperx - image.shape[1]
                upperx = i[0] - diff + int(self.window_size/2)
                lowerx = lowerx - diff

            patch = image[lowery:uppery, lowerx:upperx]
            blur = cv2.GaussianBlur(patch, (5,5),0)
            feature = cv2.resize(blur, (8,8))
            if count < 5 and save_on:
                cv2.imwrite("results/FeatureDescriptor/FD%s%s.png"%(var, count), feature)
            feature = np.reshape(feature, (feature.shape[0]*feature.shape[1]))
            mean = np.sum(feature)/feature.shape[0]
            std = ((1/feature.shape[0])*(np.sum((feature-mean)**2)))**(1/2)
            feature = (feature - (mean))/std
            features.append(feature)
            count += 1
        return features


    def matchFeatures(self, des_one, des_two):
        pair = list()
        des_one = np.array(des_one, dtype=np.int64)
        des_two = np.array(des_two, dtype=np.int64)
        for i in range(len(des_one)):
            ssd = des_two - np.reshape(des_one[i], (1,des_one.shape[1]))
            ssd = ssd**2
            ssd = np.sum(ssd, axis = 1)

            minSSD = np.min(ssd)
            if minSSD > 15:
                continue
            ind = np.where(ssd==minSSD)
            pair.append([i,ind[0][0]])

        print("matched pairs from ssd %s\n"%(len(pair)))
        ratio = len(pair)/len(des_one)
        return pair

    def drawMatch(self, points1, img1, points2, img2, pair, var, save_on):
        max_height = np.max([img1.shape[0], img2.shape[0]])
        tmp_img1 = np.zeros((max_height,img1.shape[1],img1.shape[2]),dtype=np.uint8)
        tmp_img2 = np.zeros((max_height,img2.shape[1],img1.shape[2]),dtype=np.uint8)
        tmp_img1[:img1.shape[0],:img1.shape[1]] = img1
        tmp_img2[:img2.shape[0],:img2.shape[1]] = img2
        joint_imgs = np.concatenate((tmp_img1, tmp_img2), axis=1)
        for i in range(len(pair)):
            start = (points1[pair[i][0]][0], points1[pair[i][0]][1])
            end = (points2[pair[i][1]][0]+img1.shape[1], points2[pair[i][1]][1])
            cv2.line(joint_imgs, start, end, (0,255,255), 1)
        cv2.imshow("nnew",joint_imgs)
        if save_on:
            cv2.imwrite("results/MatchingOutput/matching%s.png"%(var), joint_imgs)
        cv2.waitKey(1)
        return

class RANSAC:

    def ssd(self, match_right, nw_right):
        ssd = match_right - nw_right
        ssd = ssd**2
        ssd = np.sum(ssd, axis=1)
        inlier = np.where(ssd < 2)
        return inlier

    def check_if_multiple_points_match_to_same_point(self, projected_points):
        check = 1
        for i in range(len(projected_points)):
            point = np.reshape(projected_points[i], (1,projected_points.shape[1]))
            ssd = projected_points - point
            ssd = ssd**2
            ssd = np.sum(ssd, axis=1)
            ind = np.where(ssd<2)
            if len(ind[0]) > 1:
                check = 0
                break
        return check

    def runRANSAC(self, points1, points2, pairs):
        s = 4       # number of points required for the model
        e = 0.5
        N = 15000
        p = 0.95
        count = 0
        best_score = None
        pairs = np.array(pairs)
        ind = np.arange(pairs.shape[0])
        matched_left = points1[pairs[:,0]]
        matched_right = points2[pairs[:,1]]
        counter = 0
        print("Running RANSAC........\n")
        while True:
            count = 0
            while N > count:
                random_ind = np.random.choice(ind, size=4, replace=False)
                left = matched_left[random_ind]
                right = matched_right[random_ind]
                H = cv2.getPerspectiveTransform(np.float32(left), np.float32(right))
                tmp_left = np.concatenate((matched_left, np.ones((matched_left.shape[0],1))), axis = 1)
                new_right = np.dot(H, tmp_left.T)
                new_right[-1,:] = new_right[-1,:]+0.0001
                new_right = new_right/(new_right[-1,:])
                new_right = new_right.T
                check = self.check_if_multiple_points_match_to_same_point(new_right[:,:2])
                if check == 0:
                    count += 0.5
                    continue
                inliers = self.ssd(matched_right, new_right[:,:2])
                num_inliers = len(inliers[0])
                if best_score == None or best_score < num_inliers:
                    best_score = num_inliers
                    print("Number of inliers: ",best_score)
                    best_points1 = matched_left[inliers]
                    best_points2 = matched_right[inliers]
                    H_best = H
                count += 1

            counter += 1
            print("Executed RANSAC ",counter," times")
            if best_score > 6 or counter > 5:
                break
            else:
                print("Extended search for inliers......\n")
        return best_points1, best_points2, H_best, best_score

    def drawMatch(self, points1, img1, points2, img2, var, save_on):
        max_height = np.max([img1.shape[0], img2.shape[0]])
        tmp_img1 = np.zeros((max_height,img1.shape[1],img1.shape[2]),dtype=np.uint8)
        tmp_img2 = np.zeros((max_height,img2.shape[1],img1.shape[2]),dtype=np.uint8)
        tmp_img1[:img1.shape[0],:img1.shape[1]] = img1
        tmp_img2[:img2.shape[0],:img2.shape[1]] = img2
        joint_imgs = np.concatenate((tmp_img1, tmp_img2), axis=1)

        for i in range(len(points1)):
            start = (points1[i,0], points1[i,1])
            end = (points2[i,0]+img1.shape[1], points2[i,1])
            cv2.line(joint_imgs, start, end, (0,255,255), 1)
        cv2.imshow("final",joint_imgs)
        if save_on:
            cv2.imwrite("results/RansacOutput/RANSACmatching%s.png"%(var), joint_imgs)
        cv2.waitKey(1)
        return

    def EstimateHomography(self, src, dst):
        A = np.zeros((2*len(src),9))
        i = 0
        for a in range(len(A)):
            if a%2 == 0:
                A[a,:] = [src[i][0], src[i][1], 1, 0, 0, 0, -(dst[i][0] * src[i][0]), -(dst[i][0] * src[i][1]), -dst[i][0]]
            else:
                A[a,:] = [0, 0, 0, src[i][0], src[i][1], 1, -(dst[i][1] * src[i][0]), -(dst[i][1] * src[i][1]), -dst[i][1]]
                i += 1

        U, sigma, V = np.linalg.svd(A)
        Vt = V.T
        h = Vt[:,8]/Vt[8][8]
        H = np.reshape(h, (3,3))
        return H

class Blend:
    def __init__(self,num):
        self.num = num

    def getNewDimensions(self, img2, H):
        point = np.array([[0,0,1],[0,img2.shape[0],1],[img2.shape[1],img2.shape[0],1],[img2.shape[1],0,1]])
        border = np.dot(H, point.T)
        border = border/border[-1]
        col_min = np.min(border[0,:])
        col_max = np.max(border[0,:])
        row_min = np.min(border[1,:])
        row_max = np.max(border[1,:])

        # new_width = round(col_max - col_min)
        # new_height = round(col_max - col_min)
        if col_min < 0:
            new_width = round(col_max - col_min)
        else:
            # new_width = round(col_max)
            new_width = round(col_max - col_min)

        if row_min < 0:
            new_height = round(row_max - row_min)
        else:
            # new_height = round(row_max)
            new_height = round(row_max - row_min)
        shift = np.array([[1,0,-col_min],[0,1,-row_min],[0,0,1]])
        H = np.dot(shift, H)
        return new_height, new_width, H

    def EstimateTranslation(self, points1, points2, H):
        onet = np.ones((points2.shape[0],1))
        point = np.concatenate((points2,onet), axis=1)
        point = point.T
        transformed_point = np.dot(H,point)
        transformed_point = transformed_point/transformed_point[-1]
        points1 = points1.T
        translations = points1 - transformed_point[:2]
        translations = translations.T

        translations_sum = np.sum(translations, axis=0)
        translations_mean = translations_sum/translations.shape[0]
        translation = np.array([[translations_mean[0]],[translations_mean[1]]])
        # print("translation is ",translation)
        return translation, transformed_point

    def stitch(self, stitch_this, stitch_to, transformed_point):
        to_shape = stitch_to.shape
        this_shape = stitch_this.shape
        shape = np.array([to_shape,this_shape])
        panorama = np.zeros((np.max(shape[:,0]), np.max(shape[:,1]),3), dtype=np.uint8)
        # print("panorama shape: ",panorama.shape)
        panorama[:to_shape[0],:to_shape[1]] = stitch_to
        ind = np.where(stitch_this>0)
        panorama[ind[:2]] = stitch_this[ind[:2]]
        return panorama


    def warpAndStitch(self, img, points1, img2, points2, H):
        H = np.linalg.inv(H)
        new_height, new_width, H = self.getNewDimensions(img2, H)
        stitch_this = cv2.warpPerspective(img2, H, (new_width,new_height))

        translation, transformed_point = self.EstimateTranslation(points1, points2, H)

        if translation[0,0] < 0 and translation[1,0] < 0:
            M = np.float32([[1,0,abs(round(translation[0,0]))],[0,1,abs(round(translation[1,0]))]])
            translated_shape = (abs(round(translation[0,0]))+img.shape[1], abs(round(translation[1,0]))+img.shape[0])
            img = cv2.warpAffine(img, M, translated_shape)
            # panorama = self.stitch(stitch_this, img, transformed_point)
            panorama = self.stitch(img, stitch_this, transformed_point)
        elif translation[0,0] < 0 and translation[1,0] > 0:
            M = np.float32([[1,0,abs(round(translation[0,0]))],[0,1,0]])
            translated_shape = (abs(round(translation[0,0]))+img.shape[1], img.shape[0])
            img = cv2.warpAffine(img, M, translated_shape)

            M = np.float32([[1,0,0],[0,1,abs(round(translation[1,0]))]])
            translated_shape = (stitch_this.shape[1], abs(round(translation[1,0]))+stitch_this.shape[0])
            stitch_this = cv2.warpAffine(stitch_this, M, translated_shape)
            panorama = self.stitch(img, stitch_this, transformed_point)
        elif translation[0,0] > 0 and translation[1,0] < 0:
            M = np.float32([[1,0,abs(round(translation[0,0]))],[0,1,0]])
            translated_shape = (abs(round(translation[0,0]))+stitch_this.shape[1], stitch_this.shape[0])
            stitch_this = cv2.warpAffine(stitch_this, M, translated_shape)

            M = np.float32([[1,0,0],[0,1,abs(round(translation[1,0]))]])
            translated_shape = (img.shape[1], abs(round(translation[1,0]))+img.shape[0])
            img = cv2.warpAffine(img, M, translated_shape)
            panorama = self.stitch(img, stitch_this, transformed_point)
        else:
            M = np.float32([[1,0,round(translation[0,0])],[0,1,round(translation[1,0])]])
            translated_shape = (abs(round(translation[0,0]))+stitch_this.shape[1], abs(round(translation[1,0]))+stitch_this.shape[0])
            stitch_this = cv2.warpAffine(stitch_this, M, translated_shape)
            panorama = self.stitch(img, stitch_this, transformed_point)

        cv2.imshow("panorama", panorama)
        cv2.waitKey(1)

        return panorama
