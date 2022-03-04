# Autopano

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
CMSC 733 Project 1 - AutoPano: Repository for image stitching using traditional CV and deep learning approach

---

## Contributors

1) [Abhishek Nalawade](https://github.com/abhishek-nalawade)
Graduate Student of M.Eng Robotics at University of Maryland. 
2) [Aditya Jadhav](https://github.com/iamjadhav)
Graduate Student of M.Eng Robotics at University of Maryland.

## Overview

The repository contains our efforts aimed at stitching two or more images for creating uninterrupted Panorama. The report contains a detailed solution
to Phase 1 and Phase 2 of the Panorama Problem. Phase -1 uses traditional Computer Vision techniques and the Phase - 2 makes
use of a Deep-Learning approach. The first method uses the Homography Matrix between two images and the second method
uses Supervised and Unsupervised Neural Networks to achieve the stitching.

## Technology Used

We used the following tools and systems:

* Ubuntu 18.04 LTS
* Python Programming Language
* OpenCV Library
* NumPy Library
* TensorFlow
* Pandas

## License 

```
MIT License

Copyright (c) 2022 Abhishek Nalawade, Aditya Jadhav

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE 
SOFTWARE.
```

## Results

* Corner Detection

![cornersCustomSet12](https://user-images.githubusercontent.com/35925489/156830990-a02dbea2-04d8-4f7d-9a3c-6d744638517f.png)

* ANMS

![anmsSet12](https://user-images.githubusercontent.com/35925489/156831341-7345edf1-15f2-4350-b207-f098afebab81.png)

* Feature Matching

![matchingCustomSet2](https://user-images.githubusercontent.com/35925489/156831576-e712f08c-03da-4e62-8d9b-18458e525efa.png)

* Blending

![mypanoCustomSet2](https://user-images.githubusercontent.com/35925489/156831743-ffc2a81c-605e-4ebd-baf4-b2830549b959.png)

![mypanoSet1](https://user-images.githubusercontent.com/35925489/156831785-88d46096-1819-4b44-a6a2-650e50bb713f.png)

* Deep Learning Results

![stacked44](https://user-images.githubusercontent.com/35925489/156831998-032353dd-d635-4775-ad51-43f114a8f89c.png)

![stacked47](https://user-images.githubusercontent.com/35925489/156832028-0ea2880e-fbbd-45e2-a724-4e61206cc83f.png)



## Known Issues/Bugs 

- The Supervised Network accuracy needs much better tuning

## Dependencies

- Install OpenCV 4.0, Tensorflow 2.8, Cuda 11.2 and Cudnn 8.1.0

## How to build

```
git clone --recursive https://github.com/iamjadhav/Human_Obstacle_Detector
cd autopano
```

* Phase - 1

```
cd Phase1\Code
python Wrapper.py --path Train/Set1 --save True
```

- Results Phase1:

The results for Phase one are saved in Phase1/Code/results. The image corners.png can be found inside Corners folder at above path. The image anms.png can be found inside ANMS folder at above path

* Phase - 2

- Data Generation: 

```
cd Phase2\Code
python Wrapper.py
```
- Supervised Network: 


- Training:

```
python Train.py --BasePath ../Data/Trained_ --CheckPointPath ../Checkpoints/supervised/ --ModelType sup --NumEpochs 100 --DivTrain 1 --MiniBatchSize 64 --LoadCheckPoint 0 --LogsPath ./Logs/supervised/
```
- Testing:

```
python Test.py --ModelPath ../Checkpoints/supervised/99model.ckpt --BasePath ../Data/Validated_ --SavePath ./Results/ --ModelType sup 
```

- Unsupervised Network: 


- Training:

```
python Train.py --BasePath ../Data/Trained_ --CheckPointPath ../Checkpoints/unsupervised/ --ModelType Unsup --NumEpochs 100 --DivTrain 1 --MiniBatchSize 64 --LoadCheckPoint 0 --LogsPath ./Logs/unsupervised/
```
- Testing:

```
python Test.py --ModelPath ../Checkpoints/unsupervised/99model.ckpt --BasePath ../Data/Validated_ --CheckPointPath ../Checkpoints/unsupervised/ --SavePath ./Results/ --ModelType Unsup
```

- Results Phase2:

Phase2 Results are inside the Results folder in the Code directory.