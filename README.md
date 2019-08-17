## Advanced Lane Finding
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)
![Lanes Image](./output_images/Figure_out1.png)

## The Project

The goal of this project is to identify the lane boundaries in a video using computer vision.
The steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

The images for camera calibration are stored in the folder called `camera_cal`.  The images in `test_images` are for testing the pipeline on single frames.  If you want to extract more test images from the videos, you can simply use an image writing method like `cv2.imwrite()`, i.e., you can read the video in frame by frame as usual, and for frames you want to save for later you can write to an image file.  

I saved the examples of the output from each stage of my pipeline in the folder called `output_images`, and included a description in my writeup for the project of what each image shows.    The video called `project_video.mp4` is the first video which my pipeline works well on it.  

The `challenge_video.mp4` video is an extra challenge which I tested on my pipeline under somewhat trickier conditions. My proposed pipline works on it, too. 

## Dependencies

* Python 3.5
* Numpy
* OpenCV-Python
* Matplotlib
* Pickle

## How to run

Referencing The Project
---
If you like my code and you want to use it in your project, please refer it like this:

`Amani, Sajjad. "Advanced Lane Finding on the Road." GitHub, 16 July 2019, https://github.com/Sj-Amani/LaneLines_AdvancedDetector`


