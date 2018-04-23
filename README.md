# OpenCVCalibration
OpenCV camera calibration scheme has a few non-optimalities that make it harder to have a good camera calibration in presence of strong radial distortion:

1. Checker-board calibration targets must be fully seen in order to be detected. 
As a result most detected corners are in the center of the camera's FOV (field of view), and there are less near the edges.

2. The calculation of lens distortion parameters are not constrained. 
Due to the first issue, there are no constraints that effect the distortion parameters near the edges of the FOV.
Therefore the calculated parameters will be accurate in the middle of the FOV and less accurate near the edges.
Moreover, and more important, there is no constraint that forces the distortion parameters to be physical - there may be cases where pixels in the distorted image have no source pixel in an undistorted image given the distortion parameters.

3. In order to calculate the optimal camera matrix of the undistorted image, OpenCV calculates the inverse of the distortion function to map the edges of the distorted image to an undistorted image. if the second issue happens, there is no inverse and the result of cvGetOptimalNewCameraMatrix is very inaccurate.

This repository is of a python code that mitigates issues #1 and #3.

In order to mitigate issue #1, there is the CornerMapper class that receives the corners that OpenCV did detected which are not ordered, and finds a mapping from theses corners to the calibration checker board.

In order to mitigate issue #3, there is the NewtonRaphsonUndistort class that replaces OpenCV iterative inverse search that may fail to converge with a Newton-Raphson iterations followed by a binary search that tends to converge.

All code is written in Python, compatible with python 2.7
