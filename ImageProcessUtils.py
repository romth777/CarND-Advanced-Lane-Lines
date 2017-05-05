import os
import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle


class ImageProcessUtils:
    def __init__(self):
        # undistortion matrix from pickle file
        with open(os.path.join(os.path.dirname(__file__), "my_trial", "dist_pickle.p"), "rb") as f:
            self.dist_pkl = pickle.load(f)
        # rotation matrix to the bird eye view
        self.M = None
        # inverse rotation matrix from the bird eye view
        self.Minv = None
        # image size for this project, this value must change in the other image shape
        self.img_size = (1280, 720)
        # update rotation / inverse rotation matrix
        self.cal_warp_matrix()

    def undistort_img(self, img):
        """Undistort image using calibration matrix from the chess board"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(self.dist_pkl["objp"], self.dist_pkl["imgp"], (img.shape[1], img.shape[0]), None, None)
        undist = cv2.undistort(img, mtx, dist, None, mtx)
        return undist

    def hls_select(self, img, thresh=(0, 255), channel=2):
        """Return a channel from HLS"""
        # 1) Convert to HLS color space
        hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        # 2) Apply a threshold to the S channel
        selected_channel = hls[:, :, channel]
        binary = np.zeros_like(selected_channel)
        binary[(selected_channel > thresh[0]) & (selected_channel <= thresh[1])] = 1
        # 3) Return a binary image of threshold result
        binary_output = np.copy(binary)  # placeholder line
        return binary_output

    def rgb_select(self, img, thresh=(0, 255), channel=2):
        """Return a channel from RGB"""
        # 2) Apply a threshold to the selected channel
        selected_channel = img[:, :, channel]
        binary = np.zeros_like(selected_channel)
        binary[(selected_channel > thresh[0]) & (selected_channel <= thresh[1])] = 1
        # 3) Return a binary image of threshold result
        binary_output = np.copy(binary)  # placeholder line
        return binary_output

    # Define a function that applies Sobel x or y,
    # then takes an absolute value and applies a threshold.
    # Note: calling your function with orient='x', thresh_min=5, thresh_max=100
    # should produce output like the example image shown above this quiz.
    def abs_sobel_thresh(self, img, orient='x', sobel_kernel=3, thresh_min=20, thresh_max=100):
        """Apply single direction sobel filter and threshold of them"""
        # Apply the following steps to img
        # 1) Convert to grayscale
        # gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        gray = hls[:, :, 2]

        # 2) Take the derivative in x or y given orient = 'x' or 'y'
        if orient == 'x':
            sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
        elif orient == 'y':
            sobel = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)

        # 3) Take the absolute value of the derivative or gradient
        abs_sobel = np.absolute(sobel)

        # 4) Scale to 8-bit (0 - 255) then convert to type = np.uint8
        scaled_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))

        # 5) Create a mask of 1's where the scaled gradient magnitude
        # is > thresh_min and < thresh_max
        binary_output = np.zeros_like(scaled_sobel)
        binary_output[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1

        # 6) Return this mask as your binary_output image
        return binary_output

    # Define a function that applies Sobel x and y,
    # then computes the magnitude of the gradient
    # and applies a threshold
    def mag_thresh(self, img, sobel_kernel=3, mag_thresh=(30, 100)):
        """Apply mixed direction sobel filter with sqrt(sobelx ** 2 + sobely ** 2) and threshold of them"""
        # Apply the following steps to img
        # 1) Convert to grayscale
        # gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        gray = hls[:, :, 2]

        # 2) Take the gradient in x and y separately
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)

        # 3) Calculate the magnitude
        sobel = np.sqrt(sobelx ** 2 + sobely ** 2)
        gradmag = np.absolute(sobel)

        # 4) Scale to 8-bit (0 - 255) then convert to type = np.uint8
        scale_factor = np.max(gradmag) / 255
        gradmag = (gradmag / scale_factor).astype(np.int8)

        # 5) Create a binary mask where mag thresholds are met
        binary_output = np.zeros_like(gradmag)
        binary_output[(gradmag >= mag_thresh[0]) & (gradmag <= mag_thresh[1])] = 1

        # 6) Return this mask as your binary_output image
        return binary_output

    # Define a function that applies Sobel x and y,
    # then computes the direction of the gradient
    # and applies a threshold.
    def dir_threshold(self, img, sobel_kernel=15, thresh=(0.8, 1.2)):
        """Apply mixed direction sobel filter with arctan2(sobelx, sobely) and threshold of them"""
        # Apply the following steps to img
        # 1) Convert to grayscale
        # gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        gray = hls[:, :, 2]

        # 2) Take the gradient in x and y separately
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)

        # 3) Take the absolute value of the derivative or gradient
        abs_sobelx = np.absolute(sobelx)
        abs_sobely = np.absolute(sobely)

        # 4) Use np.arctan2(abs_sobely, abs_sobelx) to calculate the direction of the gradient
        atan_sobel = np.arctan2(abs_sobely, abs_sobelx)

        # 5) Create a binary mask where direction thresholds are met
        binary_output = np.zeros_like(atan_sobel)
        binary_output[(atan_sobel >= thresh[0]) & (atan_sobel <= thresh[1])] = 1

        # 6) Return this mask as your binary_output image
        return binary_output

    def warp(self, img):
        """convert img with the rotation matrix"""
        warped = cv2.warpPerspective(img, self.M, self.img_size, flags=cv2.INTER_LINEAR)
        return warped

    def warp_inverse(self, img):
        """convert img with the inverse rotation matrix"""
        warped = cv2.warpPerspective(img, self.Minv, self.img_size, flags=cv2.INTER_LINEAR)
        return warped

    def cal_warp_matrix(self):
        """calculate the rotation matrix for bird view image"""
        src = np.float32(
            [[770, 480], [510, 480], [0, 720], [1280, 720]])
        #        [[710, 450], [570, 450], [0, 720], [1280, 720]])

        dst = np.float32(
            [[self.img_size[0], 0], [0, 0], [0, self.img_size[1]], [self.img_size[0], self.img_size[1]]])

        self.M = cv2.getPerspectiveTransform(src, dst)
        self.Minv = cv2.getPerspectiveTransform(dst, src)
