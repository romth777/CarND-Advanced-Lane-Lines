import os
import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle
from moviepy.editor import VideoFileClip
from collections import deque

import ImageProcessUtils
import Line


class Pipeline:
    def __init__(self):
        self.ipu = ImageProcessUtils.ImageProcessUtils()
        self.left_line = Line.Line()
        self.right_line = Line.Line()
        self.isfound = False
        self.img_deque = deque(maxlen=5)

    # apply gradient using some method and combine them in one image
    def gradient_image(self, img):
        gradx = self.ipu.abs_sobel_thresh(img, orient='x', sobel_kernel=17, thresh_min=25, thresh_max=170)
        grady = self.ipu.abs_sobel_thresh(img, orient='y', sobel_kernel=3, thresh_min=25, thresh_max=220)
        mag_binary = self.ipu.mag_thresh(img, sobel_kernel=7, mag_thresh=(10, 130))
        dir_binary = self.ipu.dir_threshold(img, sobel_kernel=11, thresh=(0.5, 1.4))

        combined = np.zeros_like(dir_binary)
        combined[((gradx == 1) | (grady == 1) | (mag_binary == 1)) & (dir_binary == 1)] = 1
        return combined

    # combine the signal of R and S with thresholds
    def clearlify_image(self, img):
        s_binary = self.ipu.hls_select(img, thresh=(80, 255), channel=2)
        r_binary = self.ipu.rgb_select(img, thresh=(220, 250), channel=0)

        combined = np.zeros_like(r_binary)
        combined[((s_binary == 1) | (r_binary == 1))] = 1
        return combined

    # imput image is not warped
    def mix_image_of_grad_clearlify(self, img):
        grad = self.ipu.warp(self.gradient_image(img))
        origin = self.clearlify_image(self.ipu.warp(img))

        combined = np.zeros_like(grad)
        combined[((grad > 0) | (origin == 1))] = 1
        return combined

    # Assuming you have created a warped binary image called "binary_warped"
    def find_lane(self, binary_warped):
        # Take a histogram of the bottom half of the image
        histogram = np.sum(binary_warped[:, :], axis=0)
        # Find the peak of the left and right halves of the histogram
        # These will be the starting point for the left and right lines
        midpoint = np.int(histogram.shape[0] / 2)
        left_line_base = np.argmax(histogram[:midpoint])
        right_line_base = np.argmax(histogram[midpoint:]) + midpoint

        self.left_line.cal_line_base(left_line_base)
        self.right_line.cal_line_base(right_line_base)

        # Choose the number of sliding windows
        nwindows = 9
        # Set height of windows
        window_height = np.int(binary_warped.shape[0] / nwindows)
        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Current positions to be updated for each window
        leftx_current = left_line_base
        rightx_current = right_line_base
        # Set the width of the windows +/- margin
        margin = 100
        # Set minimum number of pixels found to recenter window
        minpix = 50
        # Create empty lists to receive left and right lane pixel indices
        left_lane_inds = []
        right_lane_inds = []

        # Step through the windows one by one
        for window in range(nwindows):
            # Identify window boundaries in x and y (and right and left)
            win_y_low = binary_warped.shape[0] - (window + 1) * window_height
            win_y_high = binary_warped.shape[0] - window * window_height
            win_xleft_low = leftx_current - margin
            win_xleft_high = leftx_current + margin
            win_xright_low = rightx_current - margin
            win_xright_high = rightx_current + margin

            # Identify the nonzero pixels in x and y within the window
            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (
            nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (
            nonzerox < win_xright_high)).nonzero()[0]
            # Append these indices to the lists
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)
            # If you found > minpix pixels, recenter next window on their mean position
            if len(good_left_inds) > minpix:
                leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > minpix:
                rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

        # Concatenate the arrays of indices
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)

        # Extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]

        return leftx, lefty, rightx, righty

    # Assume you now have a new warped binary image
    # from the next frame of video (also called "binary_warped")
    # It's now much easier to find line pixels!
    def found_lane(self, binary_warped, left_fit, right_fit):
        # Take a histogram of the bottom half of the image
        histogram = np.sum(binary_warped[:, :], axis=0)
        # Find the peak of the left and right halves of the histogram
        # These will be the starting point for the left and right lines
        midpoint = np.int(histogram.shape[0] / 2)
        left_line_base = np.argmax(histogram[:midpoint])
        right_line_base = np.argmax(histogram[midpoint:]) + midpoint

        self.left_line.cal_line_base(left_line_base)
        self.right_line.cal_line_base(right_line_base)

        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        margin = 100
        left_lane_inds = ((nonzerox > (left_fit[0] * (nonzeroy ** 2) + left_fit[1] * nonzeroy + left_fit[2] - margin)) & (
        nonzerox < (left_fit[0] * (nonzeroy ** 2) + left_fit[1] * nonzeroy + left_fit[2] + margin)))
        right_lane_inds = (
        (nonzerox > (right_fit[0] * (nonzeroy ** 2) + right_fit[1] * nonzeroy + right_fit[2] - margin)) & (
        nonzerox < (right_fit[0] * (nonzeroy ** 2) + right_fit[1] * nonzeroy + right_fit[2] + margin)))

        # Again, extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]

        return leftx, lefty, rightx, righty

    # reset function clear the data of previous analysis
    def reset(self):
        self.left_line.reset()
        self.right_line.reset()
        self.isfound = False

    # Main pipeline process of this project
    def pipeline(self, img):
        # convert dtype for uint8 for processing
        img = img.astype(np.uint8)

        # undistort image
        undistorted = self.ipu.undistort_img(img)

        # Mix the image of warped gradient and warped R and S channel
        grad = self.mix_image_of_grad_clearlify(undistorted).astype(np.uint8)

        # sum images by frame for the dotted line
        self.img_deque.append(grad)
        if len(self.img_deque) > 1:
            grad_like = np.zeros_like(grad)
            for img_q in self.img_deque:
                grad_like[(grad_like == 1) | (img_q == 1)] = 1
            grad = grad_like

        # Poly fitting to the line of the gradiented image
        # Check if the lane has founded for shoten the processing time
        if self.isfound:
            leftx, lefty, rightx, righty = self.found_lane(grad, self.left_line.current_fit, self.right_line.current_fit)
        else:
            leftx, lefty, rightx, righty = self.find_lane(grad)
            self.isfound = True

        # Update line information
        # if it can't find the lane in some times, find lane again
        if not self.left_line.update(leftx, lefty):
            self.isfound = False
        if not self.right_line.update(rightx, righty):
            self.isfound = False

        # Recast the x and y points into usable format for cv2.fillPoly()
        pts_left = np.array([np.transpose(np.vstack([self.left_line.bestx, self.left_line.ally]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([self.right_line.bestx, self.right_line.ally])))])
        pts = np.hstack((pts_left, pts_right))

        # Create an image to draw the lines on
        warp_zero = np.zeros_like(grad).astype(np.uint8)
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

        # Warp the blank back to original image space using inverse perspective matrix (Minv)
        newwarp = cv2.warpPerspective(color_warp, self.ipu.Minv, self.ipu.img_size)
        # Combine the result with the original image
        result = cv2.addWeighted(undistorted, 1, newwarp, 0.3, 0)

        # draw radius + distance from center on image
        font = cv2.FONT_HERSHEY_SIMPLEX
        radius = (self.left_line.radius_of_curvature_mean + self.right_line.radius_of_curvature_mean) / 2
        cv2.putText(result, 'radius: {:5.2f}km'.format(radius / 1000), (10, 30),
                    font, 1, (255, 255, 255), 2, cv2.LINE_AA)
        center_distance = (self.left_line.line_base_pos_mean - self.right_line.line_base_pos_mean) / 2
        cv2.putText(result, 'distance from center: {:5.2f}m'.format(center_distance), (10, 60),
                    font, 1, (255, 255, 255), 2, cv2.LINE_AA)

        return result

    # show image as arranged
    def arrange_images(self, img1, img2, title1, title2):
        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
        f.tight_layout()
        ax1.imshow(img1)
        ax1.set_title(title1, fontsize=50)
        ax2.imshow(img2)
        ax2.set_title(title2, fontsize=50)
        plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)

    # run the main pipeline for video
    def run_pipeline(self, video_file, duration=None, end=False):
        """Runs pipeline on a video and writes it to temp folder"""
        print('processing video file {}'.format(video_file))
        clip = VideoFileClip(video_file)

        if duration is not None:
            if end:
                clip = clip.subclip(clip.duration - duration)
            else:
                clip = clip.subclip(0, duration)

        fpath = 'temp/' + video_file
        if os.path.exists(fpath):
            os.remove(fpath)
        processed = clip.fl(lambda gf, t: self.pipeline(gf(t)), [])
        processed.write_videofile(fpath, audio=False)


def main():
    pl = Pipeline()

    do_images = False
    do_videos = True

    if do_images:
        images = glob.glob('test_images/*.jpg')

        for fname in images:
            image = cv2.imread(fname)
            output = pl.pipeline(image)
            plt.imshow(pl.pipeline(output))
            pl.reset()

    if do_videos:
        # video_files = ['project_video.mp4', 'challenge_video.mp4', 'harder_challenge_video.mp4']
        video_files = ['project_video.mp4']
        for video_file in video_files:
            pl.run_pipeline(video_file)
            pl.reset()

main()
