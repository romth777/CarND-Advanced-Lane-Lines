import numpy as np
from collections import deque


# Define a class to receive the characteristics of each line detection
class Line:
    def __init__(self):
        # this must be changed in the other case
        self.image_size = [1280, 720]
        # was the line detected in the last iteration?
        self.detected = False
        # number of average iteration
        self.number_of_iteration = 10
        # x values of the last n fits of the line
        self.recent_xfitted = []
        # average x values of the fitted line over the last n iterations
        self.bestx = None
        # polynomial coefficients averaged over the last n iterations
        self.best_fit = None
        # polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]
        # radius of curvature of the line in some units in current
        self.radius_of_curvature = deque(maxlen=self.number_of_iteration)
        # radius of curvature of the line in some units
        self.radius_of_curvature_mean = None
        # distance in meters of vehicle center from the line in current
        self.line_base_pos = deque(maxlen=self.number_of_iteration)
        # distance in meters of vehicle center from the line
        self.line_base_pos_mean = None
        # difference in fit coefficients between last and new fits
        self.diffs = np.array([0,0,0], dtype='float')
        # x values for detected line pixels
        self.allx = deque(maxlen=self.number_of_iteration)
        # y values for detected line pixels
        self.ally = self.get_ally()
        # Number of consecutive times that data could be successfully added
        self.count = 0

    def update(self, x, y):
        """update line information"""
        # update the current data
        self.current_fit = self.apply_poly_fit(x, y)
        self.recent_xfitted = self.get_poly(self.current_fit)

        if self.bestx is not None:
            self.diffs = np.subtract(self.bestx, self.recent_xfitted)
            diffs_max = np.max(np.abs(self.diffs))
            diffs_std = np.std(self.diffs)
            # check the difference
            if diffs_max < 50:
                self.detected = True
                self.allx.append(self.recent_xfitted)
            else:
                self.detected = False
                self.allx.clear()

        # update the best data
        if len(self.allx) < 2:
            self.detected = True
            self.allx.append(self.recent_xfitted)
            self.bestx = self.recent_xfitted
            self.best_fit = self.current_fit
        else:
            self.bestx = np.mean(self.allx, axis=0)
            self.best_fit = np.polyfit(self.ally, self.bestx, 2)

        # update curvature
        self.radius_of_curvature.append(self.cal_radius_of_curvature(fit=self.best_fit, plotx=self.bestx, ploty=self.ally))
        if len(self.radius_of_curvature) > 1:
            self.radius_of_curvature_mean = np.mean(self.radius_of_curvature, axis=0)
        else:
            self.radius_of_curvature_mean = self.radius_of_curvature[-1]

        # update line base
        if len(self.line_base_pos) > 1:
            self.line_base_pos_mean = np.mean(self.line_base_pos, axis=0)
        else:
            self.line_base_pos_mean = self.line_base_pos[-1]

        return self.detected

    def reset(self):
        """reset for switch to another image or video file"""
        self.__init__()

    def get_ally(self):
        return np.linspace(0, self.image_size[1] - 1, self.image_size[1])

    def apply_poly_fit(self, x, y):
        return np.polyfit(y, x, 2)

    def get_poly(self, current_fit):
        return current_fit[0] * self.ally ** 2 + current_fit[1] * self.ally + current_fit[2]

    # Define y-value where we want radius of curvature
    def cal_radius_of_curvature(self, fit, plotx, ploty):
        # I'll choose the maximum y-value, corresponding to the bottom of the image
        y_eval = np.max(ploty)

        # Define conversions in x and y from pixels space to meters
        ym_per_pix = self.get_ym_per_pix()  # meters per pixel in y dimension
        xm_per_pix = self.get_xm_per_pix()  # meters per pixel in x dimension

        # Fit new polynomials to x,y in world space
        fit_cr = np.polyfit(ploty * ym_per_pix, plotx * xm_per_pix, 2)
        # Calculate the new radii of curvature
        radius_of_curvature = ((1 + (2 * fit_cr[0] * y_eval * ym_per_pix + fit_cr[1]) ** 2) ** 1.5) / np.absolute(
            2 * fit_cr[0])

        # Now our radius of curvature is in meters
        return radius_of_curvature

    def get_ym_per_pix(self):
        return 30 / self.image_size[1]

    def get_xm_per_pix(self):
        return 3.7 / self.image_size[1]

    def cal_line_base(self, line_base):
        self.line_base_pos.append(np.abs(self.image_size[0] / 2 - line_base) * self.get_xm_per_pix())
        if len(self.line_base_pos) < 2:
            self.line_base_pos_mean = self.line_base_pos[-1]
        else:
            self.line_base_pos_mean = np.mean(self.line_base_pos, axis=0)
