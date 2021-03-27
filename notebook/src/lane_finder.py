
import cv2
import numpy as np


class LaneFinder:
    """
    This class will provide the following functionality:
    - lane detection
    - sliding window
    - curve fitting
    """

    def __init__(self, parameter_dict) -> None:
        self.parameter_dict = parameter_dict

    @staticmethod
    def find_peak(binary_warped):
        histogram = np.sum(
            binary_warped[binary_warped.shape[0]//2:, :], axis=0)
        # Find the peak of the left and right halves of the histogram
        # These will be the starting point for the left and right lines
        midpoint = np.int(histogram.shape[0]//2)
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint
        return leftx_base, rightx_base

    def find_lane_pixels(self, binary_warped):
        # HYPERPARAMETERS
        # Choose the number of sliding windows
        nwindows = self.parameter_dict['n_window']
        # Set the width of the windows +/- margin
        margin = self.parameter_dict['margin']
        # Set minimum number of pixels found to recenter window
        minpix = self.parameter_dict['minpix']

        # Take a histogram of the bottom half of the image
        leftx_current, rightx_current = self.find_peak(binary_warped)

        # Set height of windows - based on nwindows above and image shape
        window_height = np.int(binary_warped.shape[0]//nwindows)
        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        # Create empty lists to receive left and right lane pixel indices
        left_lane_inds = []
        right_lane_inds = []

        # Create an output image to draw on and visualize the result
        out_img = np.dstack((binary_warped, binary_warped, binary_warped))

        # Step through the windows one by one
        for window in range(nwindows):
            # Identify window boundaries in x and y (and right and left)
            win_y_low = binary_warped.shape[0] - (window+1)*window_height
            win_y_high = binary_warped.shape[0] - window*window_height

            win_xleft_low = leftx_current - margin  # Update this
            win_xleft_high = leftx_current + margin  # Update this

            win_xright_low = rightx_current - margin  # Update this
            win_xright_high = rightx_current + margin  # Update this

            # Draw the windows on the visualization image
            cv2.rectangle(out_img, (win_xleft_low, win_y_low),
                          (win_xleft_high, win_y_high), (255, 0, 0), 2)
            cv2.rectangle(out_img, (win_xright_low, win_y_low),
                          (win_xright_high, win_y_high), (0, 0, 255), 2)

            ###Identify the nonzero pixels in x and y within the window ###
            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high)
                              & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high)
                               & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]

            # Append these indices to the lists
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)

            if len(good_left_inds) > minpix:
                leftx_current = np.int(np.mean(nonzerox[good_left_inds]))

            if len(good_right_inds) > minpix:
                rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

        # Concatenate the arrays of indices (previously was a list of lists of pixels)
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)

        # Extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]

        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]

        return leftx, lefty, rightx, righty, out_img

    def search_around_poly(self, binary_warped, prev_left_fit, prev_right_fit):
        margin = self.parameter_dict['margin']

        # Grab activated pixels
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        new_left_fitx_high = prev_left_fit[0]*nonzeroy**2 + \
            prev_left_fit[1]*nonzeroy + prev_left_fit[2] + margin
        new_left_fitx_low = prev_left_fit[0]*nonzeroy**2 + \
            prev_left_fit[1]*nonzeroy + prev_left_fit[2] - margin

        new_right_fitx_high = prev_right_fit[0]*nonzeroy**2 + \
            prev_right_fit[1]*nonzeroy + prev_right_fit[2] + margin
        new_right_fitx_low = prev_right_fit[0]*nonzeroy**2 + \
            prev_right_fit[1]*nonzeroy + prev_right_fit[2] - margin

        left_lane_inds = (nonzerox < new_left_fitx_high) & (
            nonzerox >= new_left_fitx_low)
        right_lane_inds = (nonzerox < new_right_fitx_high) & (
            nonzerox >= new_right_fitx_low)

        # Again, extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]

        return leftx, lefty, rightx, righty

    def find_pixels(self, binary_warped):
        if self.left_fit is None or self.right_fit is None:
            leftx, lefty, rightx, righty, _ = self.find_lane_pixels(
                binary_warped)
        else:
            leftx, lefty, rightx, righty = self.search_around_poly(
                binary_warped, self.left_fit, self.right_fit)
        return leftx, lefty, rightx, righty

    @staticmethod
    def fit_poly(xs, ys):
        return np.polyfit(ys, xs, 2)

    @staticmethod
    def find_radius(fit, y):
        return ((1 + (2 * fit[0] * y + fit[1])**2)**(1.5)) / np.absolute(2 * fit[0])

    def find_lane(self, img_shape, leftx, lefty, rightx, righty):

        ym_per_pix = self.parameter_dict['ym_per_pix']
        xm_per_pix = self.parameter_dict['xm_per_pix']
        
        left_fit_meter = self.fit_poly(leftx * xm_per_pix, lefty * ym_per_pix)
        right_fit_meter = self.fit_poly(rightx * xm_per_pix, righty * ym_per_pix)

        y = img_shape[0]

        left_radius = self.find_radius(left_fit_meter, y * ym_per_pix)
        right_radius = self.find_radius(right_fit_meter, y * ym_per_pix)

        return left_fit_meter, right_fit_meter, left_radius, right_radius

    def plot(self, img_shape, leftx, lefty, rightx, righty, out_img):
        out_img[lefty, leftx] = [255, 0, 0]
        out_img[righty, rightx] = [0, 0, 255]

        left_fit_pixel = self.fit_poly(leftx, lefty)
        right_fit_pixel = self.fit_poly(rightx, righty)

        # Generate x and y values for plotting
        ploty = np.linspace(0, img_shape[0]-1, img_shape[0])
        ### Calc both polynomials using ploty, left_fit and right_fit ###
        left_fitx = left_fit_pixel[0]*ploty**2 + \
            left_fit_pixel[1]*ploty + left_fit_pixel[2]
        right_fitx = right_fit_pixel[0]*ploty**2 + \
            right_fit_pixel[1]*ploty + right_fit_pixel[2]

        xls, xrs, ys = left_fitx.astype(np.int32), right_fitx.astype(
            np.int32), ploty.astype(np.int32)
        t = 4

        for xl, xr, y in zip(xls, xrs, ys):
            cv2.line(out_img, (xl - t, y),
                     (xl + t, y), (255, 0, 0), int(t / 2))
            cv2.line(out_img, (xr - t, y),
                     (xr + t, y), (0, 0, 255), int(t / 2))
        return out_img, left_fit_pixel, right_fit_pixel

    def find_vehicle_position(self, img_shape, left_fit_pixel, right_fit_pixel):
        y = img_shape[0]

        mid = img_shape[1] / 2
        xl = left_fit_pixel[0] * \
            (y**2) + left_fit_pixel[1] * y + left_fit_pixel[2]
        xr = right_fit_pixel[0] * (y**2) + \
            right_fit_pixel[1] * y + right_fit_pixel[2]
        pix_pos = xl + (xr - xl) / 2
        vehicle_position = (pix_pos - mid) * self.parameter_dict['xm_per_pix']
        return vehicle_position

    def process_image(self, binary_warped):
        leftx, lefty, rightx, righty, out_img = self.find_lane_pixels(
            binary_warped)
        left_fit_meter, right_fit_meter, left_radius, right_radius = self.find_lane(
            binary_warped.shape, leftx, lefty, rightx, righty)
        out_img, left_fit_pixel, right_fit_pixel = self.plot(binary_warped.shape, leftx, lefty, rightx, righty, out_img)
        vehicle_position = self.find_vehicle_position(binary_warped.shape, left_fit_pixel, right_fit_pixel)
        
        result = {
            "image":out_img,
            "left_fit_meter" : left_fit_meter,
            "right_fit_meter" : right_fit_meter,
            "left_fit_pixel" : left_fit_pixel,
            "right_fit_pixel": right_fit_pixel,
            "left_radius": left_radius,
            "right_radius" : right_radius,
            "vehicle_position" : vehicle_position
        }

        return result
