import cv2
import numpy as np
from numpy.core.numeric import binary_repr


class ImageFilter:
    """
    This class will provide the following functionality:
    - Convert Image from color space to another color space
    - Masking image
    - Convert image to binary coded
    """

    def __init__(self, threshold_dict) -> None:
        self.threshold_dict = threshold_dict

    def color_filter(self, image):
        hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
        s_channel = hls[:, :, 2]
        l_channel = hls[:, :, 1]
        s_thresh_min = self.threshold_dict["color2_min_threshold"]
        s_thresh_max = self.threshold_dict["color2_max_threshold"]
        l_thresh_min = self.threshold_dict["color1_min_threshold"]
        l_thresh_max = self.threshold_dict["color1_max_threshold"]
        
        s_binary = np.zeros_like(s_channel)
        s_binary[(s_channel >= s_thresh_min) & (s_channel <= s_thresh_max)
                &(l_channel >= l_thresh_min) & (l_channel <= l_thresh_max)] = 1

        return s_binary

    def sobel_magnitude_threshold(self, sobel):
        abs_sobel = np.absolute(sobel)
        # 4) Scale to 8-bit (0 - 255) then convert to type = np.uint8
        scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
        # 5) Create a mask of 1's where the scaled gradient magnitude
        # is > thresh_min and < thresh_max

        thresh_min = self.threshold_dict["magnitude_min_threshold"]
        
        binary_output = np.zeros_like(scaled_sobel)
        binary_output[(scaled_sobel >= thresh_min)] = 1        
        return binary_output

    def sobel_absolute_threshold(self, sobelx, sobely):
        # Calculate the gradient magnitude
        gradmag = np.sqrt(sobelx**2 + sobely**2)
        # Rescale to 8 bit
        scaled_mag = np.uint8(255*gradmag/np.max(gradmag))
        # Create a binary image of ones where threshold is met, zeros otherwise
        binary_output = np.zeros_like(scaled_mag)

        thresh_min = self.threshold_dict["absolute_min_threshold"]
        
        binary_output[(scaled_mag >= thresh_min)] = 1
        # Return the binary image
        return binary_output

    def sobel_direction_threshold(self, sobelx, sobely):
        abs_sobelx = np.absolute(sobelx)
        abs_sobely = np.absolute(sobely)
        
        direction = np.arctan2(abs_sobely, abs_sobelx)
        
        binary_output = np.zeros_like(direction)

        thresh_min = self.threshold_dict["direction_min_threshold"]
        thresh_max = self.threshold_dict["direction_max_threshold"]

        binary_output[(direction >= thresh_min) & (direction <= thresh_max)] = 1
        
        return binary_output

    def sobel_filter(self, image):
        hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
        l_channel = hls[:, :, 1]
        sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0,
                           ksize=self.threshold_dict['kernal_size'])  # Take the derivative in x
        sobely = cv2.Sobel(l_channel, cv2.CV_64F, 0, 1,
                           ksize=self.threshold_dict['kernal_size'])  # Take the derivative in y
        grad_binary = self.sobel_absolute_threshold(sobelx, sobely)
        mag_binary = self.sobel_magnitude_threshold(sobelx)
        dir_binary = self.sobel_direction_threshold(sobelx, sobely)

        combined = np.zeros_like(dir_binary)
        combined[(grad_binary == 1) & (mag_binary == 1)
                 & (dir_binary == 1)] = 1
        return combined

    def get_binary_image(self, image):
        sxbinary = self.sobel_filter(image)
        s_binary = self.color_filter(image)
        # Combine the two binary thresholds
        combined_binary = np.zeros_like(sxbinary)
        
        combined_binary[(s_binary == 1) | (sxbinary == 1)] = 1

        filtered_img = np.dstack(( combined_binary, np.zeros_like(sxbinary), np.zeros_like(sxbinary))) * 255

        return combined_binary, filtered_img
