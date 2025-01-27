import cv2
import numpy as np
from .visualizer import Visualizer


class Camera:
    """
    This class will provide the following functionality:
    - convert image to undistort image
    - convert image from car view to bird view
    - convert image from bird view to car view
    """

    def __init__(self, calibration_dict, car_points = None, bird_points = None, do_prespective=True) -> None:
        self.camera_matrix = calibration_dict['camera_matrix']
        self.distortion_coefficient = calibration_dict['distortion_coefficient']
        self.do_prespective = do_prespective
        if do_prespective:        
            self.car_points = car_points
            self.bird_points = bird_points
            car_points_np = np.array(car_points, np.float32)
            bird_points_np = np.array(bird_points, np.float32)
            
            self.bird_view_matrix = cv2.getPerspectiveTransform(
                car_points_np, bird_points_np)
            self.car_view_matrix = cv2.getPerspectiveTransform(
                bird_points_np, car_points_np)
        self.visualizer = Visualizer()

    def get_car_view(self, bird_view_image, show_dotted = False):
        if self.do_prespective:
            img_size = (bird_view_image.shape[1], bird_view_image.shape[0])
            warped = cv2.warpPerspective(
                bird_view_image, self.car_view_matrix, img_size, flags=cv2.INTER_LINEAR)
            if show_dotted: 
                self.visualizer.show_dotted_image(warped, self.car_points)
            return warped
        return None
    
    def get_bird_view(self, car_view_image, show_dotted = False):
        if self.do_prespective:
            img_size = (car_view_image.shape[1], car_view_image.shape[0])
            warped = cv2.warpPerspective(
                car_view_image, self.bird_view_matrix, img_size, flags=cv2.INTER_LINEAR)
            
            if show_dotted: 
                self.visualizer.show_dotted_image(warped, self.bird_points)
            return warped
        return None
    
    def get_undistort_image(self, dist_image, show_dotted = False):
        undist = cv2.undistort(dist_image, self.camera_matrix,
                               self.distortion_coefficient, None, self.camera_matrix)

        if show_dotted: 
            self.visualizer.show_dotted_image(undist, self.car_points)
        return undist

    def get_projected_image(self, undist, warped, ploty, left_fitx, right_fitx):
        warp_zero = np.zeros_like(warped).astype(np.uint8)
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

        # Recast the x and y points into usable format for cv2.fillPoly()
        pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
        pts = np.hstack((pts_left, pts_right))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

        newwarp = self.get_car_view(color_warp)
        # Combine the result with the original image

        result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)
        return result