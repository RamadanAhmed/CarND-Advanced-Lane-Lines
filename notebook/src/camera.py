import cv2
import numpy as np

class Camera:
    """
    This class will provide the following functionality:
    - convert image to undistort image
    - convert image from car view to bird view
    - convert image from bird view to car view
    """
    def __init__(self, calibration_dict, car_points, bird_points) -> None:
        self.camera_matrix = calibration_dict['camera_matrix']
        self.distortion_coefficient = calibration_dict['distortion_coefficient']
        car_points_np = np.array(car_points, np.float32)
        bird_points_np = np.array(car_points, np.float32)
        self.bird_view_matrix = cv2.getPerspectiveTransform(car_points_np, bird_points_np)
        self.car_view_matrix  = cv2.getPerspectiveTransform(bird_points_np, car_points_np)

    def get_car_view(self, bird_view_image):
        img_size = (bird_view_image.shape[1], bird_view_image.shape[0])
        warped = cv2.warpPerspective(bird_view_image, self.car_view_matrix, img_size, flags=cv2.INTER_LINEAR)
        return warped

    def get_bird_view(self, car_view_image):
        img_size = (car_view_image.shape[1], car_view_image.shape[0])
        warped = cv2.warpPerspective(car_view_image, self.bird_view_matrix, img_size, flags=cv2.INTER_LINEAR)
        return warped

    def get_undistort_image(self, dist_image):
        undist = cv2.undistort(dist_image, self.camera_matrix, self.distortion_coefficient, None, self.camera_matrix)
        return undist