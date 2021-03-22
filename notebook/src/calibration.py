import cv2
import glob
import numpy as np
import pickle

class Calibration:
    """
    This class will provide the following functionality:
    - detect chessboard corners
    - calibrate the camera
    - return the camera Matrix and distortion coeff
    """
    def __init__(self, images_path='../camera_cal/calibration*.jpg', board_shape=(6,9)):
        self.object_points = np.zeros((6*9,3), np.float32)
        self.object_points[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)
        self.images_path = images_path
        self.board_shape = board_shape
        self.image_shape = (0,0)

    def detect_corner(self, image):
        return cv2.findChessboardCorners(image, self.board_shape,None)

    def read_calibration_images(self):
        return glob.glob(self.images_path)
    
    def find_points_corners(self):
        points = []
        corners = []
        images = self.read_calibration_images()
        for i, image_path in enumerate(images):
            image = cv2.imread(image_path)
            if i == 0:
                self.image_shape = image.shape[1::-1]

            found, image_corners = self.detect_corner(image)
            if found:
                points.append(self.object_points)
                corners.append(image_corners)
        
        return points, corners
    
    def calibrate_images(self):
        points, corners = self.find_points_corners()
        _, matrix, distortion_coef, _, _ = cv2.calibrateCamera(points, corners, self.image_shape, None, None)

        calibration_data = {
            "camera_matrix": matrix, 
            "distortion_coefficient": distortion_coef
        }
        
        pickle.dump(calibration_data, open("calibration_data.p", "wb" ))   
        
        return calibration_data
