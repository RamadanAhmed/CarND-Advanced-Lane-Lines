import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np
import cv2

class Visualizer:
    """
    This class will provide the following functionality:
    """

    def show_images(self, imgs, per_row=3, per_col=2, W=10, H=5, tdpi=80):
        fig, ax = plt.subplots(per_col, per_row, figsize=(
            W, H), dpi=tdpi, constrained_layout=True)
        ax = ax.ravel()

        for i in range(len(imgs)):
            img = imgs[i]
            ax[i].imshow(img)

        for i in range(per_row * per_col):
            ax[i].axis('off')

    def save_images(self, prefix, imgs, imgs_name):
        for i, img in enumerate(imgs):
            cv2.imwrite('../output_images/{}_{}'.format(prefix, imgs_name[i]), img)

    def show_dotted_image(self, this_image, points, thickness=5, color=[255, 0, 255], d=15):
        image = this_image.copy()

        cv2.line(image, points[0], points[1], color, thickness)
        cv2.line(image, points[2], points[3], color, thickness)

        fig, ax = plt.subplots(1)
        ax.set_aspect('equal')
        ax.imshow(image)

        for (x, y) in points:
            dot = Circle((x, y), d)
            ax.add_patch(dot)

        plt.show()
