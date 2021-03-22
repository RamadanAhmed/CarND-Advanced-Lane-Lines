import matplotlib.pyplot as plt
class Visualizer:
    """
    This class will provide the following functionality:
    """
    def show_images(self, imgs, per_row = 3, per_col = 2, W = 10, H = 5, tdpi = 80):    
        fig, ax = plt.subplots(per_col, per_row, figsize = (W, H), dpi = tdpi, constrained_layout=True)
        ax = ax.ravel()

        for i in range(len(imgs)):
            img = imgs[i]
            ax[i].imshow(img)
        
        for i in range(per_row * per_col):
            ax[i].axis('off')