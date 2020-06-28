import io
import numpy as np
import matplotlib.pyplot as plt
import PIL as pillow
import settings


class Utils:
    """ """
    def __init__(self):
        pass

    def show_image(self, image):
        """ """
        plt.imshow(image, vmin=0, vmax=settings.HYPER['num_classes'] - 1, cmap='jet')
        plt.show()
