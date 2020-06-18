import io
import PIL as pillow
import numpy as np
import matplotlib.pyplot as plt


class Inference:
    """ """

    def __init__(self):
        pass

    def show_image(self, image):
        """ """
        num_classes = 21
        mask = np.array(pillow.Image.open(io.BytesIO(image)))
        plt.imshow(mask, vmin=0, vmax=num_classes - 1, cmap='jet')
        plt.show()

    def infer(self, filename, predictor, show_image):
        """ """
        image = pillow.Image.open(filename)
        image.thumbnail([800, 600], pillow.Image.ANTIALIAS)
        image.save(filename, "JPEG")

        predictor.content_type = 'image/jpeg'
        predictor.accept = 'image/png'

        labelled_img = predictor.predict(image)

        if show_image is True:
            show_image(labelled_img)
