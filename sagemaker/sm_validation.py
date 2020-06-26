import numpy as np


class Validation:
    """ Validation metrics for semantic segmentation results
            - Pixel Accuracy
            - Intersection-Over-Union (Jaccard Index)
            - Dice Coefficient (F1 Score)
    """
    def __init__(self):
        pass

    def iou(self, target, prediction):
        intersection = np.logical_and(target, prediction)
        union = np.logical_or(target, prediction)
        iou_score = np.sum(intersection) / np.sum(union)

        return iou_score

