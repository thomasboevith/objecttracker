import cv2

import logging
LOG = logging.getLogger(__name__)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))


def remove_noise(fgmask):
    """
    Erode, then dilate. To remove noise.
    """
    LOG.debug("Removing noise.")
    return cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)

