import cv2
import numpy as np

import logging
LOG = logging.getLogger(__name__)

def remove_noise(fgmask):
    """
    To remove noise.

    Erode (makes the object bigger) to "swallow holes".
    then dilate (reduces the object) again.
    """
    LOG.debug("Removing noise.")

    LOG.debug("Eroding (making it smaller).")
    erode_kernel_size = max(fgmask.shape[:2])/150
    LOG.debug("Erode kernel size: '%s'."%(erode_kernel_size))
    ERODE_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erode_kernel_size,)*2)
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, ERODE_KERNEL)
    # cv2.imshow('eroded frame', fgmask)

    LOG.debug("Dilating (making it bigger again).")
    dilate_kernel_size = erode_kernel_size*3
    DILATE_KERNEL = np.ones((dilate_kernel_size,)*2, np.uint8)
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, DILATE_KERNEL, iterations=5)
    # cv2.imshow('dilated frame', fgmask)

    return fgmask
