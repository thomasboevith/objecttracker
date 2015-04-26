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
    erode_kernel_size = min(fgmask.shape[:2])/150
    ERODE_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erode_kernel_size,)*2)
    DILATE_KERNEL = np.ones((erode_kernel_size*2, )*2, np.uint8)

    LOG.debug("Removing noise.")
    LOG.debug("Eroding")
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, ERODE_KERNEL)

    LOG.debug("Dilating.")
    fgmask = cv2.morphologyEx(fgmask,
                              cv2.MORPH_CLOSE,
                              DILATE_KERNEL,
                              iterations=5
                              )
    return fgmask
