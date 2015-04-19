import cv2

import logging
LOG = logging.getLogger(__name__)
ERODE_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
DILATE_KERNEL = np.ones((7, 7), np.uint8)


def remove_noise(fgmask):
    """
    To remove noise.

    Erode (makes the object bigger) to "swallow holes".
    then dilate (reduces the object) again.
    """
    LOG.debug("Removing noise.")
    LOG.debug("Eroding")
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, ERODE_KERNEL)

    LOG.debug("Dilating.")
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, DILATE_KERNEL, iterations=5)
    return fgmask



