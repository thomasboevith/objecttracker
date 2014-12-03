import cv2
import numpy as np

import logging
LOG = logging.getLogger(__name__)


def find_contours(fgmask):
    """
    Finds the contours.
    """
    contours, hierarchy = cv2.findContours(fgmask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    return contours


def label_frame(contours, fgmask):
    """
    Create a labelled frame.
    """
    label = 0
    for contour in contours:
        label += 1
        cv2.drawContours(fgmask, [contour], 0, label, -1)  # -1 fills the image.
    return fgmask
    

def get_labelled_frame(fgmask):
    """
    Inserts labels for a fgmask.
    """
    contours = find_contours(fgmask)
    LOG.debug("%i connected components."%(np.max(fgmask)))
    return label_frame(contours, fgmask)
