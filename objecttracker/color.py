import numpy as np
import colorsys

def get_colors(num_colors):
    colors=[]

    for i in np.arange(0., 360., 360. / num_colors): 
        np.random.seed([i])
        hue = i/360.
        lightness = (50 + np.random.rand() * 10)/100.
        saturation = (90 + np.random.rand() * 10)/100.
        colors.append(tuple(i*255 for i in colorsys.hls_to_rgb(hue, lightness, saturation)))
    return colors
