""" Common utilities. """

# Logging
# =======

import logging
import os, os.path
from colorlog import ColoredFormatter

ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)

formatter = ColoredFormatter(
    "%(log_color)s[%(asctime)s] %(message)s",
#    datefmt='%H:%M:%S.%f',
    datefmt=None,
    reset=True,
    log_colors={
        'DEBUG':    'cyan',
        'INFO':     'white,bold',
        'INFOV':    'cyan,bold',
        'WARNING':  'yellow',
        'ERROR':    'red,bold',
        'CRITICAL': 'red,bg_white',
    },
    secondary_log_colors={},
    style='%'
)
ch.setFormatter(formatter)

log = logging.getLogger('videocap')
log.setLevel(logging.DEBUG)
log.handlers = []       # No duplicated handlers
log.propagate = False   # workaround for duplicated logs in ipython
log.addHandler(ch)

logging.addLevelName(logging.INFO + 1, 'INFOV')
def _infov(self, msg, *args, **kwargs):
    self.log(logging.INFO + 1, msg, *args, **kwargs)
logging.Logger.infov = _infov


# Image Processing Routine
# ========================

import numpy as np
import scipy.misc

imresize = scipy.misc.imresize

def imread(path, gray=False):
    image = scipy.misc.imread(path, gray)

    # convert grayscale into color image
    if not gray and len(image.shape) == 2:
        image = np.tile(image[:, :, np.newaxis], (1, 1, 3))
    return image

def imcrop_and_resize(img, H=224, W=224):
    img = img / 255.0
    assert (0 <= img).all() and (img <= 1.0).all()
    # we crop image from center
    short_edge = min(img.shape[:2])
    yy = int((img.shape[0] - short_edge) / 2)
    xx = int((img.shape[1] - short_edge) / 2)
    crop_img = img[yy : yy + short_edge, xx : xx + short_edge]
    resized_img = imresize(crop_img, (H, W))
    return resized_img

def imshow(a, title=None):
    import matplotlib.pyplot as plt
    plt.imshow(a)
    if title: plt.title(title)



# Maptlotlib
# ==========

from matplotlib.colors import Normalize

# http://stackoverflow.com/questions/20144529/shifted-colorbar-matplotlib
class MidpointNormalize(Normalize):
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        # I'm ignoring masked values and all kinds of edge cases to make a
        # simple example...
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))

# Etc
# ===

def mkdir_p(path):
    if not os.path.exists(path):
        os.mkdir(path)
    return path

def in_ipython_notebook():
    try:
        import ipykernel.zmqshell
        return isinstance(get_ipython(), ipykernel.zmqshell.ZMQInteractiveShell)
    except:
        return False

__all__ = (
    'log',
    'imread', 'imresize', 'imshow', 'imcrop_and_resize',
    'MidpointNormalize',
    'mkdir_p', 'in_ipython_notebook',
)
