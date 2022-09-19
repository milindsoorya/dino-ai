import numpy as np


def PSNR(gt, img):
    '''
    Compute PSNR.
    Parameters
    ----------
    gt: array
        Ground truth image.
    img: array
        Predicted image.''
    '''
    mse = np.mean(np.square(gt - img))
    return 20 * np.log10(np.max(gt)-np.min(gt)) - 10 * np.log10(mse)
