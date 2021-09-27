import numpy as np
from numpy.linalg import inv

def _interpolate_surface(pt, F, p00, p11):
    """
    Helper to perform Bilinear interpolation

    Args:
        - pt - np.array of shape (2,1) entries may be float
        - F - np.array of shape (2, 2) entries represent the pixel intensities for a given region near pt
        - p00 - Diagonal point (y0, x0) corresponding to entry F[0,0]
        - p11 - Similar to p00, corresponding to entry F[1,1]

    Returns:
        - float of the interpolated intensity at point pt
    """

    y, x = pt[:, 0] # n.b. y is down/rows, x is right/cols

    y0, x0 = p00
    y1, x1 = p11

    """
    Here we define Y which can be thought of as the normalized y coordinate
    This accounts for changes in y, when multiplied by F on the right

    note that when y0==y1 we explicity remove the effect of y (since there is no variation in that direciton)
    we scale it back by 2 to avoid double counting intensities in y
    """

    if y0 != y1:
        Y = np.array([
            [y1 - y],
            [y - y0]
        ])/(y1-y0)
    else:
        Y = np.array([
            [1],
            [1]
        ])/2


    """
    Similar to the previous comment except in the x direction
    """

    if x0 != x1:
        X = np.array([
            [x1 - x, x - x0]
        ])/(x1 - x0)
    else:
        X = np.array([
            [1, 1]
        ])/2

    # The following dot product will produce the interpolated scalar
    # This is a result from the repeated linear interpolation formulation
    # see https://en.wikipedia.org/wiki/Bilinear_interpolation

    return np.dot(np.dot(X, F),Y).item()


def bilinear_interp(I, pt):
    """
    Performs bilinear interpolation for a given image point.

    Given the (x, y) location of a point in an input image, use the surrounding
    4 pixels to conmpute the bilinearly-interpolated output pixel intensity.

    Note that images are (usually) integer-valued functions (in 2D), therefore
    the intensity value you return must be an integer (use round()).

    This function is for a *single* image band only - for RGB images, you will
    need to call the function once for each colour channel.

    Parameters:
    -----------
    I   - Single-band (greyscale) intensity image, 8-bit np.array (i.e., uint8).
    pt  - 2x1 np.array of point in input image (x, y), with subpixel precision.

    Returns:
    --------
    b  - Interpolated brightness or intensity value (whole number >= 0).
    """

    if pt.shape != (2, 1):
        raise ValueError('Point size is incorrect.')

    """
    To conduct bilinear interpolation we need to find the pixels which border our point
    We do this by finding the diagonal of the pixels through the floor and ceil of our given point
    We can then construct the off diagonals using points from the diagonals
    """

    p00 = np.floor(pt[:,0]).astype(int) # top left (x0, y0)
    p11 = np.ceil(pt[:,0]).astype(int)  # bot right (x1, y1)

    p01 = np.array([p00[0], p11[1]]) # (x0, y1)
    p10 = np.array([p11[0], p00[1]]) # (x1, y0)

    """
    N.B. The x corresponds to right/cols while y corresponds to down/rows
    Thus we construct our intensity matrix below with this in mind
    """
    F = np.array([
        [I[p00[1],p00[0]], I[p10[1],p10[0]]],
        [I[p01[1],p01[0]], I[p11[1],p11[0]]],
    ])
    return round(_interpolate_surface(pt, F, p00, p11))

# Test Code below
if __name__ == "__main__":

    from imageio import imread
    b = bilinear_interp(np.asarray(imread('../billboard/yonge_dundas_square.jpg'))[:,:,0], np.array([[100.9],[400]]))
    print(b)
