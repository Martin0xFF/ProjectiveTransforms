import numpy as np

def histogram_eq(I, bitres=256):
    """
    Histogram equalization for greyscale image.

    Perform histogram equalization on the 8-bit greyscale intensity image I
    to produce a contrast-enhanced image J. Full details of the algorithm are
    provided in the Szeliski text.

    Parameters:
    -----------
    I  - Single-band (greyscale) intensity image, 8-bit np.array (i.e., uint8).
    bitres - integer resolution of the intensity of image, typically 256

    Returns:
    --------
    J  - Contrast-enhanced greyscale intensity image, 8-bit np.array (i.e., uint8).
    """

    if I.dtype != np.uint8:
        raise ValueError('Incorrect image format!')

    # for 8-bit images, 255 is max value, 0 is min value
    # store the total number of pixels in image
    N = I.size

    # create place holder for cumulative density function
    # over discrete pixel intensities
    cdf = np.zeros(bitres)

    # Leverage recursive def of cdf
    cdf[0] = np.sum(I==0)/N
    for i in range(1, bitres):
        cdf[i] = cdf[i-1] + np.sum(I==i)/N

    # Scale the CDF so that it ranges between 255 and
    # is a discrete value, so we can use it as an intensity
    # for normalization
    J = np.round(cdf[I]*(bitres-1)).astype(np.uint8)

    return  J
# test code below
if __name__ == "__main__":
    from imageio import imread, imwrite
    I = np.asarray(imread("../billboard/uoft_soldiers_tower_dark.png"))
    J = histogram_eq(I)
    imwrite("~/remove_me.png", J)
