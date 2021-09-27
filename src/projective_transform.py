import numpy as np
from matplotlib.path import Path
from imageio import imread, imwrite

from dlt_homography import dlt_homography
from bilinear_interp import bilinear_interp
from histogram_eq import histogram_eq

def projective_transform(target_points, template_points, target_file, template_file):
    """
    Parameters:
    -----------

    Returns:
    --------
    Ihack  - Hacked RGB intensity image, 8-bit np.array (i.e., uint8).
    """
    # Bounding box in Y & D Square image.
    bbox = np.array([[404, 490, 490, 404], [38,  38, 354, 354]])

    # Point correspondences. Formated to be more visually interpretable to a human.
    Iyd_pts = target_points

    Ist_pts = template_points

    # This creates a path corresponding to the exact location we wish to fill in
    # within YD image
    sign_path = Path(Iyd_pts.T)

    Iyd = imread(target_file)
    Ist = imread(template_file)

    Ihack = np.asarray(Iyd)
    Ist = np.asarray(Ist)

    # Let's do the histogram equalization first.
    Ist = histogram_eq(Ist)

    # Compute the perspective homography we need...
    # N.B. Since we have the pixels we wish to fill in within the
    # "warped image", we need the homography from YD to ST
    # This can be thought of as the "inverse" Homography Matrix
    H, _ = dlt_homography(Iyd_pts, Ist_pts)

    # Here I create a grid using the corners of the bounding box to set up
    # a collection of [x,y].T points to warp with the H matrix
    ydx, ydy = np.meshgrid(np.arange(404, 490), np.arange(38, 354))
    ydx, ydy = ydx.flatten(), ydy.flatten()

    # Bring the bounding box points into homogenous form so we can apply H vectorally
    bbpts = np.array([ydx, ydy, np.ones(ydy.size)])

    # Find all the points which actually correspond to the region we
    # are converting. Update the points we will need to apply H to
    valid = sign_path.contains_points(bbpts[0:2,:].T)
    bbpts = bbpts.T[valid, :].T

    # Apply the Homography to get the warped real tuple point of
    # the location on ST to use for intesity
    st_pts = np.dot(H, bbpts)

    # Need to normalize each point by last entry since it should be 1
    # Now everypoint is back in Homogenous form
    st_pts = st_pts[0:2, :] / st_pts[2, :]

    # Given my implementation of bilinear_interp, I need to feed it point by point
    # we iterate over each point aquired by warping bbpts, interpolating the intensity
    # of that point at ST, then assigning the output image's bbpts pixel with that intensity
    # Since ST is greyscale, I just assign each channel of YD to be that intensity
    for i, point in enumerate(st_pts.T):
        # This is normalized already
        pix = bbpts[0:2, i].astype(int)
        Ihack[pix[1], pix[0], 0] = bilinear_interp(Ist[:,:,0], point[:, None])
        Ihack[pix[1], pix[0], 1] = bilinear_interp(Ist[:,:,1], point[:, None])
        Ihack[pix[1], pix[0], 2] = bilinear_interp(Ist[:,:,2], point[:, None])

    # Some visualization code
    '''
    import matplotlib.pyplot as plt
    plt.imshow(Ihack)
    plt.show()
    '''
    return Ihack

if __name__ == "__main__":
    # Test Code
    # call the function and uncomment visualization code to inspect result
    # add prints as necessary
    target_points = np.array([
        [416, 485, 488, 410],
        [40,  61, 353, 349]
        ])

    # Mike
    template_points = np.array([
        [2, 748, 748, 2],
        [2, 2, 998, 998]
        ])

    target_file = '../billboard/yonge_dundas_square.jpg'
    template_file = '../billboard/mike.jpg'

    img = projective_transform(target_points, template_points, target_file, template_file)
    imwrite('../billboard/yonge_dundas_mike.png', img)
    print("File has been written to ../billboard/yonge_dundas_mike.png")
