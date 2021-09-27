import numpy as np
from numpy.linalg import inv, norm
from scipy.linalg import null_space

def normalize(xi):
    '''
    Args:
        xi - 2xN array of points with x in row 0 and y in row 1
    Returns:
        Homogenous, Normalized xi and Similarity transform T

    '''
    # determine centroid
    xm = np.mean(xi, axis=1)

    s = np.mean(np.sqrt(np.sum(np.square(xi - xm[:, None]), axis=0)))/np.sqrt(2)

    T = np.array([
        [1/s, 0, -xm[0]/s],
        [0, 1/s, -xm[1]/s],
        [0, 0, 1],
    ])


    # Convert to Homogenous form
    Xi = np.vstack([xi, np.ones(xi.shape[1])])

    return np.dot(T, Xi), T

def dlt_homography(I1pts, I2pts):
    """
    Find perspective Homography between two images.

    Given 4 points from 2 separate images, compute the perspective homography
    (warp) between these points using the DLT algorithm.

    Parameters:
    -----------
    I1pts  - 2x4 np.array of points from Image 1 (each column is x, y).
    I2pts  - 2x4 np.array of points from Image 2 (in 1-to-1 correspondence).

    Returns:
    --------
    H  - 3x3 np.array of perspective homography (matrix map) between image coordinates.
    A  - 8x9 np.array of DLT matrix used to determine homography.
    """

    """
    To perform DLT, we simply just need to normalize the given points, construct matrix A, determine the nullspace of A (assuming A is overdetermined) use the single vector of said null space to construct our normalized Homography matrix, apply similarity transforms to create the correctly scaled Homography transform.

    We start here by initializing A with all zeros with the correct shape
    """
    A = np.zeros((8,9))

    """
    DLT works better with normalized points, apply our function and save the similarity matrix
    """
    X_til, T = normalize(I1pts)
    Xp_til, Tp = normalize(I2pts)
    for i in range(X_til.shape[1]):
    # Exhaust the points we have, and assume we have equal numbers of both
        # Each point contributes two rows to A, hence must index by 2 each point
        A[i*2, 0:3] = -X_til[:, i]
        A[i*2, 6:9] = Xp_til[0, i]*X_til[:, i]
        A[i*2+1, 3:6] = -X_til[:, i]
        A[i*2+1, 6:9] = Xp_til[1, i]*X_til[:, i]

    # As per Elan, the null space of A will be a vector which is our homograpy
    h = null_space(A)
    if h.shape[1] != 1:
        raise("Null Space of A contains more than one vector, some points are collinear")

    # Note that A was made from normalized X and Xprime so H needs to
    # have its similarity transformations applied to scale it properly
    H = np.matmul(inv(Tp), np.matmul(h.reshape((3,3)), T))

    # as we return, divide by bot right most element as we would like it to be one
    # Homographies are defined up to scale, so any H which is scaled by a constant is
    # technically the same Homography
    return H/H[2,2], A

if __name__ == "__main__":
    """
    Test the dlt_homography function here
    """
    # Make some fake data
    # ensure points are not co-linear
    i1pts = np.array([
        [1,2.5,3.75,4.99,],
        [5,6,7,8,]
    ])

    i2pts = i1pts.copy()

    H, A = dlt_homography(i1pts, i2pts)
    # inspect homography matrix
    # it should be really close to Identity
    print(H)
