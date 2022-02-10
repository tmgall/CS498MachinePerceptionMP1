import numpy as np
import matplotlib.pyplot as plt
from imageio import imread

# Load the image and plot the keypoints
im = imread('uiuc.png') / 255.0
keypoints_im = np.array([(604.593078169188, 583.1361439828671),
                       (1715.3135416380655, 776.304920238324),
                       (1087.5150188078305, 1051.9034760165837),
                       (79.20731171576836, 642.2524505093215)])

# print(keypoints_im)
plt.clf()
plt.imshow(im)
plt.scatter(keypoints_im[:, 0], keypoints_im[:, 1])
plt.plot(keypoints_im[[0, 1, 2, 3, 0], 0], keypoints_im[[0, 1, 2, 3, 0], 1], 'g')

for ind, corner in enumerate(keypoints_im):
    plt.text(corner[0] + 30.0, corner[1] + 30.0, '#'+str(ind), c='b', family='sans-serif', size='x-large')
plt.title("Target Image and Keypoints")
plt.show()

'''
Question 1: specify the corners' coordinates
Take point 3 as origin, the long edge as x axis and short edge as y axis
Output:
     - corners_court: a numpy array (4x2 matrix)
'''
# --------------------------- Begin your code here ---------------------------------------------

corners_court = np.array([(0, 15.24), (28.65, 15.24), (28.65, 0), (0, 0)])

# --------------------------- End your code here   ---------------------------------------------

'''
Question 2: complete the findHomography function
Arguments:
     pts_src - Each row corresponds to an actual point on the 2D plane (Nx2 matrix)
     pts_dst - Each row is the pixel location in the target image coordinate (Nx2 matrix)
Returns:
     H - The homography matrix (3x3 matrix)

Hints:
    - you might find the function vstack, hstack to be handy for getting homogenous coordinate;
    - you might find numpy.linalg.svd to be useful for solving linear system
    - directly calling findHomography in cv2 will receive zero point, but you could use it as sanity-check of your own implementation
'''
def findHomography(pts_src, pts_dst):
    n = pts_src.shape[0]
    A = np.zeros((2 * n, 9))
    for i in range(n):
        Xi = pts_src[i, 0]
        Yi = pts_src[i, 1]
        xi = pts_dst[i, 0]
        yi = pts_dst[i, 1]
        A[2 * i + 0, :] = np.array([Xi, Yi, 1,  0,  0, 0, -1 * xi * Xi, -1 * xi * Yi, -1 * xi])
        A[2 * i + 1, :] = np.array([ 0,  0, 0, Xi, Yi, 1, -1 * yi * Xi, -1 * yi * Yi, -1 * yi])
    U, sigma, V = np.linalg.svd(A)
    H = V[-1] / V[-1, -1]
    return H.reshape((3, 3))


# Calculate the homography matrix using your implementation
H = findHomography(corners_court, keypoints_im)

'''
Question 3.a: insert the logo virtually onto the state farm center image.
Specific requirements:
     - the size of the logo needs to be 3x6 meters;
     - the bottom left logo corner is at the location (23, 2) on the basketball court.
Returns:
     transform_target - The transformation matrix from logo.png image coordinate to target.png coordinate (3x3 matrix)

Hints:
     - Consider to calculate the transform as the composition of the two: H_logo_target = H_court_target @ H_logo_court
     - Given the banner size in meters and image size in pixels, could you scale the logo image coordinate from pixels to meters
     - What transform will move the logo to the target location?
     - Could you leverage the homography between basketball court to target image we computed in Q.2?
     - Image coordinate is y down ((0, 0) at bottom-left corner) while we expect the inserted logo to be y up, how would you handle this?
'''

# Read the banner image that we want to insert to the basketball court
logo = imread('logo.png') / 255.0
plt.clf()
plt.imshow(logo)
plt.title("Banner")
plt.show()

# --------------------------- Begin your code here ---------------------------------------------

court_points = np.array([(23, 5), (23, 2), (29, 2), (29, 5)])
logo_width = logo.shape[1]
logo_height = logo.shape[0]
logo_points = np.array([(0, 0), (0, logo_height), (logo_width, logo_height), (logo_width, 0)])
target_transform = H @ findHomography(logo_points, court_points)

# --------------------------- End your code here   ---------------------------------------------

'''
Question 3.b: compute the warpImage function
Arguments:
     image - the source image you may want to warp (Hs x Ws x 4 matrix, R,G,B,alpha)
     H - the homography transform from the source to the target image coordinate (3x3 matrix)
     shape - a tuple of the target image shape (Wt, Ht)
Returns:
     image_warped - the warped image (Ht x Wt x 4 matrix)

Hints:
    - you might find the function numpy.meshgrid and numpy.ravel_multi_index useful;
    - are you able to get rid of any for-loop over all pixels?
    - directly calling warpAffine or warpPerspective in cv2 will receive zero point, but you could use as sanity-check of your own implementation
'''


def warpImage(image, H, shape):
    image_warped = np.zeros((shape[1], shape[0], image.shape[2]))
    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            point = np.matmul(H, np.array([x, y, 1]).T)
            point /= point[2]
            image_warped[int(point[1]), int(point[0]), :] = image[y, x, :]
    return image_warped


# call the warpImage function
logo_warp = warpImage(logo, target_transform, (im.shape[1], im.shape[0]))

plt.clf()
plt.imshow(logo_warp)
plt.title("Warped Banner")
plt.show()

'''
Question 3.c: alpha-blend the warped logo and state farm center image

im = logo * alpha_logo + target * (1 - alpha_logo)

Hints:
    - try to avoid for-loop. You could either use numpy's tensor broadcasting or explicitly call np.repeat / np.tile
'''

alpha_logo = np.expand_dims(logo_warp[:, :, 3], axis=2)
im = logo_warp * alpha_logo + im * (1 - alpha_logo)

im *= 255.0
im = im.astype(np.uint8)

plt.clf()
plt.imshow(im)
plt.title("Blended Image")
plt.show()

# dump the results for autograde
outfile = 'solution_homography.npz'