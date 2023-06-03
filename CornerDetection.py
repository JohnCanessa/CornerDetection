# **** imports ****
from matplotlib import pyplot as plt

from skimage import data
from skimage.feature import corner_harris, corner_subpix, corner_peaks
from skimage.transform import warp, AffineTransform
from skimage.draw import ellipse, circle_perimeter              # circle_perimeter was: circle


# **** checkerboard ****
checkerboard = data.checkerboard()                              # checkerboard is a 2D array of 0s and 1s

# **** display checkerboard image ****
plt.figure(figsize=(6, 6))
plt.imshow(checkerboard, cmap='gray')
plt.title('Checkerboard')
plt.show()


# **** transform (scale, rotate, shear, and translate image) ****
transform = AffineTransform(scale=(0.9, 0.8), 
                            rotation=1, 
                            shear=0.6,
                            translation=(150, -80))

# **** generate warped_checkerboard image  ****
warped_checkerboard = warp( checkerboard, 
                            transform, 
                            output_shape=(320, 320))

# **** display warped checker_board image ****
plt.figure(figsize=(6, 6))
plt.imshow(warped_checkerboard, cmap='gray')
plt.title('Warped Checkerboard')
plt.show()


# **** detect corners in warped checkerboard image ****
corners = corner_harris(warped_checkerboard)

# **** display corners in warped checkerboard image ****
plt.figure(figsize=(6, 6))
plt.imshow(corners, cmap='gray')
plt.title('Corners in Warped Checkerboard')
plt.show()


# **** find corners in the Harris response image -
#      the result is the coordinates of the corners ****
coords_peaks = corner_peaks(corners, 
                            min_distance=10)               # min_distance is the minimum number of pixels separating

# **** display coords_peaks.shape ****
print(f'coords_peaks.shape: {coords_peaks.shape}')


# **** statistical test to determine whether the corner is
#      classified as an intersection of two edges or a single peak ****
coords_subpix = corner_subpix(  warped_checkerboard,
                                coords_peaks,
                                window_size=10)

# **** display coords_subpix ****
print(f'coords_subpix[0:11]: {coords_subpix[0:11]}')

# **** display coords_peeks ****
print(f'coords_peaks[0:11]: {coords_peaks[0:11]}')

# **** note that the corners are a little different 
#      after we use the statistical estimation techniques ****


# ****
fig, ax = plt.subplots(figsize=(8, 8))
ax.imshow(  warped_checkerboard, 
            interpolation='nearest',
            cmap='gray')
ax.plot(coords_peaks[:, 1], coords_peaks[:, 0], '.b', markersize=30)        # markersize was: 30
ax.plot(coords_subpix[:, 1], coords_subpix[:, 0], '*r', markersize=10)

# **** display the image ****
plt.tight_layout()
plt.show()

# **** blue values are original corners
#      red values are the corners after statistical estimation ****

