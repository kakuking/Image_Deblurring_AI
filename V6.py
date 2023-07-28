import numpy as np
from scipy.signal import convolve2d
from skimage import color, data, restoration
import matplotlib.pyplot as plt
from tqdm import tqdm
import cv2

def compute_blurriness(image):
    # Compute the Laplacian of the image
    laplacian = cv2.Laplacian(image, cv2.CV_64F)

    # Compute the variance of the Laplacian
    variance = np.var(laplacian)

    # Return the Blurriness Index
    return variance

def richardson_lucy(image, psf, iterations):
    # Normalize the PSF
    psf /= psf.sum()

    # Initialize the estimated image with the blurred image
    estimate = np.copy(image)

    # Compute the Laplacian operator for Tikhonov regularization
    laplacian = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])

    # Iterate over the number of iterations
    for i in tqdm(range(iterations)):
        # print(f"Iteration {i}")
        # Estimate the blurred image from the current estimate and the PSF
        blurred = convolve2d(estimate, psf, 'same')

        blurred[blurred == 0] = 1e-8
        # Divide the observed image by the estimated blurred image
        ratio = np.divide(image, blurred)

        # Estimate the next iteration of the image by convolving the current estimate with the ratio
        estimate *= convolve2d(ratio, psf, 'same')

        # Apply Tikhonov regularization to reduce noise amplification
        estimate += 0.1 * convolve2d(estimate, laplacian, 'same')


    return estimate

# Load the blurred image
image = cv2.imread("gt.jpg", cv2.IMREAD_GRAYSCALE)
# image = color.rgb2gray(data.astronaut())
psf = np.ones((5, 5)) / 25
blurred = convolve2d(image, psf, 'same')

# Apply Richardson-Lucy algorithm for blind deconvolution
deconvolved_RL = richardson_lucy(blurred, psf, iterations=50)

# Display the results
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(8, 4))
plt.gray()

ax = axes[0]
ax.imshow(image)
ax.axis('off')
ax.set_title(f'Original %.2f' %compute_blurriness(image))

ax = axes[1]
ax.imshow(blurred)
ax.axis('off')
ax.set_title('Blurred %.2f' %compute_blurriness(blurred))

ax = axes[2]
ax.imshow(deconvolved_RL)
ax.axis('off')
ax.set_title('Deconvolved (RL) %.2f' %compute_blurriness(deconvolved_RL))

fig.tight_layout()
plt.show()


