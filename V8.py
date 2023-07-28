import numpy as np

import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm

def laplace(image):
    # Compute the Laplacian of the image
    laplacian = cv2.Laplacian(image, cv2.CV_64F)

    # Compute the variance of the Laplacian
    variance = np.var(laplacian)

    # Return the Blurriness Index
    return variance

# Define the function to optimize
def target_function(psf):
    # Example function: calculate the sum of the main diagonal
    psf = np.array(psf)
    # Convolve the image with the PSF
    blurred = cv2.filter2D(blurred_image, -1, psf)
    
    # Compute the Laplacian of the blurred image
    return -((laplace(blurred) - targetedVariance)**2)

# Define the hill climbing algorithm
def hill_climbing(start_matrix, step_size, max_iterations):
    # Initialize the current matrix
    current_matrix = start_matrix
    
    # Iterate until the maximum number of iterations is reached
    for i in tqdm(range(max_iterations)):
        # print(f"Generation {i}")
        # Generate a random neighbor within the step size
        neighbor = current_matrix + np.random.uniform(-step_size, step_size, size=(current_matrix.shape))
        
        # Check if the neighbor is a better point
        if target_function(neighbor) > target_function(current_matrix):
            current_matrix = neighbor
    
    # Return the best point found
    return current_matrix

def run_Hill(blurred_path, gtPaht):
    # Test the hill climbing algorithm
    n = 3
    start_matrix = np.random.rand(n, n)
    step_size = 0.1
    max_iterations = 500

    global blurred_image
    blurred_image = cv2.imread(blurred_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.imread(gtPaht, cv2.IMREAD_GRAYSCALE)

    global targetedVariance 
    targetedVariance = 600

    best_matrix = hill_climbing(start_matrix, step_size, max_iterations)
    print("Best matrix found:\n", best_matrix)
    print("Best function value found:", target_function(best_matrix))


    best_psf = np.array(best_matrix)
    best_psf = np.reshape(best_psf, (n, n))
    print(best_psf)
    best_psf = np.reshape(best_psf, (n, n))

    result_image = cv2.filter2D(blurred_image, -1, best_psf)

    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(8, 4))
    plt.gray()

    ax = axes[0]
    ax.imshow(image)
    ax.axis('off')
    ax.set_title(f'Original %.2f' %laplace(image))

    ax = axes[1]
    ax.imshow(blurred_image)
    ax.axis('off')
    ax.set_title('Blurred %.2f' %laplace(blurred_image))

    ax = axes[2]
    ax.imshow(result_image)
    ax.axis('off')
    ax.set_title('Deconvolved (RL) %.2f' %laplace(result_image))

    fig.tight_layout()
    # plt.show()

    cv2.imwrite(f"output_Hill.jpg", result_image)
    return result_image

# run_Hill()