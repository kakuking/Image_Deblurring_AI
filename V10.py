import math
import random
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm

# Define the function to optimize
def objective_function(psf, blurred_image):
    # calculate the sum of squared differences of each row and column
    psf = psf/np.sum(psf)
    psf = np.array(psf)
    # Convolve the image with the PSF
    blurred = cv2.filter2D(blurred_image, -1, psf)
    
    # Compute the Laplacian of the blurred image
    return laplace(blurred)
    # return -((laplace(blurred) - targetedVariance)**2)

def laplace(image):
    # Compute the Laplacian of the image
    laplacian = cv2.Laplacian(image, cv2.CV_64F)

    # Compute the variance of the Laplacian
    variance = np.var(laplacian)

    # Return the Blurriness Index
    return variance

def differential_evolution(objective_function, bounds,blurred_image, pop_size=50, max_iter=1000, F=0.8, CR=0.7):
    # Initialize the population
    population = np.random.uniform(bounds[0], bounds[1], size=(pop_size, 3, 3))

    for it in population:
        it = it/np.sum(it)

    # Iterate for the specified number of generations
    for i in tqdm(range(max_iter)):
        # Generate mutant vectors
        mutants = np.zeros_like(population)
        for j in range(pop_size):
            indices = np.random.choice(pop_size, size=3, replace=False)
            mutants[j] = population[indices[0]] + F * (population[indices[1]] - population[indices[2]])

        # Clip mutant vectors to bounds
        mutants = np.clip(mutants, bounds[0], bounds[1])

        # Crossover with the population to generate trial vectors
        trials = np.zeros_like(population)
        for j in range(pop_size):
            mask = np.random.rand(3, 3) < CR
            mask[np.random.randint(0, 3), np.random.randint(0, 3)] = True
            trials[j] = np.where(mask, mutants[j], population[j])

        # Evaluate the trial vectors and update the population
        costs = np.array([objective_function(matrix, blurred_image) for matrix in population])
        trial_costs = np.array([objective_function(matrix, blurred_image) for matrix in trials])
        mask = trial_costs < costs
        population[mask] = trials[mask]
        population[mask] = population[mask]/np.sum(population[mask])

    # Return the best solution found
    best_index = np.argmax(np.array([objective_function(matrix, blurred_image) for matrix in population]))
    return population[best_index], objective_function(population[best_index], blurred_image)

def run_Differential(blurred_path, gtPaht):
    bounds = (-10, 10)

    blurred_image = cv2.imread(blurred_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.imread(gtPaht, cv2.IMREAD_GRAYSCALE)
    targetedVar = 600


    best_psf, best_cost = differential_evolution(objective_function, bounds, blurred_image, max_iter=20)

    print("Best state found: ")
    print(best_psf)
    print("Best cost found: ", best_cost)

    best_psf = np.asarray(best_psf)
    best_psf = best_psf/np.sum(best_psf)
    # best_psf = np.random.uniform(low=0.1, high=1, size=(3, 3))
    print(best_psf)
    best_psf = np.reshape(best_psf, (3, 3))
    result_image = cv2.filter2D(blurred_image, -1, best_psf)

    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(8, 4))
    plt.gray()

    ax = axes[0]
    ax.imshow(image)
    ax.axis('off')
    # r = laplace(result_image) + (random.random())*150
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

    cv2.imwrite(f"output_Differential.jpg", result_image)
    return result_image

# run_Differential()