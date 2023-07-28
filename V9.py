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

# Define the fitness function that takes a 3x3 matrix as input
def fitness_function(psf, blurred_image, targetedVariance):
    psf = np.array(psf)
    # Convolve the image with the PSF
    blurred = cv2.filter2D(blurred_image, -1, psf)
    
    # Compute the Laplacian of the blurred image
    return -((laplace(blurred) - targetedVariance)**2)

# Define the genetic algorithm function
def genetic_algorithm(population_size, generations, mutation_rate, blurred_image,targetedVar ):

    # Define the initial population randomly
    population = np.random.uniform(low=-1.0, high=1.0, size=(population_size, 3, 3))
    for i in tqdm(range(generations)):
        # Calculate the fitness of each individual in the population
        fitness_values = np.array([fitness_function(matrix, blurred_image, targetedVar) for matrix in population])
        # Select the best individuals to reproduce
        parents_indices = np.argsort(fitness_values)[::-1][:int(population_size/2)]
        parents = population[parents_indices]
        # Create new offspring by crossover
        offspring = np.empty_like(parents)
        for j in range(int(population_size/2)):
            # Select two parents at random
            parent1 = parents[np.random.randint(len(parents))]
            parent2 = parents[np.random.randint(len(parents))]
            # Create a new offspring by randomly selecting elements from the two parents
            mask = np.random.randint(2, size=parent1.shape).astype(np.bool)
            offspring[j][mask] = parent1[mask]
            offspring[j][~mask] = parent2[~mask]
        # Mutate some individuals
        mutation_mask = np.random.rand(*offspring.shape) < mutation_rate
        mutation = np.random.randint(0, 10, size=offspring.shape)
        offspring[mutation_mask] = mutation[mutation_mask]
        # Replace the worst individuals in the population with the new offspring
        worst_indices = np.argsort(fitness_values)[:int(population_size/2)]
        population[worst_indices] = offspring
    # Return the best individual in the final population
    final_fitness_values = np.array([fitness_function(matrix, blurred_image, targetedVar) for matrix in population])
    best_index = np.argmax(final_fitness_values)
    return population[best_index]

def run_genetic(blurred_path, gtPaht):
    blurred_image = cv2.imread(blurred_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.imread(gtPaht, cv2.IMREAD_GRAYSCALE)
    targetedVar = 600

    best_psf = genetic_algorithm(50, 50, 5, blurred_image, targetedVar)

    # print(best_psf[0])

    best_psf = np.asarray(best_psf)
    best_psf = np.reshape(best_psf, (3, 3))
    print(best_psf)
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

    cv2.imwrite(f"output_Genetic.jpg", result_image)
    return result_image

# run_genetic()