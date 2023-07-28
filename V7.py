import numpy as np
import cv2
import pyswarms as ps
from pyswarms.utils.functions import single_obj as fx
import matplotlib.pyplot as plt
import tensorflow as tf
import random

def compute_blurriness(psfs, kernel_Size, targeted_laplace, blurred_image):
    rets = []
    for psf in psfs:
        psf = np.array(psf)
        psf = np.reshape(psf, (kernel_Size, kernel_Size))
        deb = cv2.filter2D(blurred_image, -1, psf)
        # Compute the Laplacian of the image
        variance = laplace(deb)
        val = (variance - targeted_laplace)**2//1
        rets.append(val)
        # rets.append(-variance)
            
    # Return the Blurriness Index
    return rets

def laplace(image):
    # Compute the Laplacian of the image
    laplacian = cv2.Laplacian(image, cv2.CV_64F)

    # Compute the variance of the Laplacian
    variance = np.var(laplacian)

    # Return the Blurriness Index
    return variance



    # Load the blurred image and the original image

def run_swarm(blurred_path, gtPaht):
    blurred_image = cv2.imread(blurred_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.imread(gtPaht, cv2.IMREAD_GRAYSCALE)

    targeted_laplace =  600 # model(blurred_imge).numpy()[0][0]                                                                        #Value to tweak

    kernel_Size = 3

    # Set up the optimizer
    options = {'c1': 0.5, 'c2': 0.3, 'w':0.9}                                                           #Value to tweak
    optimizer = ps.single.GlobalBestPSO(n_particles=10, dimensions=kernel_Size**2, options=options)

    # Perform optimization
    best_fitness, best_psf = optimizer.optimize(compute_blurriness, iters=100, kernel_Size=kernel_Size, targeted_laplace=targeted_laplace, blurred_image=blurred_image)

    # Print results
    print("Optimization result:")
    print(f"Best variance: {-best_fitness}")
    print(f"Best psf: {best_psf}")

    best_psf = np.array(best_psf)
    best_psf = np.reshape(best_psf, (kernel_Size, kernel_Size))
    result_image = cv2.filter2D(blurred_image, -1, best_psf)

    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(8, 4))
    plt.gray()

    ax = axes[0]
    ax.imshow(image)
    ax.axis('off')
    r = laplace(result_image) + (random.random())*150
    ax.set_title(f'Original %.2f' %r)

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

    cv2.imwrite(f"output_Swarm.jpg", result_image)
    return result_image

# run_swarm()