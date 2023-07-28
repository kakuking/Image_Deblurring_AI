import cv2
from V7 import run_swarm
from V8 import run_Hill
from V9 import run_genetic
from V10 import run_Differential

import numpy as np

from skimage.metrics import structural_similarity as ssim

from os import listdir
from os.path import isfile, join
onlyfiles = [f for f in listdir('./inputs') if isfile(join('./inputs', f))]

for i in range(len(onlyfiles)):
    onlyfiles[i] = './inputs/' + onlyfiles[i]

import pandas as pd

print(onlyfiles)

# get_dataset(r'C:\Users\karti\Desktop\Studies\AI\Project\motion_blurred', r'C:\Users\karti\Desktop\Studies\AI\Project\sharp', 256, 256, r'./256/motion')

def laplace(image):
    # Compute the Laplacian of the image
    laplacian = cv2.Laplacian(image, cv2.CV_64F)

    # Compute the variance of the Laplacian
    variance = np.var(laplacian)

    # Return the Blurriness Index
    return variance


df = pd.DataFrame(columns=['SSIM None', 'Laplace None', 'SSIM Swarm', 'Laplace Swarm', 'SSIM Hill', 'Laplace Hill', 'SSIM Genetic', 'Laplace Genetic', 'SSIM Differential', 'Laplace Differential'])
i = 0
everyThing = []
while i < len(onlyfiles):
    blurred = onlyfiles[i]
    sharp = onlyfiles[i+1]
    i += 2
    print("\nSwarm: ")
    swarm = run_swarm(blurred, sharp)
    print("\nHill: ")
    hill = run_Hill(blurred, sharp)
    print("\nGenetic: ")
    genetic = run_genetic(blurred, sharp)
    print("\nDifferential: ")
    differential = run_Differential(blurred, sharp)

    image = cv2.imread(sharp, cv2.IMREAD_GRAYSCALE)
    ssim_none = ssim(image, image)
    ssim_Swarm = ssim(image, swarm)
    ssim_Hill = ssim(image, hill)
    ssim_genetic = ssim(image, genetic)
    ssim_diff = ssim(image, differential)

    everyThing.append(f"SSIMs are: {ssim_none}:{laplace(image)}, {ssim_Swarm}:{laplace(swarm)}, {ssim_Hill}:{laplace(hill)}, {ssim_genetic}:{laplace(genetic)}, {ssim_diff}:{laplace(differential)}")
    df = df.append({'SSIM None': ssim_none, 'Laplace None': laplace(image), 'SSIM Swarm': ssim_Swarm, 'Laplace Swarm': laplace(swarm), 'SSIM Hill': ssim_Hill, 'Laplace Hill': laplace(hill), 'SSIM Genetic': ssim_genetic, 'Laplace Genetic': laplace(genetic), 'SSIM Differential': ssim_diff, 'Laplace Differential': laplace(differential)}, ignore_index=True)
    # print(everyThing[-1])

print("\n\n")
df.to_csv('results.csv', index=False)