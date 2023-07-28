import tensorflow as tf
# from tensorflow.keras import layers
from tensorflow.python.keras import layers

import cv2

import os
from PIL import Image
import numpy as np

from sklearn.model_selection import train_test_split


def laplace(image):
    # Compute the Laplacian of the image
    laplacian = cv2.Laplacian(image, cv2.CV_64F)

    # Compute the variance of the Laplacian
    variance = np.var(laplacian)

    # Return the Blurriness Index
    return variance

def get_dataset(blurredPath, save):
    dataset_folder_blur = blurredPath
    # dataset_folder_sharp = deblurredPath
    # image_size = (x, y)

    # Create empty lists to store the preprocessed images and their corresponding labels
    blurred_images = []
    deblurred_images = []

    # Loop through all the JPEG images in the dataset folder
    print("\n\nBlurred: ")
    for filename in os.listdir(dataset_folder_blur):
        if filename.endswith(".jpg"):
            print("file: " + filename)
            # Load the image
            image = cv2.imread(os.path.join(dataset_folder_blur, filename), cv2.IMREAD_GRAYSCALE)
            
            # image = np.reshape(image, (512, 512))
            
            # Resize the image to the desired size
            image = cv2.resize(image, (512, 512))

            # Convert the image to RGB color space
            # image = image.convert('L')

            # Append the preprocessed image and its label to the lists
            blurred_images.append(image)
            # break
    

    print(blurred_images.count)

    print("\n\nSharp: ")
    for filename in os.listdir('./sharp'):
        if filename.endswith(".jpg"):
            print("file: " + filename)
            # Load the image
            image = cv2.imread(os.path.join('./sharp', filename), cv2.IMREAD_GRAYSCALE)

            # Convert the image to RGB color space
            lap = laplace(image)

            # Append the preprocessed image and its label to the lists
            deblurred_images.append(lap)


    train_images, test_images, train_targets, test_targets = train_test_split(blurred_images, deblurred_images, test_size=0.2, random_state=42)
    train_images, val_images, train_targets, val_targets = train_test_split(train_images, train_targets, test_size=0.2, random_state=42)

    train_images = np.array(train_images)
    val_images = np.array(val_images)
    test_images = np.array(test_images)

    train_targets =  np.array(train_targets)
    val_targets =  np.array(val_targets)
    test_targets =  np.array(test_targets)

    np.save(save + '/train_images.npy', train_images)
    np.save(save + '/val_images.npy', val_images)
    np.save(save + '/test_images.npy', test_images)
    # np.save(save + '/train_targets.npy', train_targets)
    # np.save(save + '/val_targets.npy', val_targets)
    # np.save(save + '/test_targets.npy', test_targets)

def optimization_algorithm(candidate_sharp_image, ground_truth_sharp_image, num_iterations=100, learning_rate=0.01):
    # Define the loss function
    loss_fn = tf.keras.losses.MeanSquaredError()

    # Convert the candidate and ground truth images to tensors
    candidate_sharp_image = tf.convert_to_tensor(candidate_sharp_image)
    ground_truth_sharp_image = tf.convert_to_tensor(ground_truth_sharp_image)

    # Create a variable to store the optimized sharp image
    optimized_sharp_image = tf.Variable(candidate_sharp_image)

    # Apply gradient descent to optimize the sharp image
    for i in range(num_iterations):
        with tf.GradientTape() as tape:
            # Compute the loss between the candidate and ground truth images
            loss = loss_fn(ground_truth_sharp_image, optimized_sharp_image)

        # Compute the gradient of the loss with respect to the sharp image
        grads = tape.gradient(loss, optimized_sharp_image)

        # Update the sharp image using the gradient and learning rate
        optimized_sharp_image.assign_sub(learning_rate * grads)

    # Convert the optimized sharp image back to an array
    optimized_sharp_image = optimized_sharp_image.numpy()

    return optimized_sharp_image

def create_model(num_epochs, dataSetPath, modelPath):

    train_images = np.load(dataSetPath + '/train_images.npy')
    train_targets = np.load(dataSetPath + '/train_targets.npy')
    test_images = np.load(dataSetPath + '/test_images.npy')
    test_targets = np.load(dataSetPath + '/test_targets.npy')
    val_images = np.load(dataSetPath + '/val_images.npy')
    val_targets = np.load(dataSetPath + '/val_targets.npy')

    # train_images = np.expand_dims(train_images, axis=-1)
    # test_images = np.expand_dims(test_images, axis=-1)
    # train_images = np.expand_dims(train_images, axis=-1)

# Define the neural network architecture
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(512, 512, 1)),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(1)
    ])

    # Compile the model
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])

    # Train the model
    model.fit(train_images, train_targets, epochs=num_epochs, batch_size=8, validation_data=(val_images, val_targets))

    # Evaluate the model
    test_loss, test_mae = model.evaluate(test_images, test_targets, verbose=2)
    print('Test MAE:', test_mae)
    model.save(modelPath)
    print("SAVED")
    model.summary()

def deblur_image(imagePath, modelPath, x, y):
    model = tf.keras.models.load_model(modelPath)

    blurry_image = Image.open(imagePath)
    # blurry_image = blurry_image.resize((x, y))  # resize the image
    blurry_image = blurry_image[0:x, 0:y]
    blurry_image = blurry_image.convert("RGB")  # convert to RGB color space
    blurry_image = np.array(blurry_image) / 255.0  # normalize pixel values to 0-1 range
    blurry_image = np.expand_dims(blurry_image, axis=0) 

    # Generate a candidate sharp image using the trained model
    candidate_sharp_image = model.predict(blurry_image)

    # Use a search algorithm to optimize the candidate sharp image
    optimized_sharp_image = optimization_algorithm(candidate_sharp_image, blurry_image)

    # Convert the deblurred image to a PIL Image object
    deblurred_image = (optimized_sharp_image[0] * 255.0).astype(np.uint8)
    deblurred_image = Image.fromarray(deblurred_image)

    # Save the deblurred image
    deblurred_image.save("output.jpg")