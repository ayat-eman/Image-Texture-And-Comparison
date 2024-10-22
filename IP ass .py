import cv2
from skimage import feature
import matplotlib.pyplot as plt
import numpy as np
import random
from sklearn.metrics.pairwise import cosine_similarity

# Function to compute LBP texture features with grayscale conversion and image resizing
def extract_lbp_texture(image, P=8, R=1):
    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Resize the image for processing
    resized_image = cv2.resize(gray_image, (150, 150))  # Resize to 150x150 pixels

    # Apply Gaussian blur to reduce noise
    blurred_image = cv2.GaussianBlur(resized_image, (5, 5), 0)  # Moderate blur
    
    # Apply LBP texture extraction
    lbp = feature.local_binary_pattern(blurred_image, P, R, method="uniform")
    
    # Normalize LBP for better visualization
    lbp_normalized = (lbp - lbp.min()) / (lbp.max() - lbp.min()) * 255
    
    # Convert LBP to uint8 for visualization
    lbp_image = lbp_normalized.astype(np.uint8)

    return gray_image, lbp_image  # Return both grayscale and LBP images

# Simulating class creation with image loading for 'horse', 'owl', and 'fish'
def load_images():
    # Each class has 5 images
    horse = [f'horse_img_{i}.jpg' for i in range(1, 6)]
    owl = [f'owl_img_{i}.jpg' for i in range(1, 6)]
    fish = [f'fish_img_{i}.jpg' for i in range(1, 6)]

    # Dictionary mapping class names to image lists
    classes = {
        'horse': horse,
        'owl': owl,
        'fish': fish
    }
    
    # Loading images (simulated with random matrices in this case)
    images = []
    for class_name, img_list in classes.items():
        for img in img_list:
            # Replace with actual image loading using cv2.imread(img)
            img_matrix = np.random.randint(0, 255, (150, 150, 3), dtype=np.uint8)  # Simulated random images
            images.append((img_matrix, class_name))
    
    return images

# Function to randomly select an image and compute its texture
def random_image_selection(images):
    selected_image, class_name = random.choice(images)
    gray_image, lbp = extract_lbp_texture(selected_image)
    return selected_image, gray_image, lbp, class_name

# Function to calculate the standard deviation of LBP texture
def calculate_std(texture):
    return np.std(texture)

# Function to compare textures using cosine similarity
def compare_textures(texture_a, texture_b):
    return cosine_similarity([texture_a.flatten()], [texture_b.flatten()])[0][0]

# Main program
def main():
    # Load the images from the classes 'horse', 'owl', and 'fish'
    images = load_images()

    # Select one specific image (first one from the list) for comparison
    selected_image, gray_selected, lbp_selected, class_selected = images[0][0], *extract_lbp_texture(images[0][0]), images[0][1]
    print(f"Selected image belongs to: {class_selected}")
    
    # Randomly select one image
    random_image, gray_random, lbp_random, class_random = random_image_selection(images)
    print(f"Randomly selected image belongs to: {class_random}")

    # Calculate standard deviations
    std_selected = calculate_std(lbp_selected)
    std_random = calculate_std(lbp_random)

    # Calculate cosine similarity between the textures
    similarity = compare_textures(lbp_selected, lbp_random)

    # Display only the selected image and random image with proper titles
    plt.figure(figsize=(10, 5))
    
    # Display selected image
    plt.subplot(1, 2, 1)
    plt.title(f"Selected Image ({class_selected})")
    plt.imshow(gray_selected, cmap='gray')  # Display grayscale image of the selected one
    plt.axis('off')

    # Display random image
    plt.subplot(1, 2, 2)
    plt.title(f"Random Image ({class_random})")
    plt.imshow(gray_random, cmap='gray')  # Display grayscale random image
    plt.axis('off')

    plt.tight_layout()
    plt.show()

    # Print the standard deviations and cosine similarity
    print(f"Standard Deviation of Selected Image: {std_selected:.4f}")
    print(f"Standard Deviation of Random Image: {std_random:.4f}")
    print(f"Cosine similarity between textures: {similarity:.4f}")

if __name__ == "__main__":
    main()
