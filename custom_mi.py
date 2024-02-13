import numpy as np
from skimage.filters import sobel

def pairwise_mi_score(images, bins=256):
    """
    Calculate the average mutual information between multiple images.

    Parameters:
    - images: List of images as TensorFlow tensors.
    - bins: Number of bins to use for histogram calculation.

    Returns:
    - average_mi: Average Mutual Information between all pairs of images.
    """
    num_images = len(images)
    total_mi = 0
    num_pairs = 0

    # Convert images to numpy and scale
    images_np = [(img.numpy().squeeze() * 255).astype(int) for img in images]

    # Calculate MI for each pair
    for i in range(num_images):
        for j in range(i + 1, num_images):
            joint_hist, _, _ = np.histogram2d(images_np[i].ravel(), images_np[j].ravel(), bins=bins)
            
            # Convert joint histogram to joint probability distribution
            joint_prob_dist = joint_hist / float(np.sum(joint_hist))
            
            # Marginal probability distributions
            prob_dist1 = np.sum(joint_prob_dist, axis=1)
            prob_dist2 = np.sum(joint_prob_dist, axis=0)
            
            # Calculate the mutual information
            mi = 0
            for x in range(bins):
                for y in range(bins):
                    if joint_prob_dist[x, y] > 0:
                        mi += joint_prob_dist[x, y] * np.log(joint_prob_dist[x, y] / (prob_dist1[x] * prob_dist2[y]))
            mi = mi / np.log(2)  # Convert to bits

            total_mi += mi
            num_pairs += 1

    # Calculate average MI
    average_mi = total_mi / num_pairs if num_pairs > 0 else 0
    return average_mi

def calculate_gradients(image):
    """
    Calculate the gradient magnitude of an image using the Sobel operator.
    
    Parameters:
    - image: Numpy array of the image.
    
    Returns:
    - Gradient magnitude of the image.
    """
    sobel_image = sobel(image)
    return sobel_image

def pairwise_mi_score_with_features(images, bins=256, features_to_use=['pixels', 'gradients']):
    """
    Calculate the average mutual information between multiple images, considering a dynamic set of features.
    
    Parameters:
    - images: List of images as TensorFlow tensors.
    - bins: Number of bins to use for histogram calculation.
    - features_to_use: List specifying which features to use for MI calculation. Possible values include 'pixels', 'gradients', etc.
    
    Returns:
    - average_mi: Average Mutual Information for the specified features across all pairs of images.
    """
    num_images = len(images)
    total_mi = 0
    num_pairs = 0

    # Prepare images and calculate any required features
    images_np = [(img.numpy().squeeze() * 255).astype(int) for img in images]
    features_dict = {'pixels': images_np}

    if 'gradients' in features_to_use:
        gradients_np = [calculate_gradients(img) for img in images_np]
        features_dict['gradients'] = gradients_np

    # Additional features can be calculated and added to features_dict here

    for i in range(num_images):
        for j in range(i + 1, num_images):
            combined_features_i = []
            combined_features_j = []

            # Construct combined feature vectors based on the selected features
            for feature in features_to_use:
                if feature in features_dict:
                    combined_features_i.append(features_dict[feature][i].ravel())
                    combined_features_j.append(features_dict[feature][j].ravel())

            features_i = np.stack(combined_features_i, axis=1) if len(combined_features_i) > 1 else combined_features_i[0]
            features_j = np.stack(combined_features_j, axis=1) if len(combined_features_j) > 1 else combined_features_j[0]

            # Calculate joint histogram and mutual information
            if len(features_to_use) == 1:
                joint_hist, _, _ = np.histogram2d(features_i, features_j, bins=bins)
            else:
                joint_hist, _ = np.histogramdd(np.concatenate([features_i, features_j], axis=1), bins=bins)
            
            joint_prob_dist = joint_hist / joint_hist.sum()
            prob_dist1 = np.sum(joint_prob_dist, axis=0 if len(features_to_use) == 1 else tuple(range(1, len(features_to_use) + 1)))
            prob_dist2 = np.sum(joint_prob_dist, axis=1 if len(features_to_use) == 1 else 0)

            # MI calculation
            mi = 0
            for x in np.ndindex(joint_prob_dist.shape):
                if joint_prob_dist[x] > 0:
                    mi += joint_prob_dist[x] * np.log(joint_prob_dist[x] / (prob_dist1[x[0]] * prob_dist2[(x[1] if len(features_to_use) == 1 else x[0])]))
            mi = mi / np.log(2)  # Convert to bits

            total_mi += mi
            num_pairs += 1

    average_mi = total_mi / num_pairs if num_pairs > 0 else 0
    return average_mi