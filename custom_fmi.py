import numpy as np
from skimage.filters import sobel

def calculate_feature(image, feature_type='edge'):
    """
    Calculate specified feature for an image.
    """
    if feature_type == 'edge':
        return sobel(image)  # Example for edge detection
    elif feature_type == 'gradient':
        # Placeholder for gradient calculation
        pass
    elif feature_type == 'none':
        return image  # No feature extraction, use raw pixels

def fmi_generalized(source_images, fused_image, feature='edge', w=3):
    """
    Calculate the generalized Feature Mutual Information (FMI) for multiple source images
    and a fused image based on specified features and sliding window size.
    """
    # Feature extraction
    features = [calculate_feature(img.numpy().squeeze(), feature) for img in source_images]
    fused_feature = calculate_feature(fused_image.numpy().squeeze(), feature)
    
    # Initialize sliding window parameters
    m, n = fused_feature.shape
    w_half = w // 2
    fmi_scores = []

    # Slide window across images
    for i in range(w_half, m - w_half):
        for j in range(w_half, n - w_half):
            window_scores = []
            fused_window = fused_feature[i-w_half:i+w_half+1, j-w_half:j+w_half+1]

            for feature in features:
                source_window = feature[i-w_half:i+w_half+1, j-w_half:j+w_half+1]
                # Calculate similarity measure (e.g., MI) between fused_window and source_window
                window_score = calculate_nmi(source_window, fused_window) 
                window_scores.append(window_score)

            # Combine scores from all source images for this window
            fmi_scores.append(np.mean(window_scores))

    # Average FMI score across all windows
    average_fmi = np.mean(fmi_scores)
    return average_fmi

def calculate_histogram(data, bins):
    """
    Calculate the normalized histogram of given data.
    """
    hist, _ = np.histogram(data, bins=bins, range=[np.min(data), np.max(data)])
    hist = hist / np.sum(hist)  # Normalize histogram
    return hist

def calculate_entropy(hist):
    """
    Calculate the entropy from a histogram.
    """
    hist = hist[hist > 0]  # Filter zero values to avoid log(0)
    return -np.sum(hist * np.log2(hist))

def calculate_joint_entropy(hist1, hist2):
    """
    Calculate joint entropy of two histograms.
    """
    joint_hist = np.outer(hist1, hist2)
    joint_hist = joint_hist[joint_hist > 0]  # Filter zero values
    return -np.sum(joint_hist * np.log2(joint_hist))

def calculate_mi(feature1, feature2, bins=256):
    """
    Calculate the mutual information between two feature arrays.
    """
    hist1 = calculate_histogram(feature1.ravel(), bins)
    hist2 = calculate_histogram(feature2.ravel(), bins)

    entropy1 = calculate_entropy(hist1)
    entropy2 = calculate_entropy(hist2)
    joint_entropy = calculate_joint_entropy(hist1, hist2)

    mi = entropy1 + entropy2 - joint_entropy
    return mi

def calculate_nmi(feature1, feature2, bins=256):
    """
    Calculate the normalized mutual information between two feature arrays.
    """
    mi = calculate_mi(feature1, feature2, bins)
    hist1 = calculate_histogram(feature1.ravel(), bins)
    hist2 = calculate_histogram(feature2.ravel(), bins)

    entropy1 = calculate_entropy(hist1)
    entropy2 = calculate_entropy(hist2)

    # Avoid division by zero; add small epsilon
    nmi = mi / (np.sqrt(entropy1 * entropy2) + np.finfo(float).eps)
    return nmi

