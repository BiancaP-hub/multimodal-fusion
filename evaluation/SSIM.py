import tensorflow as tf

def ssim(true, pred, max_val=1.0, alpha=1.0, beta=1.0, gamma=1.0):
    """
    Computes the SSIM score between two images.
    
    Parameters:
    - true: The ground truth image.
    - pred: The predicted image.
    - max_val: The dynamic range of the pixel values.
    - alpha, beta, gamma: Weights for the luminance, contrast, and structure components of SSIM.
    
    Returns:
    - The SSIM score between the two images.
    """
    C1 = (0.01 * max_val) ** 2
    C2 = (0.03 * max_val) ** 2
    C3 = C2 / 2  # Typically C3 is set to C2 / 2

    # Luminance comparison
    mu_x = tf.reduce_mean(true)
    mu_y = tf.reduce_mean(pred)
    luminance = ((2 * mu_x * mu_y + C1) / (mu_x ** 2 + mu_y ** 2 + C1)) ** alpha

    # Contrast comparison
    sigma_x = tf.sqrt(tf.math.reduce_variance(true) + C2)
    sigma_y = tf.sqrt(tf.math.reduce_variance(pred) + C2)
    contrast = ((2 * sigma_x * sigma_y + C2) / (sigma_x ** 2 + sigma_y ** 2 + C2)) ** beta

    # Structure comparison
    sigma_xy = tf.reduce_mean((true - mu_x) * (pred - mu_y))
    structure = ((sigma_xy + C3) / (sigma_x * sigma_y + C3)) ** gamma

    return luminance * contrast * structure

# Function to calculate the SSIM score between multiple comparison images and a reference image
def multi_ssim(comparison_images, reference, max_val=1.0, alpha=1.0, beta=1.0, gamma=1.0, weights=None):
    """
    Computes the SSIM score between multiple comparison images and a reference image.
    
    Parameters:
    - comparison_images: List of images to compare with the reference image.
    - reference: The reference image.
    - max_val: The dynamic range of the pixel values.
    - alpha, beta, gamma: Weights for the luminance, contrast, and structure components of SSIM.
    - weights: Optional list of weights for each comparison image.
    
    Returns:
    - The combined SSIM score between the comparison images and the reference image.
    """
    if weights is None:
        # Assign equal weight to each comparison if no weights are provided
        weights = [1.0 / len(comparison_images)] * len(comparison_images)
    
    ssim_scores = []
    for img, weight in zip(comparison_images, weights):
        ssim_score = ssim(img, reference, max_val, alpha, beta, gamma)
        ssim_scores.append(ssim_score * weight)
    
    combined_ssim = tf.reduce_sum(ssim_scores)
    return combined_ssim