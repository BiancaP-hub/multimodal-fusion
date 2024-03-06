import tensorflow as tf
print(tf.__version__)

def ssim_loss(true, pred, max_val=1.0):
    """Computes SSIM loss."""
    return 1.0 - tf.reduce_mean(tf.image.ssim(true, pred, max_val))

def get_custom_loss(alpha, beta, gamma):
    def custom_loss(y_true, y_pred):
        return custom_multi_ssim_loss(y_true, y_pred, alpha=alpha, beta=beta, gamma=gamma)
    return custom_loss

def custom_multi_ssim_loss(comparison_images : list, reference, max_val=1.0, alpha=1.0, beta=1.0, gamma=1.0, weights=None):
    """
    Computes a custom SSIM loss for comparing multiple images with a single reference image.
    
    Parameters:
    - comparison_images: A list of images to compare against the reference.
    - reference: The reference image.
    - max_val: The dynamic range of the pixel values.
    - alpha, beta, gamma: Weights for the luminance, contrast, and structure components of SSIM.
    - weights: A list of weights to balance the importance of each comparison image's SSIM loss. If None, all images are considered equally.
    
    Returns:
    - A combined custom SSIM loss value.
    """
    if weights is None:
        # Assign equal weight to each comparison if no weights are provided
        weights = [1.0 / len(comparison_images)] * len(comparison_images)
    
    ssim_losses = []
    for img, weight in zip(comparison_images, weights):
        ssim_loss = custom_ssim_loss(img, reference, max_val, alpha, beta, gamma)
        ssim_losses.append(ssim_loss * weight)
    
    combined_ssim_loss = tf.reduce_sum(ssim_losses)
    return combined_ssim_loss

def custom_ssim_loss(true, pred, max_val=1.0, alpha=1.0, beta=1.0, gamma=1.0):
    """
    Computes a custom SSIM loss, allowing adjustments to the importance of luminance, contrast, and structure.
    
    Parameters:
    - true: The ground truth image.
    - pred: The predicted image.
    - max_val: The dynamic range of the pixel values.
    - alpha, beta, gamma: Weights for the luminance, contrast, and structure components of SSIM.
    
    Returns:
    - A custom SSIM loss value.
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

    ssim_index = luminance * contrast * structure
    return 1.0 - tf.reduce_mean(ssim_index)

def gradient_loss(true, pred):
    """Computes gradient loss."""
    true_dx, true_dy = tf.image.image_gradients(true)
    pred_dx, pred_dy = tf.image.image_gradients(pred)
    
    loss_dx = tf.reduce_mean(tf.abs(true_dx - pred_dx))
    loss_dy = tf.reduce_mean(tf.abs(true_dy - pred_dy))
    
    return (loss_dx + loss_dy) / 2.0

def pixel_loss(true, pred):
    """Computes pixel loss (Mean Squared Error between the images)."""
    return tf.reduce_mean(tf.square(true - pred))

def std_dev_loss(pred, gamma=0.1):
    """
    Calculates a loss based on the standard deviation of the predicted image.
    This function is designed to encourage higher contrast in the predicted image
    by penalizing low standard deviation values.

    Parameters:
    - pred: The predicted image.
    - gamma: Weight for the standard deviation incentive.

    Returns:
    - A loss value based on the negative standard deviation of the predicted image,
      scaled by the gamma parameter.
    """
    # Calculate the standard deviation of the predicted image
    std_dev_pred = tf.math.reduce_std(pred)
    
    # The loss is negative because we want to maximize the standard deviation
    # (i.e., penalize low standard deviation, which corresponds to low contrast)
    return -gamma * std_dev_pred

def laplacian_edge_loss(true, pred, delta=0.5):
    """
    Calculates a loss based on the Laplacian (second derivative), emphasizing edges in the predicted image.
    
    Parameters:
    - true: The ground truth image.
    - pred: The predicted image.
    - delta: Weight for the Laplacian edge loss component.

    Returns:
    - A loss value based on the difference between the Laplacian of the predicted and ground truth images,
      scaled by the delta parameter.
    """
    # Compute the Laplacian for both true and predicted images
    laplacian_true = tf.image.sobel_edges(true)
    laplacian_pred = tf.image.sobel_edges(pred)
    
    # Calculate the loss as the L2 norm (mean squared error) between the Laplacian of true and pred images
    edge_loss = tf.reduce_mean(tf.square(laplacian_pred - laplacian_true))
    
    return delta * edge_loss

def combined_loss(true, pred, alpha=0.4, beta=0.3, gamma=0.1, max_val=1.0):
    """
    Combines SSIM, Gradient, Pixel Losses, Standard Deviation incentive, and Laplacian Edge Loss into a single loss function.
    
    Parameters:
    - true: The ground truth image.
    - pred: The predicted image.
    - alpha: Weight for the SSIM loss component.
    - beta: Weight for the Gradient loss component.
    - gamma: Weight for the standard deviation incentive.
    - delta: Weight for the Laplacian edge loss component.
    - max_val: The dynamic range of the pixel values (255 for 8-bit images, 1 for normalized images).
    
    Returns:
    - The weighted sum of the specified loss components.
    """
    ssim_l = ssim_loss(true, pred, max_val)
    grad_l = gradient_loss(true, pred)
    pixel_l = pixel_loss(true, pred)
    edge_l = laplacian_edge_loss(true, pred, gamma)
    
    # Ensure the weights sum to 1 or less
    total_weight = alpha + beta + gamma
    if total_weight > 1.0:
        raise ValueError("Weights for loss components must sum to 1 or less.")
    
    return alpha * ssim_l + beta * grad_l + (1 - total_weight) * pixel_l + edge_l



