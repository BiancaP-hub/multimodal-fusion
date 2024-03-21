import tensorflow as tf
from evaluation.SSIM import ssim, multi_ssim
print(tf.__version__)

# SSIM Loss between two images
def ssim_loss(true, pred, max_val=1.0, alpha=1.0, beta=1.0, gamma=1.0):
    """Computes SSIM loss."""
    return 1 - ssim(true, pred, max_val, alpha, beta, gamma)

# SSIM Loss between multiple images
def multi_ssim_loss(comparison_images, reference, max_val=1.0, alpha=1.0, beta=1.0, gamma=1.0):
    """Computes multi-SSIM loss."""
    return 1 - multi_ssim(comparison_images, reference, max_val, alpha, beta, gamma)

def custom_ssim_loss(alpha, beta, gamma):
    def custom_loss(y_true, y_pred):
        return multi_ssim_loss(y_true, y_pred, alpha=alpha, beta=beta, gamma=gamma)
    return custom_loss

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

def multi_combined_loss(true_images, pred_image, alpha=0.5, beta=0.5, max_val=1.0):
    """
    Combines multi SSIM and Pixel Losses into a single loss function for multiple images.
    
    Parameters:
    - true_images: The list of ground truth images.
    - pred_image: The predicted (fused) image.
    - alpha: Weight for the SSIM loss component.
    - beta: Weight for the Pixel loss component.
    - max_val: The dynamic range of the pixel values (255 for 8-bit images, 1 for normalized images).
    
    Returns:
    - The weighted sum of the specified loss components.
    """
    # Initialize losses
    ssim_l = 0
    pixel_l = 0

    # Calculate losses for each true image and the single predicted image
    for true in true_images:
        ssim_l += ssim_loss(true, pred_image, max_val) # components weights are 1.0, 1.0, 1.0
        pixel_l += pixel_loss(true, pred_image)

    # Average the losses
    ssim_l /= len(true_images)
    pixel_l /= len(true_images)

    # Ensure the weights sum to 1
    total_weight = alpha + beta
    if total_weight != 1.0:
        raise ValueError("Weights for loss components must sum to 1.")

    return alpha * ssim_l + beta * pixel_l

def custom_combined_loss(alpha, beta):
    def custom_loss(y_true, y_pred):
        return multi_combined_loss(y_true, y_pred, alpha=alpha, beta=beta)
    return custom_loss


