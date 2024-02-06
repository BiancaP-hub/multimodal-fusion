import tensorflow as tf

def ssim_loss(true, pred, max_val=1.0):
    """Computes SSIM loss."""
    return 1.0 - tf.reduce_mean(tf.image.ssim(true, pred, max_val))

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

def combined_loss(true, pred, alpha=0.4, beta=0.3, max_val=1.0):
    """
    Combines SSIM, Gradient, and Pixel Losses into a single loss function.
    
    Parameters:
    - true: The ground truth image.
    - pred: The predicted image.
    - alpha: Weight for the SSIM loss component.
    - beta: Weight for the Gradient loss component.
    - max_val: The dynamic range of the pixel values (255 for 8-bit images, 1 for normalized images).
    
    The weight for the Pixel loss component will be (1 - alpha - beta).
    
    Returns:
    - The weighted sum of SSIM, Gradient, and Pixel Loss.
    """
    ssim_l = ssim_loss(true, pred, max_val)
    grad_l = gradient_loss(true, pred)
    pixel_l = pixel_loss(true, pred)
    return alpha * ssim_l + beta * grad_l + (1 - alpha - beta) * pixel_l
