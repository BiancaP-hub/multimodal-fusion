import tensorflow as tf

def ssim_loss(true, pred, max_val=1.0):
    """Computes SSIM loss.
    Measures the similarity between two images in terms of luminance, contrast, and structure. 
    """
    return 1.0 - tf.reduce_mean(tf.image.ssim(true, pred, max_val))

def gradient_loss(true, pred):
    """Computes gradient loss.
    Preserves edge information by penalizing the difference in gradients (edge information) 
    between the fused image and the target image.
    """
    true_dx, true_dy = tf.image.image_gradients(true)
    pred_dx, pred_dy = tf.image.image_gradients(pred)
    
    loss_dx = tf.reduce_mean(tf.abs(true_dx - pred_dx))
    loss_dy = tf.reduce_mean(tf.abs(true_dy - pred_dy))
    
    return (loss_dx + loss_dy) / 2.0

def combined_loss(true, pred, alpha=0.5, max_val=1.0):
    """
    Combines SSIM and Gradient Loss into a single loss function.
    
    Parameters:
    - true: The ground truth image.
    - pred: The predicted image.
    - alpha: Weight for the SSIM loss component. (1 - alpha) will be the weight for the gradient loss component.
    - max_val: The dynamic range of the pixel values (255 for 8-bit images, 1 for normalized images).
    
    Returns:
    - The weighted sum of SSIM and Gradient Loss.
    """
    ssim = ssim_loss(true, pred, max_val)
    grad = gradient_loss(true, pred)
    return alpha * ssim + (1 - alpha) * grad