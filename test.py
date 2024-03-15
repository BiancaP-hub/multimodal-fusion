import numpy as np
import tensorflow as tf
from data.dataset import get_datasets
from models.loss_functions import custom_ssim_loss
from evaluation.FMI import fmi
from evaluation.SSIM import multi_ssim as ssim

import argparse

# Create a parser object and add arguments for alpha, beta, and gamma, modalities (list of strings, by default ['t2w', 't1w']) and batch size (default 1)
parser = argparse.ArgumentParser(description='Test a model on a dataset')
parser.add_argument('--alpha', type=float, default=1, help='Weight for the luminance component of the SSIM loss')
parser.add_argument('--beta', type=float, default=1, help='Weight for the contrast component of the SSIM loss')
parser.add_argument('--gamma', type=float, default=1, help='Weight for the structure component of the SSIM loss')
parser.add_argument('--modalities', nargs='+', default=['t2w', 't1w'], help='List of modalities to use for training')
parser.add_argument('--batch_size', type=int, default=1, help='Batch size for testing')
args = parser.parse_args()

def test_model(modalities, model, test_dataset, alpha, beta, gamma):
    """
    Test the model on the test dataset and calculate the average FMI and SSIM scores.

    Args:
        modalities: List of strings representing the modalities used for training
        model: The trained model
        test_dataset: The test dataset
        alpha: The alpha parameter for the SSIM loss
        beta: The beta parameter for the SSIM loss
        gamma: The gamma parameter for the SSIM loss
    """

    fmi_scores = []
    ssim_scores = []

    for images in test_dataset:
        image1, image2 = images[modalities[0]], images[modalities[1]]  
        test_output = model([image1, image2], training=False)  # Generate predictions
        fmi_score = fmi(image1, image2, test_output)  # Calculate FMI score between ground truth and prediction
        ssim_score = ssim([image1, image2], test_output, alpha=alpha, beta=beta, gamma=gamma)
        fmi_scores.append(fmi_score)
        ssim_scores.append(ssim_score)

    average_fmi = np.mean(fmi_scores)
    average_ssim = np.mean(ssim_scores)

    print("Average FMI on test set:", average_fmi)
    print("Average SSIM on test set:", average_ssim)

if __name__ == "__main__":
    # Load the test dataset
    _, _, test_dataset = get_datasets(args.modalities, args.batch_size)

    # Recreate the custom loss function with the same parameters used during training
    custom_loss = custom_ssim_loss(alpha=args.alpha, beta=args.beta, gamma=args.gamma)

    # Load the trained model
    model = tf.keras.models.load_model(f"saved_models/fusion_model_alpha{args.alpha}_beta{args.beta}_gamma{args.gamma}.h5", 
                                       custom_objects={'custom_loss': custom_loss})

    # Test the model
    test_model(args.modalities, model, test_dataset, args.alpha, args.beta, args.gamma)