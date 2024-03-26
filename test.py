import numpy as np
import tensorflow as tf
from data.dataset import get_datasets
from models.loss_functions import custom_ssim_loss, custom_combined_loss
from evaluation.FMI import fmi
from evaluation.SSIM import multi_ssim as ssim
from util.image_utils import save_fused_image
import matlab.engine
import os

import argparse

# Create a parser object and add arguments for alpha, beta, and gamma, modalities (list of strings, by default ['t2w', 't1w']) and batch size (default 1)
test_parser = argparse.ArgumentParser(description='Test a model on a dataset')
test_parser.add_argument('--alpha', type=float, default=0.5, help='Weight for the luminance component of the SSIM loss')
test_parser.add_argument('--beta', type=float, default=0.5, help='Weight for the contrast component of the SSIM loss')
test_parser.add_argument('--gamma', type=float, default=1, help='Weight for the structure component of the SSIM loss')
test_parser.add_argument('--modalities', nargs='+', default=['T2w', 'T1w'], help='List of modalities to use for training')
test_parser.add_argument('--batch_size', type=int, default=1, help='Batch size for testing')
# Add argument for the loss function (default 'ssim')
test_parser.add_argument('--loss_function', type=str, default='combined', help='Loss function to use for testing')
# Add argument for the multi-scale feature usage (default False)
test_parser.add_argument('--use_multi_scale', action='store_false', help='Use multi-scale features in the model')

def test_model(modalities, model, test_dataset):
    """
    Test the model on the test dataset and calculate the average FMI and SSIM scores.

    Args:
        modalities: List of strings representing the modalities used for training
        model: The trained model
        test_dataset: The test dataset
    """

    fmi_scores = []
    ssim_scores = []

    # Start MATLAB engine
    eng = matlab.engine.start_matlab()
    # Change to the directory of the FMI MATLAB script : ./evaluation/fmi.m
    eng.cd(os.path.dirname(os.path.realpath(__file__)) + '/evaluation') 

    for images in test_dataset:
        image1, image2 = images[modalities[0]], images[modalities[1]]  
        test_output = model([image1, image2], training=False)  # Generate predictions
        # save_fused_image(modalities, test_output, images['patient_id'])  # Save the fused image
        fmi_score = fmi(image1, image2, test_output, eng)  # Calculate FMI score between ground truth and prediction
        ssim_score = ssim([image1, image2], test_output)
        fmi_scores.append(fmi_score)
        ssim_scores.append(ssim_score)

    # Stop MATLAB engine
    eng.quit()

    average_fmi = np.mean(fmi_scores)
    average_ssim = np.mean(ssim_scores)

    print("Average FMI on test set:", average_fmi)
    print("Average SSIM on test set:", average_ssim)

if __name__ == "__main__":
    args = test_parser.parse_args()

    # Load the test dataset
    _, _, test_dataset = get_datasets(args.modalities, args.batch_size)

    # Recreate the custom loss function with the same parameters used during training
    if args.loss_function == 'ssim':
        custom_loss = custom_ssim_loss(alpha=args.alpha, beta=args.beta, gamma=args.gamma)
        if args.use_multi_scale:
            model = tf.keras.models.load_model(f"saved_models/fusion_model_alpha{args.alpha}_beta{args.beta}_gamma{args.gamma}_multi_ssim_loss_multi_scale.h5", 
                                       custom_objects={'custom_loss': custom_loss})
            print("Loaded model with ssim loss and multi-scale features")
        else:
            model = tf.keras.models.load_model(f"saved_models/fusion_model_alpha{args.alpha}_beta{args.beta}_gamma{args.gamma}_multi_ssim_loss.h5", 
                                       custom_objects={'custom_loss': custom_loss})
            print("Loaded model with ssim loss")
    else:
        custom_loss = custom_combined_loss(alpha=args.alpha, beta=args.beta)
        if args.use_multi_scale:
            model = tf.keras.models.load_model(f"saved_models/fusion_model_alpha{args.alpha}_beta{args.beta}_multi_combined_loss_multi_scale.h5", 
                                       custom_objects={'custom_loss': custom_loss})
            print("Loaded model with combined loss and multi-scale features")
        else:
            model = tf.keras.models.load_model(f"saved_models/fusion_model_alpha{args.alpha}_beta{args.beta}_multi_combined_loss.h5", 
                                       custom_objects={'custom_loss': custom_loss})
            print("Loaded model with combined loss")    

    # Test the model
    test_model(args.modalities, model, test_dataset)