import tensorflow as tf
import time
from models.fusion_model import build_model
from models.loss_functions import multi_ssim_loss, custom_ssim_loss
from util.image_utils import display_images_and_histograms
from data.dataset import get_datasets

import argparse

# Create a parser object and add arguments for modalities (list of strings, by default ['T2w', 'T1w'])
parser = argparse.ArgumentParser(description='Train a model on a dataset')
# Use multi-scale features argument (default False)
parser.add_argument('--use_multi_scale', action='store_true', help='Use multi-scale features in the model')
parser.add_argument('--modalities', nargs='+', default=['T2w', 'T1w'], help='List of modalities to use for training')
# Batch size argument (default 32)
parser.add_argument('--batch_size', type=int, default=8, help='Batch size for training')
# Learning rate argument (default 0.0001)
parser.add_argument('--learning_rate', type=float, default=0.0001, help='Learning rate for training')
# Optimizer argument (default 'adam')
parser.add_argument('--optimizer', type=str, default='adam', help='Optimizer for training')
# Max epochs argument (default 100)
parser.add_argument('--max_epochs', type=int, default=100, help='Maximum number of epochs for training')
# Alpha argument (default 1)
parser.add_argument('--alpha', type=float, default=1, help='Weight for the luminance component of the SSIM loss')
# Beta argument (default 1)
parser.add_argument('--beta', type=float, default=1, help='Weight for the contrast component of the SSIM loss')
# Gamma argument (default 1)
parser.add_argument('--gamma', type=float, default=1, help='Weight for the structure component of the SSIM loss')
args = parser.parse_args()

def train_model(modalities, model, train_dataset, val_dataset, alpha, beta, gamma, max_epochs, learning_rate, use_multi_scale=False):
    # Training configuration
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    patience = 5
    display_step = 5  # Display images every N epochs

    model.compile(optimizer=optimizer, loss=custom_ssim_loss(alpha, beta, gamma))

    best_val_loss = float('inf')
    wait = 0
    start_time = time.time()    

    # Train model on GPU if available, otherwise use CPU
    with tf.device('/GPU:0' if tf.config.list_physical_devices('GPU') else '/CPU:0'):
        for epoch in range(max_epochs):
            train_loss = 0
            val_loss = 0
            num_batches_train = 0
            num_batches_val = 0

            # Training loop
            for images in train_dataset:
                image1, image2 = images[modalities[0]], images[modalities[1]]
                with tf.GradientTape() as tape:
                    predictions = model([image1, image2], training=True)
                    loss = multi_ssim_loss([image1, image2], predictions, alpha=alpha, beta=beta, gamma=gamma)
                grads = tape.gradient(loss, model.trainable_weights)
                optimizer.apply_gradients(zip(grads, model.trainable_weights))
                train_loss += loss
                num_batches_train += 1

            # Validation loop
            for images in val_dataset:
                image1, image2 = images[modalities[0]], images[modalities[1]]
                predictions = model([image1, image2], training=False)
                loss = multi_ssim_loss([image1, image2], predictions, alpha=alpha, beta=beta, gamma=gamma)
                val_loss += loss
                num_batches_val += 1

            train_loss /= num_batches_train
            val_loss /= num_batches_val

            print(f"Epoch {epoch+1}/{max_epochs}, Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")

            # Early stopping check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                wait = 0
            else:
                wait += 1
                if wait >= patience:
                    print(f'Early stopping at epoch {epoch}')
                    break

            # Display progress
            if (epoch + 1) % display_step == 0:
                images = next(iter(val_dataset))
                image1, image2, patient_id = images[modalities[0]], images[modalities[1]], images['patient_id']
                val_output = model([image1, image2], training=False)
                display_images_and_histograms(image1[0], image2[0], val_output[0], patient_id[0])

        print(f'Training finished in {time.time() - start_time} seconds')

        multi_scale = 'multi_scale' if use_multi_scale else ''
        model.save(f"saved_models/fusion_model_alpha{alpha}_beta{beta}_gamma{gamma}_{multi_scale}.h5")


if __name__ == '__main__':
    print(args)
    # Load the training and validation datasets
    train_dataset, val_dataset, _ = get_datasets(args.modalities, args.batch_size)

    # Take one sample from the dataset
    sample = next(iter(train_dataset))

    # Get the shapes of the images for all modalities, excluding the batch size dimension
    image_shapes = [sample[modality].shape[1:] for modality in args.modalities]

    # Build the model with the dynamically created list of image shapes
    model = build_model(image_shapes, use_multi_scale=args.use_multi_scale)

    train_model(args.modalities, model, train_dataset, val_dataset, args.alpha, args.beta, args.gamma, args.max_epochs, args.learning_rate, args.use_multi_scale)
