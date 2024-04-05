import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import skew, kurtosis
import os 
import cv2

def calculate_histogram_statistics(image_np):
    # Flatten the image to turn it into a 1D array of pixel values
    pixels = image_np.ravel()
    
    # Calculate statistics
    mean = np.mean(pixels)
    std_dev = np.std(pixels)
    skewness = skew(pixels)
    kurt = kurtosis(pixels)
    
    return mean, std_dev, skewness, kurt

def display_images_and_histograms(image1, image2, output, patient_id):
    # Assuming images are normalized in the range [0, 1]; scale to [0, 255]
    image1_np = (image1.numpy().squeeze() * 255).astype(int)
    image2_np = (image2.numpy().squeeze() * 255).astype(int)
    output_np = (output.numpy().squeeze() * 255).astype(int)

    # Set up the subplot for images and histograms
    fig, axs = plt.subplots(2, 3, figsize=(18, 8))
    
    # Display each image
    axs[0, 0].imshow(image1_np, cmap='gray', vmin=0, vmax=255)
    axs[0, 0].set_title(f'Patient {patient_id} - T2W')
    axs[0, 0].axis('off')
    
    axs[0, 1].imshow(image2_np, cmap='gray', vmin=0, vmax=255)
    axs[0, 1].set_title(f'Patient {patient_id} - T1W')
    axs[0, 1].axis('off')
    
    axs[0, 2].imshow(output_np, cmap='gray', vmin=0, vmax=255)
    axs[0, 2].set_title(f'Patient {patient_id} - Output')
    axs[0, 2].axis('off')
    
    # Plot histograms with axes set from 0 to 255
    axs[1, 0].hist(image1_np.ravel(), bins=256, range=[0,255], color='gray')
    axs[1, 0].set_title(f'Patient {patient_id} - T2W Histogram')
    
    axs[1, 1].hist(image2_np.ravel(), bins=256, range=[0,255], color='gray')
    axs[1, 1].set_title(f'Patient {patient_id} - T1W Histogram')
    
    axs[1, 2].hist(output_np.ravel(), bins=256, range=[0,255], color='gray')
    axs[1, 2].set_title(f'Patient {patient_id} - Output Histogram')
    
    plt.tight_layout()
    plt.show()

def calculate_histogram_statistics(image1_np, image2_np, output_np, patient_id):
    # Calculate statistics for each image
    stats_image1 = calculate_histogram_statistics(image1_np)
    stats_image2 = calculate_histogram_statistics(image2_np)
    stats_output = calculate_histogram_statistics(output_np)
    
    # Display statistics
    print(f"Statistics for Patient {patient_id}:")
    print(f"T2W - Mean: {stats_image1[0]:.2f}, Std Dev: {stats_image1[1]:.2f}, Skewness: {stats_image1[2]:.2f}, Kurtosis: {stats_image1[3]:.2f}")
    print(f"T1W - Mean: {stats_image2[0]:.2f}, Std Dev: {stats_image2[1]:.2f}, Skewness: {stats_image2[2]:.2f}, Kurtosis: {stats_image2[3]:.2f}")
    print(f"Output - Mean: {stats_output[0]:.2f}, Std Dev: {stats_output[1]:.2f}, Skewness: {stats_output[2]:.2f}, Kurtosis: {stats_output[3]:.2f}")

    # Analyze edge concentration
    edge_proportion1, middle_proportion1 = analyze_edge_concentration(image1_np)
    edge_proportion2, middle_proportion2 = analyze_edge_concentration(image2_np)
    edge_proportion_output, middle_proportion_output = analyze_edge_concentration(output_np)

    print(f"Edge Proportion - T2W: {edge_proportion1:.2f}, T1W: {edge_proportion2:.2f}, Output: {edge_proportion_output:.2f}")
    print(f"Middle Proportion - T2W: {middle_proportion1:.2f}, T1W: {middle_proportion2:.2f}, Output: {middle_proportion_output:.2f}")

def analyze_edge_concentration(image_np):
    # Calculate the histogram for the image
    histogram, bin_edges = np.histogram(image_np, bins=256, range=(0, 255))
    
    # Define the edges
    low_edge_range = (0, 10)
    high_edge_range = (246, 255)
    
    # Calculate sums of the histogram counts at the edges and in the middle
    low_edge_sum = np.sum(histogram[low_edge_range[0]:low_edge_range[1] + 1])
    high_edge_sum = np.sum(histogram[high_edge_range[0]:high_edge_range[1] + 1])
    middle_sum = np.sum(histogram[low_edge_range[1] + 1:high_edge_range[0]])
    
    # Calculate the proportion of pixels at the edges compared to the middle
    edge_sum = low_edge_sum + high_edge_sum
    total_sum = edge_sum + middle_sum
    edge_proportion = edge_sum / total_sum
    middle_proportion = middle_sum / total_sum
    
    return edge_proportion, middle_proportion

def save_fused_image(modalities, fused_image, patient_id, slice_number, output_dir='results'):
    # Create a directory to save the results
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert the fused image to a numpy array if it's a tensor
    if hasattr(fused_image, 'numpy'):
        fused_image = fused_image.numpy()

    # Squeeze the image to remove dimensions of size 1, particularly batch and channel dimensions
    fused_image = np.squeeze(fused_image)

    # Check if the image is in float format, assumes the image is normalized between 0 and 1
    if fused_image.dtype == np.float32:
        # Scale to 0-255 and convert to uint8
        fused_image = np.clip(fused_image * 255, 0, 255).astype(np.uint8)

    # Assuming patient_id is a numpy array with a single byte string element,
    # extract and decode the first element to get the patient ID string
    if hasattr(patient_id, 'numpy'):
        patient_id = patient_id.numpy()
    if patient_id.size > 0:
        patient_id_str = patient_id[0].decode('utf-8')
    else:
        raise ValueError("patient_id array is empty")

    # Use slice_number for the filename
    filename_base = f'fused_image_{patient_id_str}_{modalities[0]}_{modalities[1]}'
    output_path = os.path.join(output_dir, f'{filename_base}_{slice_number}.png')

    # Save the fused image
    success = cv2.imwrite(output_path, fused_image)
    
    if success:
        print(f"Saved fused image for patient {patient_id_str} at {output_path}")
    else:
        print(f"Failed to save fused image for patient {patient_id_str} at {output_path}")
