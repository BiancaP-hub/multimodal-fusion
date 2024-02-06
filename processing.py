import numpy as np
import matplotlib.pyplot as plt
import cv2
import ants
import os

# Function to draw contours
def draw_contours(original_image, contours):
    if len(original_image.shape) == 2 or (len(original_image.shape) == 3 and original_image.shape[2] == 1):
        original_image = cv2.cvtColor(original_image, cv2.COLOR_GRAY2BGR)
    colors = [tuple(np.random.randint(0, 255, 3).tolist()) for _ in range(len(contours))]
    for i, contour in enumerate(contours):
        cv2.drawContours(original_image, [contour], -1, colors[i], 1)
    return original_image

# Function to find contours and compute overlap height
def find_contours_and_overlap_height(edge_image, original_image):
    contours, _ = cv2.findContours(edge_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:  # Check if contours list is empty
        print("No contours found.")
        return original_image, 0  # Return the original image and 0 as overlap height
    contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[3], reverse=True)[:4]  # Sort by vertical extent
    contoured_image = draw_contours(original_image.copy(), contours)
    overlap_height = calculate_overlap_height(contours)
    return contoured_image, overlap_height

# Function to calculate overlap height
def calculate_overlap_height(contours):
    bounding_boxes = [cv2.boundingRect(c) for c in contours]
    vertical_ranges = [(y, y+h) for (_, y, _, h) in bounding_boxes]
    overlap_range = vertical_ranges[0]
    for i in range(1, len(vertical_ranges)):
        overlap_range = (max(overlap_range[0], vertical_ranges[i][0]), min(overlap_range[1], vertical_ranges[i][1]))
    return max(0, overlap_range[1] - overlap_range[0])

def process_patient_images(data_dir='../../data-multi-subject', save_dir='paired_images'):
    patients = [f for f in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, f)) and f.startswith('sub-')]
    for patient_id in patients:
        t2w_image_path = os.path.join(data_dir, patient_id, 'anat', f'{patient_id}_T2w.nii.gz')
        t1w_image_path = os.path.join('registered_images', f'{patient_id}_T1w_to_T2w.nii.gz')

        patient_save_dir = os.path.join(save_dir, patient_id)
        os.makedirs(patient_save_dir, exist_ok=True)

        t2w_img = ants.image_read(t2w_image_path, reorient='LSA')
        t1w_img = ants.image_read(t1w_image_path)

        t2w_array = t2w_img.numpy()
        t1w_array = t1w_img.numpy()

        t2w_normalized = ((t2w_array - np.min(t2w_array)) / (np.max(t2w_array) - np.min(t2w_array)) * 255).astype(np.uint8)
        t1w_normalized = ((t1w_array - np.min(t1w_array)) / (np.max(t1w_array) - np.min(t1w_array)) * 255).astype(np.uint8)

        overlap_height_threshold = 80
        
        min_index, max_index = -1, -1
        for i in range(t2w_normalized.shape[0]):
            enhanced_image = cv2.convertScaleAbs(t2w_normalized[i], alpha=0.8, beta=0)
            edge_image = cv2.Canny(enhanced_image, 100, 200, apertureSize=3, L2gradient=True)
            _, overlap_height = find_contours_and_overlap_height(edge_image, t2w_normalized[i])
            
            if overlap_height > overlap_height_threshold:
                min_index = i if min_index == -1 else min(min_index, i)
                max_index = max(max_index, i)
                
        if min_index != -1 and max_index != -1:
            for i in range(min_index, max_index + 1):
                t2w_slice_resized = cv2.resize(t2w_normalized[i], (256, 256))
                t1w_slice_resized = cv2.resize(t1w_normalized[i], (256, 256))
                
                t2w_slice_save_path = os.path.join(patient_save_dir, f't2w_slice_{i}.png')
                t1w_slice_save_path = os.path.join(patient_save_dir, f't1w_slice_{i}.png')
                
                cv2.imwrite(t2w_slice_save_path, t2w_slice_resized)
                cv2.imwrite(t1w_slice_save_path, t1w_slice_resized)
                print(f"Saved resized slices for {patient_id}, slice index: {i}")

if __name__ == '__main__':
    process_patient_images()
