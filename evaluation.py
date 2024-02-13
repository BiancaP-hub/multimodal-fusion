import numpy as np
import matlab.engine
import scipy.io as sio

def mutual_information_between_two_images(image1, image2, bins=256):
    """
    Calculate the mutual information between two images.

    Parameters:
    - image1_np: Numpy array of the first image.
    - image2_np: Numpy array of the second image.
    - bins: Number of bins to use for histogram calculation.

    Returns:
    - mi: Mutual Information between the two images.
    """
    image1_np = (image1.numpy().squeeze() * 255).astype(int)
    image2_np = (image2.numpy().squeeze() * 255).astype(int)

    joint_hist, _, _ = np.histogram2d(image1_np.ravel(), image2_np.ravel(), bins=bins)
    
    # Convert joint histogram to joint probability distribution
    joint_prob_dist = joint_hist / float(np.sum(joint_hist))
    
    # Marginal probability distributions
    prob_dist1 = np.sum(joint_prob_dist, axis=1)
    prob_dist2 = np.sum(joint_prob_dist, axis=0)
    
    # Calculate the mutual information
    mi = 0
    for i in range(bins):
        for j in range(bins):
            if joint_prob_dist[i, j] > 0:
                mi += joint_prob_dist[i, j] * np.log(joint_prob_dist[i, j] / (prob_dist1[i] * prob_dist2[j]))
    mi = mi / np.log(2)  # Convert to bits
    return mi

def save_images_to_mat(ima, imb, imf):
    """
    Save ima, imb, and imf to ima.mat, imb.mat, and imf.mat respectively.

    Parameters:
    - ima: Numpy array of the first image.
    - imb: Numpy array of the second image.
    - imf: Numpy array of the fused image.
    """
    # Convert to uint8
    ima = (ima.numpy().squeeze() * 255).astype(np.uint8)
    imb = (imb.numpy().squeeze() * 255).astype(np.uint8)
    imf = (imf.numpy().squeeze() * 255).astype(np.uint8)
    
    sio.savemat('ima.mat', {'ima': ima})
    sio.savemat('imb.mat', {'imb': imb})
    sio.savemat('imf.mat', {'imf': imf})

def calculate_fmi(batch1, batch2, output_batch):
    fmi_batch = []
    # Start MATLAB engine
    eng = matlab.engine.start_matlab()

    # Save ima, imb, and imf to .mat files
    for ima, imb, imf in zip(batch1, batch2, output_batch):
        save_images_to_mat(ima, imb, imf)

        # Load ima.mat, imb.mat, and imf.mat
        ima = sio.loadmat('ima.mat')['ima']
        imb = sio.loadmat('imb.mat')['imb']
        imf = sio.loadmat('imf.mat')['imf']

        # Use fmi.m to calculate the FMI
        fmi_batch.append(eng.fmi(ima, imb, imf))

    # Stop MATLAB engine
    eng.quit()

    return np.mean(fmi_batch)



