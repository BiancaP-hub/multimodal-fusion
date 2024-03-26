import os
import numpy as np
import matlab.engine
import scipy.io as sio

def save_images_to_mat(ima, imb, imf, dir_path='tmp'):
    """
    Save ima, imb, and imf to ima.mat, imb.mat, and imf.mat respectively.

    Parameters:
    - ima: Numpy array of the first image.
    - imb: Numpy array of the second image.
    - imf: Numpy array of the fused image.
    - dir_path: The directory where to save the .mat files.
    """
    # Convert to uint8
    ima = (ima.numpy().squeeze() * 255).astype(np.uint8)
    imb = (imb.numpy().squeeze() * 255).astype(np.uint8)
    imf = (imf.numpy().squeeze() * 255).astype(np.uint8)
    
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    sio.savemat(os.path.join(dir_path, 'ima.mat'), {'ima': ima})
    sio.savemat(os.path.join(dir_path, 'imb.mat'), {'imb': imb})
    sio.savemat(os.path.join(dir_path, 'imf.mat'), {'imf': imf})

def fmi(images1, images2, output_images, eng):
    """
    Calculate the FMI score between the input images and the output images.

    Parameters:
    - images1: List of Numpy arrays of the first set of images.
    - images2: List of Numpy arrays of the second set of images.
    - output_images: List of Numpy arrays of the output images.

    Returns:
    - fmi: FMI score between the input images and the output images.
    """
    fmi_scores = []
    # # Start MATLAB engine
    # eng = matlab.engine.start_matlab()
    # eng.cd(os.path.dirname(os.path.realpath(__file__)))

    # Save ima, imb, and imf to .mat files
    for ima, imb, imf in zip(images1, images2, output_images):
        save_images_to_mat(ima, imb, imf)

        # Load ima.mat, imb.mat, and imf.mat
        ima = sio.loadmat('tmp/ima.mat')['ima']
        imb = sio.loadmat('tmp/imb.mat')['imb']
        imf = sio.loadmat('tmp/imf.mat')['imf']

        # Use fmi.m to calculate the FMI
        fmi_scores.append(eng.fmi(ima, imb, imf))

    # # Stop MATLAB engine
    # eng.quit()

    # Delete the tmp directory and all its contents
    os.system('rm -rf tmp')

    return np.mean(fmi_scores)