% Load the images
I1 = imread('paired_images/sub-amu01/t2w_slice_25.png');
I2 = imread('paired_images/sub-amu01/t1w_slice_25.png');
I3 = imread('paired_images/sub-amu01/t2w_slice_26.png');

% Compute the mutual information
mi12 = fmi(ima, imb, imf, 'none');
