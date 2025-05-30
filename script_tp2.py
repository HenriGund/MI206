import os
import numpy as np
from skimage.morphology import (erosion, dilation, closing, thin, disk)
from PIL import Image
from matplotlib import pyplot as plt
from skimage.filters import  gaussian

def my_segmentation(img, img_mask, seuil):
    """
    Multi-scale segmentation of retinal vessels using pure morphological operators.
    """
    
    # 1. Preprocessing - smoothing
    img_smooth = gaussian(img, sigma=1.2)  # Gaussian smoothing for noise reduction
    # sigma-1.2 after testing, it seems to work well for the images provided
    
    # 2. Multi-scale approach with top-hat using explicit operations
    scales = [1, 2, 3, 4, 5, 6, 7] # multiple scales to capture vessels of varying thickness
    # Each scale represents a different number of pixels for the structuring element
    
    multi_scale_result = np.zeros_like(img, dtype=np.float64) # Initialize the result array
    
    for scale in scales:
        # Circular structuring element for each scale
        se = disk(scale)
        
        # Black top-hat to emphasize dark features (in the images, each vessel is darker than the background)
        closed_img = closing(img_smooth, se)
        tophat = closed_img.astype(np.float64) - img_smooth.astype(np.float64)
        
        # Accumulate results with decreasing weight for larger scales
        weight = 1.0 / (1 + scale * 0.03)
        multi_scale_result += tophat * weight
    
    # 3. Normalization of the multi-scale result
    if multi_scale_result.max() > 0:
        multi_scale_result = multi_scale_result / multi_scale_result.max()
    
    # 4. Adaptive thresholding to create a binary mask
    # Using a percentile threshold to adapt to varying illumination
    threshold_value = np.percentile(multi_scale_result[img_mask], seuil)
    binary_vessels = multi_scale_result > threshold_value
    
    #  Opening to remove small isolated objects: erosion followed by dilation
    se_opening = disk(1) 
    eroded_clean = erosion(binary_vessels, se_opening)
    binary_vessels = dilation(eroded_clean, se_opening)

    # 5. Closing to connect nearby structures: dilation followed by erosion
    se_cleanup = disk(2)
    dilated = dilation(binary_vessels, se_cleanup)
    binary_vessels = erosion(dilated, se_cleanup)
    

    # Opening to remove small isolated objects: erosion followed by dilation
    se_opening = disk(1)  
    eroded_clean = erosion(binary_vessels, se_opening)
    binary_vessels = dilation(eroded_clean, se_opening)

    
    # 6. Remove small objects using iterative erosion-dilation
    # Multiple erosions to remove small objects, then reconstruct
    se_small = disk(1)
    temp_img = binary_vessels.copy()
    for i in range(3):
        temp_img = erosion(temp_img, se_small)
    
    # Now perform morphological reconstruction manually using iterative dilation
    reconstructed = temp_img.copy()
    se_recon = disk(10)
    
    # Iterative reconstruction: dilate marker but keep it within the mask
    for _ in range(150):  # Maximum iterations to avoid infinite loop
        previous = reconstructed.copy()
        reconstructed = dilation(reconstructed, se_recon)
        # Keep reconstruction within original binary image
        reconstructed = reconstructed & binary_vessels
        
        # Check for convergence
        if np.array_equal(reconstructed, previous):
            break
    
    # 7. Final refinement with additional morphological operations
    # Small closing to ensure connectivity
    se_final = disk(1)
    dilated_final = dilation(reconstructed, se_final)
    final_result = erosion(dilated_final, se_final)
    
    # 8. Apply retina mask
    final_result = final_result.astype(bool)
    img_mask = img_mask.astype(bool)
    final_result = final_result & img_mask
    
    return final_result


def evaluate(img_out, img_GT):
    GT_skel  = thin(img_GT, max_num_iter = 15) # On suppose que la demie épaisseur maximum 
    img_out_skel  = thin(img_out, max_num_iter = 15) # d'un vaisseau est de 15 pixels...
    TP = np.sum(img_GT & img_out) # Vrais positifs
    FP = np.sum(img_out_skel & ~img_GT) # Faux positifs (relaxes)
    FN = np.sum(GT_skel & ~img_out) # Faux negatifs (relaxes)

    ACCU = TP / (TP + FP) # Precision
    RECALL = TP / (TP + FN) # Rappel
    return ACCU, RECALL, img_out_skel, GT_skel


def return_images(directory, begin):
    """
    Returns a list of image file names in the specified directory.
    """
    return [f for f in os.listdir(directory) if f.startswith(begin)]

directory = './images_IOSTAR/'
images = return_images(directory, 'star')
GT_images = return_images(directory, 'GT_')

def relation_star_gt(images, GT_images):
    corresponding_GT = {img : '' for img in images}
    for img in images:
        for gt in GT_images:
            # Assuming the GT image name is derived from the star image name
            if img.split('_')[0].replace('star', '') in gt.split('_')[1]:
                corresponding_GT[img] = gt
                break
    return corresponding_GT

corresponding_GT = relation_star_gt(images, GT_images)

ACCU = []
RECALL = []
count = 0

for image in images:

    #Ouvrir l'image originale en niveau de gris
    img =  np.asarray(Image.open(directory + image)).astype(np.uint8)
    print(img.shape)

    nrows, ncols = img.shape
    row, col = np.ogrid[:nrows, :ncols]
    #On ne considere que les pixels dans le disque inscrit 
    img_mask = (np.ones(img.shape)).astype(np.bool_)
    invalid_pixels = ((row - nrows/2)**2 + (col - ncols/2)**2 > (nrows / 2)**2)
    img_mask[invalid_pixels] = 0

    img_out = my_segmentation(img,img_mask,85)  # Seuil de 85 pour l'adaptation

    #Ouvrir l'image Verite Terrain en booleen
    img_GT =  np.asarray(Image.open(directory + corresponding_GT[image])).astype(np.bool_)
    ACCU.append(0)
    RECALL.append(0)
    ACCU[count], RECALL[count], img_out_skel, GT_skel = evaluate(img_out, img_GT)
    print('Results for image:', image)
    print('Accuracy =', ACCU[count],', Recall =', RECALL[count])

    count += 1

    plt.subplot(231)
    plt.imshow(img,cmap = 'gray')
    plt.title('Image Originale')
    plt.subplot(232)
    plt.imshow(img_out)
    plt.title('Segmentation')
    plt.subplot(233)
    plt.imshow(img_out_skel)
    plt.title('Segmentation squelette')
    plt.subplot(235)
    plt.imshow(img_GT)
    plt.title('Verite Terrain')
    plt.subplot(236)
    plt.imshow(GT_skel)
    plt.title('Verite Terrain Squelette')
    plt.show()

    print('------------------------------------------------------------')


print('\n------------------------------------------------------------')
print('------------------------------------------------------------')
print('Average Accuracy:', np.mean(ACCU))
print('Average Recall:', np.mean(RECALL))
print('------------------------------------------------------------')
print('------------------------------------------------------------')
      