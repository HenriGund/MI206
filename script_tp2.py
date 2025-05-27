import numpy as np
from skimage.morphology import (binary_closing, erosion, dilation, binary_erosion, 
                               opening, closing, white_tophat, reconstruction, 
                               black_tophat, skeletonize, convex_hull_image, thin, 
                               disk, remove_small_objects, square, diamond, octagon, 
                               rectangle, star)
from skimage.filters.rank import entropy, enhance_contrast_percentile
from PIL import Image
from scipy import ndimage as ndi
from skimage.util import img_as_ubyte
import math
from skimage import data, filters
from matplotlib import pyplot as plt
from skimage.filters import threshold_otsu, gaussian

def my_segmentation(img, img_mask, seuil):
    """
    Multi-scale segmentation of retinal vessels using morphological operators
    and filtering specific to tubular structures.
    """
    
    # 1. Preprocessing - smoothing to reduce noise
    img_smooth = gaussian(img, sigma=1.7)
    
    # 2. Multi-scale approach with top-hat
    # Different scales to capture vessels of varying thickness
    scales = [1, 2, 3, 4, 5, 6, 7]  # Smaller scales for thin vessels, larger for thick ones
    multi_scale_result = np.zeros_like(img, dtype=np.float64)
    
    for scale in scales:
        # Circular structuring element for each scale
        se = disk(scale)
        
        # White top-hat enhances bright structures (vessels appear dark)
        # We use black top-hat for dark structures
        tophat = black_tophat(img_smooth, se)
        
        # Accumulate results with decreasing weight for larger scales
        weight = 1.0 / (1 + scale * 0.03)
        multi_scale_result += tophat * weight
    
    # 3. Normalization of the multi-scale result
    if multi_scale_result.max() > 0:
        multi_scale_result = multi_scale_result / multi_scale_result.max()
    
    # 4. Adaptive thresholding
    # Use percentile instead of Otsu for better control
    threshold_value = np.percentile(multi_scale_result[img_mask], 85)
    binary_vessels = multi_scale_result > threshold_value
    
    # 5. Morphological operations for cleanup using connected operators
    # Closing to connect nearby structures
    binary_vessels = binary_closing(binary_vessels, disk(1))
    
    # Opening to remove small isolated objects
    binary_vessels = opening(binary_vessels, disk(1.8))
    
    # 6. Remove small objects (noise)
    binary_vessels = remove_small_objects(binary_vessels, min_size=20)
    
    # 7. Refinement using morphological reconstruction to preserve connectivity
    # Erosion followed by reconstruction to retain only connected structures
    eroded = erosion(binary_vessels, disk(1))
    reconstructed = reconstruction(eroded, binary_vessels)
    
    # 8. Ensure both are boolean and apply retina mask
    reconstructed = reconstructed.astype(bool)
    img_mask = img_mask.astype(bool)
    final_result = reconstructed & img_mask
    
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

#Ouvrir l'image originale en niveau de gris
img =  np.asarray(Image.open('./images_IOSTAR/star01_OSC.jpg')).astype(np.uint8)
print(img.shape)

nrows, ncols = img.shape
row, col = np.ogrid[:nrows, :ncols]
#On ne considere que les pixels dans le disque inscrit 
img_mask = (np.ones(img.shape)).astype(np.bool_)
invalid_pixels = ((row - nrows/2)**2 + (col - ncols/2)**2 > (nrows / 2)**2)
img_mask[invalid_pixels] = 0

img_out = my_segmentation(img,img_mask,80)

#Ouvrir l'image Verite Terrain en booleen
img_GT =  np.asarray(Image.open('./images_IOSTAR/GT_01.png')).astype(np.bool_)

ACCU, RECALL, img_out_skel, GT_skel = evaluate(img_out, img_GT)
print('Accuracy =', ACCU,', Recall =', RECALL)

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

