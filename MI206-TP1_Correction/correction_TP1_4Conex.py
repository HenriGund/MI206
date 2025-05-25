import numpy as np
from skimage.morphology import erosion, dilation,binary_erosion, opening, closing, reconstruction
from skimage.morphology import square, diamond, octagon, rectangle, star, disk
from PIL import Image
from scipy import ndimage as ndi
from matplotlib import pyplot as plt

# Ouvrir une image binaire et conversion en tableau numpy
# au format booléen (codé sur 0 et 1)
imgBin =  np.asarray(Image.open('./images/particules.png')).astype(np.bool_)

# Ouverture par rayon
distance_map = np.asarray(ndi.distance_transform_cdt(imgBin,metric='chessboard')).astype(np.int16)
distance_cut = (distance_map > 10)
#distance_seed=np.asarray(distance).astype(np.uint8)
imgRec = reconstruction(distance_cut,np.asarray(imgBin))
# Affichage avec matplotlib
plt.subplot(141)
plt.imshow(imgBin,cmap = 'gray',vmin = 0.0,vmax = 1.0)
plt.title('Originale')
plt.subplot(142)
plt.imshow(distance_map,cmap = 'gray')
plt.title('Transformée en distance')
plt.subplot(143)
plt.imshow(distance_cut,cmap = 'gray',vmin = 0.0,vmax = 1.0)
plt.title('Erosion')
plt.subplot(144)
plt.imshow(imgRec,cmap = 'gray',vmin = 0.0,vmax = 1.0)
plt.title('Reconstruction')
plt.show()
# Fermer la fenêtre pour passer aux traitements suivants 

# Ouvrir une image binaire et conversion en tableau numpy
# au format booléen (codé sur 0 et 1)
img_ori =  np.asarray(Image.open('./images/coffee.png')).astype(np.bool_)
distance = np.asarray(ndi.distance_transform_cdt(img_ori,metric='chessboard')).astype(np.int16)
img_rec = reconstruction(distance,distance+1)
img_max = (img_rec != distance+1)
img_display = 127*img_ori + 128*img_max
# Affichage avec matplotlib
plt.subplot(131)
plt.imshow(img_ori,cmap = 'gray',vmin = 0.0,vmax = 1.0)
plt.title('Originale')
plt.subplot(132)
plt.imshow(distance,cmap = 'gray')
plt.title('Transformée en distance')
plt.subplot(133)
plt.imshow(img_display,cmap = 'gray',vmin = 0.0,vmax = 255.0)
plt.title('Erodés ultimes')
plt.show()
