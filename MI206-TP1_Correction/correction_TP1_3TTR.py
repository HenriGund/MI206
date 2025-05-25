import numpy as np
from skimage.morphology import erosion, dilation
from PIL import Image
from matplotlib import pyplot as plt

from skimage.util import img_as_ubyte
# Ouvrir et afficher une image en niveau de gris

# Definition des éléments structurants
se1 = np.array([[1, 0, 1]], dtype=np.bool_)
se2 = np.array([[1], [0], [1]], dtype=np.bool_)

img =  np.asarray(Image.open('./images/texte_bruit.png')).astype(np.bool_)

#Calcul TTR1 (origine à 0) et épaississement
imEro1 = erosion(img, se1)
imPts0Hisoles = imEro1 & ~img
img2 = img | imPts0Hisoles
plt.subplot(131)
plt.imshow(img,cmap = 'gray',vmin = 0.0,vmax = 1.0)
plt.title('Originale')
plt.subplot(132)
plt.imshow(imPts0Hisoles,cmap = 'gray',vmin = 0.0,vmax = 1.0)
plt.title('Points noirs H-isolés')
plt.subplot(133)
plt.imshow(img2,cmap = 'gray',vmin = 0.0,vmax = 1.0)
plt.title('Image filtrée 1')
plt.show()

#Calcul TTR2 (origine à 1) et amincissement
imEro2 = erosion(~img2, se2)
imPts1Visoles = imEro2 & img
img3 = img2 & ~imPts1Visoles
plt.subplot(131)
plt.imshow(img2,cmap = 'gray',vmin = 0.0,vmax = 1.0)
plt.title('Image filtrée 1')
plt.subplot(132)
plt.imshow(imPts1Visoles,cmap = 'gray',vmin = 0.0,vmax = 1.0)
plt.title('Points blancs V-isolés')
plt.subplot(133)
plt.imshow(img3,cmap = 'gray',vmin = 0.0,vmax = 1.0)
plt.title('Image filtrée 2')
plt.show()

#Etc....


